#!/usr/bin/env python3
"""Assemble a standalone BF16 MTP draft checkpoint from the stock
nvidia/GLM-5.3-Flash-NVFP4 checkpoint.

The stock checkpoint already stores the MTP layer (layers.45) in BF16,
so no dequantization is needed — the tensors are copied as-is. What the
script adds:

- Draft keys the checkpoint does not carry (the vLLM draft does not
  share embeddings with the target, draft_model.py:108-112):
      model.language_model.layers.45.embed_tokens.weight   <- embed_tokens
      model.language_model.layers.45.shared_head.head.weight <- lm_head
  The loader (glm5next/nvidia/mtp.py load_weights) strips the
  "model.language_model." prefix and maps embed->model.embed_tokens,
  head->shared_head.head.
- A flattened text-only config.json: text_config promoted to top level
  (Glm5NextMTP reads attributes off hf_config directly), quantization
  dropped (the mtp-bf16 overlay builds the draft with quant_config=None,
  so it must be BF16). model_type stays "glm5_next" — vLLM rewrites it
  to glm5_next_mtp for the draft (config/speculative.py:1029-1035).

Usage (each node needs the draft dir locally; build on one and rsync):
  build-mtp-draft.py --target /path/GLM-5.3-Flash-NVFP4 --out /path/draft
Expected output ends with: "OK: 891 tensors, 16.2 GiB, parts=...".
"""
import argparse
import json
import os
import shutil
import sys
import time

from safetensors.torch import save_file, safe_open

SPEC_LAYER = 45  # measured num_hidden_layers; index of the MTP layer
BUCKET_BYTES = 4 * 1024**3  # target shard size; tensors read one at a time
PREFIX = f"model.language_model.layers.{SPEC_LAYER}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="stock NVFP4 checkpoint dir")
    ap.add_argument("--out", required=True, help="output draft dir")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve keys and report; write nothing")
    args = ap.parse_args()

    t0 = time.time()
    idx = json.load(open(os.path.join(args.target, "model.safetensors.index.json")))
    wm = idx["weight_map"]

    layer45 = sorted(k for k in wm if k.startswith(PREFIX))
    embed_key = "model.language_model.embed_tokens.weight"
    head_key = "lm_head.weight"
    for k in (embed_key, head_key):
        if k not in wm:
            sys.exit(f"FAIL: {k} not in the index")
    # The draft needs a BF16 head. A requantized checkpoint (route h and
    # anything else that touches lm_head) stores it packed as uint8, and
    # copying that through would produce a draft that loads but is wrong.
    # Point --target at the stock NVFP4 checkpoint.
    with safe_open(os.path.join(src, wm[head_key]), framework="pt") as f:
        head_dtype = str(f.get_slice(head_key).get_dtype())
    if "BF16" not in head_dtype.upper() and "BFLOAT" not in head_dtype.upper():
        sys.exit(
            f"FAIL: {head_key} is {head_dtype}, not bfloat16. --target must be "
            "the stock NVFP4 checkpoint, not a requantized one: the draft "
            "needs an unpacked head."
        )
    rename = {
        embed_key: f"{PREFIX}embed_tokens.weight",
        head_key: f"{PREFIX}shared_head.head.weight",
    }
    needed = layer45 + [embed_key, head_key]
    n_expert = sum(1 for k in layer45 if ".mlp.experts." in k)
    print(f"layers.{SPEC_LAYER} keys={len(layer45)} (experts={n_expert}) "
          f"+ embed/head = {len(needed)}")

    # config.json: flatten text_config to top level, drop quantization
    cfg = json.load(open(os.path.join(args.target, "config.json")))
    text = cfg["text_config"]
    flat = {"architectures": cfg.get("architectures",
                                     ["Glm5NextForConditionalGeneration"]),
            "model_type": cfg["model_type"],
            "dtype": cfg.get("dtype", "bfloat16")}
    flat.update(text)
    flat["model_type"] = cfg["model_type"]  # text_config holds glm5_next_text;
    # the rewrite condition is "glm5_next", so restore it
    flat.pop("quantization_config", None)
    assert flat.get("num_nextn_predict_layers") == 1, \
        "unexpected num_nextn_predict_layers"
    os.makedirs(args.out, exist_ok=True)
    if not args.dry_run:
        with open(os.path.join(args.out, "config.json"), "w") as fh:
            json.dump(flat, fh, indent=2, sort_keys=True)
        for f in ("tokenizer.json", "tokenizer_config.json"):
            src = os.path.join(args.target, f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(args.out, f))

    # copy tensors into fresh shards (safe_open reads one at a time)
    by_file = {}
    for k in needed:
        by_file.setdefault(wm[k], []).append(k)
    out_index = {"metadata": {"total_size": 0}, "weight_map": {}}
    total_bytes = 0
    part_no, buf, buf_bytes = 0, {}, 0

    def flush():
        nonlocal part_no, buf, buf_bytes, total_bytes
        if not buf:
            return
        part_no += 1
        name = f"model-{part_no:05d}.safetensors"
        if not args.dry_run:
            save_file(buf, os.path.join(args.out, name),
                      metadata={"format": "pt"})
        for k in buf:
            out_index["weight_map"][k] = name
            out_index["metadata"]["total_size"] += \
                buf[k].numel() * buf[k].element_size()
        total_bytes += sum(t.numel() * t.element_size() for t in buf.values())
        buf, buf_bytes = {}, 0

    for f in sorted(by_file):
        if args.dry_run:
            continue
        with safe_open(os.path.join(args.target, f), framework="pt") as fh:
            for k in by_file[f]:
                t = fh.get_tensor(k)
                out_k = rename.get(k, k)
                buf[out_k] = t
                buf_bytes += t.numel() * t.element_size()
                if buf_bytes >= BUCKET_BYTES:
                    flush()
    flush()
    if not args.dry_run:
        with open(os.path.join(args.out,
                               "model.safetensors.index.json"), "w") as fh:
            json.dump(out_index, fh, indent=2)

    print(f"OK: {len(out_index['weight_map'])} tensors, "
          f"{total_bytes / 2**30:.2f} GiB, parts={part_no}, "
          f"{time.time() - t0:.0f}s, out={args.out} (dry-run={args.dry_run})")


if __name__ == "__main__":
    main()
