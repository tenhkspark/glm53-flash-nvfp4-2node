#!/usr/bin/env python3
"""patch-mla.py — build the mla-quant overlay from the image's model.py.

The serve image's glm5next model.py builds Glm5NextMLAAttention with a
literal `quant_config=None` ("MLA projections are BF16 in checkpoint").
Every MLA/indexer child — fused_qkv_a_proj, q_b/kv_b/o_proj and
indexer.wq_b — is therefore constructed Unquantized and can never
resolve a quantized_layers key: a requantized member scale tensor has
no registered parameter and model.load_weights dies with a params_dict
KeyError (same failure class as the b4 KDA boot crash).

Route g declares those modules W4A16_NVFP4, so the overlay flips that
one argument to `quant_config` (the decoder layer already holds
`vllm_config.quant_config` in a local). Everything else is untouched:
undeclared children still resolve to UnquantizedLinearMethod, the
indexer's wk_weights_proj keeps its own hardcoded quant_config=None
(attention.py), and the vision tower's `quant_config=None` must
survive verbatim.

Fail closed: the anchor must match exactly once, the patched source
must compile, and the vision tower's None must still be present. Any
other shape exits non-zero so the runner never ships a broken overlay.

usage: patch-mla.py <image model.py> <out path>
"""
import sys

MARKER = "mla-quant-overlay"

# model.py @ glm53-flash-arm64-cu130, Glm5NextDecoderLayer.__init__ --
# the Glm5NextMLAAttention(...) call.
ANCHOR = """\
                quant_config=None,  # MLA projections are BF16 in checkpoint
"""

REPLACEMENT = """\
                # %s: was quant_config=None ("MLA projections are BF16 in
                # checkpoint"). Route g declares the MLA/indexer linears
                # W4A16 and needs the MIXED config to reach them; the
                # indexer's wk_weights_proj keeps its own hardcoded None.
                quant_config=quant_config,
""" % MARKER

REQUIRED = ("Glm5NextMLAAttention(", MARKER)


def patch(src: str) -> str:
    if MARKER in src:
        raise ValueError("input already carries the mla-quant patch")
    n = src.count(ANCHOR)
    if n != 1:
        raise ValueError(f"anchor matched {n} times (want 1) -- "
                         "image model.py drifted; re-derive the patch")
    out = src.replace(ANCHOR, REPLACEMENT, 1)
    for needle in REQUIRED:
        if needle not in out:
            raise ValueError(f"patched source lost {needle!r}")
    if "quant_config=None" not in out:
        # the vision tower's quant_config=None must survive untouched
        raise ValueError("patched source lost the vision tower's "
                         "quant_config=None")
    compile(out, "<mla-quant overlay>", "exec")
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src = open(sys.argv[1]).read()
    try:
        out = patch(src)
    except (ValueError, SyntaxError) as e:
        print(f"FAIL: mla patch refused: {e}")
        return 1
    with open(sys.argv[2], "w") as f:
        f.write(out)
    print(f"mla-quant overlay written: {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
