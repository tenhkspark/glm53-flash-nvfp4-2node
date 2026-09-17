#!/usr/bin/env python3
"""Requantize the BF16 dense linears of nvidia/GLM-5.3-Flash-NVFP4.

Weight-only, tensor-by-tensor repack into the on-disk formats the vLLM
v0.29 modelopt loader expects (modelopt.py). No calibration forward pass:
all chosen algos are weight-only (W4A16_NVFP4, FP8_PB_WO) or need only
weight-derived scales, so no activation scales are required at all.

Targets:
  a  shared_experts.{gate,up,down}_proj -> FP8_PB_WO (fp8 weight + f32
     scale per 128x128 block, dynamic per-token activation quant at
     runtime); lm_head -> W4A16_NVFP4 (see below: PB_WO cannot load on a
     vocab-parallel head); all of self_attn stays BF16 (see KDA_IN_FUSED
     -- no FP8-expressible fused group exists there); mlp.gate stays
     BF16 -- FP8 would perturb expert selection
  b  KDA in_proj (q/k/v/b/f_a/g_a) + shared_experts + lm_head ->
     W4A16_NVFP4 (packed fp4 weight, fp8 e4m3 per-16 group scale, f32
     global scale shared across each fused group); everything else stays
     BF16
  c  KDA fused in_proj only -> W4A16_NVFP4; shared_experts and lm_head
     stay BF16 exactly as NVIDIA shipped (isolates the fused-KDA delta
     on otherwise-stock precision). NOTE: redefined 2026-09-15f -- c was
     previously "in_proj+lm_head W4A16 + shared PB_WO"
  d  KDA in_proj + lm_head -> FP8 (static per-tensor, algo "FP8"; no
     new NVFP4 modules). Scalar weight_scale/input_scale shard
     element-wise, so fused members of any width and the vocab head
     both load. input_scale is a calibrated proxy: 6x the layer's NVFP4
     expert input_scale (amax/2688 -> amax/448; the loader takes .max()
     over fused shards), global max for lm_head
  e  lm_head only -> W4A16_NVFP4 (last-mile-only arm)
  g  c + every remaining attention-side linear -> W4A16_NVFP4: KDA
     out projections (f_b/g_b/o_proj), MLA q/kv/o (the fused_qkv_a_proj
     members q_a/kv_a plus q_b/kv_b/o_proj) and indexer.wq_b. NOT the
     router, norms, embed, shared experts, lm_head, or the indexer's
     wk_weights_proj (its ctor hardcodes quant_config=None). Bootable
     only with the mla-quant overlay (patch-mla.py): the image builds
     Glm5NextMLAAttention with quant_config=None.
  h  g + the remaining BF16 dense linears -> W4A16_NVFP4:
     shared_experts (gate/up/down members of the fused gate_up trio +
     down_proj) and lm_head (route e's algo -- PB_WO cannot shard the
     vocab-parallel head, see the note below). Router/gate, norms and
     embed stay BF16. Same overlays as g (kda-quant + mla-quant);
     lm_head needs none -- its per-row GroupQuantScale and scalar
     PerTensorScale shard exactly like vocab rows.

lm_head note: FP8_PB_WO cannot load under TP>1 -- the scale param is a
BlockQuantScaleParameter routed through VocabParallelEmbedding.weight_
loader, which asserts loaded.shape[0] == org_vocab_size (block rows can
never match vocab rows). FP8_PER_CHANNEL_PER_TOKEN would shard fine but
is not dispatched under MIXED_PRECISION (modelopt.py get_quant_method
covers FP8, FP8_PB_WO, NVFP4, W4A16_NVFP4, MXFP8 only). W4A16_NVFP4 is
the only weight-only algo that loads: its per-row GroupQuantScale and
scalar PerTensorScale shard exactly like vocab rows.

Output: a NEW checkpoint dir, all tensors rewritten into fresh
model-NNNNN-of-MMMMM.safetensors shards (non-target tensors byte-copied),
plus rewritten model.safetensors.index.json / config.json /
hf_quant_config.json (quant_algo=MIXED_PRECISION + quantized_layers), and
copied tokenizer/processor files. Source is opened read-only.

Usage:
  requant.py --target a|b|c|d|e|g|h --src SRC --dst DST
  requant.py --target a --src SRC --dst DST --dry-run
  requant.py --target a --src SRC --dst DST --selfcheck-only
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

FP8_MAX = 448.0
FP4_MAX = 6.0
NVFP4_GS = 16
PB_BLOCK = 128          # FP8_PB_WO weight block (both dims)
SHARD_BUDGET = 6 << 30  # bytes per output shard

# Checkpoint-side linear suffixes (under self_attn / mlp).
KDA_IN = ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj")
KDA_OUT = ("f_b_proj", "g_b_proj", "o_proj")
DSA = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj",
       "indexer.wq_b", "indexer.wk", "indexer.weights_proj")
SHARED = ("mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj",
          "mlp.shared_experts.down_proj")
# vLLM-side fused name the KDA_IN members load into (serve image model.py
# stacked_params_mapping). The MIXED resolver cannot unfuse member
# quantized_layers keys back to it: Glm5Next* carries no
# packed_modules_mapping and the resolver's fused_projection_shards lists
# only qkv_proj/gate_up_proj. The group therefore quantizes only under
# its FUSED name, and only as W4A16_NVFP4: PB_WO block scales cannot
# shard the sub-128-row members (b=64, f_a/g_a=128 rows at TP=2), and a
# PB_WO member tensor under an unresolvable parent dies as KeyError on
# ...in_proj_qkvbfg_a.weight_scale. 2026-09-15: a2 crashed at boot on
# PB_WO member keys here; b2 crashed the same way on W4A16 member keys.
KDA_IN_FUSED = "self_attn.in_proj_qkvbfg_a"
# DSA-layer self_attn is unquantizable on the STOCK image: model.py
# builds Glm5NextMLAAttention with quant_config=None and its fp8
# projections (q_a/kv_a/q_b/o_proj, indexer.wk) are DeepSeek
# weight+weight_scale_inv pairs the loader dequantizes to BF16
# (_try_load_fp8_attn_proj / _try_load_fp8_indexer_wk). A modelopt-scale
# member tensor there either never reaches a param (swallowed by the
# dequant buffer) or hits params_dict -> KeyError. Route g lifts this
# with the mla-quant overlay (patch-mla.py): one quant_config argument
# flipped to pass the MIXED config into the MLA ctor.
# DSA member suffixes under self_attn: fused_qkv_a_proj is the
# vLLM-side fused name for the q_a/kv_a members -- like in_proj_qkvbfg_a
# it must be declared under the fused name (no packed_modules_mapping
# to unfuse). indexer.wk/weights_proj stay BF16: the indexer fuses them
# into wk_weights_proj, built with a hardcoded quant_config=None inside
# Indexer.__init__ -- declared keys there could never attach.
DSA_FUSED = "self_attn.fused_qkv_a_proj"
DSA_IN = ("q_a_proj", "kv_a_proj_with_mqa")
DSA_FREE = ("q_b_proj", "kv_b_proj", "o_proj", "indexer.wq_b")
# Already-NVFP4 modules kept in quantized_layers so MIXED_PRECISION still
# resolves them to NVFP4.
MOE_EXPERTS = "mlp.experts"          # parent key (RoutedExperts prefix)
DENSE_MLP = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")

# Scale buckets: (canonical module suffix, ckpt member suffixes). All
# members of one bucket are quantized with a single shared global scale
# (weight_scale_2) -- they load into one fused vLLM param (in_proj
# sextet, fused_qkv_a pair, shared_experts trio) whose loader collapses
# per-shard scales to .max(). Standalone modules (route g's
# o_proj/f_b/g_b/q_b/kv_b, indexer.wq_b, lm_head) are in no group and
# keep their own amax.
FUSE_GROUPS = [
    ("mlp.shared_experts", SHARED),
    ("self_attn.in_proj_qkvbfg_a", KDA_IN),
    ("self_attn.fused_qkv_a_proj", DSA_IN),
]


def _scale_group(mod):
    """convert-member module -> the scale bucket it is quantized under."""
    for canon, members in FUSE_GROUPS:
        for s in members:
            sfx = "." + s
            if mod.endswith(sfx):
                return mod[: -len(sfx)] + "." + canon
    return mod

# The only weight-only algos that can actually load on glm5next (v0.29,
# MIXED dispatch + the model's fused/dequant-intercept layout):
#   FP8_PB_WO     -- standalone or gate_up members with all dims
#                    128*TP-aligned (shared_experts trio)
#   W4A16_NVFP4   -- everything else that is requantized: its group/global
#                    scales shard element-wise, so fused members of any
#                    width load (in_proj sextet), and ParallelLMHead's
#                    vocab loader accepts its per-row scale
#   FP8           -- static per-tensor (route d): scalar weight_scale +
#                    input_scale shard element-wise / land via shard_id,
#                    and on the vocab head take the loader's
#                    output_dim=None scalar branch, so fused members and
#                    lm_head both load. Needs an input_scale that cannot
#                    calibrate -> expert-proxy value (FP8_INS_FALLBACK)
W4A16 = "W4A16_NVFP4"

# Per-target algo assignments (2026-09-15f: c/d/e added for the
# speed/quality sweep; c redefined to in_proj-only; 2026-09-16: h = g +
# shared_experts + lm_head).
KDA_ALGO = {"b": W4A16, "c": W4A16, "d": "FP8", "g": W4A16, "h": W4A16}
# Route g/h: the remaining attention-side linears (KDA out projections,
# DSA/MLA fused+standalone, indexer.wq_b).
ATTN_ALGO = {"g": W4A16, "h": W4A16}
SHARED_ALGO = {"a": "FP8_PB_WO", "b": W4A16, "h": W4A16}
HEAD_ALGO = {"a": W4A16, "b": W4A16, "d": "FP8", "e": W4A16, "h": W4A16}
# FP8 static needs an input_scale (activation amax/448). No calibration
# pass exists, so route d borrows the layer's calibrated NVFP4 expert
# input_scale (amax/2688) x6; fallback 1.0 when no expert scale exists.
FP8_INS_FALLBACK = 1.0

LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")

_E2M1_VALS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MID = torch.tensor(
    [(a + b) / 2 for a, b in zip(_E2M1_VALS, _E2M1_VALS[1:])])


# ---------------------------------------------------------------- plan ---

def classify_layers(weight_map):
    """layer -> 'kda' | 'dsa' from which self_attn weights exist; MoE set."""
    kinds, moe = {}, set()
    for name in weight_map:
        m = LAYER_RE.match(name)
        if not m:
            continue
        layer, sub = int(m.group(1)), m.group(2)
        if sub == "self_attn.q_proj.weight":
            kinds.setdefault(layer, "kda")
        elif sub == "self_attn.q_a_proj.weight":
            kinds.setdefault(layer, "dsa")
        elif sub.startswith("mlp.experts."):
            moe.add(layer)
    return kinds, moe


def build_plan(index, num_layers, target):
    """(qlayers, keep, convert): three module-prefix -> algo maps.

    qlayers/keep are written into quantized_layers: keys are the names
    the MIXED resolver sees -- member names for gate_up_proj (the
    resolver unfuses them via its hardcoded fused_projection_shards) and
    standalone modules, but the FUSED name for KDA in_proj (member keys
    there can never resolve). convert lists the ckpt member tensors to
    actually rewrite; fused members load into the fused param by
    shard_id, so member names are what must exist on disk.
    """
    kinds, moe = classify_layers(index["weight_map"])
    algo_of, keep, convert = {}, {}, {}

    def put(layer, suffix, algo):
        key = f"model.language_model.layers.{layer}.{suffix}"
        algo_of[key] = algo
        convert[key] = algo

    for layer in sorted(kinds):
        if layer >= num_layers:
            continue  # MTP draft layer 45: leave BF16
        if kinds[layer] == "kda" and target in KDA_ALGO:
            fused = f"model.language_model.layers.{layer}.{KDA_IN_FUSED}"
            algo_of[fused] = KDA_ALGO[target]
            for s in KDA_IN:
                convert[f"model.language_model.layers.{layer}."
                        f"self_attn.{s}"] = KDA_ALGO[target]
            if target in ATTN_ALGO:
                for s in KDA_OUT:
                    put(layer, f"self_attn.{s}", ATTN_ALGO[target])
        elif kinds[layer] == "dsa" and target in ATTN_ALGO:
            base = f"model.language_model.layers.{layer}.self_attn"
            algo_of[f"model.language_model.layers.{layer}.{DSA_FUSED}"] = \
                ATTN_ALGO[target]
            for s in DSA_IN:
                convert[f"{base}.{s}"] = ATTN_ALGO[target]
            for s in DSA_FREE:
                put(layer, f"self_attn.{s}", ATTN_ALGO[target])
        if layer in moe:
            if target in SHARED_ALGO:
                for s in SHARED:
                    put(layer, s, SHARED_ALGO[target])
            keep[f"model.language_model.layers.{layer}.{MOE_EXPERTS}"] = \
                "NVFP4"
        else:
            for s in DENSE_MLP:
                keep[f"model.language_model.layers.{layer}.{s}"] = "NVFP4"

    if target in HEAD_ALGO:
        algo_of["lm_head"] = HEAD_ALGO[target]
        convert["lm_head"] = HEAD_ALGO[target]
    return algo_of, keep, convert


# --------------------------------------------------------- quantizers ---

def e2m1_codes(x):
    """float -> uint8 E2M1 codes, round-nearest-even on exact midpoints."""
    xa = x.abs()
    idx = torch.bucketize(xa, _E2M1_MID, right=True)  # ties -> lower index
    mid = _E2M1_MID[(idx - 1).clamp(min=0)]
    # right=True puts an exact midpoint at idx=i+1 (i = midpoint index).
    # RNE wants code i when i is even, code i+1 when odd; idx odd <=> i even,
    # so an odd idx on a tie steps back one to land on the even code.
    tie = (idx > 0) & (xa == mid) & (idx % 2 == 1)
    idx = idx - tie.long()
    sign = (x < 0).to(torch.uint8) << 3
    return (sign | idx.to(torch.uint8))


def e2m1_values(codes):
    """uint8 E2M1 codes -> float tensor."""
    vals = torch.tensor(_E2M1_VALS, dtype=torch.float32)
    mag = vals[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -mag, mag)


def quant_nvfp4(w, gs=None, group_size=NVFP4_GS):
    """BF16/float [n,k] -> (packed U8 [n,k/2], F8 [n,k/16], F32 scalar).

    `gs` overrides the global scale (weight_scale_2 = amax/2688) so fused
    shard groups can share one value.
    """
    n, k = w.shape
    assert k % group_size == 0, f"k={k} not divisible by {group_size}"
    w = w.float()
    if gs is None:
        gs = w.abs().max().clamp_min(1e-12) / (FP4_MAX * FP8_MAX)
    g = w.view(n, k // group_size, group_size)
    bmax = g.abs().amax(-1)
    # clamp_min keeps all-zero blocks representable: without it bs==0 makes
    # g/(bs*gs) produce NaN codes instead of cleanly quantized zeros.
    bs = (bmax / (FP4_MAX * gs)).clamp(min=2**-9, max=FP8_MAX) \
        .to(torch.float8_e4m3fn)
    wq = (g / (bs.float().unsqueeze(-1) * gs)).clamp(-FP4_MAX, FP4_MAX)
    codes = e2m1_codes(wq.reshape(n, k))
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    if not torch.is_tensor(gs):
        gs = torch.tensor(float(gs), dtype=torch.float32)
    # clone: a group-shared gs tensor is handed to every shard of the fused
    # parent; safetensors rejects tensors sharing memory in one save_file.
    return packed, bs.view(n, k // group_size).contiguous(), \
        gs.float().reshape(()).clone()


def dequant_nvfp4(packed, bs, gs):
    """(U8 [n,k/2], F8 [n,k/16], F32) -> float [n,k]. Even k = low nibble."""
    n, k2 = packed.shape
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).view(n, k2 * 2)
    v = e2m1_values(codes).view(n, k2 * 2 // NVFP4_GS, NVFP4_GS)
    return (v * bs.float().unsqueeze(-1) * gs).view(n, k2 * 2)


def quant_fp8_pbwo(w, block=PB_BLOCK):
    """BF16 [n,k] -> (F8 [n,k], F32 [out_blk,1,in_blk,1])."""
    n, k = w.shape
    assert k % block == 0, f"k={k} not divisible by {block}"
    ob = -(-n // block)
    w = w.float()
    wp = torch.zeros(ob * block, k)
    wp[:n] = w
    g = wp.view(ob, block, k // block, block)
    amax = g.abs().amax(dim=(1, 3)).clamp_min(1e-12)
    sc = (amax / FP8_MAX).view(ob, 1, k // block, 1)
    wq = (g / sc.squeeze(1).squeeze(-1)[:, None, :, None]) \
        .clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return wq.view(ob * block, k)[:n].contiguous(), sc


def dequant_fp8_pbwo(wq, sc, n, block=PB_BLOCK):
    """(F8 [n,k], F32 [ob,1,ib,1]) -> float [n,k]."""
    ob, _, ib, _ = sc.shape
    g = wq.float()
    if g.shape[0] < ob * block:
        g = torch.nn.functional.pad(g, (0, 0, 0, ob * block - g.shape[0]))
    g = g.view(ob, block, ib, block)
    return (g * sc.squeeze(1).squeeze(-1)[:, None, :, None]) \
        .view(ob * block, ib * block)[:n]


def quant_fp8_pertensor(w, ws=None):
    """BF16 [n,k] -> (F8 [n,k], F32 scalar weight_scale=amax/448).

    `ws` overrides the scale so all members of a fused group share one
    value -- the loader otherwise requantizes per-shard scales to the
    group max at boot (requantize_with_max_scale).
    """
    if ws is None:
        ws = w.float().abs().max().clamp_min(1e-12) / FP8_MAX
    wq = (w.float() / ws).clamp(-FP8_MAX, FP8_MAX) \
        .to(torch.float8_e4m3fn)
    if not torch.is_tensor(ws):
        ws = torch.tensor(float(ws), dtype=torch.float32)
    return wq.contiguous(), ws.float().reshape(()).clone()


def dequant_fp8_pertensor(wq, s):
    """(F8 [n,k], F32 scalar) -> float [n,k]."""
    return wq.float() * float(s)


def quantize_module(w, algo, group_amax=None, input_scale=None):
    """BF16 weight -> {suffix: tensor} on-disk set for one module."""
    if algo == "W4A16_NVFP4" or algo == "NVFP4":
        gs = group_amax / (FP4_MAX * FP8_MAX) \
            if group_amax is not None else None
        packed, bscale, s2 = quant_nvfp4(w, gs=gs)
        return {"weight": packed, "weight_scale": bscale,
                "weight_scale_2": s2}
    if algo == "FP8_PB_WO":
        wq, sc = quant_fp8_pbwo(w)
        return {"weight": wq, "weight_scale": sc}
    if algo == "FP8":
        ws = group_amax / FP8_MAX if group_amax is not None else None
        wq, s = quant_fp8_pertensor(w, ws=ws)
        ins = FP8_INS_FALLBACK if input_scale is None else input_scale
        return {"weight": wq, "weight_scale": s,
                "input_scale": torch.tensor(float(ins),
                                            dtype=torch.float32)
                .reshape(())}
    raise ValueError(f"unknown algo {algo}")


def expert_input_scales(src, weight_map):
    """layer -> max calibrated NVFP4 expert gate_proj input_scale on disk.

    The shipped experts are W4A4, so their input_scale = act_amax/2688 is
    a real calibrated value for the same hidden stream the KDA in_proj
    reads (post-layernorm activations). Returns (per_layer, global_max).
    """
    by_shard = {}
    for name, shard in weight_map.items():
        if name.endswith(".input_scale") and ".mlp.experts." in name \
                and name.split(".")[-2] == "gate_proj":
            m = LAYER_RE.match(name)
            if m:
                by_shard.setdefault(shard, []).append(
                    (int(m.group(1)), name))
    per_layer, gmax = {}, 0.0
    for shard, items in by_shard.items():
        with safe_open(os.path.join(src, shard), framework="pt") as f:
            for layer, name in items:
                v = float(f.get_tensor(name))
                per_layer[layer] = max(per_layer.get(layer, 0.0), v)
                gmax = max(gmax, v)
    return per_layer, gmax


# ------------------------------------------------------------ configs ---

def ignore_after(ignore, covered):
    """Drop exclude patterns that cover any module now quantized.

    Mirrors the loader's substring+wildcard exclusion semantics: a pattern
    like `...self_attn*` or `...self_attn` matches every converted child,
    so it must go entirely; leftover unconverted children fall back to
    UnquantizedLinearMethod and keep their BF16 weights.
    """
    out = []
    for p in ignore:
        hit = any(fnmatch.fnmatch(m, p) or p in m for m in covered)
        if not hit:
            out.append(p)
    mtp = "model.language_model.layers.45*"
    if mtp not in out and not any(fnmatch.fnmatch(mtp, p) for p in out):
        out.append(mtp)
    return out


def write_configs(src, dst, algo_of, keep, orig_ignore, covered, target):
    """Write config.json + hf_quant_config.json into dst."""
    ignore = ignore_after(orig_ignore, covered | set(algo_of))
    merged = dict(keep)
    merged.update(algo_of)
    qlayers = {m: {"quant_algo": a, "group_size": NVFP4_GS}
               for m, a in sorted(merged.items())}

    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    qc = cfg.get("quantization_config")
    if qc:
        qc["quant_algo"] = "MIXED_PRECISION"
        qc["ignore"] = ignore
        qc["quantized_layers"] = qlayers
        prod = qc.setdefault("producer", {})
        prod["requant"] = "requant.py (weight-only NVFP4 quantization)"
        # guards against reuse of a stale output directory across target
        # redefinitions (a stale dir could still pass `verify.py config`
        # under the new plan)
        prod["requant_target"] = target
    _atomic_write_json(os.path.join(dst, "config.json"), cfg)

    hq_path = os.path.join(src, "hf_quant_config.json")
    if os.path.exists(hq_path):
        with open(hq_path) as f:
            hq = json.load(f)
    else:
        hq = {"producer": {"name": "modelopt"}, "quantization": {}}
    q = hq.setdefault("quantization", {})
    q["quant_algo"] = "MIXED_PRECISION"
    q["kv_cache_quant_algo"] = q.get("kv_cache_quant_algo", "FP8")
    q["group_size"] = NVFP4_GS
    q["exclude_modules"] = ignore
    q["quantized_layers"] = qlayers
    _atomic_write_json(os.path.join(dst, "hf_quant_config.json"), hq)
    return ignore


def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


# ------------------------------------------------------------- driver ---

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", choices=("a", "b", "c", "d", "e", "g", "h"),
                    required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selfcheck-only", action="store_true")
    args = ap.parse_args()

    with open(os.path.join(args.src, "model.safetensors.index.json")) as f:
        index = json.load(f)
    with open(os.path.join(args.src, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    num_layers = int(tc.get("num_hidden_layers", 45))
    hq_path = os.path.join(args.src, "hf_quant_config.json")
    orig_ignore = []
    if os.path.exists(hq_path):
        orig_ignore = json.load(open(hq_path))["quantization"] \
            .get("exclude_modules", [])
    else:
        orig_ignore = (cfg.get("quantization_config") or {}).get("ignore", [])

    if os.path.realpath(args.src) == os.path.realpath(args.dst):
        raise SystemExit("src and dst resolve to the same directory")

    algo_of, keep, convert = build_plan(index, num_layers, args.target)
    weight_map = index["weight_map"]
    covered = {m for m in convert
               if m + ".weight" in weight_map}

    # Every planned member tensor must exist on disk; an absent one means
    # the name list drifted from the real checkpoint and that tensor would
    # silently stay BF16 while quantized_layers claims it is quantized.
    missing = sorted(set(convert) - covered)
    if missing:
        for m in missing:
            print(f"  MISSING {convert[m]:<13} {m}", file=sys.stderr)
        raise SystemExit(f"plan names {len(missing)} module(s) absent from "
                         "the checkpoint index; aborting")

    # keep entries are parent keys (e.g. mlp.experts) or direct modules;
    # require at least one weight_map entry below each.
    keep_missing = [m for m in keep
                    if not any(k == m + ".weight" or k.startswith(m + ".")
                               for k in weight_map)]
    if keep_missing:
        for m in keep_missing:
            print(f"  MISSING-KEEP {keep[m]:<13} {m}", file=sys.stderr)
        raise SystemExit(f"{len(keep_missing)} keep-entries absent from "
                         "the checkpoint index; aborting")

    by_algo = {}
    for a in convert.values():
        by_algo[a] = by_algo.get(a, 0) + 1
    print(f"target={args.target} qlayers={len(algo_of)} members="
          f"{len(convert)} (in ckpt: {len(covered)}) algos={by_algo}")
    for m in sorted(algo_of):
        print(f"  {algo_of[m]:<13} {m}")

    if args.dry_run:
        print(f"dry-run done: {len(covered)} tensors would be repacked")
        return

    # group amax per scale bucket (FUSE_GROUPS). The loader collapses a
    # fused linear's per-shard global scales to .max() (Marlin
    # weight_scale_2 without rescaling; FP8 weight_scale with
    # requantize), so all W4A16/FP8 members of a fused group (self_attn
    # in_proj sextet, MLA fused_qkv_a pair, shared_experts trio) share
    # one global scale. Standalone modules (route g's o_proj/f_b/g_b/
    # q_b/kv_b/wq_b, lm_head) keep their own amax.
    group_amax = {}
    parents = {}
    for m in convert:
        if convert[m] not in ("W4A16_NVFP4", "FP8") \
                or m + ".weight" not in weight_map:
            continue
        parents.setdefault(_scale_group(m), []).append(m)
    for mods in parents.values():
        amax = 0.0
        for m in mods:
            with safe_open(os.path.join(args.src,
                                        weight_map[m + ".weight"]),
                           framework="pt") as f:
                amax = max(amax, float(
                    f.get_tensor(m + ".weight").abs().max()))
        for m in mods:
            group_amax[m] = amax

    # target d input_scale: 6x the calibrated NVFP4 expert input_scale
    # (amax/2688 -> amax/448). Per-layer max for layer modules; the
    # global max for lm_head (deepest hidden state) and for layers with
    # no experts of their own. One shared value per fused group -- the
    # input tensor is shared and the loader takes .max() anyway.
    input_scales = {}
    if args.target == "d":
        per_layer, gmax = expert_input_scales(args.src, weight_map)
        if gmax <= 0:
            print("WARN: no expert input_scale found; FP8 input_scale "
                  f"falls back to {FP8_INS_FALLBACK}", file=sys.stderr)
        for m in convert:
            if convert[m] != "FP8":
                continue
            lm = LAYER_RE.match(m)
            base = per_layer.get(int(lm.group(1)), gmax) if lm else gmax
            input_scales[m] = 6.0 * base if base > 0 else FP8_INS_FALLBACK

    if args.selfcheck_only:
        import tempfile
        n_ok = rel_worst = 0
        n_bytes = 0
        for m in sorted(covered):
            with safe_open(os.path.join(args.src, weight_map[m + ".weight"]),
                           framework="pt") as f:
                w = f.get_tensor(m + ".weight")
            t = quantize_module(w, convert[m], group_amax.get(m),
                                input_scales.get(m))
            if convert[m] == "FP8_PB_WO":
                d = dequant_fp8_pbwo(t["weight"], t["weight_scale"],
                                     w.shape[0])
            elif convert[m] == "FP8":
                d = dequant_fp8_pertensor(t["weight"], t["weight_scale"])
            else:
                d = dequant_nvfp4(t["weight"], t["weight_scale"],
                                  t["weight_scale_2"])
            rel = float((d - w.float()).abs().max() /
                        w.float().abs().max().clamp_min(1e-9))
            rel_worst = max(rel_worst, rel)
            # serialization round-trip: bytes must survive save/load
            with tempfile.NamedTemporaryFile(
                    suffix=".safetensors", delete=False) as tf:
                tmp = tf.name
            save_file(t, tmp)
            back = load_file(tmp)
            os.unlink(tmp)
            for k in t:
                n_bytes += t[k].numel() * t[k].element_size()
                if not torch.equal(back[k], t[k]):
                    raise SystemExit(f"selfcheck FAIL: {m}.{k} round-trip "
                                     "mismatch")
            n_ok += 1
        print(f"selfcheck done: {n_ok}/{len(covered)} modules quantized, "
              f"worst rel-err {rel_worst:.4f}, byte-match "
              f"({n_bytes} B verified)")
        return

    # ---- full conversion: rewrite every tensor into fresh shards
    os.makedirs(args.dst, exist_ok=True)
    buf, buf_bytes, shard_idx = {}, 0, 0
    new_map, tmp_names = {}, []
    total_bytes = 0

    def emit(name, t):
        nonlocal buf_bytes, total_bytes
        buf[name] = t
        n = t.numel() * t.element_size()
        buf_bytes += n
        total_bytes += n

    def flush():
        nonlocal buf, buf_bytes, shard_idx
        if not buf:
            return
        shard_idx += 1
        tmp = os.path.join(args.dst, f".shard-{shard_idx:05d}.safetensors")
        save_file(buf, tmp)
        tmp_names.append(tmp)
        for name in buf:
            new_map[name] = os.path.basename(tmp)
        buf, buf_bytes = {}, 0

    for shard in sorted(set(weight_map.values())):
        with safe_open(os.path.join(args.src, shard), framework="pt") as f:
            for name in f.keys():
                mod = name[:-len(".weight")] \
                    if name.endswith(".weight") else None
                if mod in covered:
                    for sfx, t in quantize_module(
                            f.get_tensor(name), convert[mod],
                            group_amax.get(mod),
                            input_scales.get(mod)).items():
                        emit(mod + "." + sfx, t)
                else:
                    emit(name, f.get_tensor(name))
        if buf_bytes > SHARD_BUDGET:
            flush()
        print(f"[requant] {shard} done", file=sys.stderr)
    flush()

    total = len(tmp_names)
    for i, tmp in enumerate(tmp_names, 1):
        final = f"model-{i:05d}-of-{total:05d}.safetensors"
        os.replace(tmp, os.path.join(args.dst, final))
        for n in list(new_map):
            if new_map[n] == os.path.basename(tmp):
                new_map[n] = final
    out_idx = {"metadata": {"total_size": total_bytes},
               "weight_map": new_map}
    _atomic_write_json(
        os.path.join(args.dst, "model.safetensors.index.json"), out_idx)
    write_configs(args.src, args.dst, algo_of, keep, orig_ignore,
                  covered, args.target)
    for entry in os.listdir(args.src):
        s = os.path.join(args.src, entry)
        if not os.path.isfile(s):
            continue
        if entry.endswith(".safetensors") or entry in (
                "model.safetensors.index.json", "config.json",
                "hf_quant_config.json"):
            continue
        d = os.path.join(args.dst, entry)
        if not os.path.exists(d):
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)
    print("conversion done")


if __name__ == "__main__":
    main()
