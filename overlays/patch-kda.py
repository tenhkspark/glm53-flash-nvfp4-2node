#!/usr/bin/env python3
"""patch-kda.py — build the kda-quant overlay from the image's own kda.py.

The serve image's glm5next kda.py nulls vllm_config.quant_config around
GatedDeltaNetAttention.__init__ ("KDA projections remain BF16 because fp8
checkpoints omit their scales"). That call is where the superclass
snapshots `self.quant_config = vllm_config.quant_config`, so every KDA
projection — including the fused `in_proj_qkvbfg_a` — is constructed with
quant_config=None and can never resolve a `quantized_layers` entry. A
requantized member scale tensor (weight_scale / weight_scale_2) then has
no registered parameter and model.load_weights dies with
`KeyError: '...in_proj_qkvbfg_a.weight_scale_2'` (2026-09-15 b4 boot).

The patch leaves the BF16 hack in place (conv1d/A_log/dt_bias stay
unquantized either way — none of them take a quant_config) and restores
`self.quant_config` after the finally block. Children not named in
quantized_layers still resolve to UnquantizedLinearMethod, so stock /
route-a/e behaviour is identical; declared members get their method.

Fail closed: the anchor must match exactly once, the patched source must
compile, and the fused-constructor call sites must still be present.
Any other shape exits non-zero so a broken overlay is never shipped.

usage: patch-kda.py <image kda.py> <out path>
"""
import sys

MARKER = "kda-quant-overlay"

# kda.py @ v0.28.1rc1.dev580+g385dce36b, Glm5NextLinearAttention.__init__
ANCHOR = """\
        # KDA projections remain BF16 because fp8 checkpoints omit their scales.
        saved_quant_config = vllm_config.quant_config
        try:
            vllm_config.quant_config = None
            super().__init__(config, vllm_config, prefix)
        finally:
            vllm_config.quant_config = saved_quant_config
"""

REPLACEMENT = ANCHOR + """\
        # %s: the hack above also left self.quant_config=None -- it is
        # snapshotted inside GatedDeltaNetAttention.__init__ while
        # vllm_config.quant_config was temporarily None. Without this the
        # fused in_proj_qkvbfg_a is built UnquantizedLinearMethod and a
        # declared quantized_layers entry can never attach (member scale
        # tensors then KeyError in params_dict at load). Restoring it is
        # safe: undeclared children still resolve to unquantized.
        self.quant_config = saved_quant_config
""" % MARKER

REQUIRED = ("in_proj_qkvbfg_a", "_Glm5NextMergedColumnParallelLinear",
            "self.quant_config = saved_quant_config")


def patch(src: str) -> str:
    if MARKER in src:
        raise ValueError("input already carries the kda-quant patch")
    n = src.count(ANCHOR)
    if n != 1:
        raise ValueError(f"anchor matched {n} times (want 1) -- "
                         "image kda.py drifted; re-derive the patch")
    out = src.replace(ANCHOR, REPLACEMENT, 1)
    for needle in REQUIRED:
        if needle not in out:
            raise ValueError(f"patched source lost {needle!r}")
    compile(out, "<kda-quant overlay>", "exec")
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src = open(sys.argv[1]).read()
    try:
        out = patch(src)
    except (ValueError, SyntaxError) as e:
        print(f"FAIL: kda patch refused: {e}")
        return 1
    with open(sys.argv[2], "w") as f:
        f.write(out)
    print(f"kda-quant overlay written: {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
