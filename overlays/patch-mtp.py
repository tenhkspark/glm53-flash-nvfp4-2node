#!/usr/bin/env python3
"""patch-mtp.py — build the mtp-bf16 overlay from the image's own mtp.py.

The official MTP draft cannot boot on the NVFP4 checkpoint: the draft
vllm_config inherits the target's modelopt_fp4 quant_config
(vllm/v1/spec_decode/llm_base_proposer.py _create_draft_vllm_config never
clears it), so RoutedExperts allocates NVFP4-packed expert buffers while
the layer-45 MTP weights are plain BF16 -- the copy dies with
"The size of tensor a (1024) must match the size of tensor b (2048)".
The draft-model proposer already does this fix for draft-model methods
(vllm/v1/spec_decode/draft_model.py sets quant_config=None); the MTP
path lacks it.

The patch inserts a _DraftBf16Config shim -- a delegating VllmConfig
view that reports quant_config=None -- and wraps vllm_config at the top
of Glm5NextMTP.__init__, so the whole draft subtree builds unquantized.
Everything else is untouched.

Fail closed: each anchor must match exactly once and the patched source
must compile. Any other shape exits non-zero.

usage: patch-mtp.py <image mtp.py> <out path>
"""
import sys

MARKER = "mtp-bf16-overlay"

# mtp.py @ glm53-flash-arm64-cu130 (v0.28.1rc1.dev580+g385dce36b).
ANCHOR_CLS = "class Glm5NextMultiTokenPredictorLayer(nn.Module):"

SHIM = '''\
class _DraftBf16Config:
    """Delegating VllmConfig view with ``quant_config`` forced to None.

    The MTP draft checkpoints (the layer-45 weights of the target
    checkpoint and a repacked BF16 draft dir alike) hold plain BF16
    expert weights, while the inherited modelopt_fp4 quant_config makes
    RoutedExperts allocate NVFP4-packed buffers they cannot copy into.
    Forcing quant_config to None here mirrors what DraftModelProposer
    does for draft-model methods. Every other attribute is delegated
    unchanged. (%s)
    """

    def __init__(self, inner: VllmConfig) -> None:
        object.__setattr__(self, "_inner", inner)
        self.quant_config = None

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_inner"), name)


''' % MARKER

ANCHOR_INIT = (
    "class Glm5NextMTP(nn.Module, DeepseekV2MixtureOfExperts):\n"
    "    def __init__(self, *, vllm_config: VllmConfig, prefix: str = \"\"):\n"
    "        super().__init__()\n"
)

REPL_INIT = ANCHOR_INIT + (
    "        # %s: build the whole draft subtree unquantized -- the shim\n"
    "        # delegates every attribute except quant_config, pinned to\n"
    "        # None. Downstream readers (Glm5NextDecoderLayer reads\n"
    "        # vllm_config.quant_config at its own init) all receive None.\n"
    "        vllm_config = _DraftBf16Config(vllm_config)\n"
) % MARKER

REQUIRED = ("_DraftBf16Config", "VllmConfig",
            "vllm_config = _DraftBf16Config(vllm_config)")


def patch(src: str) -> str:
    if MARKER in src:
        raise ValueError("input already carries the mtp-bf16 patch")
    for anchor in (ANCHOR_CLS, ANCHOR_INIT):
        n = src.count(anchor)
        if n != 1:
            raise ValueError(f"anchor matched {n} times (want 1): "
                             f"{anchor[:60]!r} -- image mtp.py drifted; "
                             "re-derive the patch")
    out = src.replace(ANCHOR_CLS, SHIM + ANCHOR_CLS, 1)
    out = out.replace(ANCHOR_INIT, REPL_INIT, 1)
    for needle in REQUIRED:
        if needle not in out:
            raise ValueError(f"patched source lost {needle!r}")
    compile(out, "<mtp-bf16 overlay>", "exec")
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src = open(sys.argv[1]).read()
    try:
        out = patch(src)
    except (ValueError, SyntaxError) as e:
        print(f"FAIL: mtp patch refused: {e}")
        return 1
    with open(sys.argv[2], "w") as f:
        f.write(out)
    print(f"mtp-bf16 overlay written: {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
