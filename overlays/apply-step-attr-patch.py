#!/usr/bin/env python3
"""Build the step-attribution overlay files from the deployed image's files.

Goal:
attribute the ~95 ms stock decode step by recording CUDA event pairs around
each layer's attention block, each MoE block, every TP all-reduce, the
sampler, and the whole worker-side step. Events accumulate device-side and
flush (one sync + one stderr line) every 256 steps:

    [step-attr] steps=256 total=.. attn=.. moe=.. ar=.. sampler=.. other=..

This script takes the image's exact files -- snapshotted by
fetch-image-file.sh into _image/ next to this script -- and applies
anchored source transforms, so each result is ABI-identical to the image
file except for the patch:

1. model.py (models/glm5next/nvidia/): wraps the self_attn call and the
   mlp call of Glm5NextDecoderLayer.forward in _sa.span("attn"/"moe"),
   both the mHC path (incl. the sp_all_gather/sp_reduce_scatter pair that
   brackets attention) and the non-mHC/MTP path.
2. model_runner.py (v1/worker/gpu/): _sa.step_begin() at the top of the
   `not dummy_run` branch of execute_model (worker-side step wall start,
   dummy/capture runs excluded), _sa.span("sampler") around the
   self.sample() call (compute_logits + sampling), and _sa.step_end()
   right before `return async_output` in sample_tokens.
3. cuda_communicator.py (distributed/device_communicators/): renames
   CudaCommunicator.all_reduce to _all_reduce_impl and adds a thin
   all_reduce wrapper that records the call in the "ar" bucket -- a
   single choke point covering every TP all-reduce on all backends.

The runtime module itself is a static file, step_attr.py in this
directory (mounted at dist-packages root); the patcher does not build it.

Usage:
    apply-step-attr-patch.py [IMAGE_DIR] [OUT_DIR]

    IMAGE_DIR default: <this dir>/_image
    OUT_DIR   default: <this dir>/_build

Outputs keep the image basename (model.py, model_runner.py,
cuda_communicator.py). Exit 0 on success; on any anchor/verify failure
prints "FAIL: apply-step-attr-patch failed at step: <stage>" to stderr
and exits 1.
"""

import datetime
import difflib
import hashlib
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE_DIR = os.path.join(_HERE, "_image")
DEFAULT_OUT_DIR = os.path.join(_HERE, "_build")

IMAGE = "vllm/vllm-openai:glm53-flash-arm64-cu130"


class PatchError(Exception):
    def __init__(self, stage, msg):
        super().__init__(msg)
        self.stage = stage


# --- anchored substitutions (each old must occur exactly once) ---------

SUBS = {
    "model.py": [
        (
            "import torch\nfrom torch import nn\n",
            "import torch\nfrom torch import nn\n\nimport step_attr as _sa\n",
        ),
        (
            "            attn_output = self.self_attn(\n"
            "                hidden_states=hidden_states,\n"
            "                positions=positions,\n"
            "            )",
            "            with _sa.span(\"attn\"):\n"
            "                attn_output = self.self_attn(\n"
            "                    hidden_states=hidden_states,\n"
            "                    positions=positions,\n"
            "                )",
        ),
        (
            "            hidden_states = self.mlp(hidden_states)\n"
            "            if self.is_mtp_layer:",
            "            with _sa.span(\"moe\"):\n"
            "                hidden_states = self.mlp(hidden_states)\n"
            "            if self.is_mtp_layer:",
        ),
        (
            "        if self.is_sequence_parallel:\n"
            "            x = sp_all_gather(x)[: positions.shape[0]]\n"
            "\n"
            "        x = self.self_attn(\n"
            "            hidden_states=x,\n"
            "            positions=positions,\n"
            "        )\n"
            "\n"
            "        if self.is_sequence_parallel:\n"
            "            x = sp_reduce_scatter(x)",
            "        with _sa.span(\"attn\"):\n"
            "            if self.is_sequence_parallel:\n"
            "                x = sp_all_gather(x)[: positions.shape[0]]\n"
            "\n"
            "            x = self.self_attn(\n"
            "                hidden_states=x,\n"
            "                positions=positions,\n"
            "            )\n"
            "\n"
            "            if self.is_sequence_parallel:\n"
            "                x = sp_reduce_scatter(x)",
        ),
        (
            "        if self._mlp_is_moe:\n"
            "            x = self.mlp(x, already_sequence_parallel=self.is_sequence_parallel)\n"
            "        else:\n"
            "            x = self.mlp(x)",
            "        if self._mlp_is_moe:\n"
            "            with _sa.span(\"moe\"):\n"
            "                x = self.mlp(\n"
            "                    x, already_sequence_parallel=self.is_sequence_parallel\n"
            "                )\n"
            "        else:\n"
            "            with _sa.span(\"moe\"):\n"
            "                x = self.mlp(x)",
        ),
    ],
    "model_runner.py": [
        (
            "import torch\nimport torch.nn as nn\n",
            "import torch\nimport torch.nn as nn\n\nimport step_attr as _sa\n",
        ),
        (
            "        if not dummy_run:\n"
            "            # Update the request states.\n"
            "            self.update_pp_decode_requests()",
            "        if not dummy_run:\n"
            "            _sa.step_begin()\n"
            "            # Update the request states.\n"
            "            self.update_pp_decode_requests()",
        ),
        (
            "        sampler_output, num_sampled, num_rejected = self.sample(\n"
            "            hidden_states, input_batch, grammar_output\n"
            "        )",
            "        with _sa.span(\"sampler\"):\n"
            "            sampler_output, num_sampled, num_rejected = self.sample(\n"
            "                hidden_states, input_batch, grammar_output\n"
            "            )",
        ),
        (
            "        model_runner_output.ec_connector_output = ec_connector_output\n"
            "\n"
            "        return async_output",
            "        model_runner_output.ec_connector_output = ec_connector_output\n"
            "\n"
            "        _sa.step_end()\n"
            "        return async_output",
        ),
    ],
    "cuda_communicator.py": [
        (
            "from .base_device_communicator import DeviceCommunicatorBase\n"
            "\n"
            "logger = init_logger(__name__)",
            "from .base_device_communicator import DeviceCommunicatorBase\n"
            "\n"
            "import step_attr as _sa\n"
            "\n"
            "logger = init_logger(__name__)",
        ),
        (
            "    def all_reduce(self, input_):\n"
            "        fi_ar_comm = self.fi_ar_comm",
            "    def all_reduce(self, input_):\n"
            "        with _sa.span(\"ar\"):\n"
            "            return self._all_reduce_impl(input_)\n"
            "\n"
            "    def _all_reduce_impl(self, input_):\n"
            "        fi_ar_comm = self.fi_ar_comm",
        ),
    ],
}

# Mount targets inside the image (used by the header + the arm script).
MOUNT_TARGET = {
    "model.py": "vllm/models/glm5next/nvidia/model.py",
    "model_runner.py": "vllm/v1/worker/gpu/model_runner.py",
    "cuda_communicator.py": (
        "vllm/distributed/device_communicators/cuda_communicator.py"
    ),
}

HEADER = """\
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Overlay (step-attr): CUDA-event step attribution. {what}
# Base file: vllm/{target} from image
# {image}, snapshotted {date} by fetch-image-file.sh
# into _image/{base} (sha256:{sha}).
# Regenerated by apply-step-attr-patch.py; do not
# hand-edit -- fix the patcher and regenerate instead.
#
# Enable by bind-mounting the patched files plus step_attr.py at
# dist-packages root and setting STEP_ATTR=1.
"""

WHAT = {
    "model.py": (
        "wraps Glm5NextDecoderLayer.forward's self_attn calls (mHC path "
        "including the sp_all_gather/sp_reduce_scatter pair) and mlp "
        "calls in _sa.span(\"attn\"/\"moe\")."
    ),
    "model_runner.py": (
        "_sa.step_begin() on real (non-dummy) execute_model, "
        "_sa.span(\"sampler\") around self.sample(), _sa.step_end() "
        "before returning async_output."
    ),
    "cuda_communicator.py": (
        "all_reduce renamed to _all_reduce_impl behind a thin wrapper "
        "recording every TP all-reduce in the \"ar\" bucket."
    ),
}

MAX_CHANGED = 60


def lines_of(src):
    return src.splitlines(keepends=True)


def apply_patch(src, subs):
    """Apply anchored substitutions. Raises PatchError on anchor failure."""
    for i, (old, new) in enumerate(subs):
        n = src.count(old)
        if n != 1:
            raise PatchError(
                "anchor", f"anchor {i} count {n} != 1: {old[:70]!r}"
            )
        src = src.replace(old, new, 1)
    return src


def _verify(src, new):
    compile(new, "<patched>", "exec")
    diff = list(difflib.unified_diff(lines_of(src), lines_of(new), n=1))
    changed = sum(
        1
        for l in diff
        if l[:1] in ("+", "-") and not l.startswith(("+++", "---"))
    )
    if changed > MAX_CHANGED:
        raise PatchError("verify", f"patch is not minimal ({changed} lines)")
    return changed


def process(base_path, out_path, base_name):
    try:
        with open(base_path) as f:
            src = f.read()
    except OSError as e:
        raise PatchError("read", f"{base_path}: {e}")
    patched = apply_patch(src, SUBS[base_name])
    changed = _verify(src, patched)

    sha = hashlib.sha256(src.encode()).hexdigest()[:16]
    date = datetime.date.today().isoformat()
    final = (
        HEADER.format(
            what=WHAT[base_name],
            target=MOUNT_TARGET[base_name],
            image=IMAGE,
            date=date,
            sha=sha,
            base=base_name,
        )
        + patched.lstrip("\n")
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        f.write(final)
    if os.path.exists(out_path):
        shutil.copymode(out_path, tmp)
    os.replace(tmp, out_path)
    return sha, changed


def main():
    image_dir = os.path.abspath(
        sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_DIR
    )
    out_dir = os.path.abspath(
        sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT_DIR
    )
    try:
        for base_name in SUBS:
            base = os.path.join(image_dir, base_name)
            out = os.path.join(out_dir, base_name)
            sha, changed = process(base, out, base_name)
            print(
                f"apply-step-attr-patch: {out} "
                f"(base sha256:{sha}, {changed} changed lines)"
            )
    except PatchError as e:
        print(
            f"FAIL: apply-step-attr-patch failed at step: {e.stage} ({e})",
            file=sys.stderr,
        )
        return 1
    except SyntaxError as e:
        print(
            f"FAIL: apply-step-attr-patch failed at step: verify "
            f"(output does not compile: {e})",
            file=sys.stderr,
        )
        return 1
    print("apply-step-attr-patch done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
