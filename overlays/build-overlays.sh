#!/bin/bash
# build-overlays.sh — produce every overlay file under _build/ from the
# image's own sources.
#
#   1. snapshot the needed image files into _image/ (fetch-image-file.sh)
#   2. patch-kda.py    _image/kda.py    -> _build/kda-quant.py
#      patch-mla.py    _image/model.py  -> _build/mla-quant.py
#      patch-mtp.py    _image/mtp.py    -> _build/glm5next-mtp-bf16.py
#   3. step-attribution: apply-step-attr-patch.py runs on a staged dir in
#      which model.py is the mla-quant output, so _build/model.py carries
#      BOTH patches (mount it instead of mla-quant.py when profiling)
#   4. static files copied through: step_attr.py,
#      flashinfer_mla_sparse_sm120.py
#
# usage: build-overlays.sh [ssh-host]
#   With no argument the image files are read from a local docker daemon;
#   pass an ssh host to read them on a remote docker host instead.
#   IMAGE (env) overrides the source image.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
IMG="$HERE/_image"
BUILD="$HERE/_build"
SSH_HOST=${1:-}
mkdir -p "$IMG" "$BUILD"

fetch() {  # $1 = relpath under dist-packages/vllm
  local rel=$1 base
  base=$(basename "$rel")
  [ -f "$IMG/$base" ] && return 0
  if [ -n "$SSH_HOST" ]; then
    "$HERE/fetch-image-file.sh" "$rel" "$SSH_HOST"
  else
    "$HERE/fetch-image-file.sh" "$rel"
  fi
}

# model code (glm5next/nvidia) + step-attr engine files
fetch models/glm5next/nvidia/kda.py
fetch models/glm5next/nvidia/model.py
fetch models/glm5next/nvidia/mtp.py
fetch v1/worker/gpu/model_runner.py
fetch distributed/device_communicators/cuda_communicator.py

python3 "$HERE/patch-kda.py" "$IMG/kda.py" "$BUILD/kda-quant.py"
python3 "$HERE/patch-mla.py" "$IMG/model.py" "$BUILD/mla-quant.py"
python3 "$HERE/patch-mtp.py" "$IMG/mtp.py" "$BUILD/glm5next-mtp-bf16.py"

# step-attribution: model.py staged as the mla-quant output so the
# combined file keeps the quant fix; the other two inputs are stock.
STAGE="$BUILD/.sa-stage"
mkdir -p "$STAGE"
cp "$BUILD/mla-quant.py" "$STAGE/model.py"
cp "$IMG/model_runner.py" "$IMG/cuda_communicator.py" "$STAGE/"
python3 "$HERE/apply-step-attr-patch.py" "$STAGE" "$BUILD"
rm -rf "$STAGE"
cp "$HERE/step_attr.py" "$HERE/flashinfer_mla_sparse_sm120.py" "$BUILD/"

echo "build-overlays done -> $BUILD"
ls -1 "$BUILD"
