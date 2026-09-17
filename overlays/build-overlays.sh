#!/bin/bash
# build-overlays.sh — produce every overlay file under _build/ from the
# image's own sources.
#
#   1. snapshot the needed image files into _image/ (fetch-image-file.sh)
#   2. patch-kda.py    _image/kda.py    -> _build/kda-quant.py
#      patch-mla.py    _image/model.py  -> _build/mla-quant.py
#      patch-mtp.py    _image/mtp.py    -> _build/glm5next-mtp-bf16.py
#   3. step-attribution (profiling only): apply-step-attr-patch.py runs
#      on a staged dir in which model.py is the mla-quant output, so
#      _build/step-attr/model.py carries BOTH patches. It lands in its
#      own subdir so the serve scripts never mount the instrumented
#      build by accident -- they read it only under STEP_ATTR=1.
#   4. static files copied through: flashinfer_mla_sparse_sm120.py into
#      _build/, step_attr.py into _build/step-attr/ (runtime of step 3)
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
# Kept out of $BUILD's top level: an instrumented model.py sitting next
# to mla-quant.py would be mounted by default and every measurement
# would carry the CUDA-event overhead.
SA="$BUILD/step-attr"
STAGE="$BUILD/.sa-stage"
mkdir -p "$STAGE" "$SA"
cp "$BUILD/mla-quant.py" "$STAGE/model.py"
cp "$IMG/model_runner.py" "$IMG/cuda_communicator.py" "$STAGE/"
python3 "$HERE/apply-step-attr-patch.py" "$STAGE" "$SA"
rm -rf "$STAGE"
cp "$HERE/step_attr.py" "$SA/"
cp "$HERE/flashinfer_mla_sparse_sm120.py" "$BUILD/"

echo "build-overlays done -> $BUILD"
ls -1 "$BUILD"
ls -1 "$SA" | sed 's|^|step-attr/|'
