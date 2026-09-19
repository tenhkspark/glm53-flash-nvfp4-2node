#!/bin/bash
# start-worker.sh — run ON the worker node: join the Ray cluster and
# sleep; the head's vLLM process drives this rank remotely.
#
# Required env (see serve.env.example):
#   HEAD_IP    head node IP (ray start --address=$HEAD_IP:6399)
#   MY_IP      this node's IP on the fast interconnect
#   IFNAME     netdev carrying RoCE on this node
#   MODEL_DIR  path to the requantized checkpoint on this node
# Optional: same IMAGE / NCCL_IB / NCCL_IB_IMAGE / OVERLAY_DIR / MTP_DIR /
# STEP_ATTR / STEP_ATTR_EVERY / CONTAINER (default glm53-worker) knobs as
# start-head.sh. The worker mounts the same overlays so its model code
# matches the head's -- STEP_ATTR must be set the same way on both nodes.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/nccl-ib.sh"

: "${HEAD_IP:?set HEAD_IP}" "${MY_IP:?set MY_IP}" "${IFNAME:?set IFNAME}" \
  "${MODEL_DIR:?set MODEL_DIR}"
IMAGE=${IMAGE:-vllm/vllm-openai:glm53-flash-arm64-cu130}
NCCL_IB=${NCCL_IB:-1}
NCCL_IB_IMAGE=${NCCL_IB_IMAGE:-$IMAGE-nccl-ib}
# docker needs an absolute bind source, so resolve the repo root rather
# than handing it a path with ".." in it.
REPO="$(cd "$HERE/.." && pwd)"
OVERLAY_DIR=${OVERLAY_DIR:-$REPO/overlays/_build}
CONTAINER=${CONTAINER:-glm53-worker}
STEP_ATTR=${STEP_ATTR:-0}
PKG=/usr/local/lib/python3.12/dist-packages

# every -v source below must be absolute
abs() { case "$1" in /*) ;; *) echo "FAIL: $2 must be an absolute path: $1" >&2; exit 1;; esac; }
abs "$MODEL_DIR" MODEL_DIR
abs "$OVERLAY_DIR" OVERLAY_DIR
if [ -n "${MTP_DIR:-}" ]; then abs "$MTP_DIR" MTP_DIR; fi

[ "$NCCL_IB" = "0" ] || [ "$NCCL_IB" = "1" ] || { echo "FAIL: NCCL_IB must be 0 or 1" >&2; exit 1; }
[ "$STEP_ATTR" = "0" ] || [ "$STEP_ATTR" = "1" ] || { echo "FAIL: STEP_ATTR must be 0 or 1" >&2; exit 1; }
NCCL_IB_ENV=""; NCCL_IB_ARGS=""
if [ "$NCCL_IB" = "1" ]; then
  nccl_ib_setup "$IFNAME"
  IMAGE=$NCCL_IB_IMAGE
fi

# Overlay mounts: each built overlay is bind-mounted over the image file
# it patches. A missing file is reported and not mounted.
MOUNTS=""
mnt() {  # $1 = file in OVERLAY_DIR, $2 = target path in the image
  # A plain `[ -f ... ] && ...` returns 1 when the overlay is absent,
  # which under set -e kills the script with no message; report the
  # skip and return 0 instead.
  if [ -f "$OVERLAY_DIR/$1" ]; then
    MOUNTS="$MOUNTS -v $OVERLAY_DIR/$1:$2:ro"
  else
    echo "WARN: $1 not in $OVERLAY_DIR -- serving the image's stock file" >&2
  fi
}
mnt kda-quant.py                 "$PKG/vllm/models/glm5next/nvidia/kda.py"
mnt glm5next-mtp-bf16.py         "$PKG/vllm/models/glm5next/nvidia/mtp.py"
mnt flashinfer_mla_sparse_sm120.py "$PKG/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"
# Step attribution is opt-in (STEP_ATTR=1). Its model.py is the
# mla-quant output plus CUDA-event instrumentation, so mounting it by
# default would make every reader measure the instrumented build; it
# replaces the mla-quant mount (same target) rather than adding to it.
STEP_ENV=""
if [ "$STEP_ATTR" = "1" ]; then
  SA=$OVERLAY_DIR/step-attr
  for f in model.py model_runner.py cuda_communicator.py step_attr.py; do
    [ -f "$SA/$f" ] || { echo "FAIL: STEP_ATTR=1 but $SA/$f is missing (run overlays/build-overlays.sh)" >&2; exit 1; }
  done
  MOUNTS="$MOUNTS -v $SA/model.py:$PKG/vllm/models/glm5next/nvidia/model.py:ro"
  MOUNTS="$MOUNTS -v $SA/model_runner.py:$PKG/vllm/v1/worker/gpu/model_runner.py:ro"
  MOUNTS="$MOUNTS -v $SA/cuda_communicator.py:$PKG/vllm/distributed/device_communicators/cuda_communicator.py:ro"
  MOUNTS="$MOUNTS -v $SA/step_attr.py:$PKG/step_attr.py:ro"
  STEP_ENV="-e STEP_ATTR=1 -e STEP_ATTR_EVERY=${STEP_ATTR_EVERY:-256}"
else
  mnt mla-quant.py "$PKG/vllm/models/glm5next/nvidia/model.py"
fi

# JIT_CACHE_DIR (optional): host dir to keep the Triton / TorchInductor /
# FlashInfer JIT caches in. Without it every boot recompiles into the
# container layer and throws the result away on `docker rm`, and vLLM's
# own jit_monitor then reports compilations happening during inference.
# Unset = the stock behaviour, i.e. this knob changes nothing unless set.
JIT_MNT=""; JIT_ENV=""
if [ -n "${JIT_CACHE_DIR:-}" ]; then
  abs "$JIT_CACHE_DIR" JIT_CACHE_DIR
  mkdir -p "$JIT_CACHE_DIR"
  JIT_MNT="-v $JIT_CACHE_DIR:/jit-cache"
  JIT_ENV="-e TRITON_CACHE_DIR=/jit-cache/triton"
  JIT_ENV="$JIT_ENV -e TORCHINDUCTOR_CACHE_DIR=/jit-cache/inductor"
  JIT_ENV="$JIT_ENV -e FLASHINFER_WORKSPACE_DIR=/jit-cache/flashinfer"
  JIT_ENV="$JIT_ENV -e VLLM_CACHE_ROOT=/jit-cache/vllm"
fi

MTP_MNT=""
if [ -n "${MTP_DIR:-}" ]; then
  [ -d "$MTP_DIR" ] || { echo "FAIL: MTP_DIR is not a directory: $MTP_DIR" >&2; exit 1; }
  MTP_MNT="-v $MTP_DIR:/checkpoint-mtp:ro"
fi

[ -d "$MODEL_DIR" ] || { echo "FAIL: MODEL_DIR is not a directory: $MODEL_DIR" >&2; exit 1; }
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER" --network host --gpus all \
  --shm-size=32g --ipc=host \
  -v "$MODEL_DIR":/checkpoint:ro \
  $MOUNTS $MTP_MNT $JIT_MNT $STEP_ENV $JIT_ENV $NCCL_IB_ENV $NCCL_IB_ARGS \
  --entrypoint bash \
  "$IMAGE" -lc "python3 -c 'import ray' 2>/dev/null || pip install -q ray; \
    export VLLM_HOST_IP=$MY_IP RAY_memory_usage_threshold=0.99 NCCL_SOCKET_IFNAME=$IFNAME \
      GLOO_SOCKET_IFNAME=$IFNAME && \
    ray start --address=$HEAD_IP:6399 --node-ip-address=$MY_IP \
      --num-gpus=1 && sleep infinity"
echo "started $CONTAINER on $MY_IP ($IFNAME); joined ray at $HEAD_IP:6399"
