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
# CONTAINER (default glm53-worker) knobs as start-head.sh. The worker
# mounts the same overlays so its model code matches the head's.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/nccl-ib.sh"

: "${HEAD_IP:?set HEAD_IP}" "${MY_IP:?set MY_IP}" "${IFNAME:?set IFNAME}" \
  "${MODEL_DIR:?set MODEL_DIR}"
IMAGE=${IMAGE:-vllm/vllm-openai:glm53-flash-arm64-cu130}
NCCL_IB=${NCCL_IB:-1}
NCCL_IB_IMAGE=${NCCL_IB_IMAGE:-$IMAGE-nccl-ib}
OVERLAY_DIR=${OVERLAY_DIR:-$HERE/../overlays/_build}
CONTAINER=${CONTAINER:-glm53-worker}
PKG=/usr/local/lib/python3.12/dist-packages

[ "$NCCL_IB" = "0" ] || [ "$NCCL_IB" = "1" ] || { echo "FAIL: NCCL_IB must be 0 or 1" >&2; exit 1; }
NCCL_IB_ENV=""; NCCL_IB_ARGS=""
if [ "$NCCL_IB" = "1" ]; then
  nccl_ib_setup "$IFNAME"
  IMAGE=$NCCL_IB_IMAGE
fi

MOUNTS=""
mnt() { [ -f "$OVERLAY_DIR/$1" ] && MOUNTS="$MOUNTS -v $OVERLAY_DIR/$1:$2:ro"; }
mnt kda-quant.py                 "$PKG/vllm/models/glm5next/nvidia/kda.py"
mnt glm5next-mtp-bf16.py         "$PKG/vllm/models/glm5next/nvidia/mtp.py"
mnt flashinfer_mla_sparse_sm120.py "$PKG/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"
mnt model_runner.py              "$PKG/vllm/v1/worker/gpu/model_runner.py"
mnt cuda_communicator.py         "$PKG/vllm/distributed/device_communicators/cuda_communicator.py"
mnt step_attr.py                 "$PKG/step_attr.py"
STEP_ENV=""
[ -f "$OVERLAY_DIR/step_attr.py" ] && STEP_ENV="-e STEP_ATTR=${STEP_ATTR:-1} -e STEP_ATTR_EVERY=${STEP_ATTR_EVERY:-256}"
[ -f "$OVERLAY_DIR/model.py" ] \
  && MOUNTS="$MOUNTS -v $OVERLAY_DIR/model.py:$PKG/vllm/models/glm5next/nvidia/model.py:ro" \
  || mnt mla-quant.py "$PKG/vllm/models/glm5next/nvidia/model.py"

MTP_MNT=""
[ -n "${MTP_DIR:-}" ] && { test -d "$MTP_DIR"; MTP_MNT="-v $MTP_DIR:/checkpoint-mtp:ro"; }

test -d "$MODEL_DIR"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER" --network host --gpus all \
  --shm-size=32g --ipc=host \
  -v "$MODEL_DIR":/checkpoint:ro \
  $MOUNTS $MTP_MNT $STEP_ENV $NCCL_IB_ENV $NCCL_IB_ARGS \
  --entrypoint bash \
  "$IMAGE" -lc "python3 -c 'import ray' 2>/dev/null || pip install -q ray; \
    export VLLM_HOST_IP=$MY_IP NCCL_SOCKET_IFNAME=$IFNAME \
      GLOO_SOCKET_IFNAME=$IFNAME && \
    ray start --address=$HEAD_IP:6399 --node-ip-address=$MY_IP \
      --num-gpus=1 && sleep infinity"
echo "started $CONTAINER on $MY_IP ($IFNAME); joined ray at $HEAD_IP:6399"
