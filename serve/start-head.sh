#!/bin/bash
# start-head.sh — run ON the head node: Ray head + vLLM serve for the
# requantized GLM-5.3-Flash-NVFP4 checkpoint, TP=2 across two nodes.
#
# Required env (see serve.env.example):
#   HEAD_IP    this node's IP on the fast interconnect
#   IFNAME     netdev carrying RoCE on this node (e.g. enp1s0f0np0)
#   MODEL_DIR  path to the requantized checkpoint on this node
# Optional:
#   PORT            API port (default 8000)
#   IMAGE           base image (default the official image below)
#   NCCL_IB=1       use the derived rdma-core image + NET/IB (default 1)
#   NCCL_IB_IMAGE   derived image tag (default <IMAGE>-nccl-ib)
#   OVERLAY_DIR     dir holding built overlay files (default ../overlays/_build)
#                   kda-quant.py / mla-quant.py / glm5next-mtp-bf16.py /
#                   flashinfer_mla_sparse_sm120.py are mounted when present
#   MTP_DIR         BF16 MTP draft dir -> enables official MTP
#   MTP_K           num_speculative_tokens (default 2, used iff MTP_DIR)
#   CONTAINER       container name (default glm53-head)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/nccl-ib.sh"

: "${HEAD_IP:?set HEAD_IP}" "${IFNAME:?set IFNAME}" "${MODEL_DIR:?set MODEL_DIR}"
PORT=${PORT:-8000}
IMAGE=${IMAGE:-vllm/vllm-openai:glm53-flash-arm64-cu130}
NCCL_IB=${NCCL_IB:-1}
NCCL_IB_IMAGE=${NCCL_IB_IMAGE:-$IMAGE-nccl-ib}
OVERLAY_DIR=${OVERLAY_DIR:-$HERE/../overlays/_build}
MTP_K=${MTP_K:-2}
CONTAINER=${CONTAINER:-glm53-head}
PKG=/usr/local/lib/python3.12/dist-packages

[ "$NCCL_IB" = "0" ] || [ "$NCCL_IB" = "1" ] || { echo "FAIL: NCCL_IB must be 0 or 1" >&2; exit 1; }
NCCL_IB_ENV=""; NCCL_IB_ARGS=""
if [ "$NCCL_IB" = "1" ]; then
  nccl_ib_setup "$IFNAME"
  IMAGE=$NCCL_IB_IMAGE
fi

# Overlay mounts: each built overlay is bind-mounted over the image file
# it patches. Missing files are simply not mounted.
MOUNTS=""
mnt() { [ -f "$OVERLAY_DIR/$1" ] && MOUNTS="$MOUNTS -v $OVERLAY_DIR/$1:$2:ro"; }
mnt kda-quant.py                 "$PKG/vllm/models/glm5next/nvidia/kda.py"
mnt glm5next-mtp-bf16.py         "$PKG/vllm/models/glm5next/nvidia/mtp.py"
mnt flashinfer_mla_sparse_sm120.py "$PKG/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"
# step-attribution overlay (optional; enable with STEP_ATTR=1 in the
# container). model.py is shared with mla-quant: build a combined file
# first if both overlays are wanted.
mnt model_runner.py              "$PKG/vllm/v1/worker/gpu/model_runner.py"
mnt cuda_communicator.py         "$PKG/vllm/distributed/device_communicators/cuda_communicator.py"
mnt step_attr.py                 "$PKG/step_attr.py"
STEP_ENV=""
[ -f "$OVERLAY_DIR/step_attr.py" ] && STEP_ENV="-e STEP_ATTR=${STEP_ATTR:-1} -e STEP_ATTR_EVERY=${STEP_ATTR_EVERY:-256}"
# mla-quant mounts model.py only when the step-attr build did not
# already provide a combined model.py.
[ -f "$OVERLAY_DIR/model.py" ] \
  && MOUNTS="$MOUNTS -v $OVERLAY_DIR/model.py:$PKG/vllm/models/glm5next/nvidia/model.py:ro" \
  || mnt mla-quant.py "$PKG/vllm/models/glm5next/nvidia/model.py"

MTP_MNT=""; SPEC_ARG=""
if [ -n "${MTP_DIR:-}" ]; then
  test -d "$MTP_DIR"
  MTP_MNT="-v $MTP_DIR:/checkpoint-mtp:ro"
  SPEC="{\"method\":\"mtp\",\"model\":\"/checkpoint-mtp\",\"num_speculative_tokens\":$MTP_K}"
  SPEC_ARG="--speculative-config $SPEC"
fi

test -d "$MODEL_DIR"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
KC='{"enable_flashinfer_autotune": false}'
docker run -d --name "$CONTAINER" --network host --gpus all \
  --shm-size=32g --ipc=host \
  -v "$MODEL_DIR":/checkpoint:ro \
  $MOUNTS $MTP_MNT $STEP_ENV $NCCL_IB_ENV $NCCL_IB_ARGS \
  --entrypoint bash \
  "$IMAGE" -lc "python3 -c 'import ray' 2>/dev/null || pip install -q ray; \
    export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_HOST_IP=$HEAD_IP \
      NCCL_SOCKET_IFNAME=$IFNAME GLOO_SOCKET_IFNAME=$IFNAME && \
    ray start --head --node-ip-address=$HEAD_IP --port=6399 --num-gpus=1 \
      --dashboard-host=127.0.0.1 && \
    exec vllm serve /checkpoint \
    --served-model-name GLM-5.3-Flash-NVFP4 \
    --host 0.0.0.0 --port $PORT \
    --tensor-parallel-size 2 --data-parallel-size 1 \
    --distributed-executor-backend ray \
    --reasoning-parser glm45 --kernel-config '$KC' \
    --kv-cache-dtype fp8 \
    --model-loader-extra-config '{\"enable_multithread_load\": true, \"num_threads\": 32}' \
    --max-num-batched-tokens 8192 --enable-chunked-prefill \
    --max-num-seqs 20 --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --limit-mm-per-prompt '{\"image\":0,\"video\":0}' $SPEC_ARG"
echo "started $CONTAINER on $HEAD_IP ($IFNAME); API will listen on :$PORT"
