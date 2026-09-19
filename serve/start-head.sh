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
#   STEP_ATTR=1     profiling run: mount OVERLAY_DIR/step-attr/ instead
#                   of mla-quant.py (default 0 = uninstrumented)
#   STEP_ATTR_EVERY flush interval in steps (default 256, iff STEP_ATTR=1)
#   MTP_DIR         BF16 MTP draft dir -> enables official MTP
#   MTP_K           num_speculative_tokens (default 2, used iff MTP_DIR)
#   CONTAINER       container name (default glm53-head)
#
# Serving window: this script ships --max-model-len 204800 with
# --max-num-seqs 20, --gpu-memory-utilization 0.85 and --enable-prefix-caching.
# 0.85 rather than 0.88: at 0.88 the host had 2.11% free, which is the line the
# node's OOM reaper fires at; 0.85 leaves 8.6%. RAY_memory_usage_threshold=0.99
# is set on the head and the worker both (each node runs its own raylet, and the
# default 0.95 kills the vLLM worker on a unified-memory node). Those are one
# setting with three names, not three knobs: at 307200 the engine does
# not come up with 20 sequence slots, and 0.89 was refused at boot on
# this pair. Edit them together on the vllm serve line below.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/nccl-ib.sh"

: "${HEAD_IP:?set HEAD_IP}" "${IFNAME:?set IFNAME}" "${MODEL_DIR:?set MODEL_DIR}"
PORT=${PORT:-8000}
IMAGE=${IMAGE:-vllm/vllm-openai:glm53-flash-arm64-cu130}
NCCL_IB=${NCCL_IB:-1}
NCCL_IB_IMAGE=${NCCL_IB_IMAGE:-$IMAGE-nccl-ib}
# docker needs an absolute bind source, so resolve the repo root rather
# than handing it a path with ".." in it.
REPO="$(cd "$HERE/.." && pwd)"
OVERLAY_DIR=${OVERLAY_DIR:-$REPO/overlays/_build}
MTP_K=${MTP_K:-2}
CONTAINER=${CONTAINER:-glm53-head}
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

MTP_MNT=""; SPEC_ARG=""
if [ -n "${MTP_DIR:-}" ]; then
  [ -d "$MTP_DIR" ] || { echo "FAIL: MTP_DIR is not a directory: $MTP_DIR" >&2; exit 1; }
  MTP_MNT="-v $MTP_DIR:/checkpoint-mtp:ro"
  SPEC="{\"method\":\"mtp\",\"model\":\"/checkpoint-mtp\",\"num_speculative_tokens\":$MTP_K}"
  # The JSON is expanded into the container's `bash -lc` string, so it
  # has to carry its own quotes like the other JSON flags below: bare
  # {"a":1,"b":2} is a brace expansion to that shell, and vllm then dies
  # with argparse exit 2 about 20 s after ray start -- long after this
  # script has printed "started".
  SPEC_ARG="--speculative-config '$SPEC'"
fi

[ -d "$MODEL_DIR" ] || { echo "FAIL: MODEL_DIR is not a directory: $MODEL_DIR" >&2; exit 1; }
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
KC='{"enable_flashinfer_autotune": false}'
docker run -d --name "$CONTAINER" --network host --gpus all \
  --shm-size=32g --ipc=host \
  -v "$MODEL_DIR":/checkpoint:ro \
  $MOUNTS $MTP_MNT $JIT_MNT $STEP_ENV $JIT_ENV $NCCL_IB_ENV $NCCL_IB_ARGS \
  --entrypoint bash \
  "$IMAGE" -lc "python3 -c 'import ray' 2>/dev/null || pip install -q ray; \
    export VLLM_ENGINE_READY_TIMEOUT_S=3600 RAY_memory_usage_threshold=0.99 VLLM_HOST_IP=$HEAD_IP \
      NCCL_SOCKET_IFNAME=$IFNAME GLOO_SOCKET_IFNAME=$IFNAME && \
    ray start --head --node-ip-address=$HEAD_IP --port=6399 --num-gpus=1 \
      --dashboard-host=127.0.0.1 && \
    exec vllm serve /checkpoint \
    --served-model-name GLM-5.3-Flash-NVFP4-Wabi \
    --host 0.0.0.0 --port $PORT \
    --tensor-parallel-size 2 --data-parallel-size 1 \
    --distributed-executor-backend ray \
    --reasoning-parser glm45 --kernel-config '$KC' \
    --enable-auto-tool-choice --tool-call-parser glm45 \
    --kv-cache-dtype fp8 \
    --model-loader-extra-config '{\"enable_multithread_load\": true, \"num_threads\": 32}' \
    --max-num-batched-tokens 8192 --enable-chunked-prefill \
    --max-num-seqs 20 --max-model-len 204800 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --limit-mm-per-prompt '{\"image\":0,\"video\":0}' $SPEC_ARG"
echo "started $CONTAINER on $HEAD_IP ($IFNAME); API will listen on :$PORT"
