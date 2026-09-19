#!/bin/bash
# verify-serve.sh <container-name> <env-file>
#
# Compares what a launched serving container actually runs -- its image,
# command line, environment and mounts, i.e. everything `docker inspect`
# exposes -- against what an env file in serve.env format declares.
# Prints one line per finding:
#
#   MISMATCH:      the declared value and the actual value disagree
#   UNVERIFIABLE:  the env file names it but this container does not
#                  expose it (reported, never silently passed)
#
# Exit status: 0 = no mismatch, 1 = at least one MISMATCH, 2 = usage or
# fetch error. UNVERIFIABLE lines do not fail the run by themselves; the
# RESULT line prints both counts so a caller can gate on either number.
#
# Expected values come from two places, matching how a container is
# actually launched:
#   * variables the env file sets: MODEL_DIR, PORT, MTP_DIR/MTP_K,
#     IMAGE/NCCL_IB/NCCL_IB_IMAGE, HEAD_IP/MY_IP/IFNAME,
#     STEP_ATTR/STEP_ATTR_EVERY, OVERLAY_DIR, CONTAINER
#   * constants start-head.sh / start-worker.sh hardcode: TP=2,
#     --max-model-len 204800, --max-num-seqs 20,
#     --gpu-memory-utilization 0.85,
#     --served-model-name GLM-5.3-Flash-NVFP4-Wabi, expert-parallel OFF
#     (no --enable-expert-parallel), RAY_memory_usage_threshold=0.99,
#     model bind-mounted at /checkpoint, MTP draft at /checkpoint-mtp.
#
# Offline testing: fetch_state() is the only step that touches docker.
# Set INSPECT_FIXTURE=<file> to feed it a captured `docker inspect` JSON
# instead -- see fixtures/*.inspect.json.
set -euo pipefail

[ $# -eq 2 ] || { echo "usage: $0 <container-name> <env-file>" >&2; exit 2; }
CHECK_CONTAINER=$1
ENV_FILE=$2
[ -f "$ENV_FILE" ] || { echo "FAIL: env file not found: $ENV_FILE" >&2; exit 2; }

MISM=0; UNV=0
mm() { echo "MISMATCH: $1"; MISM=$((MISM+1)); }
uv() { echo "UNVERIFIABLE: $1"; UNV=$((UNV+1)); }

# ---- declared side: source the env file exactly like the operator does ----
. "$ENV_FILE"
D_MODEL_DIR=${MODEL_DIR:-}
D_MTP_DIR=${MTP_DIR:-}
D_MTP_K=${MTP_K:-2}
D_PORT=${PORT:-8000}
D_NCCL_IB=${NCCL_IB:-1}
D_IMAGE=${IMAGE:-vllm/vllm-openai:glm53-flash-arm64-cu130}
D_NCCL_IB_IMAGE=${NCCL_IB_IMAGE:-$D_IMAGE-nccl-ib}
D_STEP_ATTR=${STEP_ATTR:-0}
D_STEP_ATTR_EVERY=${STEP_ATTR_EVERY:-256}
D_HEAD_IP=${HEAD_IP:-}
D_MY_IP=${MY_IP:-}
D_IFNAME=${IFNAME:-}
D_ENV_CONTAINER=${CONTAINER:-}
HERE=$(cd "$(dirname "$0")" && pwd)
D_OVERLAY_DIR=${OVERLAY_DIR:-$(cd "$HERE/.." && pwd)/overlays/_build}
if [ "$D_NCCL_IB" = "1" ]; then D_WANT_IMAGE=$D_NCCL_IB_IMAGE; else D_WANT_IMAGE=$D_IMAGE; fi

# ---- actual side: the one substitutable fetch ----
fetch_state() {
  if [ -n "${INSPECT_FIXTURE:-}" ]; then cat "$INSPECT_FIXTURE";
  else docker inspect "$CHECK_CONTAINER"; fi
}
STATE=$(fetch_state) \
  || { echo "FAIL: cannot read state of container '$CHECK_CONTAINER'" >&2; exit 2; }

# Flatten the inspect JSON to tag<TAB>fields lines (host python3, stdlib).
FLAT=$(python3 -c '
import json, sys
doc = json.load(sys.stdin)
c = doc[0] if isinstance(doc, list) else doc
cfg = c.get("Config") or {}
def emit(tag, *fields):
    clean = [str(f).replace("\t", " ").replace("\n", " ") for f in fields]
    print(tag + "\t" + "\t".join(clean))
emit("IMAGE", cfg.get("Image") or "")
emit("RUNNING", (c.get("State") or {}).get("Running"))
for e in cfg.get("Env") or []: emit("ENV", e)
for a in c.get("Args") or cfg.get("Cmd") or []: emit("ARG", a)
for m in c.get("Mounts") or []:
    emit("MOUNT", m.get("Source", ""), m.get("Destination", ""), m.get("Mode", ""))
' <<<"$STATE") \
  || { echo "FAIL: cannot parse container state (needs host python3)" >&2; exit 2; }

C_IMAGE=""; C_RUNNING=""; CMDLINE=""
ENV_PAIRS=(); MOUNTS=()
while IFS=$'\t' read -r tag rest; do
  case "$tag" in
    IMAGE)   C_IMAGE=$rest;;
    RUNNING) C_RUNNING=$rest;;
    ENV)     ENV_PAIRS+=("$rest");;
    ARG)     CMDLINE="${CMDLINE:+$CMDLINE }$rest";;
    MOUNT)   MOUNTS+=("$rest");;
  esac
done <<<"$FLAT"

cenv() {  # container env value by name, empty+1 if absent
  local p
  for p in ${ENV_PAIRS[@]+"${ENV_PAIRS[@]}"}; do
    case "$p" in "$1="*) echo "${p#*=}"; return 0;; esac
  done
  return 1
}
mount_src() {  # host source mounted at container path $1, empty+1 if absent
  local m s d md
  for m in ${MOUNTS[@]+"${MOUNTS[@]}"}; do
    IFS=$'\t' read -r s d md <<<"$m"
    [ "$d" = "$1" ] && { echo "$s"; return 0; }
  done
  return 1
}
ckv()   { grep -m1 -oE -- "$1=[^ ]+"      <<<"$CMDLINE"    | sed "s/^$1=//"       || true; }
sflag() { grep -m1 -oE -- "--$1[= ][^ ]*" <<<"$SERVE_ARGS" | sed -E "s/^--$1[= ]//" || true; }

# the vllm serve argument tail (empty for the worker container, which
# only joins ray and sleeps)
SERVE_ARGS=""
case "$CMDLINE" in *"vllm serve "*) SERVE_ARGS=${CMDLINE##*"vllm serve "};; esac
ROLE=""
case "$CMDLINE" in
  *"ray start --head"*) ROLE=head;;
  *"--address="*)       ROLE=worker;;
esac

# ---------- checks ----------

[ "$C_RUNNING" = "True" ] \
  || uv "container '$CHECK_CONTAINER' is not running; settings describe a stopped container"
if [ -n "$D_ENV_CONTAINER" ] && [ "$D_ENV_CONTAINER" != "$CHECK_CONTAINER" ]; then
  mm "container name: env declares CONTAINER=$D_ENV_CONTAINER, checked '$CHECK_CONTAINER'"
fi

# image (env: IMAGE, NCCL_IB, NCCL_IB_IMAGE)
[ "$C_IMAGE" = "$D_WANT_IMAGE" ] \
  || mm "image: env implies '$D_WANT_IMAGE' (NCCL_IB=$D_NCCL_IB), container runs '$C_IMAGE'"
if [ "$D_NCCL_IB" = "1" ]; then
  [ "$(cenv NCCL_IB_DISABLE || true)" = "0" ] \
    || mm "NCCL_IB=1 declared but container env lacks NCCL_IB_DISABLE=0"
elif cenv NCCL_IB_DISABLE >/dev/null; then
  mm "NCCL_IB=0 declared but container carries NCCL_IB_DISABLE=$(cenv NCCL_IB_DISABLE)"
fi

# STEP_ATTR (env: -e STEP_ATTR=1 plus instrumented overlay mounts)
if [ "$D_STEP_ATTR" = "1" ]; then
  [ "$(cenv STEP_ATTR || true)" = "1" ] \
    || mm "STEP_ATTR=1 declared but container env lacks STEP_ATTR=1"
  [ "$(cenv STEP_ATTR_EVERY || true)" = "$D_STEP_ATTR_EVERY" ] \
    || mm "STEP_ATTR_EVERY: declared $D_STEP_ATTR_EVERY, container '$(cenv STEP_ATTR_EVERY || echo none)'"
elif [ "$(cenv STEP_ATTR || true)" = "1" ]; then
  mm "container runs STEP_ATTR=1 (instrumented build) but env does not declare STEP_ATTR"
fi

# declared mounts: model dir and MTP draft dir
if [ -z "$D_MODEL_DIR" ]; then
  uv "MODEL_DIR: not declared in env file"
elif s=$(mount_src /checkpoint); then
  [ "$s" = "$D_MODEL_DIR" ] \
    || mm "model dir: env declares '$D_MODEL_DIR', container mounts '$s' at /checkpoint"
else
  mm "model dir: env declares '$D_MODEL_DIR' but container has nothing mounted at /checkpoint"
fi
if [ -n "$D_MTP_DIR" ]; then
  if s=$(mount_src /checkpoint-mtp); then
    [ "$s" = "$D_MTP_DIR" ] \
      || mm "MTP dir: env declares '$D_MTP_DIR', container mounts '$s' at /checkpoint-mtp"
  else
    mm "MTP dir: env declares '$D_MTP_DIR' but container has nothing mounted at /checkpoint-mtp"
  fi
elif s=$(mount_src /checkpoint-mtp); then
  mm "MTP dir: env declares none but container mounts '$s' at /checkpoint-mtp"
fi

# overlay mounts must come from the declared OVERLAY_DIR
for m in ${MOUNTS[@]+"${MOUNTS[@]}"}; do
  IFS=$'\t' read -r s d md <<<"$m"
  case "$d" in
    /usr/local/lib/python3.12/dist-packages/*)
      case "$s" in
        "$D_OVERLAY_DIR"/*) ;;
        *) mm "overlay mount '$d': source '$s' is not under declared OVERLAY_DIR '$D_OVERLAY_DIR'";;
      esac;;
  esac
done

# reaper line both scripts export inside the container command
v=$(ckv RAY_memory_usage_threshold)
[ "$v" = "0.99" ] \
  || mm "RAY_memory_usage_threshold: launch scripts set 0.99, container has '${v:-<absent>}'"

# node identity: head exports HEAD_IP, worker exports MY_IP and joins HEAD_IP
if [ -z "$D_HEAD_IP" ]; then uv "HEAD_IP: not declared in env file"; fi
if [ -z "$D_IFNAME" ];  then uv "IFNAME: not declared in env file"; fi
case "$ROLE" in
  head)   WANT_NODE_IP=$D_HEAD_IP; NODE_LABEL=HEAD_IP;;
  worker) WANT_NODE_IP=$D_MY_IP;   NODE_LABEL=MY_IP;;
  *)      WANT_NODE_IP="";         NODE_LABEL="";;
esac
if [ "$ROLE" = head ] && [ -n "$D_MY_IP" ]; then
  uv "MY_IP is declared in env file but the head container does not expose it (worker-only setting)"
fi
if [ "$ROLE" = worker ]; then
  [ -n "$D_MY_IP" ] || uv "MY_IP: worker container needs it but env file does not declare it"
  v=$(ckv --address); v=${v%%:*}
  if [ -n "$v" ] && [ -n "$D_HEAD_IP" ]; then
    [ "$v" = "$D_HEAD_IP" ] \
      || mm "ray --address: env declares HEAD_IP=$D_HEAD_IP, container joins '$v'"
  fi
fi
for var in VLLM_HOST_IP --node-ip-address; do
  v=$(ckv "$var")
  if [ -z "$v" ]; then uv "$var: container cmdline does not expose it"; continue; fi
  if [ -z "$NODE_LABEL" ]; then
    uv "$var=$v: cannot tell head from worker in container cmdline"
  elif [ -z "$WANT_NODE_IP" ]; then
    : # the matching D_* was already reported undeclared above
  else
    [ "$v" = "$WANT_NODE_IP" ] \
      || mm "$var: env declares $NODE_LABEL=$WANT_NODE_IP, container has '$v'"
  fi
done
for var in NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME; do
  v=$(ckv "$var")
  if [ -z "$v" ]; then uv "$var: container cmdline does not expose it"; continue; fi
  [ -z "$D_IFNAME" ] && continue
  [ "$v" = "$D_IFNAME" ] \
    || mm "$var: env declares IFNAME=$D_IFNAME, container has '$v'"
done

# serve-line checks (only meaningful on the head container)
if [ -n "$SERVE_ARGS" ]; then
  mpath=${SERVE_ARGS%% *}
  [ "$mpath" = "/checkpoint" ] \
    || mm "model path: serve line reads '$mpath', expected /checkpoint (the MODEL_DIR mount)"

  v=$(sflag served-model-name)
  [ "$v" = "GLM-5.3-Flash-NVFP4-Wabi" ] \
    || mm "served-model-name: start-head.sh sets GLM-5.3-Flash-NVFP4-Wabi, container has '${v:-<none>}'"

  v=$(sflag port)
  [ "$v" = "$D_PORT" ] \
    || mm "port: env declares PORT=$D_PORT, container serves '${v:-<none>}'"

  v=$(sflag tensor-parallel-size)
  [ "$v" = "2" ] \
    || mm "tensor-parallel-size: start-head.sh sets 2, container has '${v:-<none>}'"

  v=$(sflag max-model-len)
  [ "$v" = "204800" ] \
    || mm "max-model-len: start-head.sh sets 204800, container has '${v:-<none>}'"

  v=$(sflag max-num-seqs)
  [ "$v" = "20" ] \
    || mm "max-num-seqs: start-head.sh sets 20, container has '${v:-<none>}'"

  v=$(sflag gpu-memory-utilization)
  [ "$v" = "0.85" ] \
    || mm "gpu-memory-utilization: start-head.sh sets 0.85, container has '${v:-<none>}'"

  case "$SERVE_ARGS" in
    *--enable-expert-parallel*)
      mm "expert-parallel: container has --enable-expert-parallel; adopted configuration is EP-off";;
  esac

  if [ -n "$D_MTP_DIR" ]; then
    sc=$(grep -m1 -oE -- "--speculative-config[= ]+'?\{[^}]*\}" <<<"$SERVE_ARGS" || true)
    if [ -n "$sc" ]; then
      grep -q '"model" *: *"/checkpoint-mtp"' <<<"$sc" \
        || mm "speculative-config: expected \"model\":\"/checkpoint-mtp\", got '$sc'"
      k=$(grep -m1 -oE '"num_speculative_tokens" *: *[0-9]+' <<<"$sc" | grep -oE '[0-9]+' || true)
      [ "$k" = "$D_MTP_K" ] \
        || mm "num_speculative_tokens: env declares MTP_K=$D_MTP_K, container has '${k:-<none>}'"
    else
      mm "speculative-config: env declares MTP_DIR but container has no --speculative-config"
    fi
  else
    case "$SERVE_ARGS" in
      *--speculative-config*)
        mm "speculative-config: container has one but env declares no MTP_DIR";;
    esac
  fi
else
  for what in "model path" "served-model-name" "port" "tensor-parallel-size" \
              "max-model-len" "max-num-seqs" "gpu-memory-utilization" \
              "expert-parallel" "speculative-config"; do
    uv "$what: container cmdline has no 'vllm serve' to check against"
  done
fi

echo "RESULT: $MISM mismatch(es), $UNV unverifiable -- $CHECK_CONTAINER vs $ENV_FILE"
[ "$MISM" -eq 0 ]
