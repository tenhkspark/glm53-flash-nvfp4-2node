#!/bin/bash
# agent-run.sh — reproduce the whole repo30 setup on two DGX Spark nodes.
#
#   scripts/agent-run.sh --dry-run           print every command, run nothing
#   scripts/agent-run.sh                     run all steps in order
#   scripts/agent-run.sh prereqs pull ...    run only the named steps
#   scripts/agent-run.sh --env FILE          use a different env file
#
# Reads setup.env (cp setup.env.example setup.env first). Every step ends
# with a greppable line: "STEP <n> OK" / "STEP <n> SKIP (...)" /
# "STEP <n> FAIL: <cause>". Steps are idempotent: anything already done
# on the nodes is skipped. AGENTS.md lists the pass criteria and the
# failure modes.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

DRY=0
ENV_FILE="$ROOT/setup.env"
WANT=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --env) [ $# -ge 2 ] || { echo "--env needs a file" >&2; exit 2; }
           ENV_FILE=$2; shift 2 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) WANT+=("$1"); shift ;;
  esac
done

step() { printf '\n== STEP %s %s ==\n' "$1" "$2"; }
ok()   { printf 'STEP %s OK\n' "$1"; }
skip() { printf 'STEP %s SKIP (%s)\n' "$1" "$2"; }
fail() { printf 'STEP %s FAIL: %s\n' "$1" "$2" >&2; exit 1; }

# --- remote helpers -------------------------------------------------
# The operator may be the head node itself (the clean-room boot test runs
# this script on the head). A host equal to $(hostname), localhost,
# 127.0.0.1, or ::1 means "run locally, no ssh"; relative remote paths
# still resolve under $HOME, matching what an ssh session would see.
LOCAL_HOST=$(hostname 2>/dev/null || true)
is_local() { # $1 = host -> 0 when it names this machine
  case "$1" in localhost|127.0.0.1|::1) return 0 ;; esac
  [ -n "$LOCAL_HOST" ] || return 1
  [ "$1" = "$LOCAL_HOST" ] || [ "${1%%.*}" = "${LOCAL_HOST%%.*}" ]
}
# xcmd host 'cmd'     run cmd on host; local hosts run under bash with
#                     cwd=$HOME, matching an ssh session's start dir
xcmd() {
  if is_local "$1"; then (cd "$HOME" && bash -c "$2")
  else ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "$2"; fi
}
# drytag host         "ssh host" or "local" for --dry-run prints
drytag() { is_local "$1" && printf 'local' || printf 'ssh %s' "$1"; }
# rpath host 'path'   host:path for rsync; a $HOME-anchored path when local
rpath() {
  if ! is_local "$1"; then printf '%s:%s' "$1" "$2"; return; fi
  case "$2" in /*) printf '%s' "$2" ;; *) printf '%s/%s' "$HOME" "$2" ;; esac
}
# rsh host 'cmd'      run cmd on host (prints it under --dry-run)
rsh() {
  if [ "$DRY" = 1 ]; then printf 'DRY$ %s %q\n' "$(drytag "$1")" "$2"; return 0; fi
  xcmd "$1" "$2"
}
# rout host 'cmd'     like rsh but for $(capture) — stdout is the output
rout() {
  if [ "$DRY" = 1 ]; then printf 'DRY$ %s %q\n' "$(drytag "$1")" "$2" >&2; return 0; fi
  xcmd "$1" "$2"
}
# rdone host 'check'  0 = already done -> caller skips the action.
#                     Under --dry-run always returns 1 so the action prints.
rdone() {
  if [ "$DRY" = 1 ]; then
    printf 'DRY$ %s %q   # step skipped when this exits 0\n' "$(drytag "$1")" "$2" >&2
    return 1
  fi
  xcmd "$1" "$2" >/dev/null 2>&1
}
# rassert host 'cmd' step msg   cmd must exit 0 or the step FAILs
rassert() {
  if [ "$DRY" = 1 ]; then printf 'DRY$ %s %q   # must exit 0\n' "$(drytag "$1")" "$2"; return 0; fi
  xcmd "$1" "$2" || fail "$3" "$4"
}
# rcheck host 'cmd' step regex msg   cmd output must match regex
rcheck() {
  local h=$1 cmd=$2 n=$3 re=$4 msg=$5
  if [ "$DRY" = 1 ]; then
    printf 'DRY$ %s %q   # pass iff output matches /%s/\n' "$(drytag "$h")" "$cmd" "$re"
    return 0
  fi
  local out
  out=$(xcmd "$h" "$cmd" 2>&1) || fail "$n" "$msg (ssh failed)"
  printf '%s\n' "$out" | sed 's/^/  /'
  printf '%s\n' "$out" | grep -Eq "$re" || fail "$n" "$msg"
}
# run argv...         local command (prints under --dry-run)
run() {
  if [ "$DRY" = 1 ]; then printf 'DRY$ '; printf '%q ' "$@"; printf '\n'; return 0; fi
  "$@"
}
# WEIGHTS_CHECK: remote python3 (stdlib only) — exit 0 iff dir holds the
# full checkpoint: a parseable model.safetensors.index.json, every shard
# it names present and non-empty, and shard sizes summing to at least
# metadata.total_size.
WEIGHTS_CHECK='import json, os, sys
d = sys.argv[1]
try:
    idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
except Exception:
    sys.exit(1)
fs = set(idx.get("weight_map", {}).values())
sizes = [os.path.getsize(os.path.join(d, f))
         if os.path.isfile(os.path.join(d, f)) else 0 for f in fs]
sys.exit(0 if fs and all(sizes)
         and sum(sizes) >= idx.get("metadata", {}).get("total_size", 0)
         else 1)'
# weights_complete host dir -> 0 when the checkpoint at dir is whole
weights_complete() { rdone "$1" "python3 -c '$WEIGHTS_CHECK' '$2'"; }
# absdir host         absolute path of REMOTE_DIR on host (resolved, cached)
absdir() {
  if [ "$DRY" = 1 ]; then
    case "$REMOTE_DIR" in /*) echo "$REMOTE_DIR";; *) echo "<remote-home>/$REMOTE_DIR";; esac
    return 0
  fi
  rsh "$1" "mkdir -p -- '$REMOTE_DIR' && cd '$REMOTE_DIR' && pwd"
}

# --- step 1: env -----------------------------------------------------
step_env() {
  step 1 env
  [ -f "$ENV_FILE" ] || fail 1 "no $ENV_FILE (cp setup.env.example setup.env)"
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  local v val
  for v in HEAD_HOST WORK_HOST HEAD_IP HEAD_IF WORK_IP WORK_IF WEIGHTS_ROOT; do
    val=${!v:-}
    case "$val" in
      ""|*CHANGEME*) fail 1 "set $v in $ENV_FILE (now '${val:-unset}')" ;;
    esac
  done
  REQUANT_HOST=${REQUANT_HOST:-$HEAD_HOST}
  case "$REQUANT_HOST" in *CHANGEME*) fail 1 "set REQUANT_HOST";; esac
  STOCK_DIR=${STOCK_DIR:-$WEIGHTS_ROOT/GLM-5.3-Flash-NVFP4}
  MODEL_DIR=${MODEL_DIR:-$WEIGHTS_ROOT/GLM-5.3-Flash-NVFP4-h}
  MTP_DIR=${MTP_DIR-$WEIGHTS_ROOT/GLM-5.3-Flash-MTP-bf16}
  REMOTE_DIR=${REMOTE_DIR:-glm53-repo30}
  MODEL_ID=${MODEL_ID:-nvidia/GLM-5.3-Flash-NVFP4}
  MODEL_REVISION=${MODEL_REVISION:-main}
  IMAGE=${IMAGE:-vllm/vllm-openai:glm53-flash-arm64-cu130}
  NCCL_IB_IMAGE=${NCCL_IB_IMAGE:-$IMAGE-nccl-ib}
  NCCL_IB=${NCCL_IB:-1}
  MTP_K=${MTP_K:-2}
  PORT=${PORT:-8000}
  VERIFY_TOL=${VERIFY_TOL:-0.07}
  LABEL=${LABEL:-agent-run}
  RESULTS_DIR=${RESULTS_DIR:-results}
  case "$NCCL_IB" in 0|1) ;; *) fail 1 "NCCL_IB must be 0 or 1";; esac
  case "$MTP_K" in ''|*[!0-9]*) fail 1 "MTP_K must be a small integer (2)";;
    *) [ "$MTP_K" -le 3 ] || fail 1 "MTP_K=$MTP_K: K>=4 does not boot on 128 GB unified memory (AGENTS.md)";;
  esac
  case "$REQUANT_HOST" in
    "$HEAD_HOST") OTHER_HOST=$WORK_HOST ;;
    "$WORK_HOST") OTHER_HOST=$HEAD_HOST ;;
    *) fail 1 "REQUANT_HOST must be HEAD_HOST or WORK_HOST" ;;
  esac
  local t
  for t in ssh rsync python3; do
    command -v "$t" >/dev/null || fail 1 "local tool missing: $t"
  done
  ok 1
}

# --- step 2: prereqs --------------------------------------------------
step_prereqs() {
  step 2 prereqs
  local h ifn
  for h in "$HEAD_HOST" "$WORK_HOST"; do
    [ "$h" = "$HEAD_HOST" ] && ifn=$HEAD_IF || ifn=$WORK_IF
    rcheck "$h" 'cat /etc/dgx-release 2>/dev/null || lsb_release -ds 2>/dev/null || uname -srm' \
      2 '.' "no OS release info on $h"
    rcheck "$h" 'uname -m' 2 'aarch64' "$h is not aarch64 (expected a DGX Spark GB10)"
    rassert "$h" 'docker info >/dev/null 2>&1' \
      2 "docker not usable on $h (user in the docker group? daemon up?)"
    rcheck "$h" 'nvidia-smi --query-gpu=name --format=csv,noheader' \
      2 'GB10' "no GB10 GPU on $h (nvidia-smi)"
    rassert "$h" 'test -d /dev/infiniband && ls /dev/infiniband/uverbs* >/dev/null 2>&1' \
      2 "no uverbs under /dev/infiniband on $h (host RDMA stack down)"
    rcheck "$h" "(ibdev2netdev 2>/dev/null || rdma link 2>/dev/null) | grep -w '$ifn'" \
      2 "$ifn" "netdev $ifn not mapped to an RDMA HCA on $h"
  done
  # disk: >=800 GB on the requant host (stock 191G + requant out + draft
  # + two docker images); >=400 GB on the other node (weights copy + image)
  local n avail min
  for n in "$REQUANT_HOST:800" "$OTHER_HOST:400"; do
    h=${n%%:*}; min=${n##*:}
    avail=$(rout "$h" "df -BG --output=avail '$WEIGHTS_ROOT' 2>/dev/null | tail -1 | tr -dc 0-9")
    if [ "$DRY" = 0 ]; then
      [ -n "$avail" ] || fail 2 "cannot stat $WEIGHTS_ROOT on $h (create it first)"
      [ "$avail" -ge "$min" ] || fail 2 "$h has ${avail}G free under $WEIGHTS_ROOT; need >=${min}G"
      printf '  %s: %sG free (need %sG)\n' "$h" "$avail" "$min"
    fi
  done
  # inter-node ssh by hostname (weights rsync rides the fast link)
  rassert "$REQUANT_HOST" "ssh -o BatchMode=yes -o ConnectTimeout=5 '$OTHER_HOST' true" \
    2 "$REQUANT_HOST cannot ssh to $OTHER_HOST by hostname (ssh keys + /etc/hosts on the nodes)"
  # weights: a complete stock checkpoint under STOCK_DIR means STEP 4
  # skips its download and nothing else is needed here. Otherwise STEP 4
  # fetches it with snapshot_download inside $IMAGE, so the host never
  # needs an hf CLI or a pip install (externally-managed distros refuse
  # pip --user — PEP 668). Manual fallback: pip --user + hf download,
  # documented in AGENTS.md section 0.
  if weights_complete "$REQUANT_HOST" "$STOCK_DIR"; then
    printf '  %s: weights complete at %s\n' "$REQUANT_HOST" "$STOCK_DIR"
    printf 'STEP 2 OK (weights present)\n'
    return
  fi
  [ "$DRY" = 1 ] || printf '  %s: no complete checkpoint at %s — STEP 4 downloads inside %s\n' \
    "$REQUANT_HOST" "$STOCK_DIR" "$IMAGE"
  ok 2
}

# --- step 3: sync repo -------------------------------------------------
step_sync() {
  step 3 sync
  local h
  for h in "$HEAD_HOST" "$WORK_HOST"; do
    rsh "$h" "mkdir -p -- '$REMOTE_DIR'"
    run rsync -az --delete-excluded \
      --exclude .git --exclude results --exclude __pycache__ \
      --exclude setup.env --exclude serve/serve.env \
      --exclude overlays/_image --exclude overlays/_build \
      "$ROOT/" "$(rpath "$h" "$REMOTE_DIR/")"
  done
  ok 3
}

# --- step 4: pull image + checkpoint -----------------------------------
step_pull() {
  step 4 pull
  local h
  for h in "$HEAD_HOST" "$WORK_HOST"; do
    if rdone "$h" "docker image inspect '$IMAGE' >/dev/null 2>&1"; then
      printf '  %s: image present\n' "$h"
    else
      rsh "$h" "docker pull '$IMAGE'" || fail 4 "docker pull $IMAGE failed on $h"
    fi
  done
  if weights_complete "$REQUANT_HOST" "$STOCK_DIR"; then
    printf '  %s: checkpoint present\n' "$REQUANT_HOST"
  else
    # the download runs inside the serving image: nothing is installed
    # on the host (pip --user + hf download is the documented fallback).
    # --entrypoint python3 on every in-image helper run: the image's
    # ENTRYPOINT is the vllm CLI, which needs a GPU just to build its
    # serve parser and would eat this argv anyway — the helpers (this
    # download, requant.py, build-mtp-draft.py) are pure CPU file work.
    # --user <remote uid:gid> keeps the written files owned by the
    # operator: root-owned shards fail the follow-up rsync with
    # "Permission denied" (2026-09-15 requant a-2/b-2, 2026-09-16
    # cleanroom-h-boot-j). -e HOME=/tmp gives the uid-less container
    # user a writable HOME.
    rsh "$REQUANT_HOST" "mkdir -p -- '$STOCK_DIR' && docker run --rm \
      --user \$(id -u):\$(id -g) -e HOME=/tmp \
      --entrypoint python3 \
      -v '$STOCK_DIR':/w '$IMAGE' -c 'from huggingface_hub import snapshot_download; snapshot_download(\"$MODEL_ID\", revision=\"$MODEL_REVISION\", local_dir=\"/w\")'" \
      || fail 4 "checkpoint download failed on $REQUANT_HOST"
  fi
  rassert "$REQUANT_HOST" "test -f '$STOCK_DIR/config.json' && test -f '$STOCK_DIR/model.safetensors.index.json'" \
    4 "checkpoint incomplete at $STOCK_DIR"
  # every file the helpers wrote must be readable by the operator —
  # the step-8 rsync runs unprivileged
  rcheck "$REQUANT_HOST" "find '$STOCK_DIR' ! -readable -print | head" \
    4 '^$' "files under $STOCK_DIR unreadable by the operator (root-owned? chown and re-run)"
  ok 4
}

# --- step 5: RDMA derived image ----------------------------------------
step_rdma() {
  step 5 rdma
  if [ "$NCCL_IB" != 1 ]; then skip 5 "NCCL_IB=0 (sockets)"; return; fi
  local h
  for h in "$HEAD_HOST" "$WORK_HOST"; do
    if rdone "$h" "docker image inspect '$NCCL_IB_IMAGE' >/dev/null 2>&1"; then
      printf '  %s: derived image present\n' "$h"
    else
      # the Dockerfile self-verifies the MLX5_1.25 symbols after install
      rsh "$h" "cd '$REMOTE_DIR' && docker build -t '$NCCL_IB_IMAGE' - < docker/Dockerfile.nccl-ib" \
        || fail 5 "derived image build failed on $h (libmlx5 ABI check in the Dockerfile)"
    fi
  done
  ok 5
}

# --- step 6: overlays ----------------------------------------------------
step_overlays() {
  step 6 overlays
  if rdone "$HEAD_HOST" "cd '$REMOTE_DIR' && test -f overlays/_build/kda-quant.py -a -f overlays/_build/mla-quant.py -a -f overlays/_build/glm5next-mtp-bf16.py -a -f overlays/_build/flashinfer_mla_sparse_sm120.py"; then
    skip 6 "overlays already built on $HEAD_HOST"
  else
    rsh "$HEAD_HOST" "cd '$REMOTE_DIR' && overlays/build-overlays.sh" \
      || fail 6 "overlay build failed on $HEAD_HOST (a patcher refused: image drifted from the pinned tag)"
  fi
  ok 6
}

# --- step 7: requant route h + config gate -------------------------------
step_requant() {
  step 7 requant
  local rdir; rdir=$(absdir "$REQUANT_HOST")
  local mparent; mparent=$(dirname "$MODEL_DIR")
  if rdone "$REQUANT_HOST" "test -f '$MODEL_DIR/model.safetensors.index.json' -a -f '$MODEL_DIR/hf_quant_config.json'"; then
    skip 7 "requant output present at $MODEL_DIR"
  else
    rsh "$REQUANT_HOST" "mkdir -p -- '$MODEL_DIR' && docker run --rm \
      --user \$(id -u):\$(id -g) -e HOME=/tmp \
      --entrypoint python3 \
      -v '$STOCK_DIR':/src:ro -v '$mparent':'$mparent' -v '$rdir':/repo:ro \
      '$IMAGE' /repo/requant/requant.py --target h --src /src --dst '$MODEL_DIR'" \
      || fail 7 "requant died on $REQUANT_HOST (disk full? ENOSPC leaves a partial dir — free space and re-run)"
    rassert "$REQUANT_HOST" "test -f '$MODEL_DIR/model.safetensors.index.json'" \
      7 "requant finished without an index (incomplete output)"
  fi
  # static gate (pure stdlib; runs on the host python3)
  if [ "$DRY" = 1 ]; then
    printf 'DRY$ %s %q\n' "$(drytag "$REQUANT_HOST")" "cd '$rdir' && python3 requant/verify.py config '$MODEL_DIR'   # must print PASS, exit 0"
  else
    local gout
    gout=$(rsh "$REQUANT_HOST" "cd '$rdir' && python3 requant/verify.py config '$MODEL_DIR'" 2>&1) \
      || { printf '%s\n' "$gout"; fail 7 "verify.py config FAILED (a missing NVFP4 global scale shows as KeyError weight_scale_2 at load — this gate catches it earlier)"; }
    printf '%s\n' "$gout" | tail -3 | sed 's/^/  /'
    printf '%s\n' "$gout" | grep -q '^PASS' || fail 7 "verify.py config did not print PASS"
  fi
  rcheck "$REQUANT_HOST" "find '$MODEL_DIR' ! -readable -print | head" \
    7 '^$' "files under $MODEL_DIR unreadable by the operator (root-owned? chown and re-run)"
  ok 7
}

# --- step 8: MTP draft + weights to the worker ---------------------------
step_stage() {
  step 8 stage
  local rdir; rdir=$(absdir "$REQUANT_HOST")
  if [ -n "$MTP_DIR" ]; then
    local pparent; pparent=$(dirname "$MTP_DIR")
    if rdone "$REQUANT_HOST" "test -f '$MTP_DIR/model.safetensors.index.json'"; then
      printf '  %s: draft dir present\n' "$REQUANT_HOST"
    else
      rsh "$REQUANT_HOST" "mkdir -p -- '$MTP_DIR' && docker run --rm \
        --user \$(id -u):\$(id -g) -e HOME=/tmp \
        --entrypoint python3 \
        -v '$STOCK_DIR':/src:ro -v '$pparent':'$pparent' -v '$rdir':/repo:ro \
        '$IMAGE' /repo/requant/build-mtp-draft.py --target /src --out '$MTP_DIR'" \
        || fail 8 "draft build failed on $REQUANT_HOST"
    fi
    rassert "$REQUANT_HOST" "test -f '$MTP_DIR/config.json'" 8 "draft dir incomplete at $MTP_DIR"
    rcheck "$REQUANT_HOST" "find '$MTP_DIR' ! -readable -print | head" \
      8 '^$' "files under $MTP_DIR unreadable by the operator (root-owned? chown and re-run)"
  else
    printf '  MTP_DIR empty: serving without speculation\n'
  fi
  # weights + draft ride the fast link between the nodes
  rsh "$REQUANT_HOST" "rsync -az --partial '$MODEL_DIR/' '$OTHER_HOST:$MODEL_DIR/'" \
    || fail 8 "weights rsync $REQUANT_HOST -> $OTHER_HOST failed"
  [ -z "$MTP_DIR" ] || \
    rsh "$REQUANT_HOST" "rsync -az --partial '$MTP_DIR/' '$OTHER_HOST:$MTP_DIR/'" \
    || fail 8 "draft rsync failed"
  # built overlays to the worker (built on the head in step 6)
  local hdir; hdir=$(absdir "$HEAD_HOST")
  rsh "$HEAD_HOST" "rsync -az '$hdir/overlays/_build/' '$WORK_HOST:$REMOTE_DIR/overlays/_build/'" \
    || fail 8 "overlay rsync $HEAD_HOST -> $WORK_HOST failed"
  ok 8
}

# --- step 9: serve -------------------------------------------------------
write_serve_env() { # $1 = node role (head|worker); prints file path
  local role=$1 f
  f=$(mktemp "${TMPDIR:-/tmp}/serve-env.XXXXXX")
  {
    echo "export HEAD_IP=$HEAD_IP"
    if [ "$role" = worker ]; then
      echo "export MY_IP=$WORK_IP"
      echo "export IFNAME=$WORK_IF"
    else
      echo "export IFNAME=$HEAD_IF"
    fi
    echo "export MODEL_DIR=$MODEL_DIR"
    [ -z "$MTP_DIR" ] || echo "export MTP_DIR=$MTP_DIR"
    echo "export MTP_K=$MTP_K"
    echo "export IMAGE=$IMAGE"
    echo "export NCCL_IB=$NCCL_IB"
    echo "export NCCL_IB_IMAGE=$NCCL_IB_IMAGE"
    echo "export PORT=$PORT"
  } > "$f"
  echo "$f"
}

step_serve() {
  step 9 serve
  if [ "$DRY" = 0 ] && \
     rsh "$HEAD_HOST" "curl -sf --max-time 5 'http://127.0.0.1:$PORT/v1/models' | grep -q GLM" \
     >/dev/null 2>&1; then
    skip 9 "API already READY on $HEAD_HOST:$PORT"
    return
  fi
  local hf wf
  hf=$(write_serve_env head); wf=$(write_serve_env worker)
  if [ "$DRY" = 1 ]; then
    printf 'DRY$ # serve.env (head):\n'; sed 's/^/DRY$ #   /' "$hf"
    printf 'DRY$ # serve.env (worker):\n'; sed 's/^/DRY$ #   /' "$wf"
  fi
  run rsync -az "$hf" "$(rpath "$HEAD_HOST" "$REMOTE_DIR/serve/serve.env")"
  run rsync -az "$wf" "$(rpath "$WORK_HOST" "$REMOTE_DIR/serve/serve.env")"
  rm -f "$hf" "$wf"
  rsh "$HEAD_HOST" "cd '$REMOTE_DIR' && set -a; . serve/serve.env; set +a; bash serve/start-head.sh" \
    || fail 9 "head start failed on $HEAD_HOST"
  # worker joins the head's Ray cluster: wait for the Ray port first
  if [ "$DRY" = 1 ]; then
    printf 'DRY$ %s %q   # poll every 5s, up to 5min: Ray head port up\n' \
      "$(drytag "$HEAD_HOST")" "bash -c '</dev/tcp/127.0.0.1/6399'"
  else
    local deadline=$((SECONDS + 300)) ray=0
    while [ $SECONDS -lt $deadline ]; do
      if rsh "$HEAD_HOST" "bash -c '</dev/tcp/127.0.0.1/6399' 2>/dev/null"; then
        ray=1; break
      fi
      sleep 5
    done
    [ "$ray" = 1 ] || fail 9 "Ray head port 6399 never came up on $HEAD_HOST (docker logs glm53-head)"
  fi
  rsh "$WORK_HOST" "cd '$REMOTE_DIR' && set -a; . serve/serve.env; set +a; bash serve/start-worker.sh" \
    || fail 9 "worker start failed on $WORK_HOST"
  # READY: weight load ~13 min + engine init; poll up to 40 min
  if [ "$DRY" = 1 ]; then
    printf 'DRY$ %s %q   # poll every 20s, up to 40min, until it prints GLM\n' \
      "$(drytag "$HEAD_HOST")" "curl -sf http://127.0.0.1:$PORT/v1/models"
  else
    local deadline=$((SECONDS + 2400)) ready=0
    while [ $SECONDS -lt $deadline ]; do
      if rsh "$HEAD_HOST" "curl -sf --max-time 5 'http://127.0.0.1:$PORT/v1/models' 2>/dev/null | grep -q GLM"; then
        ready=1; break
      fi
      sleep 20
    done
    [ "$ready" = 1 ] || fail 9 "not READY after 40min (docker logs glm53-head on $HEAD_HOST; see AGENTS.md failure modes)"
  fi
  if [ "$NCCL_IB" = 1 ]; then
    if [ "$DRY" = 1 ]; then
      printf 'DRY$ %s %q   # must exit 0: NET/IB bound, not sockets\n' \
        "$(drytag "$HEAD_HOST")" "cd '$REMOTE_DIR/serve' && . ./nccl-ib.sh && nccl_ib_gate glm53-head"
    else
      rsh "$HEAD_HOST" "cd '$REMOTE_DIR/serve' && . ./nccl-ib.sh && nccl_ib_gate glm53-head" \
        || fail 9 "NCCL NET/IB gate failed (libmlx5 ABI? missing /dev/infiniband devices? see AGENTS.md)"
    fi
  fi
  ok 9
}

# --- step 10: ruler ------------------------------------------------------
step_ruler() {
  step 10 ruler
  [ "$DRY" = 1 ] || mkdir -p "$ROOT/$RESULTS_DIR"
  # run the ruler on the head node (stdlib-only script), pull the json back
  rsh "$HEAD_HOST" "cd '$REMOTE_DIR' && mkdir -p results && python3 bench/measure.py \
    --url 'http://127.0.0.1:$PORT' --label '$LABEL' \
    --prompts bench/prompts-64.jsonl --levels 1:64 --outdir results" \
    || fail 10 "ruler failed on $HEAD_HOST"
  rassert "$HEAD_HOST" "test -f '$REMOTE_DIR/results/measure-$LABEL.json'" \
    10 "ruler wrote no result file"
  run rsync -az "$(rpath "$HEAD_HOST" "$REMOTE_DIR/results/measure-$LABEL.json")" "$ROOT/$RESULTS_DIR/"
  [ "$DRY" = 1 ] || [ -f "$ROOT/$RESULTS_DIR/measure-$LABEL.json" ] \
    || fail 10 "result file did not come back"
  ok 10
}

# --- step 11: verify -----------------------------------------------------
step_verify() {
  step 11 verify
  local ref=h-nospec
  [ -n "$MTP_DIR" ] && ref=h-rdma-mtp
  [ "$NCCL_IB" = 1 ] || printf '  note: reference rows assume RDMA; NCCL_IB=0 expects below-band\n'
  run python3 "$HERE/verify-result.py" \
    --result "$ROOT/$RESULTS_DIR/measure-$LABEL.json" \
    --ref "$ref" --tol "$VERIFY_TOL" \
    || fail 11 "ruler result outside the ${VERIFY_TOL} band (see AGENTS.md verify)"
  ok 11
}

# --- driver --------------------------------------------------------------
ORDERED=(env prereqs sync pull rdma overlays requant stage serve ruler verify)
[ ${#WANT[@]} -eq 0 ] && WANT=("${ORDERED[@]}")
ran_env=0
for s in "${WANT[@]}"; do
  case " ${ORDERED[*]} " in *" $s "*) ;; *) echo "unknown step: $s" >&2; exit 2;; esac
done
for s in "${WANT[@]}"; do
  [ "$s" != env ] && [ "$ran_env" = 0 ] && { step_env >/dev/null; ran_env=1; }
  "step_$s"
  [ "$s" = env ] && ran_env=1
done
[ "$DRY" = 1 ] && printf '\nDRY-RUN done (nothing executed)\n'
exit 0
