#!/bin/bash
# nccl-ib.sh — derive the NCCL NET/IB (RoCE v2) docker args for a node.
#
# Source this file, then call:
#   nccl_ib_setup <netdev>     -> sets NCCL_IB_ENV and NCCL_IB_ARGS for
#                                 `docker run`; exits non-zero with a
#                                 FAIL line when the host cannot do RDMA.
#   nccl_ib_gate <container>   -> after the engine is READY, verify the
#                                 NET/IB plugin actually bound a device
#                                 (otherwise NCCL silently used sockets).
#
# The derived image (docker/Dockerfile.nccl-ib) is required: the base
# image's Ubuntu noble rdma-core 50.0 lacks the MLX5_1.25 symbols the
# NCCL plugin dlvsym's; host bind-mounts cannot satisfy the ABI because
# the hosts run the same 50.0 userspace.

nccl_ib_setup() {  # $1 = netdev name carrying RoCE (e.g. enp1s0f0np0)
  local ifname=$1 hca gid g ibp d
  # netdev -> RDMA device (ibdev2netdev, rdma link fallback)
  hca="$(ibdev2netdev 2>/dev/null \
      | awk -v n="$ifname" '$5==n {print $1; exit}' || true)"
  [ -z "$hca" ] && hca="$(rdma link 2>/dev/null \
      | awk -v n="$ifname" '$NF==n {sub("/.*","",$2); print $2; exit}' || true)"
  [ -n "$hca" ] || { echo "FAIL: no RDMA HCA mapped to netdev $ifname" >&2; return 1; }

  # RoCE v2 GID index: pick the IPv4-mapped entry in sysfs
  # (::ffff:a.b.c.d). sysfs index == NCCL_IB_GID_INDEX.
  gid=""
  ibp="${NCCL_IB_SYSFS:-/sys/class/infiniband}/$hca/ports/1"
  for g in "$ibp/gid_attrs/types/"*; do
    [ "$(cat "$g" 2>/dev/null)" = "RoCE v2" ] || continue
    grep -qi 'ffff:' "$ibp/gids/${g##*/}" 2>/dev/null || continue
    gid=${g##*/}; break
  done
  [ -n "$gid" ] || { echo "FAIL: no IPv4-mapped RoCE v2 GID on $hca port 1" >&2; return 1; }

  NCCL_IB_ENV="-e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET"
  NCCL_IB_ENV="$NCCL_IB_ENV -e NCCL_IB_DISABLE=0"
  NCCL_IB_ENV="$NCCL_IB_ENV -e NCCL_IB_HCA=$hca -e NCCL_IB_GID_INDEX=$gid"

  # NCCL_NCHANNELS (optional): cap the channel count. NCCL sizes its
  # per-channel buffers out of the same unified memory the engine serves
  # from, so on this part the default (64 coll / 64 p2p here) is host RAM
  # the model cannot use. Unset = the stock default, i.e. this knob
  # changes nothing unless you set it.
  if [ -n "${NCCL_NCHANNELS:-}" ]; then
    NCCL_IB_ENV="$NCCL_IB_ENV -e NCCL_MAX_NCHANNELS=$NCCL_NCHANNELS"
    NCCL_IB_ENV="$NCCL_IB_ENV -e NCCL_MAX_P2P_NCHANNELS=$NCCL_NCHANNELS"
  fi

  # --gpus all does not pass uverbs/rdma_cm: without --device the plugin
  # reports "No device found". memlock -1 is required for ibv_reg_mr
  # pinning (container default is only 8 MiB).
  NCCL_IB_ARGS="--cap-add IPC_LOCK --ulimit memlock=-1:-1"
  for d in "${NCCL_IB_DEVDIR:-/dev/infiniband}/"uverbs* \
           "${NCCL_IB_DEVDIR:-/dev/infiniband}/"rdma_cm; do
    [ -e "$d" ] && NCCL_IB_ARGS="$NCCL_IB_ARGS --device $d"
  done
  case "$NCCL_IB_ARGS" in
    *" --device "*) ;;
    *) echo "FAIL: no uverbs/rdma_cm under ${NCCL_IB_DEVDIR:-/dev/infiniband}" >&2; return 1;;
  esac
  echo "[nccl-ib] netdev=$ifname hca=$hca gid=$gid" >&2
}

nccl_ib_gate() {  # $1 = head container name; 0 iff NET/IB is live
  local lines
  lines="$(docker logs "$1" 2>&1 | grep -E 'NCCL INFO' \
    | grep -E 'NET/|Using network|GPU Direct RDMA|dlvsym|NCCL version' || true)"
  printf '%s\n' "$lines" | grep . | sed 's/^/[nccl-ib] /' >&2 || true
  grep -q 'NET/IB : No device found' <<<"$lines" && return 1
  grep -q 'Failed to initialize NET plugin IB' <<<"$lines" && return 1
  grep -qE 'NET/IB : Using|Initialized NET plugin IB|Assigned NET plugin IB|Using network IB' \
    <<<"$lines" || return 1
  grep -q 'via NET/IB' <<<"$lines" || return 1
  if grep -q 'GPU Direct RDMA Enabled' <<<"$lines"; then
    echo "[nccl-ib] GDR: enabled" >&2
  else
    echo "[nccl-ib] GDR: disabled (report only; not gated)" >&2
  fi
}
