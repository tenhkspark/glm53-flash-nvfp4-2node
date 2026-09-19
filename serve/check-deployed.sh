#!/bin/bash
# check-deployed.sh <node> [<node>...]
#
# Compares the serving scripts in this serve/ directory against the copies
# deployed on the named nodes, by SHA-256 digest. Read-only: it only runs
# sha256sum over ssh (BatchMode, 8 s connect timeout); it never writes to a
# node and never touches containers.
#
# Prints one line per node and file:
#
#   OK:           the deployed digest matches this repository's copy
#   STALE:        the deployed file exists but its digest differs
#   MISSING:      the deployed file is absent (or unreadable)
#   UNREACHABLE:  ssh could not reach the node (never reported as matching)
#
# Exit status: 0 = everything matched, 1 = at least one STALE/MISSING,
# 2 = usage error, or every problem was only UNREACHABLE nodes.
#
# What is compared: every *.sh in this directory except this script itself.
# serve.env is site-specific (each pair carries its own) and deliberately
# not compared.
#
# Where the copies live: the deployment directory differs per pair of
# nodes, so it is not guessed. REMOTE_DIR_PER_NODE in setup.env (next to
# this script's repo root, the same file scripts/agent-run.sh sources)
# maps a node name (or the pair name agent-run's setup.env gives it) to
# its REMOTE_DIR; when a node is not named there the documented default
# REMOTE_DIR (setup.env.example: "relative to the remote $HOME") applies.
# Paths may be absolute or relative to the remote $HOME.
set -u

[ $# -ge 1 ] || { echo "usage: $0 <node> [<node>...]" >&2; exit 2; }
SELF=$(basename "$0")

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)

# ---- scripts to compare: the .sh files next to this script ----
SCRIPTS=""
for f in "$HERE"/*.sh; do
  [ -f "$f" ] || continue
  b=$(basename "$f")
  [ "$b" = "$SELF" ] && continue
  SCRIPTS="$SCRIPTS $b"
done
[ -n "$SCRIPTS" ] || { echo "FAIL: no serve scripts found in $HERE" >&2; exit 2; }

# ---- per-node deployment dir: REMOTE_DIR_PER_NODE from setup.env ----
# Format (shell sourceable):   REMOTE_DIR_PER_NODE="node1=repo30
#  node2=repo30-alt"           -- absolute paths allowed; $HOME-relative
# otherwise the plain REMOTE_DIR default below applies to every node.
PER_NODE=""
DEFAULT_DIR=""
if [ -f "$ROOT/setup.env" ]; then
  # shellcheck disable=SC1090
  . "$ROOT/setup.env"
  PER_NODE=${REMOTE_DIR_PER_NODE:-}
  DEFAULT_DIR=${REMOTE_DIR:-}
fi
DEFAULT_DIR=${DEFAULT_DIR:-glm53-repo30}

dir_for() { # $1 = node name -> deployment dir; absolute or $HOME-relative
  local n rest
  rest="$PER_NODE "   # trailing space so ${rest#* } always advances
  while [ -n "$rest" ]; do
    case "$rest" in
      "$1="*)
        rest=${rest#*=}
        n=${rest%% *}
        [ -n "$n" ] && { echo "$n"; return; }
        ;;
    esac
    rest=${rest#* }
  done
  echo "$DEFAULT_DIR"
}

STALE=0; UNREACH=0
for node in "$@"; do
  DIR=$(dir_for "$node")
  case "$DIR" in
    /*) SRV="$DIR/serve" ;;
    *)  SRV="\$HOME/$DIR/serve" ;;
  esac
  # one ssh round trip per node: cd to the deployed serve dir, digest all.
  # $SRV is deliberately unquoted so the remote shell expands $HOME.
  REMOTE_CMD="cd $SRV 2>/dev/null && sha256sum"
  for f in $SCRIPTS; do REMOTE_CMD="$REMOTE_CMD '$f'"; done
  OUT=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$node" "$REMOTE_CMD; true" 2>/dev/null)
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "UNREACHABLE: $node (ssh exit $rc)"
    UNREACH=$((UNREACH+1))
    continue
  fi
  seen=""
  while read -r h name; do
    [ -n "${h:-}" ] || continue
    seen="$seen $name"
    lh=$(shasum -a 256 "$HERE/$name" 2>/dev/null | awk '{print $1}')
    if [ "$h" = "$lh" ]; then
      echo "OK: $node $name $(echo "$h" | cut -c1-12)"
    else
      echo "STALE: $node $name repo=${lh:0:12} node=${h:0:12}"
      STALE=$((STALE+1))
    fi
  done <<EOF2
$OUT
EOF2
  for f in $SCRIPTS; do
    case " $seen " in *" $f "*) ;; *) echo "MISSING: $node $f"; STALE=$((STALE+1));; esac
  done
done

echo "RESULT: $STALE stale/missing, $UNREACH unreachable -- $# node(s) checked"
[ "$STALE" -eq 0 ]
