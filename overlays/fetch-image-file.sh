#!/bin/bash
# fetch-image-file.sh — snapshot one file out of the deployed vLLM image
# into _image/ next to this script.
#
# Run it on a node that has the image (plain docker), or pass an ssh host
# as the second argument to read from a remote node. It runs
# `docker run --rm --entrypoint cat <image> <path>`: a throwaway container
# that only reads a file, so any running serving containers are untouched.
#
# usage: fetch-image-file.sh <relpath-under-dist-packages/vllm> [ssh-host]
#   e.g.  fetch-image-file.sh models/glm5next/nvidia/model.py
#         fetch-image-file.sh models/glm5next/nvidia/mtp.py my-node
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REL="${1:?usage: fetch-image-file.sh <relpath> [ssh-host]}"
HOST="${2:-}"
IMAGE="${IMAGE:-vllm/vllm-openai:glm53-flash-arm64-cu130}"
PKG=/usr/local/lib/python3.12/dist-packages/vllm
OUT_DIR="$DIR/_image"
OUT="$OUT_DIR/$(basename "$REL")"

fail() { echo "FAIL: fetch-image-file failed at step: $1 (exit ${2:-1})" >&2; exit "${2:-1}"; }

mkdir -p "$OUT_DIR"
tmp="$OUT.tmp.$$"
trap 'rm -f "$tmp"' EXIT

if [ -n "$HOST" ]; then
  ssh -o ConnectTimeout=10 -o BatchMode=yes "$HOST" \
      "docker run --rm --entrypoint cat $IMAGE $PKG/$REL" >"$tmp" \
      || fail fetch $?
else
  docker run --rm --entrypoint cat "$IMAGE" "$PKG/$REL" >"$tmp" \
      || fail fetch $?
fi
[ -s "$tmp" ] || fail empty 1
mv -f "$tmp" "$OUT"
trap - EXIT
printf 'fetch-image-file done: %s (%s lines, sha256 %s)\n' \
    "$OUT" "$(wc -l <"$OUT" | tr -d ' ')" \
    "$( (shasum -a 256 "$OUT" 2>/dev/null || sha256sum "$OUT") | cut -d' ' -f1)"
