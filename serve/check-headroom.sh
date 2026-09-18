#!/bin/bash
# check-headroom.sh — run ON each node right after the API reports READY
# and BEFORE you send the first request. Refuses the configuration when the
# node did not come up with enough free host memory to survive its own
# warm-up.
#
# Order matters: the warm-up cost below is charged by the first requests,
# so a node that legitimately passed at READY will report a lower number
# once it has served traffic. Running this against a warmed engine gives a
# FAIL that means nothing. Gate at boot, not during a session.
#
# Why this exists. GB10 is unified memory: the GPU allocates out of host
# RAM, and about 100 GB of that reservation appears in no standard kernel
# counter (nvidia-smi reports FB Memory Usage as N/A on this part). So
# MemAvailable is the only honest margin gauge, and the engine's peak is
# not paid at READY -- it is paid the first time each larger prefill shape
# is seen. Measured on this pair, route h, 204800/seqs 20/util 0.85 with
# the MTP K=2 draft, sampling MemAvailable once a second on both nodes
# while climbing 21k -> 32k -> 64k -> 128k -> 197k tokens:
#
#   READY 10270 MiB (8.24%) -> low water 5579 MiB (4.48%)   = 4691 MiB paid
#
# That 4691 MiB is a one-time high-water mark, not a leak: repeating the
# same length costs nothing more. But it is charged after READY, so a node
# that looks fine at READY can still die later. The same configuration has
# come up with as little as 5470 MiB free (4.39%) -- 4.8 GiB less than the
# best boot of the identical script -- and that boot was killed mid-session
# by the host OOM reaper during a 21k-token agent prompt. The engine budget
# is sized from whatever happened to be free when it profiled, so the
# margin is a lottery you have to check, not a constant you can assume.
#
# FLOOR is therefore warm-up (4691) + the reaper's own line (2% = 2492)
# rounded up: a node below it is expected to die under a long prompt, and
# the fix is to restart the engine, not to tune a flag.
#
# NOTE FOR READERS WITHOUT AN OOM DAEMON: DGX OS ships neither earlyoom nor
# systemd-oomd. On our nodes earlyoom turned this into a clean process kill
# after 6 hours of agent use. Without one, the same pressure is a hard node
# hang on this hardware. That is why this is a startup gate and not a
# monitoring threshold -- you cannot rely on something reaping you.
#
# usage: check-headroom.sh            # gate with the measured floor, at boot
#        FLOOR_MIB=9000 check-headroom.sh
set -euo pipefail
FLOOR_MIB=${FLOOR_MIB:-7500}
TOTAL_KB=$(awk '/MemTotal/{print $2}' /proc/meminfo)
AVAIL_KB=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
AVAIL_MIB=$((AVAIL_KB / 1024))
TOTAL_MIB=$((TOTAL_KB / 1024))
PCT=$(awk -v a="$AVAIL_MIB" -v t="$TOTAL_MIB" 'BEGIN{printf "%.2f", 100*a/t}')
echo "$(hostname): MemAvailable ${AVAIL_MIB} MiB of ${TOTAL_MIB} MiB (${PCT}%), floor ${FLOOR_MIB} MiB"
if [ "$AVAIL_MIB" -lt "$FLOOR_MIB" ]; then
  cat >&2 <<MSG
FAIL: this node came up with less host memory than its own warm-up needs.
(If the engine has already served requests, this number is expected to be
lower and this FAIL is not meaningful -- gate at boot, before traffic.)
  free now:  ${AVAIL_MIB} MiB (${PCT}%)
  needed:    ${FLOOR_MIB} MiB = 4691 MiB warm-up + 2492 MiB reaper line, rounded
Restart the engine (serve/start-head.sh, then serve/start-worker.sh) and
check again: the same script has produced anywhere from 5470 to 10270 MiB
free at READY on the same node. Do not lower --gpu-memory-utilization to
compensate: at 204800 the limiting rank only gets 3.48 GiB of KV, and 0.01
of utilization is 1.2 GiB, so two steps down and the engine no longer has
enough KV to open the declared window at all.
MSG
  exit 1
fi
echo "OK: enough headroom for the measured warm-up."
