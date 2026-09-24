# Serving pair bring-up runbook

Written 2026-09-19 against a live four-node fleet. Every value below was
verified that day against a running container (`docker inspect`), an
on-node file, or command output — the per-value evidence is in the
table at the end. Where an older document disagrees with this file,
this file is right; the disagreements are listed in the last section
with the document that should be corrected.

**Placeholders.** Site-specific names are written as placeholders.
Substitute your own values, consistently, wherever they appear:

| placeholder | substitute with |
|---|---|
| `node1` … `node4` | ssh aliases for the nodes — `node1`/`node2` are pair 1 (head/worker), `node3`/`node4` are pair 2 (head/worker) |
| `<user>` | the login account on the nodes; every `~`/`$HOME` path resolves under that account's home directory |
| `192.0.2.x`, `198.51.100.x`, `203.0.113.x` | RFC 5737 documentation addresses standing in for the nodes' point-to-point link addresses |
| `<weights-root>` | an absolute directory on each node that holds the checkpoint trees (the requant node needs ≥800 GB free under it) |
| `<mdns-name>` | a node's factory mDNS name — the only name the nodes resolve for each other on the LAN |

## 0. The production configuration, in one sentence

Production is `serve/start-head.sh`, `serve/start-worker.sh` and
`serve/nccl-ib.sh` from this repository run unmodified — expert
parallel OFF — on two nodes over a direct ConnectX link, driven by a
per-node `serve/serve.env` that sets the requant-h checkpoint, the BF16
MTP draft (`MTP_K=2`), `PORT=8888`, `NCCL_IB=1`, `NCCL_NCHANNELS=8` and
`JIT_CACHE_DIR`, serving `GLM-5.3-Flash-NVFP4-Wabi` at
`http://<head>:8888/v1`.

What is **not** production, no matter what older docs say:

- `tools/dgx/llm/glm-serve.sh` and its `serve-official-tp2-*` scripts
  launch `glm53-official-head` / `glm53-official-worker` containers on
  the *stock* checkpoint — no MTP, no prefix caching, a 16384-token
  window, 32 seqs, and no `RAY_memory_usage_threshold=0.99`. That is a
  measurably worse, different serve, and `tools/dgx/mon/pair-relaunch.sh`
  calls it by default. Do not use either to restore this pair.
- The retired EXL3 stack (`~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`,
  containers pinned by digest) is dead state; `nodes.tsv` still
  describes it.
- `--enable-expert-parallel` was hand-added to the node copies on
  2026-09-17 and measured slower than off (35.05 vs 37.33 tok/s on the
  same pair); commit `3e42776` shipped EP off and the flag is no longer
  on either live head. Do not re-add it — `verify-serve.sh` flags it as
  a MISMATCH.
- `NCCL_IB=0` (sockets) is a supported fallback for bring-up on
  hardware without the RDMA link, not the production transport.

## 1. Prerequisites

### 1a. Operator machine

- Non-interactive ssh to both nodes (`ssh -o BatchMode=yes <node>`
  must work). Concretely that means: the operator's public key is
  authorized for `<user>` on each node (a key comment identifying the
  operator machine helps), the operator's ssh client config carries
  one `Host` block per node (`User <user>`, `HostName` set to the
  node's LAN address or `<mdns-name>`), and `/etc/hosts` or DNS pins
  the aliases to addresses that reach the nodes — a wired fallback
  path between the operator and a node, independent of the usual
  route, is worth having (the reference fleet keeps a direct wire to
  node4 for that). Substitute your own account, key, name resolution
  and L2 path; nothing else changes.
- Node account: `<user>` — any name works. The repo30 scripts are
  `$HOME`-relative and do not require a particular account name;
  helper scripts outside this repository may hardcode one — check
  before reusing them.
- Local `ssh`, `rsync`, `python3`. **macOS warning:** `/usr/bin/rsync`
  on macOS is openrsync, which crashes `agent-run.sh` STEP 3 against
  the nodes' rsync 3.2.7 and misparses `host:path` destinations —
  deploy with `tar` over ssh or install rsync 3 (recorded in
  the pair-1 rebuild log, internal).

### 1b. The pair link (on-node state until now — all verified)

Each node needs its ConnectX port addressed on a point-to-point /30
with `mtu: 9000`, via `/etc/netplan/99-qsfp.yaml` (`dhcp4: false`; NOPASSWD
sudo to `netplan apply`). Verified live values:

| node | serving netdev | link address | peer |
|---|---|---|---|
| node1 | `enp1s0f0np0` | `192.0.2.2/30` | node2 `enp1s0f1np1` `192.0.2.1/30` |
| node2 | `enp1s0f1np1` | `192.0.2.1/30` | node1 `enp1s0f0np0` `192.0.2.2/30` |
| node3 | `enp1s0f0np0` | `198.51.100.2/30` | node4 `enp1s0f1np1` `198.51.100.1/30` |
| node4 | `enp1s0f1np1` | `198.51.100.1/30` | node3 `enp1s0f0np0` `198.51.100.2/30` |

The fleet also runs two cross-links (same netplan files, same MTU,
unused by the serve): node2↔node3 `203.0.113.0/30` (node2
`enp1s0f0np0`=.1, node3 `enp1s0f1np1`=.2) and node4↔node1
`203.0.113.4/30` (node4 `enp1s0f0np0`=.2, node1 `enp1s0f1np1`=.1) — a
4-node ring. `enP2p1s0f0np0`/`enP2p1s0f1np1` are alternate logical
interfaces of the same physical ports; do not configure them.

On fresh hardware you must re-derive which physical port faces which
peer before writing the file — the procedure is the `ping6 ff02::1`
multicast + EUI-64 method described in the live yaml comments. A
hand-rolled `ip addr add` without `mtu 9000` silently runs 1500 and
"works" wrong.

The maintainer's setup tree (not shipped) copies of `netplan/99-qsfp-*.yaml` are
stale (pre-ring, one port each) and the node4 copy is missing —
use the table above.

### 1c. Node-to-node ssh (needed for weights staging and `agent-run.sh` STEP 2)

- The nodes **cannot resolve each other's bare aliases**. Only
  `<mdns-name>` resolves (mDNS, on the LAN) — and the node keys are
  bound to link IPs, not to that name. Verified mechanism: each node's
  ssh client config carries `Host` entries keyed by peer link IP whose
  `IdentityFile` points at the private key authorized for `<user>`
  (node1 uses a dedicated key, node3 the account default), and the
  peer's public key is authorized for `<user>`. ssh to the /30
  address works; ssh to the mDNS name does not.
- `agent-run.sh` STEP 2 additionally requires `$REQUANT_HOST` to
  `ssh $OTHER_HOST` *by the setup.env hostname*. Provision that before
  running it — e.g. a `Host node2 → HostName 192.0.2.1` block in the
  node's ssh client config, or an `/etc/hosts` line — or STEP 2 fails
  with `cannot ssh to <host> by hostname`.

### 1d. Daemons the serve relies on

- **earlyoom** — `apt install earlyoom` (fleet runs 1.7-2), then
  `/etc/default/earlyoom` carrying
  `EARLYOOM_ARGS="-m 2 -s 100,100 --prefer ^(vllm|VLLM::) --avoid ^(sshd|systemd)$"`
  and `systemctl enable --now earlyoom`. The
  maintainer's copy of `earlyoom.default` (verified byte-identical
  args). DGX OS ships no OOM daemon; without one, the memory pressure
  this serve produces is a hung node, not a killed process.
- `docker` usable by the account (docker group membership — `nodes.tsv`
  claims node4 lacks it; that note is stale, verified working),
  `ibdev2netdev`/`rdma` for the RDMA path, uverbs under
  `/dev/infiniband`.
- Other units in the maintainer's setup tree (not shipped) (`temp-guard.service`,
  `gpu-clock-limit.service`, `watchdog-connectx.sh`, `governor/`) have
  no recorded per-node activation state; the serve does not require
  them.

### 1e. Artifacts on both nodes

| artifact | path on each node | notes |
|---|---|---|
| requant-h checkpoint | `<weights-root>/GLM-5.3-Flash-NVFP4-requant-h` | 181 GB; sha256-verified against `../hf-route-h/SHA256SUMS` (published alongside this repo); also published as HF `tenhkspark/GLM-5.3-Flash-NVFP4-Wabi` |
| MTP draft | `<weights-root>/GLM-5.3-Flash-MTP-ja-base` | 17 GB; 891 tensors / 16.21 GiB — the same artifact `requant/build-mtp-draft.py` produces as `GLM-5.3-Flash-MTP-bf16` (measured equivalent; the headline row ran on it) |
| base image | `vllm/vllm-openai:glm53-flash-arm64-cu130` | image ID `b0501f99fec5` on all four nodes |
| derived image | `vllm/vllm-openai:glm53-flash-arm64-cu130-nccl-ib` | built per node by `docker/Dockerfile.nccl-ib`; IDs differ per node, that is expected |
| built overlays | `<REMOTE_DIR>/overlays/_build/` | produced by `overlays/build-overlays.sh`; the serve scripts bind-mount from it |
| repo copy | `~/glm53-repo30` (pair 1), `~/dogfood/repo30` (pair 2) | `REMOTE_DIR_PER_NODE` in `setup.env` maps these for `check-deployed.sh` |

Disk floor enforced by `agent-run.sh`: ≥800 GB free under
`WEIGHTS_ROOT` on the requant node, ≥400 GB on the other. Outbound
network is needed at serve time too: every boot runs `pip install -q
ray` inside the container (PyPI), and STEP 4 pulls the image and the
stock checkpoint from HF when absent.

## 2. Layout of the live fleet (verified)

| node | role | container | deploy dir |
|---|---|---|---|
| node1 | pair 1 head | `glm53-head` | `~/glm53-repo30` |
| node2 | pair 1 worker | `glm53-worker` | `~/glm53-repo30` |
| node3 | pair 2 head | `glm53-head` | `~/dogfood/repo30` |
| node4 | pair 2 worker | `glm53-worker` | `~/dogfood/repo30` |

RoCE details derived by `nccl-ib.sh` at launch: HCA `rocep1s0f0` on the
`enp1s0f0np0` side, `rocep1s0f1` on `enp1s0f1np1`, `NCCL_IB_GID_INDEX=3`
(IPv4-mapped RoCE v2) on all four.

## 3. Bring-up: head, then worker

### 3.0 Sync the repo, then confirm with the drift checker

Deploy this repository to each node's `REMOTE_DIR` (exclude `.git`,
`results`, `setup.env`, `serve/serve.env`, `overlays/_image`,
`overlays/_build` — the same set `agent-run.sh` STEP 3 uses), then
from the operator:

```bash
bash serve/check-deployed.sh node1 node2   # pair 1
bash serve/check-deployed.sh node3 node4   # pair 2
```

It compares each deployed `serve/*.sh` by SHA-256 against this repo and
prints `OK` / `STALE` / `MISSING` / `UNREACHABLE` per file; `RESULT: 0
stale/missing` is the pass line. It reads `REMOTE_DIR_PER_NODE` from
`setup.env` at the repo root (`node1=glm53-repo30
node2=glm53-repo30 node3=dogfood/repo30 node4=dogfood/repo30`;
`setup.env` is gitignored, so create it on a fresh clone from
`setup.env.example` — without it every node is checked under the
default `glm53-repo30`, which is wrong for pair 2). **A `STALE` line
means someone edited the node copy — that is exactly how the EP drift
should have been caught.** Fix by re-syncing, not by editing on the
node.

### 3.1 Write `serve/serve.env` on each node

Per-node file, deliberately excluded from repo sync. These are the
verified live bodies with placeholders per the table at the top —
copy the matching one and substitute:

```bash
# pair 1 head — node1:~/glm53-repo30/serve/serve.env
export HEAD_IP=192.0.2.2
export IFNAME=enp1s0f0np0
export MODEL_DIR=<weights-root>/GLM-5.3-Flash-NVFP4-requant-h
export IMAGE=vllm/vllm-openai:glm53-flash-arm64-cu130
export NCCL_IB=1
export NCCL_IB_IMAGE=${IMAGE}-nccl-ib
export PORT=8888
export MTP_DIR=<weights-root>/GLM-5.3-Flash-MTP-ja-base
export MTP_K=2
export NCCL_NCHANNELS=8
export JIT_CACHE_DIR=$HOME/jit-cache
```

```bash
# pair 1 worker — node2:~/glm53-repo30/serve/serve.env
export HEAD_IP=192.0.2.2
export MY_IP=192.0.2.1
export IFNAME=enp1s0f1np1
export MODEL_DIR=<weights-root>/GLM-5.3-Flash-NVFP4-requant-h
export IMAGE=vllm/vllm-openai:glm53-flash-arm64-cu130
export NCCL_IB=1
export NCCL_IB_IMAGE=${IMAGE}-nccl-ib
export PORT=8888
export MTP_DIR=<weights-root>/GLM-5.3-Flash-MTP-ja-base
export MTP_K=2
export NCCL_NCHANNELS=8
export JIT_CACHE_DIR=$HOME/jit-cache
```

```bash
# pair 2 head — node3:~/dogfood/repo30/serve/serve.env
export HEAD_IP=198.51.100.2
export MY_IP=198.51.100.2                 # set but unused by start-head.sh
export IFNAME=enp1s0f0np0
export MODEL_DIR=<weights-root>/GLM-5.3-Flash-NVFP4-requant-h
export MTP_DIR=<weights-root>/GLM-5.3-Flash-MTP-ja-base
export MTP_K=2
export IMAGE=vllm/vllm-openai:glm53-flash-arm64-cu130
export NCCL_IB=1
export NCCL_IB_IMAGE=${IMAGE}-nccl-ib
export PORT=8888
export NCCL_NCHANNELS=8
export JIT_CACHE_DIR=$HOME/dogfood/jit-cache
```

```bash
# pair 2 worker — node4:~/dogfood/repo30/serve/serve.env
export HEAD_IP=198.51.100.2
export MY_IP=198.51.100.1
export IFNAME=enp1s0f1np1
export MODEL_DIR=<weights-root>/GLM-5.3-Flash-NVFP4-requant-h
export MTP_DIR=<weights-root>/GLM-5.3-Flash-MTP-ja-base
export MTP_K=2
export IMAGE=vllm/vllm-openai:glm53-flash-arm64-cu130
export NCCL_IB=1
export NCCL_IB_IMAGE=${IMAGE}-nccl-ib
export PORT=8888
export NCCL_NCHANNELS=8
export JIT_CACHE_DIR=$HOME/dogfood/jit-cache
```

What each variable does and where it must be set:

| variable | set on | required? | meaning / consequence |
|---|---|---|---|
| `HEAD_IP` | both | yes | head's link IP; worker joins it at `:6399`, vLLM binds `VLLM_HOST_IP` to it |
| `MY_IP` | worker only | yes there | worker's own link IP (`--node-ip-address`, `VLLM_HOST_IP`) |
| `IFNAME` | both | yes | that node's own netdev; drives `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME` and the HCA/GID derivation |
| `MODEL_DIR` | both | yes | absolute path to the requant-h dir on *that* node; bind-mounted to `/checkpoint` |
| `MTP_DIR` | both | for production | absolute path to the draft dir; mounted to `/checkpoint-mtp` and adds `--speculative-config …num_speculative_tokens=$MTP_K`. Unset = no speculation, ~20% slower — the single most common silent deviation |
| `MTP_K` | both | yes | 2. `agent-run.sh` refuses K≥4 (does not boot on 128 GB) |
| `PORT` | both | yes | 8888. Script default is 8000; every client (e.g. a base URL of `http://node1:8888/v1`) assumes 8888 |
| `IMAGE` | both | yes | `vllm/vllm-openai:glm53-flash-arm64-cu130` |
| `NCCL_IB` | both | yes | 1 = derived image + NET/IB (production); 0 = stock image, sockets |
| `NCCL_IB_IMAGE` | both | yes | `${IMAGE}-nccl-ib` |
| `NCCL_NCHANNELS` | both | production value | 8 → `NCCL_MAX_NCHANNELS=8`/`NCCL_MAX_P2P_NCHANNELS=8`; stock default 64 costs ~1.6 GiB of host memory per node (measured on pair 2). `agent-run.sh` does **not** write this line — add it by hand |
| `JIT_CACHE_DIR` | both | production value | host dir for Triton/Inductor/FlashInfer JIT caches → `/jit-cache`; unset recompiles every boot. `agent-run.sh` does **not** write this line either |

`RAY_memory_usage_threshold=0.99`, `VLLM_ENGINE_READY_TIMEOUT_S=3600`,
TP=2, `--max-model-len 204800`, `--max-num-seqs 20`,
`--gpu-memory-utilization 0.85`, `--max-num-batched-tokens 8192`,
`--enable-chunked-prefill`, `--enable-prefix-caching`, FP8 KV,
`--enable-auto-tool-choice --tool-call-parser glm45`,
`--served-model-name GLM-5.3-Flash-NVFP4-Wabi` are hardcoded in the
scripts — no env needed, and `verify-serve.sh` checks them.

### 3.2 Head, wait for Ray, worker, wait for READY

```bash
# on the head node: pair 1 uses ~/glm53-repo30; pair 2 uses ~/dogfood/repo30
cd ~/glm53-repo30 && set -a; . serve/serve.env; set +a
bash serve/start-head.sh

# then — the worker joins HEAD_IP:6399 and fails if nothing listens:
until bash -c '</dev/tcp/192.0.2.2/6399' 2>/dev/null; do sleep 5; done   # from the worker; or ss -ltn | grep 6399 on the head

# on the worker node: pair 1 uses ~/glm53-repo30; pair 2 uses ~/dogfood/repo30
cd ~/glm53-repo30 && set -a; . serve/serve.env; set +a
bash serve/start-worker.sh
```

For pair 2, use `cd ~/dogfood/repo30` in both commands above and set
`HEAD_IP=198.51.100.2` in its worker environment. The pair-1 path and
link address shown in the example commands are not valid for pair 2.

READY takes ~15 min (observed 842–921 s; weight load alone ~13 min).
Poll the API, budget 40 min:

```bash
until curl -sf http://<head>:8888/v1/models | grep -q GLM; do sleep 20; done
```

`scripts/agent-run.sh` automates all of this (writes both serve.env
files, waits Ray 5 min, READY 40 min, then runs the NET/IB gate and the
headroom check) — but it needs the STEP-2 hostname ssh from §1c and it
does not write `NCCL_NCHANNELS`/`JIT_CACHE_DIR`, so a serve it brings
up lacks those two knobs.

## 4. Verify it came up correctly

Run all of these; the first four are the acceptance for "came up":

1. **API identity**
   `curl http://<head-host>:8888/v1/models` → `id =
   GLM-5.3-Flash-NVFP4-Wabi`, `max_model_len = 204800`, `root =
   /checkpoint`.
2. **Deployed files match the repo — `serve/check-deployed.sh`**
   Usage: `check-deployed.sh <node> [<node>...]` — `bash
   serve/check-deployed.sh node1 node2 node3 node4` → `OK`
   per file, `RESULT: 0 stale/missing`. Options: the node list is its
   only argument; the deploy dir per node comes from
   `REMOTE_DIR_PER_NODE`/`REMOTE_DIR` in `setup.env` (§3.0). Anything
   else means node-side drift; restore by re-syncing (§3.0).
3. **Container matches the declared env — `serve/verify-serve.sh`**
   Usage: `verify-serve.sh <container-name> <env-file>` — on each node,
   `cd <REMOTE_DIR> && bash serve/verify-serve.sh glm53-head
   serve/serve.env` (worker: `glm53-worker`). Pass =
   `RESULT: 0 mismatch(es)`. Options: `INSPECT_FIXTURE=<file>` feeds a
   captured `docker inspect` JSON instead of a live container (see
   `serve/fixtures/`). It compares the container's image, mounts,
   env and serve flags against the env file plus the script constants.
   `UNVERIFIABLE` lines are report-only and expected: the worker prints
   nine (its container has no `vllm serve` to check — by design), and a
   head with `MY_IP` set prints one (`MY_IP` is worker-only).
4. **NCCL actually bound IB** — `cd <REMOTE_DIR>/serve &&
   . ./nccl-ib.sh && nccl_ib_gate glm53-head` (exit 0), or
   `docker logs glm53-head 2>&1 | grep 'via NET/IB'`. Without this,
   NCCL silently ran on sockets.
5. **Boot headroom — `serve/check-headroom.sh`** on each node, right
   after READY and before any traffic: floor 7500 MiB MemAvailable
   (option: `FLOOR_MIB=<n>` overrides the floor). This is a
   diagnostic, not a required gate; the floor was calibrated
   on EP-off boots, which is again the live regime. Under the floor,
   restart rather than tune.

## 5. When it does not come up

**Preserve evidence before restarting.** `start-head.sh` and
`start-worker.sh` both begin with `docker rm -f <container>`, which
destroys the failed container's logs. Before re-running anything:

```bash
docker logs glm53-head   > /tmp/head.log   2>&1   # on the head
docker logs glm53-worker > /tmp/worker.log 2>&1   # on the worker
docker inspect glm53-head > /tmp/head.inspect.json
```

Then match the symptom:

| symptom | where to look | what it means / fix |
|---|---|---|
| script exits with a `FAIL:`/`set VAR` line | stderr | the message names it: unset env var, non-absolute path, `MODEL_DIR`/`MTP_DIR` missing, `no RDMA HCA mapped to netdev`, `no IPv4-mapped RoCE v2 GID`, `no uverbs under /dev/infiniband` — fix serve.env or the host RDMA state, not the script |
| head container dies ~20 s after "started" | `docker logs glm53-head` | usually a bad flag or missing `/checkpoint-mtp` mount; argparse exits after `ray start` succeeded |
| Ray port 6399 never opens | `docker logs glm53-head`, `docker inspect .State` | head container exited (see above) or the image is still pulling |
| worker exits / never joins | worker ran before head's 6399 was up | re-run `start-worker.sh` after the port listens |
| `curl /v1/models` not READY in 40 min | `docker logs glm53-head` tail | weight load is ~13 min; a truncated shard dies about then — re-stage weights with `rsync --partial` (agent-run STEP 8 checks integrity) |
| `nccl_ib_gate` fails / no `via NET/IB` | gate output | NCCL fell back to sockets: derived `-nccl-ib` image missing on that node, `/dev/infiniband` devices not passed, or wrong `IFNAME` |
| `check-deployed.sh` STALE | the named file | node-side edit; re-sync the repo copy — do not keep the edit |
| `verify-serve.sh` MISMATCH | the named field | env/container drift; rewrite `serve.env` to §3.1 and restart |
| `check-headroom.sh` FAIL at boot | its printed numbers | this boot drew a bad hand (observed range 5.4–10.3 GiB on identical inputs). Restart with the same two commands; do **not** lower `--gpu-memory-utilization` (0.01 = 1.2 GiB and the limiting rank has ~3.5 GiB KV at 204800) |
| engine dies mid-session (earlyoom SIGTERM `ray::RayWorkerP`) | `journalctl -u earlyoom`, `/tmp/head.log` | the boot lottery or an over-window prompt, not a config bug. Same restart procedure; resume client work from saved state. READY ≠ stays up — the four failure families are in AGENTS.md |
| someone suggests `glm-serve.sh up` / `pair-relaunch.sh` | §0 | they launch the older `glm53-official-*` stack on stock weights — a different serve, not this one |

## 6. Documents that contradict this runbook

Correct the source document, not this file:

- `tools/dgx/spark/nodes.tsv` — `role` column says node1 worker /
  node2 head / node3-4 spare (live: node1 and node3 are heads);
  node4 "docker グループ未所属" is stale (docker works); node3/4 notes
  describe the retired EXL3 stack. The `roce_ip` column is accurate.
- `tools/dgx/spark/assets.tsv` — `役割`/`サービス` columns stale
  (node1 `sglang_qwen38fn`, node3/4 "spare"); the port↔peer↔IP
  mapping is accurate.
- `tools/dgx/llm/README-glm-serve.md`, `tools/dgx/llm/README.md`,
  `tools/dgx/llm/README-glm-routes.md`, `tools/dgx/spark/README.md` —
  all steer the reader to `glm-serve.sh` / `serve-official-tp2-*`,
  which produces the different, worse serve described in §0.
  `tools/dgx/spark/README.md` also claims `ssh-config.sh` writes
  three node blocks; the script writes all four.
- netplan `99-qsfp-*.yaml` — pre-ring copies (maintainer tree, not shipped)
  (one port each); live files carry two ports and the node4 copy is
  missing (§1b).
- `scripts/agent-run.sh` `write_serve_env` — omits `NCCL_NCHANNELS`
  and `JIT_CACHE_DIR`, so it alone cannot emit the live serve.env.

## 7. Verified values and how each was verified (2026-09-19)

| value | how verified |
|---|---|
| roles node1/3 head, node2/4 worker; containers `glm53-head`/`glm53-worker`; image `…-nccl-ib` | `docker ps` on all four nodes |
| no `--enable-expert-parallel`; full serve flag list incl. TP=2, 204800, seqs 20, util 0.85, batched 8192, chunked prefill, prefix caching, fp8 KV, tool-call glm45, MTP `/checkpoint-mtp` K=2, `--port 8888`, `RAY_memory_usage_threshold=0.99`, `VLLM_ENGINE_READY_TIMEOUT_S=3600` | `docker inspect glm53-head --format '{{json .Args}}'` on node1 and node3 |
| link IPs, /30, MTU 9000, per-node netdevs, ring cross-links | `ip -4 addr` + `ip link` on all four; `cat /etc/netplan/99-qsfp.yaml` on all four |
| HCA `rocep1s0f0`/`rocep1s0f1`, `NCCL_IB_GID_INDEX=3`, `NCCL_MAX_NCHANNELS=8`, `NCCL_MAX_P2P_NCHANNELS=8` | `docker inspect .Config.Env` on all four containers |
| mounts: requant-h → `/checkpoint`, MTP-ja-base → `/checkpoint-mtp`, overlays `_build` → dist-packages, `jit-cache` → `/jit-cache` | `docker inspect .Mounts` on node1 and node3 |
| serve.env bodies (all four) | `cat <REMOTE_DIR>/serve/serve.env` on each node |
| deployed scripts byte-identical to this repo | `sha256sum` on all four nodes + `serve/check-deployed.sh node1 node2 node3 node4` → `RESULT: 0 stale/missing` |
| served name + window + port | `curl http://127.0.0.1:8888/v1/models` on node1 and node3 |
| NET/IB bound, 8 channels | `docker logs glm53-head` → `via NET/IB/0`, `8 coll channels … 8 p2p channels` |
| `verify-serve.sh` passes | run on all four containers: 0 mismatches (worker: 9 expected UNVERIFIABLE, node3 head: 1 expected MY_IP UNVERIFIABLE) |
| weights 181 GB + draft 17 GB on all four | `du -sh` per node |
| base image `b0501f99fec5` on all four | `docker images` per node |
| earlyoom active, args `-m 2 -s 100,100 --prefer ^(vllm|VLLM::) --avoid ^(sshd|systemd)$`, pkg 1.7-2 | `systemctl is-active`, `ps`, `/etc/default/earlyoom`, `dpkg -l` |
| node→node ssh via link IP + bound key; bare aliases unresolvable; the mDNS name resolves but key auth fails | `ssh node1 'ssh 192.0.2.1 hostname'` → ok; `getent hosts` / ssh to `<mdns-name>` → fails; ssh client config + authorized-key comments |
| operator side: `Host` blocks for node1–4 in the ssh client config; `/etc/hosts` pins the aliases to LAN addresses | ssh client config, `/etc/hosts`, `dscacheutil` |
| `check-deployed.sh` needs `REMOTE_DIR_PER_NODE`; default `glm53-repo30` | script source + repo `setup.env` (gitignored) |
| `agent-run.sh` waits: Ray 6399 poll 5 s/5 min, READY poll 20 s/40 min; writes serve.env without NCCL_NCHANNELS/JIT_CACHE_DIR | script source (`step_serve`, `write_serve_env`) |
| `verify-serve.sh` usage `<container> <env-file>`; hardcodes EP-off expectation | script source + live runs |
| `check-headroom.sh` floor `FLOOR_MIB=7500`, calibrated on EP-off boots | script source |
| `glm-serve.sh`/`pair-relaunch.sh` launch `glm53-official-*` on stock weights | `glm-serve.sh` source (`HEAD_NAME`, `MODEL_DIR`), `pair-relaunch.sh` `PR_GLM_SERVE` default |
| client assumes 8888 + `-Wabi` name | `tools/dgx/llm/config.sh` `DGXLLM_BASE_URL`/`DGXLLM_MODEL` |
