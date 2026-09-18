# Reproduction runbook

This runbook exists in English only; the translated READMEs
(`README.ja.md`, `README.ko.md`, `README.zh.md`) link back to this file.

Below, I provide the end-to-end steps to reproduce the Stage-1 result
(35.09 tok/s, C=1) on two NVIDIA DGX Spark nodes. I assume that this
repository is copied to both nodes (or accessed over SSH) and that the
source checkpoint is already available on local disk.

## Prerequisites

| | |
|---|---|
| Nodes | 2x NVIDIA DGX Spark (GB10, `sm_121`, 128 GB unified memory each) |
| Interconnect | RoCE-capable link between the nodes (ConnectX-7 class), cabled port to port with no switch in the path — that direct-attached pair is what the numbers below were measured on. Optional — `NCCL_IB=0` falls back to sockets — but the headline number uses it |
| Parallelism | Ray TP=2 (one GPU per node) |
| Image | `vllm/vllm-openai:glm53-flash-arm64-cu130` + derived RDMA image |
| Serve flags | `--max-model-len 204800`, `--max-num-seqs 20`, `--gpu-memory-utilization 0.85`, `--max-num-batched-tokens 8192`, `--enable-chunked-prefill`, `--enable-prefix-caching`, FP8 KV cache, expert parallel off (no `--enable-expert-parallel` on the serve line), `RAY_memory_usage_threshold=0.99` on both nodes. Read "What this configuration is, and is not" below before relying on the window or on concurrency |
| Host tools | docker on both nodes; a host `python3` — stdlib only, since `requant/verify.py` and `bench/measure.py` import nothing else; `ibdev2netdev`/`rdma` on both nodes for the RDMA path. torch and safetensors come from the image (see step 1), so nothing is pip-installed on the host |
| Disk | the source checkpoint (~204 GB) where the requant runs; the route-h output (~190 GB) on **both** nodes; the MTP draft dir is ~16 GiB per node. `scripts/agent-run.sh` gates on >= 800 GB free under `WEIGHTS_ROOT` on the requant node (stock + output + draft + two images) and >= 400 GB on the other node (weights copy + image). The upstream BF16 release is ~599 GB and is *not* needed |

The operator that runs `scripts/agent-run.sh` may be a separate machine
or the head node itself: when `HEAD_HOST` equals `$(hostname)`,
`localhost`, `127.0.0.1`, or `::1`, the script runs head-node commands
locally instead of over ssh (relative `REMOTE_DIR` paths still resolve
under `$HOME`, as an ssh session would). Only the worker then needs to
be reachable by ssh — from the head, over the pair link.

## What this configuration is, and is not

The configured ceiling is 204,800 tokens. The longest input I have
confirmed on the existing expert-parallel-off serving configuration is
197,485 tokens. Stability across the full window, and under concurrent
long-context load, is not something I have accepted yet. Additional
memory is claimed *after* READY. The both-node check at startup is
mandatory, but passing it is not a no-crash guarantee. The operating
mode I recommend is a single stream. A next version combining a finite
boot retry with a single-stream setting is still under acceptance.

I do not claim the crash is fixed, that a retry makes boots 99.2% safe,
that 20 concurrent requests are supported, or that a 1M window works.
Four distinct failure families are on record, and no single change
addresses all of them:

1. **Boot headroom.** The margin is set when the engine profiles, and
   the same script on the same node has reached READY with anywhere from
   5470 MiB to 10270 MiB free. A boot that starts low is on a death
   course before the first request. The startup check screens for this;
   it does not change the boot.
2. **Ray's own OOM monitor.** Ray's default `memory_usage_threshold` of
   0.95 killed the TP0 worker at about 5770 MiB free — above the reaper
   line the startup floor is calibrated against. Both serve scripts
   export `RAY_memory_usage_threshold=0.99`; if that export is missing,
   the floor is the wrong number and a passing boot can still be killed.
3. **Demand above supply.** Prompts of roughly 259k tokens died on every
   configuration tried, including a lower `--gpu-memory-utilization` and
   a smaller prefill chunk; the only change was how long they took to
   die. Those lengths are outside the 204,800 window this configuration
   advertises, but they show the window is a ceiling, not a promise.
4. **Failures that are not about memory.** `--max-num-batched-tokens
   2048` hits an illegal memory access in the KDA kernel on the first
   long prefill, with host memory still 5.33% free and no reaper
   involved. Separately, the NCCL low-latency and symmetric-memory
   all-reduce variants in `results/failures.tsv` booted but returned
   empty bodies. Neither is reached by any memory setting.

Observed wall times on my setup (excluding download time):

| step | wall time |
|---|---|
| requant route h (rewrites the BF16 dense tensors inside the ~204 GB tree; ~190 GB written) | ~30 min |
| `verify.py config` | seconds |
| `build-mtp-draft.py` (~16 GiB out) | ~3 min |
| RDMA image build | ~5 min per node |
| `build-overlays.sh` | ~2 min |
| serve: boot to READY | ~15 min (842-921 s observed) |
| gate `check` on a running server | ~15-20 min |
| ruler, C=1 x 64 prompts | ~15-30 min per configuration |

## 0. Put the checkpoints in place

```bash
# on the build node: NVIDIA GLM-5.3-Flash-NVFP4 (~204 GB)
# after step 1, copy the route-h output to the second node:
rsync -aP /path/to/GLM-5.3-Flash-NVFP4-h/ worker-node:/path/to/GLM-5.3-Flash-NVFP4-h/
```

`scripts/agent-run.sh` downloads the stock checkpoint with
`huggingface_hub.snapshot_download` inside the vLLM image — nothing is
installed on the host. If docker cannot be used for the download, the
manual fallback is `pip3 install --user "huggingface_hub[cli]"` plus
`hf download nvidia/GLM-5.3-Flash-NVFP4 --local-dir <STOCK_DIR>` (needs
`--break-system-packages` or a venv on externally-managed distros).

## 1. Requantize (route h)

`requant.py` needs torch and safetensors. I do not install them on the
host: the serving image already carries both, so I run the script inside
it (this is what `scripts/agent-run.sh` does, step 7). From the repo
root on the requant node:

```bash
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp \
    --entrypoint python3 \
    -v /path/to/GLM-5.3-Flash-NVFP4:/src:ro \
    -v /path/to:/path/to \
    -v "$PWD":/repo:ro \
    vllm/vllm-openai:glm53-flash-arm64-cu130 \
    /repo/requant/requant.py --target h --src /src \
    --dst /path/to/GLM-5.3-Flash-NVFP4-h
```

`--entrypoint python3` because the image's entrypoint is the vLLM CLI,
which wants a GPU just to build its parser; this step is CPU file work.
`--user $(id -u):$(id -g)` with `HOME=/tmp` keeps the written shards
owned by the operator — root-owned output breaks the later unprivileged
rsync. The second `-v` mounts the output's parent at the same path so
`--dst` can stay an absolute host path.

On a host that already has torch and safetensors, the plain form is the
same script:

```bash
python3 requant/requant.py --target h \
    --src /path/to/GLM-5.3-Flash-NVFP4 \
    --dst /path/to/GLM-5.3-Flash-NVFP4-h
```

Reads the source read-only, rewrites the target tensors into fresh
safetensors shards, and rewrites `config.json` / `hf_quant_config.json`
/ the index for the modelopt MIXED_PRECISION loader. Other targets
(`a`–`e`, `g`) are earlier, narrower routes; `h` is the released one.
Takes ~30 min and writes ~190 GB; the free-space figures are in the
Prerequisites table (800 GB on this node, 400 GB on the other).

Statically check the output:

```bash
python3 requant/verify.py config /path/to/GLM-5.3-Flash-NVFP4-h
```

For route `h`, the verifier demotes the "DSA-layer self_attn" check to a
warning when the checkpoint carries `producer.requant_target` = `g`/`h`,
because those routes boot with the `mla-quant` overlay that makes the
keys live config.

## 2. Build the MTP draft dir (the headline number needs it)

```bash
python3 requant/build-mtp-draft.py \
    --target /path/to/GLM-5.3-Flash-NVFP4 \
    --out /path/to/GLM-5.3-Flash-MTP-bf16
# copy it to the second node too (or run the script there)
rsync -aP /path/to/GLM-5.3-Flash-MTP-bf16/ worker-node:/path/to/GLM-5.3-Flash-MTP-bf16/
```

This script also needs torch and safetensors, so the same in-image form
as step 1 applies (`--entrypoint python3`, `-v <stock>:/src:ro`, the
output parent mounted at its own path, `-v "$PWD":/repo:ro`, then
`/repo/requant/build-mtp-draft.py --target /src --out <out>`).

The checkpoint's MTP layer is already BF16; this step copies
`layers.45.*` plus the embedding and lm_head tensors that the draft
loader expects under the layer prefix, and writes a flattened text-only
`config.json`. ~16 GiB out, ~3 min.

## 3. Build the RDMA image (on each node)

```bash
docker build -t vllm/vllm-openai:glm53-flash-arm64-cu130-nccl-ib \
    - < docker/Dockerfile.nccl-ib
```

Why: the base image's Ubuntu noble `rdma-core` 50.0 lacks the MLX5_1.25
symbols that the NCCL NET/IB plugin looks up; host bind-mounts cannot
fix it because the hosts run the same 50.0 userspace. Questing's 56.1
resolves all deps on noble; the Dockerfile pins only the rdma packages
to questing and verifies the symbols after install.

The knob: `NCCL_IB=1` (default in `serve/`) runs the derived image and
derives HCA/GID per node; `NCCL_IB=0` serves the stock image over
sockets.

## 4. Build the overlays

```bash
overlays/build-overlays.sh            # read the image from the local docker
# or, when the image lives on another machine, name it as an ssh host:
overlays/build-overlays.sh node-with-the-image
```

With no argument the patchers read the image files through the local
docker daemon; the optional argument is the ssh host whose docker holds
the image (`fetch-image-file.sh` runs there and the files come back).

Produces `overlays/_build/` containing `kda-quant.py`, `mla-quant.py`,
`glm5next-mtp-bf16.py` and `flashinfer_mla_sparse_sm120.py`. The
step-attribution build lands one level down, in `_build/step-attr/`, so
its instrumented `model.py` is never mounted unless `STEP_ATTR=1` asks
for it — otherwise every measurement would carry the profiling
overhead. Each patcher anchors on the image's own source and fails
closed if the image drifts. You need the same `_build/` output on both
nodes — run it per node or copy the directory.

## 5. Serve

**Read this before the first start: the API has no authentication.**
`start-head.sh` serves with `--host 0.0.0.0`, so the endpoint answers on
every interface of the head node, and anything that can reach the port
can send requests to it. Run this on a trusted network only; do not
expose the port to a public one. Access control is yours to provide --
a firewall, a private subnet, or an ssh tunnel -- and if the endpoint
has to be reachable from further away, put authentication and TLS in
front of it. Ray needs separate handling: the head's dashboard is
already pinned to `127.0.0.1`, but the cluster port the worker joins on
(`HEAD_IP:6399`) and the ports the two raylets negotiate are not behind
whatever guards the API, so authenticating the API does not protect
Ray. Keep those on the pair link and off any untrusted network. The
stakes here are not only reading: this configuration serves a window of
204800 tokens out of 128 GB of unified memory per node, so a long
enough input can exhaust host memory and take the pair down.

`serve.env` is per node, not shared: `IFNAME` is that node's own netdev,
and the worker additionally needs `MY_IP` (`start-worker.sh` requires
`HEAD_IP`, `MY_IP`, `IFNAME`, `MODEL_DIR`; `start-head.sh` requires
`HEAD_IP`, `IFNAME`, `MODEL_DIR`). So I write one file on each node:

```bash
# on the head node
cp serve/serve.env.example serve/serve.env
#   HEAD_IP=<head IP on the fast link>
#   IFNAME=<head netdev carrying that IP>     MY_IP is unused here
#   MODEL_DIR=<route-h dir on this node>
#   MTP_DIR=CHANGEME                          uncomment it and point it at
#                                             the step-2 draft dir; the
#                                             headline speed needs it
. serve/serve.env
serve/start-head.sh
```

```bash
# on the worker node
cp serve/serve.env.example serve/serve.env
#   HEAD_IP=<same head IP as above>
#   MY_IP=<this node's IP on the fast link>
#   IFNAME=<this node's netdev>               may differ from the head's
#   MODEL_DIR=<route-h dir on this node>
#   MTP_DIR=CHANGEME                          uncomment it here too: the
#                                             same draft dir, copied over
. serve/serve.env
serve/start-worker.sh
```

Start the head first, and wait for its Ray port to open before
starting the worker: `start-worker.sh` joins the head's cluster at
`HEAD_IP:6399` and fails if nothing is listening there yet. The head
needs tens of seconds to get that far, so poll instead of counting:

```bash
# on the worker node, before serve/start-worker.sh
until bash -c "</dev/tcp/$HEAD_IP/6399" 2>/dev/null; do sleep 5; done
```

`scripts/agent-run.sh` does this itself, polling every 5 s for up to
5 min (step 9). After READY, confirm NCCL actually bound IB (not
sockets):

```bash
docker logs glm53-head 2>&1 | grep -E 'NET/IB|via NET/IB'
```

Then gate the host headroom on **both** nodes, before you send the first
request:

```bash
serve/check-headroom.sh
```

This is not optional on this hardware, and it has to run before traffic.
GB10 is unified memory: the engine sizes its budget from whatever was
free when it profiled, so the margin is set by the boot, not by the
flags. The same script on the same node has reached READY with anywhere
from 5470 MiB to 10270 MiB free. About 4.7 GiB of that is then spent
*after* READY -- a one-time high-water mark paid the first time each
larger prefill shape is seen, not a leak -- so a boot that looks healthy
can still be killed mid-session by one long prompt. A node under the
floor should be restarted, not tuned: at 204800 the limiting rank only
gets about 3.5 GiB of KV, and 0.01 of `--gpu-memory-utilization` is
1.2 GiB, so two steps down and the engine can no longer open the window
it advertises. The script prints the measured numbers behind the floor.

The floor itself is a provisional screening value, derived from the
expert-parallel-off ladder, and it is not a safety guarantee. It was
calibrated from five boots, and the post-READY high-water figure behind
it comes from about two. A node that passes has not been shown to be
safe; it has only been shown not to be obviously short. A node that
fails should be restarted rather than tuned, and the check is only
meaningful after READY and before traffic — run against a warmed engine
it fails for reasons that say nothing about the boot.

**`MTP_DIR` is what the headline number needs.** Set it (in both files)
to the draft dir from step 2 to enable MTP (`MTP_K=2`). Left unset, the
pair serves without speculation: about 28 tok/s on this route over RDMA
(28.28 measured; the published `requant-h1-p1-rdma-nodraft` row in
`results/results.tsv` is 27.15) against the 35.09 headline, ~20%
slower. `serve.env.example` ships that line commented out, since the
path differs per machine, so it is the one setting a serve brought up
from the example silently does without -- uncomment it on both nodes.
`NCCL_IB=0` falls back to the stock image over sockets.
`scripts/agent-run.sh` writes both files itself — see section 7.

Once `/v1/models` answers, "Using the served model" in the README says
what a client has to send: the served model name, the tool-call flags
this script ships, and the `reasoning` field the answer arrives in.
That section also has the same caveat this runbook should carry: the
serve line's `--max-num-seqs 20` is a scheduler ceiling, not a measured
concurrent-request count, and not a supported concurrency level -- the
KV pool decides how many of those 20 admitted slots actually run
together, and on this pair at around 25,000 tokens per prompt that
number is **6** -- measured by sending 20 requests at once, which queued
14 and ran 6 concurrently with zero failures and zero preemptions at
128% of KV pool capacity. That measurement was taken at about 25,000
tokens per prompt; it says nothing about 20 streams near the top of the
window, which I have not measured. The 204,800 window and the 20-slot
ceiling do not multiply into a supported workload. The serve line ships
`--max-num-seqs 20` unchanged, since it caps what the scheduler accepts,
not what it decodes together, but the operating mode I recommend is a
single stream.

## 6. Gate, then measure

I need a stock baseline for the gate, so I serve the stock checkpoint
once and capture against it: same `serve.env` as step 5 but with
`MODEL_DIR` pointing at the stock `GLM-5.3-Flash-NVFP4` dir and
`MTP_DIR` left unset (comment that line out, or `unset MTP_DIR` after
sourcing), so the baseline is the unmodified checkpoint without
speculation. Then restart with the route-h `MODEL_DIR` and `MTP_DIR`
back in place and run `check` against the saved baseline:

```bash
# stock serving up:
python3 requant/verify.py capture --url http://HEAD:8000 --out baseline.jsonl
# route-h serving up:
python3 requant/verify.py check --url http://HEAD:8000 --baseline baseline.jsonl
```

Then the ruler run that produced the headline number — C=1 streaming
over all 64 prompts (the default `--levels` would run the short mixed
ladder instead):

```bash
python3 bench/measure.py --url http://HEAD:8000 --label route-h-rdma-k2 \
    --prompts bench/prompts-64.jsonl --levels 1:64 --outdir results
```

Both `capture`/`check` and `measure.py` accept an ssh-tunnelled URL;
`verify.py` additionally takes `--ssh HOST` to spawn the tunnel itself.

## 7. One-command path

`scripts/agent-run.sh` runs sections 0-5, then the ruler and a speed
check against the published row, over ssh from a single operator shell
(the head node itself counts as an operator — see the note under
Prerequisites). The quality gate of section 6 stays manual, because it
needs the stock checkpoint served once.

### (a) Fill in `setup.env`

```bash
cp setup.env.example setup.env
```

Then replace every `CHANGEME`:

| key | what goes there |
|---|---|
| `HEAD_HOST`, `WORK_HOST` | ssh hostnames of the two nodes; the head runs the Ray head and the API |
| `HEAD_IP`, `WORK_IP` | each node's IP on the point-to-point link |
| `HEAD_IF`, `WORK_IF` | the netdev carrying that IP on each node (`ibdev2netdev` shows the mapping) |
| `REQUANT_HOST` | which node runs the requant; must equal `HEAD_HOST` or `WORK_HOST`, and must be able to ssh to the other one by hostname |
| `WEIGHTS_ROOT` | absolute dir present on both nodes; the disk check measures free space here |
| `STOCK_DIR`, `MODEL_DIR`, `MTP_DIR` | download target, requant output, BF16 draft dir. Absolute paths only — they are bind-mounted into docker under the same path. An empty `MTP_DIR` means serve without speculation |
| `REMOTE_DIR` | where the repo copy lands on each node (absolute, or relative to the remote `$HOME`) |
| `MODEL_ID`, `MODEL_REVISION` | what step 4 downloads; pin a commit sha to freeze the checkpoint |
| `IMAGE`, `NCCL_IB_IMAGE` | base image and the derived RDMA tag built in step 5 |
| `NCCL_IB` | 1 = derived image + NET/IB, 0 = stock image over sockets |
| `MTP_K` | speculative tokens; step 1 refuses K >= 4, which does not boot on 128 GB unified memory |
| `PORT` | API port on the head node |
| `VERIFY_TOL` | the step 11 band, default 0.07 |
| `LABEL`, `RESULTS_DIR` | result file `results/measure-<LABEL>.json` and the local dir it is copied back to |

### (b) Read the plan first

```bash
scripts/agent-run.sh --dry-run
```

Every command is printed with a `DRY$ ` prefix and nothing runs; the
last line is `DRY-RUN done (nothing executed)`. The two `serve.env`
bodies the script would write (head and worker) are printed here too,
which is the cheapest way to check the addressing before a 40-minute
boot.

### (c) Run it

```bash
scripts/agent-run.sh                        # all 11 steps, in order
scripts/agent-run.sh serve ruler verify     # only some steps
scripts/agent-run.sh --env /path/to/other.env
```

Step 1 always runs first, even when I name later steps. Every step is
idempotent — work already done on the nodes is detected and skipped — so
after fixing a failure I re-run the same command.

### (d) Pass criteria per step

Each step ends with one greppable line: `STEP <n> OK`, `STEP <n> SKIP
(<reason>)`, or `STEP <n> FAIL: <cause>` — that last one on stderr,
followed by exit 1. Steps 6 and 7 print their `SKIP` line and then still reach `OK`, because
they re-check the existing artifact; steps 5 and 9 stop at the `SKIP`
line, since there is nothing left to check.

| step | what it does | passes when | when it fails |
|---|---|---|---|
| 1 env | loads `setup.env` | `STEP 1 OK` | the message names the offending key: an unset or `CHANGEME` value, a relative weights path, `NCCL_IB` not 0/1, `MTP_K` >= 4, a `REQUANT_HOST` that is neither node, or a missing local `ssh`/`rsync`/`python3` |
| 2 prereqs | per node: `aarch64`, usable docker, a GB10 in `nvidia-smi`, uverbs under `/dev/infiniband`, the netdev mapped to an HCA; then free space (800 GB on the requant node, 400 GB on the other) and inter-node ssh by hostname | `STEP 2 OK`, or `STEP 2 OK (weights present)` when the stock checkpoint is already complete | an OS that is neither DGX OS nor Ubuntu only prints `WARN` and continues; the other checks are hard. Disk failures print the measured free space next to the requirement |
| 3 sync | rsyncs the repo to `REMOTE_DIR` on both nodes (`setup.env`, `serve/serve.env`, `results/`, `overlays/_image`, `overlays/_build` excluded) | `STEP 3 OK` | ssh or rsync to one of the nodes |
| 4 pull | `docker pull` the base image on both nodes; download the checkpoint with `snapshot_download` inside that image when it is not already complete; then assert `config.json` + the index exist and that no file is unreadable | `STEP 4 OK` | `checkpoint incomplete` = a partial download, re-run the step. `unreadable by the operator` = root-owned files from an earlier run without `--user`; chown them and re-run |
| 5 rdma | builds `NCCL_IB_IMAGE` from `docker/Dockerfile.nccl-ib` on each node | `STEP 5 OK`, or `STEP 5 SKIP (NCCL_IB=0 (sockets))` | the Dockerfile's own libmlx5 ABI check refused the result — the questing `rdma-core` did not install as expected |
| 6 overlays | runs `overlays/build-overlays.sh` on the head node | `STEP 6 OK`, preceded by `STEP 6 SKIP (overlays already built on <host>)` when `_build/` is complete | a patcher refused its input: the image drifted from the pinned tag. The patchers are fail-closed by design; do not force them |
| 7 requant | route-h requant inside the image, then `verify.py config` must print `PASS`, then a readability check | `STEP 7 OK`, preceded by `STEP 7 SKIP (requant output present at ...)` on a re-run | `requant died` is usually ENOSPC, which leaves a partial dir: free space and re-run. `verify.py config FAILED` catches the class of writer bug that otherwise surfaces as `KeyError ...weight_scale_2` at load |
| 8 stage | builds the MTP draft dir, rsyncs weights and draft to the other node over the fast link, re-checks the copy there, and rsyncs `overlays/_build/` to the worker | `STEP 8 OK` | `weights on <host> are incomplete after rsync` means a truncated copy — re-run the step, `rsync --partial` resumes. Left unfixed it shows up ~13 min into the next weight load |
| 9 serve | writes `serve/serve.env` on each node, starts the head, waits up to 5 min for Ray port 6399, starts the worker, polls `/v1/models` for up to 40 min, and with `NCCL_IB=1` runs the NET/IB gate on the head container | `STEP 9 OK`, or `STEP 9 SKIP (API already READY on <host>:<port>)` | `Ray head port 6399 never came up` and `not READY after 40min` both point at `docker logs glm53-head`. `NCCL NET/IB gate failed` means NCCL fell back to sockets: derived image missing, or `/dev/infiniband` not passed into the container |
| 10 ruler | runs `bench/measure.py` on the head (C=1, all 64 prompts) and copies `measure-<LABEL>.json` back | `STEP 10 OK` | `ruler failed` = the API stopped answering mid-run; `result file did not come back` = the rsync back to the operator |
| 11 verify | compares the C=1 row against the reference row | `STEP 11 OK`, after `VERIFY PASS` | `ruler result outside the <tol> band`, with the per-metric deltas printed just above |

### (e) What step 11 accepts

The reference row follows the configuration: `h-rdma-mtp` (35.09 tok/s,
27.9 ms TPOT) with a draft dir and `NCCL_IB=1`, `h-sockets-mtp` (24.01,
40.7) with a draft dir over sockets, `h-nospec` (18.98, 51.5) without
one. Gated metrics are the C=1 aggregate tok/s and the C=1 median TPOT;
TTFT and MTP acceptance are printed for context only.

Each of those numbers belongs to a configuration, and the rows in
`results/results.tsv` carry the log each came from. The 35.09 reference
was measured on an earlier internal launcher with expert parallel on.
The serve scripts in this repository ship expert parallel off, and on
that path I measured 37.33 tok/s (`ship-script (ep off)`, source log
`logs/route-h-shipscript-ep0-p2-rdma.log`) against an expert-parallel-on
control of 35.05 (`ship-script + ep (control)`,
`logs/route-h-shipscript-ep1-p2-rdma.log`) on the same script and the
same pair. So a reproduction that lands near 37 rather than 35 is the
expected result of the shipped flags, and it sits inside the 7% band
rather than indicating a problem. I do not carry any of these figures
over to a configuration they were not measured on.

The band is `VERIFY_TOL`, default 0.07. That number is measured, not
chosen for comfort: two identical node pairs running the same ruler came
out 7% apart, while the same pair re-measured moves about 4%. So a
reproduction on different hardware is judged at 7%, and a repeat on the
same pair that drifts past ~4% is worth a second look even when it
passes.

This step gates speed only. The four quality criteria (degenerate
outputs, perplexity ratio, eval-200 accuracy, TTFT) are section 6 and
are not covered here.
