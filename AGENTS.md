# Reproduction runbook

Below, I provide the end-to-end steps to reproduce the Stage-1 result
(35.09 tok/s, C=1) on two NVIDIA DGX Spark nodes. I assume that this
repository is copied to both nodes (or accessed over SSH) and that the
source checkpoint is already available on local disk.

## Prerequisites

| | |
|---|---|
| Nodes | 2x NVIDIA DGX Spark (GB10, `sm_121`, 128 GB unified memory each) |
| Interconnect | RoCE-capable link between the nodes (ConnectX-7 class). Optional — `NCCL_IB=0` falls back to sockets — but the headline number uses it |
| Parallelism | Ray TP=2 (one GPU per node) |
| Image | `vllm/vllm-openai:glm53-flash-arm64-cu130` + derived RDMA image |
| Serve flags | `--max-model-len 16384`, `--gpu-memory-utilization 0.85`, FP8 KV cache |
| Host tools | docker on both nodes; python3 + torch + safetensors where `requant/` runs; `ibdev2netdev`/`rdma` on both nodes for the RDMA path |
| Disk | the source checkpoint (~204 GB) where the requant runs; the route-h output (~190 GB) on **both** nodes; >= ~220 GB free for the rewrite itself; the MTP draft dir is ~16 GiB per node. The upstream BF16 release is ~599 GB and is *not* needed |

The operator that runs `scripts/agent-run.sh` may be a separate machine
or the head node itself: when `HEAD_HOST` equals `$(hostname)`,
`localhost`, `127.0.0.1`, or `::1`, the script runs head-node commands
locally instead of over ssh (relative `REMOTE_DIR` paths still resolve
under `$HOME`, as an ssh session would). Only the worker then needs to
be reachable by ssh — from the head, over the pair link.

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

```bash
python3 requant/requant.py --target h \
    --src /path/to/GLM-5.3-Flash-NVFP4 \
    --dst /path/to/GLM-5.3-Flash-NVFP4-h
```

Reads the source read-only, rewrites the target tensors into fresh
safetensors shards, and rewrites `config.json` / `hf_quant_config.json`
/ the index for the modelopt MIXED_PRECISION loader. Other targets
(`a`–`e`, `g`) are earlier, narrower routes; `h` is the released one.
Takes ~30 min and writes ~190 GB, so keep >= ~220 GB free.

Statically check the output:

```bash
python3 requant/verify.py config /path/to/GLM-5.3-Flash-NVFP4-h
```

For route `h`, the verifier demotes the "DSA-layer self_attn" check to a
warning when the checkpoint carries `producer.requant_target` = `g`/`h`,
because those routes boot with the `mla-quant` overlay that makes the
keys live config.

## 2. Build the MTP draft dir (only if you want speculation)

```bash
python3 requant/build-mtp-draft.py \
    --target /path/to/GLM-5.3-Flash-NVFP4 \
    --out /path/to/GLM-5.3-Flash-MTP-bf16
# copy it to the second node too (or run the script there)
rsync -aP /path/to/GLM-5.3-Flash-MTP-bf16/ worker-node:/path/to/GLM-5.3-Flash-MTP-bf16/
```

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
overlays/build-overlays.sh            # needs docker access to the image
# or, on a host that has the image:
overlays/build-overlays.sh my-node
```

Produces `overlays/_build/` containing `kda-quant.py`, `mla-quant.py`,
`glm5next-mtp-bf16.py`, the step-attribution files, and
`flashinfer_mla_sparse_sm120.py`. Each patcher anchors on the image's
own source and fails closed if the image drifts. You need the same
`_build/` output on both nodes — run it per node or copy the directory.

## 5. Serve

```bash
cp serve/serve.env.example serve/serve.env   # fill in CHANGEME values
. serve/serve.env
# on the head node:
serve/start-head.sh
# on the worker node:
serve/start-worker.sh
# after READY, confirm NCCL actually bound IB (not sockets):
docker logs glm53-head 2>&1 | grep -E 'NET/IB|via NET/IB'
```

Set `MTP_DIR` to the draft dir from step 2 to enable MTP (`MTP_K=2`).
`NCCL_IB=0` falls back to the stock image over sockets.

## 6. Gate, then measure

I need a stock baseline for the gate, so I capture it while the stock
checkpoint is still being served:

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
