# glm53-flash-nvfp4-2node

Latest release: **v1** (git tag `v1`). The requantized checkpoint is
published on Hugging Face as `tenhkspark/GLM-5.3-Flash-NVFP4-Wabi` —
one repo, git tags `v1` and `v2`; the top of this file always names
the latest.

My recipe for serving `GLM-5.3-Flash-NVFP4` on two NVIDIA DGX Spark
nodes with Ray tensor parallelism (TP=2), plus the weight-only
requantization I use to recover single-stream decode speed. Everything
here is my own implementation and my own measurements on my own
hardware — *enjoying the incomplete*.

No model weights are included. The input checkpoint is the NVIDIA
`GLM-5.3-Flash-NVFP4` release; the requantization script rewrites the
BF16 dense linears of that checkpoint instead of redistributing
anything. Point the scripts at your own copy.

Several improvement ideas are still on my list, but the model has
reached a level I find usable, so I am releasing it now. It is a work
in progress: please take it and move it toward your own idea of
complete. I will keep working and publish updates as results come in,
and I would be glad to see your versions on Hugging Face and GitHub
too.

## What you get

In the measured single-stream setting — one user, one stream, C=1,
thinking off — the served model answers at about 35 tok/s on two DGX
Sparks. The exact figure is 35.09 tok/s, measured on pair 1 over
NCCL/IB (RDMA) with the MTP draft at K=2 on a fixed ruler of 64
Japanese prose prompts at `max_tokens=512`, `temperature=0`. The stock
checkpoint's reference row — pair 2, sockets, no draft, the first 8
prompts of the same ruler — gives 10.75 tok/s; the closest rows are in
the results table below (stock + MTP K=2 on pair 1 over sockets ran
19.05 tok/s with expert parallel off; the same route-h + MTP K=2 config
on pair 1 over sockets ran 24.01 tok/s with it on — see the `ep`
column).

## Where it comes from

| ingredient | what it is | measured |
|---|---|---|
| base speed | route h requant (W4A16 NVFP4 on the BF16 dense side) | 18.98 tok/s, no draft — pair 1, sockets, all 64 prompts (the no-draft RDMA row is pending; the stock no-draft row is 10.75 tok/s on pair 2, sockets, first 8 prompts) |
| speculation | the checkpoint's own MTP head, K=2 | acceptance 0.62, ~2.24 output tokens per cycle |
| combined | requant + MTP at K=2 + NCCL over RDMA | **35.09 tok/s** — pair 1, RDMA, all 64 prompts, K=2 |

With greedy decoding, the speculation step accepts only tokens the base
model would have chosen, so it does not change the output distribution.
The quality delta therefore belongs to the requantization alone, which
is what the next section measures.

## What it costs

The requant trades a bounded quality delta for speed. The gate was
declared before any speed run; the released configuration had to pass
all four criteria:

| criterion | threshold | stock | route h | result |
|---|---|---|---|---|
| degenerate outputs over the 64-prompt ruler | 0 | 0 | 0 | PASS |
| perplexity ratio on held-out passages | <= 1.10 | 1.0 | 1.051 | PASS |
| eval-200 live accuracy (150 items, the 0/50 tool floor excluded) | no more than 0.02 below stock | 0.447 | 0.453 (+0.0067) | PASS |
| TTFT on a ~2000-token probe | within 1.2x of stock | 2.380 s | 2.436 s (x1.02) | PASS |

The perplexity ratio of 1.051 is a 5.1% increase over stock — inside the
declared gate, but an increase, not parity. Eval-200 moved +0.005 on the
200-item raw score and +0.0067 on the 150 live items the gate scores;
a set this size cannot resolve a difference that small, and it does not
by itself prove equivalence.

The full delta — both scores and what each metric means in use — rendered
from `results/quality.tsv`:

<!-- quality:start -->
| metric | stock | route h | what it means for a user |
|---|---|---|---|
| Perplexity, held-out passages | 11.94 | 12.54 (ratio 1.051 = +5.1%) | How surprised the model is by fresh text; +5.1% is a real but small regression inside the declared 1.10 gate. |
| eval-200: reason | 32/50 | 27/50 | Multi-step reasoning items solved; -5 of 50 on a 50-item column -- too few items to separate a real regression from noise. |
| eval-200: trap | 13/50 | 16/50 | Trick-question resistance; +3 of 50 on a 50-item column -- same limit applies. |
| eval-200: tool | 0/50 | 0/50 | Tool-call items score zero on both checkpoints -- the prompts never name a callable tool, so the column is 0 by construction and cannot judge either side. |
| eval-200: longread | 22/50 | 25/50 | Long-document comprehension; +3 of 50 on a 50-item column -- same limit applies. |
| eval-200: total | 67/200 | 68/200 (+0.005 acc) | Overall accuracy moved +0.005 raw (+0.0067 on the 150 live items the gate scores, tool floor excluded), inside the declared gate; by itself it does not prove equivalence. |
| TTFT, ~2000-token probe | 2.380 s | 2.436 s (x1.02) | Delay before the first token on a long prompt; a 2% increase measured on this probe -- no user-perception test was run. |
| Degenerate outputs, 64-prompt ruler | 0 | 0 | Empty or looping completions; zero on both sides. |
| 13-item evaluate suite | -- | pending | The end-to-end serve evaluation (short/long decode, parallel-4, agentic tool-use, long-context, trick questions); queued on this configuration -- this row fills in when it lands. |
<!-- quality:end -->

## How to reproduce

The full runbook — prerequisites, exact commands, expected wall times,
and disk needs — is in [AGENTS.md](AGENTS.md). In short: requantize
the checkpoint with `requant/requant.py --target h`, build the RDMA
image and the overlays, serve with `serve/start-head.sh` and
`serve/start-worker.sh`, gate with `requant/verify.py check`, then
measure with `bench/measure.py`.

## What did not work, and when

Measured on the same ruler; listed so the search space is on record.

<!-- failures:start -->

| item | what | cause | date |
|---|---|---|---|
| **moe-dsl-kernel-overlay** | Three successive failures. First, the patched kernel path was dead code: the backend is gated to device family 100 and never instantiates on sm_121, so an apparent -4 ms TPOT delta was run-to-run noise. Second, after re-hooking the class the deployment selects, every call failed a static eligibility check (the model's SwiGLU clamp limit) and silently fell back to stock: 256 calls, zero dispatches. Third, a host-side cute.make_layout call raised a TypeError at first use. | Instrumented, then fixed or abandoned each time; the overlay is not in the released config. | 2026-09-15 |
| **fp8-dense** | FP8-quantized dense linear variants ran slower than W4A16 NVFP4 on this stack. | Measured slower; not adopted. | 2026-09-15 |
| **mtp-k-ge-3** | K=3 gave 19.39 tok/s and K=4 gave 17.44 tok/s vs K=2's 19.83 on an earlier requant route (sockets, pair 1); on stock, K=4 gave 15.2 and K=5 gave 13.1. | Per-position acceptance drops more quickly than the extra draft tokens save steps. K=2 was the best of the K values measured on this route; on route h + RDMA a first sweep-A pass measured K=1 at 34.28 and K=3 at 31.98 tok/s, and the full same-checkpoint sweep in both K orders is still pending. | 2026-09-16 |
| **ep-off** | Removing --enable-expert-parallel gained ~0.7%. | Inside the pair-to-pair offset; not a lever. | 2026-09-15 |
| **eager-mode** | --enforce-eager is ~10 ms/step faster on the stock single-stream step, but the gain mostly does not carry into MTP decode. | The step it speeds up is not the step speculation runs. Not adopted. | 2026-09-15 |
| **k4-k5-boot-oom** | K=4/K=5 failed to boot for the route-h checkpoint at these memory settings (128 GB unified memory); the stock checkpoint booted both. One K=4 attempt was killed externally by an orphaned pipeline's timeout cleanup; the genuine cause: the Ray memory monitor's default 95% threshold OOM-killed the worker right after speculator CUDA-graph capture, and at K=5 the capture took 103 s and 3.03 GiB, leaving 0.89 GiB of KV cache where 0.92 GiB was required. | Memory accounting and capture cost scale with K; the fix direction is a smaller drafter, not bigger K. | 2026-09-15 |
| **requant-loader-write** | A fused-KDA write missing the NVFP4 global scale produced KeyError ...weight_scale_2 at load. | Writer bug; now covered by verify.py config and the kda-quant overlay. | 2026-09-15 |
| **nccl-ll-symm-ar** | NCCL low-latency envs and symmetric-memory all-reduce booted but returned empty response bodies or died under concurrency. | Broken output path; unsafe, not adopted. | 2026-09-15 |
| **ev-ordering** | The expensive drafter-training data chain (extraction, two generation passes, conversion; ~5 h of wall time on the 2-GPU pair, ~10 GPU-hours) was queued ahead of the cheap K sweep; the trained drafter's holdout top-1 came out ~0.08, and the sweep showed K=2 was already the best of the measured values. | Sequenced by pipeline momentum instead of expected information per GPU-hour. | 2026-09-15 |

<!-- failures:end -->

Timeline of the measurements above and below:

| date | milestone |
|---|---|
| 2026-09-13 | stock baseline on pair 2: 10.75 tok/s C=1 |
| 2026-09-15 | official MTP at K=2 on stock: 19.05 tok/s on pair 1, 17.84 on pair 2; most alternative levers failed in this window |
| 2026-09-16 | route h requant, RDMA image, headline 35.09 tok/s on pair 1 |

## Why I started from the NVIDIA NVFP4 checkpoint

The official `nvidia/GLM-5.3-Flash-NVFP4` release is quantized with
NVIDIA's own toolchain, and its model card publishes BF16-vs-NVFP4
benchmark numbers that show essentially no accuracy loss (model card
revision `09b04e5e74bca08ca8549fc736d4cdd8624bfde3`): GPQA Diamond
0.9217 -> 0.9211, SciCode 0.5621 -> 0.5769, MMMU Pro 0.7688 -> 0.763,
AA-LCR 0.71 -> 0.7106, IFBench 0.613 -> 0.6054, Terminal Bench 2.1
0.8258 -> 0.8315. That makes it a trustworthy base. My route h only
extends the same 4-bit treatment to the layers the official release left
in BF16, and I keep the gate numbers next to the speed numbers so the
trade-off stays visible.

## Stage 1 — the released configuration

- **Route `h` requantization** — attention linears (KDA fused
  `in_proj`/`out` projections, MLA q/kv/o, indexer `wq_b`),
  shared-expert `gate`/`up`/`down`, and `lm_head` converted to W4A16
  NVFP4. Router, norms, and embeddings stay BF16.
- **NCCL over RDMA** — a derived image (`docker/Dockerfile.nccl-ib`)
  installs Ubuntu questing `rdma-core` over the official image so the
  NCCL NET/IB plugin can init; `serve/nccl-ib.sh` derives the HCA and
  RoCE v2 GID per node and passes `uverbs*`/`rdma_cm` devices plus
  `IPC_LOCK`/unlimited `memlock`.
- **Official MTP draft at K=2** — `num_speculative_tokens=2` against the
  model's own MTP layer, enabled by the `mtp-bf16` overlay which builds
  the draft subtree unquantized (the stock image makes the draft inherit
  the target's NVFP4 quant config and crashes loading BF16 MTP weights).

## Speed results

Fixed ruler: 64 Japanese prose prompts, `temperature=0`,
`max_tokens=512`, thinking skipped via an empty assistant continuation.
Unless noted, C=1 streams all 64 prompts sequentially with per-prompt
TTFT/TPOT. The shipped `serve/` scripts pin `--max-model-len 16384`,
`--gpu-memory-utilization 0.85`, FP8 KV cache; the measured rows ran
variants — the 2026-09-13 stock baseline used `--max-model-len 131072`,
the c1pair and route-h rows ran `--gpu-memory-utilization 0.86`, and
expert parallel was off on the two stock + MTP rows marked `ep` = `off`
(17.84 and 19.05 tok/s) and on everywhere else. Each row's exact flags
live in the file its `source_log` entry in `results/results.tsv` points
to under `results/logs/`.

How I count:

- **tok/s** — total completion tokens / total wall time of the pass,
  including TTFT. Whole-window throughput.
- **TPOT med** — median across prompts of
  `(wall - TTFT) / (tokens - 1)`.
- **weighted TPOT** — `sum(wall - TTFT) / sum(tokens - 1)` across
  prompts; a token-weighted mean. The reciprocal of TPOT med is a
  different statistic from tok/s and is not expected to match it (51.5
  ms -> 19.42 vs the measured 18.98 on the same run).
- **accept** — spec-decode counter deltas over the pass: accepted /
  draft tokens. Mean accepted length (outputs per cycle) was 2.244 on
  the headline row: 1.244 accepted draft tokens plus the bonus token.
- **quality gate / TTFT gate** — the four-criterion gate above, split
  into two columns per row. The degenerate / PPL / eval-200 legs attach
  to the checkpoint (the "PASS (PPL x1.051)" cell). The TTFT leg is
  transport-dependent and was probed once, on the route-h sockets
  serving: sockets rows read "verified on sockets" and RDMA rows read
  "verified on sockets; not re-run under RDMA" — no row claims a
  four-condition PASS on a transport the TTFT probe never saw. Each
  row's own TTFT med column carries the per-transport number.

<!-- results:start -->

| checkpoint | pair | transport | K | ep | tok/s | TPOT med ms | weighted TPOT ms | TTFT med s | accept | quality gate | TTFT gate |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| stock | 2 | sockets | - | on | 10.75 † | 95.3 † | 92.5 † | 0.361 † | - | ref | — |
| stock + MTP | 2 | sockets | 2 | on | 17.71 | 55.5 | 55.7 | 0.447 | 0.6128 | ref | — |
| stock + MTP | 2 | sockets | 2 | off | 17.84 | 55.2 | n/m | 0.441 | 0.6183 | ref | — |
| stock + MTP | 1 | sockets | 2 | off | 19.05 | 51.9 | 51.7 | 0.430 | 0.6175 | ref | — |
| stock | 2 | NCCL/IB (RDMA) | - | on | 14.32 † | 69.3 † | 69.3 † | 0.349 † | - | ref | — |
| stock + MTP | 2 | NCCL/IB (RDMA) | 2 | on | 23.93 | 41.0 | 41.2 | 0.366 | 0.6137 | ref | — |
| route h | 1 | sockets | - | on | 18.98 | 51.5 | 52.1 | 0.330 | - | PASS (PPL x1.051) | verified on sockets |
| route h + MTP | 1 | sockets | 2 | on | 24.01 | 40.7 | 41.0 | 0.389 | 0.6221 | PASS (PPL x1.051) | verified on sockets |
| **route h + MTP** | **1** | **NCCL/IB (RDMA)** | **2** | **on** | **35.09** | **27.9** | **27.9** | **0.320** | **0.6221** | **PASS (PPL x1.051)** | **verified on sockets; not re-run under RDMA** |
| route g + MTP | 2 | NCCL/IB (RDMA) | 2 | on | 29.46 | 33.3 | 33.3 | 0.340 | 0.6182 | PASS (PPL x0.999) | verified on sockets; not re-run under RDMA |
| route h | 1 | NCCL/IB (RDMA) | 0/1/3/4 | on | pending |  |  |  |  |  |  |
| route h + MTP | 2 | NCCL/IB (RDMA) | 2 | on | 34.47 | 28.2 | 28.4 | 0.313 | 0.6221 | PASS (PPL x1.051) | verified on sockets; not re-run under RDMA |
| route h + MTP | 1 | NCCL/IB (RDMA) | 2 | on | 34.48 | 28.3 | 28.4 | 0.317 | 0.6193 | PASS (PPL x1.051) | verified on sockets; not re-run under RDMA |
| route h + MTP | 1 | sockets | 2 | on | 23.75 | 41.2 | 41.4 | 0.394 | 0.6213 | PASS (PPL x1.051) | verified on sockets |
| route h + MTP | 1 | NCCL/IB (RDMA) | 2 | on | 35.12 | 27.9 | 27.9 | 0.313 | 0.6193 | PASS (PPL x1.051) | verified on sockets; not re-run under RDMA |
| route h + MTP | 1 | sockets | 2 | on | 24.35 | 40.4 | 40.4 | 0.39 | 0.6174 | PASS (PPL x1.051) | verified on sockets |
| route c + MTP | 1 | sockets | 2 | on | 19.83 | 49.6 | n/m | 0.435 | 0.6218 | PASS | verified on sockets |
| route c + MTP | 1 | sockets | 3 | on | 19.39 | 50.9 | n/m | 0.461 | 0.4925 | PASS | verified on sockets |
| route c + MTP | 1 | sockets | 4 | on | 17.44 | 56.7 | n/m | 0.446 | 0.3925 | PASS | verified on sockets |

<!-- results:end -->

† I ran these rows with C=1 over the first 8 prompts of the ruler, not
all 64.

Repeat measurements (both landed 2026-09-16):

- **Second pair** — the same headline configuration on the other node
  pair read 34.47 tok/s, inside the roughly 5-7% pair-to-pair offset I
  see on repeated runs of identical configs.
- **Publication re-measurement** — an interleaved TCP/RDMA re-run of the
  headline config on a fresh, non-overlapping 64-prompt set with
  prefix caching disabled and warm-up counted separately: 34.48 and
  35.12 tok/s over RDMA, 23.75 and 24.35 tok/s over TCP. The RDMA
  re-runs sit within ~2% of the original 35.09.
- Per-pass GPU telemetry (nvidia-smi at 2 s on both nodes; SM clock
  median / power median–max / temp max): the 34.48 pass — 2190 MHz,
  24.9–27.1 W, 68 °C; the 23.75 pass — 2190 MHz, 20.5–23.6 W, 64 °C;
  the 35.12 pass — 2190 MHz, 24.5–26.7 W, 65 °C; the 24.35 pass —
  2190 MHz, 20.6–24.3 W, 65 °C. The second-pair repeat captured no
  telemetry (n/m).

Still pending:

- **K sweep** — the full K=0/1/3/4 sweep on route h + RDMA against the
  same checkpoint, queued in both K orders for warm-up/order effects.
  A first sweep-A pass has already landed — K=1 gave 34.28 tok/s and
  K=3 gave 31.98 on pair 1 over RDMA — but until the full sweep is
  verified K=2 remains the best of the values I measured, not a proven
  optimum.

The headline run (pair 1, 2026-09-16): 32407 completion tokens over
923.43 s of wall, zero failed requests, finish reasons 61 `length` / 3
`stop`. C=32 aggregate on the stock checkpoint was 94.66 tok/s (sockets)
and 109.03 tok/s (RDMA); no C=32 row exists for route h yet.

## About the name

Wabi (侘び) is the Japanese sense of accepting what is imperfect or
plain and finding richness in it; this release is a work in progress
that I publish as it is, improvements included when they are measured.

## Stage 2

A faster configuration may follow.

## Files

```
AGENTS.md                   reproduction runbook (prereqs, commands, times)
CONTRIBUTORS.md             who worked on this
requant/requant.py          weight-only repack, targets a-e/g/h
requant/verify.py           config verifier + gate v2 (capture/check)
requant/build-mtp-draft.py  BF16 MTP draft dir from the checkpoint
overlays/                   image-source patchers (fail-closed anchors):
  patch-kda.py              restore quant_config in fused KDA members
  patch-mla.py              pass quant_config into Glm5NextMLAAttention
  patch-mtp.py              build the MTP draft subtree unquantized
  apply-step-attr-patch.py + step_attr.py   CUDA-event step attribution
  fetch-image-file.sh       snapshot a file out of the image
  build-overlays.sh         fetch + patch everything into _build/
docker/Dockerfile.nccl-ib   derived image: questing rdma-core on noble
serve/                      start-head.sh / start-worker.sh / nccl-ib.sh
bench/                      measure.py + prompts-64.jsonl + eval-200.jsonl
tests/                      run-tests.sh (offline smoke) +
                            check-md-invariants.py (wording-only edit check)
```

Two `eval-200.jsonl` prompts were re-quoted for publication (ASCII-safe
quoting / reworded instruction); expected answers and grading are
unchanged.

## Contributors

- tenhkspark (the maintainer) — the one who worried, watched, and
  said go (anonymous handle; no real name is published).
- Claude Fable 5.1 (Anthropic) — direction, experiment design,
  review and acceptance.
- Claude Opus 5 (Anthropic) — pre-release review.
- Astra (OpenAI, via pi) — adversarial review of the measurements
  and the plan.
- Devin SWE-2 (Cognition) — implementation, diagnostics, queue
  staging, repository drafting.
- GLM-5.3 (Z.ai) — implementation seat.
- GLM-5.3-Flash (Z.ai) — implementation seat and summarisation.

## License

Apache-2.0, see LICENSE. Model weights are not part of this repository;
requantize the NVIDIA checkpoint yourself with `requant/requant.py`.
