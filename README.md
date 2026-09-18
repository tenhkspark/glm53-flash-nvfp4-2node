# glm53-flash-nvfp4-2node

[English](README.md) · [日本語](README.ja.md) · [简体中文](README.zh.md) · [한국어](README.ko.md)

Two NVIDIA DGX Spark nodes serving `GLM-5.3-Flash-NVFP4` at **35.09
tok/s** of single-stream decode — one user, one stream, C=1, thinking
off — measured on node pair 1 over NCCL/IB (RDMA) with the
checkpoint's own MTP draft at K=2, on a fixed ruler of 64 Japanese
prose prompts at `max_tokens=512`, `temperature=0`. In use that means
Japanese prose arrives at about the pace I read it on screen instead
of in visible bursts.

This repository is the recipe behind that number: Ray tensor
parallelism across the two nodes (TP=2), the weight-only
requantization I use to recover single-stream decode speed (route h),
an RDMA-capable derived image, and the overlays that make the MTP
draft load. That work is my own implementation and my own
measurements on my own hardware. One shipped file is not mine to
claim: `overlays/flashinfer_mla_sparse_sm120.py` is the serving
image's own vLLM source file, licensed Apache-2.0 and carrying its
upstream copyright header, modified here to support no-rope MLA; the
other overlays are patchers I wrote that rewrite that same image's own
sources at build time. [NOTICE](NOTICE) draws the line file by file —
*enjoying the incomplete*.

Every number below carries its conditions (pair, transport, prompt
count, K, expert parallel, `max_tokens`), and every row in the speed
table names the log it came from. The stock row that makes the
comparison strictly like-for-like landed on 2026-09-17 — 14.39 tok/s,
pair 1 over RDMA, no draft, all 64 prompts — so the requant and the
draft now each separate under a single condition change.

The recipe has also been run back from the outside. Starting from an
empty working directory on the other node pair, with only the published
[AGENTS.md](AGENTS.md) and `scripts/agent-run.sh` to go on, the run
reached a serving configuration and measured 37.08 tok/s on its
10-prompt smoke — +5.7% against the 35.09 headline, inside the 7% band
the shipped checker allows. "Reproduced from the published runbook"
below says what that test does and does not show.

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

Release tag: **v1**. The requantized checkpoint is published on
Hugging Face as `tenhkspark/GLM-5.3-Flash-NVFP4-Wabi`, one repo, at
the same tag. There is no v2 yet: a v2 tag will appear only if some
other configuration passes the same gate and measures faster on the
same ruler.

## Why two nodes

The checkpoint is about 204 GB on disk (about 190 GB after the route-h
rewrite) and does not fit in one node's 128 GB of unified memory, so
the model is split across two nodes with TP=2 and every decode step
crosses the link between them. The two nodes are cabled port to port
with a single 200GbE QSFP copper cable on their ConnectX-7-class
ports, with no switch in the path; that direct-attached pair is what
every number here was measured on. The same netdev carries both
transports: `NCCL_IB=1` runs NCCL over RoCE v2 ("NCCL/IB (RDMA)" in
the tables), `NCCL_IB=0` falls back to TCP ("sockets").

Because the link sits inside the decode loop, transport is not a
detail here — it is one of the three levers below.

## The released configuration

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

## What each change buys

The honest decomposition needs four rows on one pair, one transport
and one ruler. All four are measured now:

| step | what it isolates | row | measured |
|---|---|---|---|
| stock, no draft, RDMA | the starting point on the fast transport | pair 1, 64 prompts | 14.39 tok/s |
| route h, no draft, RDMA | the requant alone | pair 1, 64 prompts | 27.15 tok/s |
| route h + MTP K=2, RDMA | the draft on top of the requant | pair 1, 64 prompts | 35.09 tok/s |
| route h + MTP K=2, sockets | the same checkpoint and draft on the slow transport | pair 1, 64 prompts | 24.01 tok/s |

The first three rows now share every condition — pair 1, NCCL/IB (RDMA),
the same 64 Japanese prose prompts, `temperature=0`, `max_tokens=512`,
expert parallel on, thinking off — so the requant and the draft each
separate cleanly from everything else:

- stock, no draft: 14.39 tok/s, TPOT median 69.0 ms, TTFT median
  0.293 s, zero failed requests.
- route h, no draft: 27.15 tok/s, TPOT median 35.7 ms, weighted TPOT
  36.4 ms, TTFT median 0.269 s, zero failed requests.
- route h + MTP K=2: 35.09 tok/s, TPOT median 27.9 ms, TTFT median
  0.320 s, acceptance 0.6221, zero failed requests.

27.15 / 14.39 = 1.89 and 35.09 / 27.15 = 1.29, so on this configuration
the requant is worth about 1.89x and the draft about 1.29x on top of
it. These are the only two ratios in this README taken across a single
condition change; every other pair of rows differs in more than one
thing, so I do not divide them. Each side of both ratios is a single
pass, and run-to-run spread on identical configs is about 2%.

What I can say today, each within one pair and one transport:

- **The draft.** Route h, pair 1, 64 prompts: over RDMA 27.15 tok/s
  without a draft and 35.09 with MTP at K=2 (x1.29); over sockets 18.98
  and 24.01, expert parallel on in all four. Acceptance on the K=2
  passes runs 0.6174–0.6221, i.e. about 2.24 output tokens per draft
  cycle.
- **The transport.** Route h + MTP K=2, pair 1, 64 prompts: 24.01 tok/s
  over sockets against 35.09 over RDMA; the two publication re-runs
  read 23.75 / 24.35 over sockets and 34.48 / 35.12 over RDMA on the
  same pair, so the transport gap reproduces.
- **The requant.** Stock against route h, pair 1, 64 prompts, over
  RDMA with no draft on either side and expert parallel on in both:
  14.39 tok/s stock against 27.15 route h (x1.89). The older stock
  reference — pair 2, sockets, the first 8 prompts of the ruler at
  `--max-model-len 131072`, 10.75 tok/s — is not a denominator for
  this; dividing by it would mix three condition changes at once, so I
  do not.

## What it costs

The requant trades quality for speed. The trade is bounded only on the
four criteria below — all Japanese, single-turn, thinking off — and not
on anything outside them. The gate was declared before any speed run;
the released configuration had to pass all four:

| criterion | threshold | stock | route h | result |
|---|---|---|---|---|
| degenerate outputs over the 64-prompt ruler | 0 | 0 | 0 | PASS |
| perplexity ratio on the 8-sentence probe | <= 1.10 | 1.0 | 1.051 | PASS |
| eval-200 live accuracy (150 items, the 0/50 tool floor excluded) | no more than 0.02 below stock | 0.447 | 0.453 (+0.0067) | PASS |
| TTFT on a ~2000-token probe | within 1.2x of stock | 2.380 s | 2.436 s (x1.02) | PASS |

**How perplexity is measured.** Perplexity is measured on eight short
Japanese sentences (249 characters in total) written for this test and
defined inline in `requant/verify.py`; they are not drawn from any
public corpus and no training-set exclusion is claimed. Each sentence
is sent to the served endpoint as a prefill-only request
(`/v1/completions` with `echo=true`, `max_tokens=1`,
`prompt_logprobs=1`, temperature 0); the log-probabilities of the
actual prompt tokens are accumulated across all eight sentences and
the mean negative log-likelihood over that single pooled token stream
is exponentiated into one number. Stock and candidate are measured by
the identical function against the same served model, and the gate is
the ratio candidate/stock <= 1.10 — the released route h scored 12.543
against 11.938 stock, a ratio of 1.051. The sample is small, so this
checks for gross degradation, not for parity.

**How eval-200 is scored.** eval-200 is a fixed set of 200 Japanese
single-turn prompts shipped in this repository (`bench/eval-200.jsonl`):
50 each of reason, trap, tool and longread, mixed easy/medium/hard.
Every item is answered greedily on the served endpoint (temperature 0,
`max_tokens` 384, thinking skipped via an empty assistant continuation)
and graded by a deterministic rule in `requant/verify.py`:
whitespace-insensitive exact match when the item's rubric demands the
answer alone, substring containment otherwise; trap items must carry a
refusal and no digits, tool items must name the required functions in
order plus the expected final value. The tool column is 0/50 on every
checkpoint by construction — the prompts never name a callable function
and no tool schema is sent — so the gate scores the remaining 150 live
items and requires the candidate to stay within 0.02 of stock with zero
request errors (0.447 stock -> 0.453 route h). The raw totals 67/200 ->
68/200 are reported for transparency only; they include that permanent
zero column and are not the pass/fail criterion. The category moves are
not independent: the trap grader rewards refusal, so a more hedging
model scores higher there and lower on reason. A net +1 total can be
produced by a directional degradation.

**The TTFT probe** is those same eight sentences concatenated twelve
times, sent as one prompt; the gate compares the candidate's
time-to-first-token on it against stock, on the route-h sockets
serving.

The perplexity ratio of 1.051 is a 5.1% increase over stock — inside the
declared gate, but an increase, not parity. Eval-200 moved +0.005 on the
200-item raw score and +0.0067 on the 150 live items the gate scores. A
150-item set resolves accuracy to roughly +-0.08 at 95% confidence
(paired, assuming ~20% of items flip), so the +0.0067 I measured is
indistinguishable from zero — and so would be a true regression of 0.05.
The 0.02 gate threshold is finer than what this set can resolve: read
that +-0.08 half-width as a normal sigma of 0.041, and a checkpoint
genuinely 0.06 worse than stock still clears the accuracy condition
about 16% of the time, and one genuinely 0.08 worse about 7%. Two things
that arithmetic is not: 0.06 and 0.08 are percentage points of accuracy,
not relative drops (0.06 below 0.447 is 0.387, not a 6% decline), and the
about-16% / about-7% figures are a normal approximation for the accuracy
condition on its own, not the probability of passing the four-criterion
gate as a whole.
The gate is still weak here — a +-0.08 interval against a 0.02 threshold
is four times wider than the thing it is supposed to catch. Reading this
gate as 'no regression' is wrong; it only rules out a large one.

Speculative decoding with greedy sampling is designed to accept only
tokens the target model would have produced, so in principle the draft
cannot move the output distribution and the quality delta belongs to
the requantization alone. I did not measure that equivalence, so treat
it as the design argument it is, not as a measured result.

The full delta — both scores and what each metric means in use —
rendered from `results/quality.tsv`:

<!-- quality:start -->
| metric | stock | route h | what it means for a user |
|---|---|---|---|
| Perplexity, 8-sentence Japanese probe | 11.938 | 12.543 (ratio 1.051 = +5.1%) | How surprised the model is by the probe text; +5.1% is a real but small regression inside the declared 1.10 gate. The probe is 249 characters of Japanese written for this test -- nothing was held out from training, and a sample this small checks for gross degradation, not for parity. |
| eval-200: reason | 32/50 | 27/50 | Multi-step reasoning items solved; -5 of 50 on a 50-item column -- too few items to separate a real regression from noise. |
| eval-200: trap | 13/50 | 16/50 | Trick-question resistance, graded mechanically: an answer passes if it contains a refusal marker and no digits. That rule rewards hedging, so a checkpoint that became more evasive would gain here while losing on reason -- which is the direction this pair of columns actually moved (+3 trap, -5 reason). I did not test whether the two moves share that cause; do not read the +1 net total as 'no change'. |
| eval-200: tool | 0/50 | 0/50 | Tool-call items score zero on both checkpoints -- the prompts never name a callable tool, so the column is 0 by construction and cannot judge either side. |
| eval-200: longread | 22/50 | 25/50 | Long-document comprehension; +3 of 50 on a 50-item column -- same limit applies. |
| eval-200: total | 67/200 | 68/200 (+0.005 acc) | Overall accuracy moved +0.005 raw (+0.0067 on the 150 live items the gate scores, tool floor excluded), inside the declared gate; by itself it does not prove equivalence. The 95% interval on this difference is about +-0.08, which is four times wider than the 0.02 gate threshold. |
| TTFT, ~2000-token probe | 2.380 s | 2.436 s (x1.02) | Delay before the first token on a long prompt. The 2% gap is the median of three runs of the same prompt and is the same size as the run-to-run spread I measure on identical configurations, so this probe shows no TTFT regression it could have detected -- it does not show that TTFT is unchanged. No user-perception test was run. |
| Degenerate outputs, 64-prompt ruler | 0 | 0 | Empty or looping completions; zero on both sides. Zero out of 64 is consistent with a true rate of up to about 5% (rule of three), and the detector only catches empty output, repeated-token runs and exactly periodic loops -- it cannot see a fluent answer that is wrong, truncated or off-topic. |
| 13-item evaluate suite | -- | -- | The end-to-end serve evaluation (short/long decode, parallel-4, agentic tool-use, long-context, trick questions). It ran on both node pairs on 2026-09-16 and both runs finished, but 4 of the 13 items produced no result on either run: the two agentic tool-use items (3-step tool success, 4th-step final answer) and both 108K long-context items (en, ja) came back blank. The two runs were also not the same configuration -- pair 1 ran with speculative decoding off, pair 2 ran with it on at a 78.0% acceptance rate -- so the 9 items that did produce numbers cannot be read as a route h vs stock comparison, and I do not report them as one here. Agentic tool use and 108K long context therefore remain untested. |
<!-- quality:end -->

Every row above is Japanese, single-turn and thinking off. The one row
that would cover long context, parallel requests and agentic tool use —
the 13-item suite — did run, on 2026-09-16, on both node pairs, but 4 of
its 13 items returned nothing in either run: agentic tool success,
agentic final answer, integrated task en 108K and integrated task ja
108K. So agentic tool use and 108K long context are still unmeasured on
this checkpoint — not because the suite is waiting to run, but because
the items that cover them produced no result when it did. The two runs
are also not a like-for-like route h vs stock comparison: pair 1 ran with
no speculation and pair 2 ran with speculative decoding at 78.0%
acceptance, so even the items that did return cannot be read as a
difference between the two checkpoints.

**What this gate does not measure.** Every number above is Japanese,
single-turn, thinking off, greedy, under 2k tokens of context and one
request at a time. I have no measurement of this checkpoint on English
or any other language, on code correctness, on multi-turn conversations,
on tool calling (the tool column is a structural zero), on instruction
following, on long-context quality — the needle probe in "Long context,
measured" scores 40/40 up to 194,544 tokens, but that is retrieval, and
every quality probe above fits in about 2k — on
safety behaviour, or with thinking enabled — which is how this model
family is normally used. The requant rewrites the dense linears and
lm_head, so those are exactly the places a regression could hide from
these four probes. If you depend on any of them, measure it yourself
before adopting this checkpoint; requant/verify.py takes a different
prompt set with one flag.

### Route g, if you want the smaller quality delta

`requant/requant.py --target g` writes a more conservative
requantization: on pair 2 over RDMA with MTP at K=2 and the same
64-prompt ruler, route g measured 29.46 tok/s at a perplexity ratio of
0.999, against route h's 34.47 tok/s at 1.051 on that same pair,
transport, K and ruler. I released h because the gate's job is to
bound the quality delta, not to minimise it, and 1.051 is inside the
bound I declared before measuring. If you would rather spend the speed
on the smaller delta, g is the one flag change.

## Before you serve: no authentication, all interfaces

`serve/start-head.sh` puts the API up with `--host 0.0.0.0`, so it
listens on every interface of the node, and the server checks no API key
— any client that can reach the port can send requests. The rest of this
file assumes you have read that.

- Keep this configuration on a network you trust and do not expose the
  port to a public one.
- Access control is yours to provide. If the endpoint has to be reachable
  from beyond the node, put your own firewall rules in front of it, and
  an authenticating proxy with TLS if the traffic needs either.
- Ray's management ports and the ports the two nodes use between
  themselves are not the API port and need isolating on their own. The
  shipped script binds the Ray dashboard to 127.0.0.1, but that is one
  port out of several: putting authentication in front of the API does
  not protect Ray.
- A long enough prompt can exhaust host memory on this configuration
  ("Long context, measured" below measures where that happens), so an
  endpoint left open is a way to take the nodes down, not only a way to
  read them.

## How to reproduce

The full runbook — prerequisites, exact commands, expected wall times,
and disk needs — is in [AGENTS.md](AGENTS.md). In short: requantize
the checkpoint with `requant/requant.py --target h`, build the RDMA
image and the overlays, serve with `serve/start-head.sh` and
`serve/start-worker.sh`, gate with `requant/verify.py check`, then
measure with `bench/measure.py`. The prompt sets are described in
[bench/README.md](bench/README.md).

**Reproducing the headline speed needs `MTP_DIR`.** The 35.09 tok/s row
is route h *with* the MTP draft at K=2, and the draft is loaded only if
`MTP_DIR` points at a draft directory. `serve/serve.env.example` carries
that line commented out with a `CHANGEME` path — on purpose, because the
path differs per machine — so a serve brought up from the example as it
ships runs with no draft and decodes at draft-free speed: I measured
28.28 tok/s that way, consistent with the 27.15 no-draft row above (+4%),
against 34.99 tok/s on the same configuration with the draft — TPOT
median 27.9 ms, TTFT median 0.309 s, acceptance 0.6223, 0 of 64 requests failed.
Uncomment the line and put your own path in it before measuring.
`requant/build-mtp-draft.py` builds the draft directory from the stock
checkpoint; [AGENTS.md](AGENTS.md) gives that step.

## Reproduced from the published runbook

Every number above was measured by the person who wrote the recipe, on
the machine the recipe was written on — the weakest part of any speed
claim. So I ran the recipe back from the outside: an empty working
directory on the other node pair, the repository skeleton as published,
[AGENTS.md](AGENTS.md) and `scripts/agent-run.sh` as the only
instructions, and one of that pair's own nodes as the operator node.
The run built the RDMA image and the overlays, filled `setup.env` for
its own two nodes, brought up head and worker, ran the config gate and
measured — nothing taken from my working tree. It skipped one step: the
requantization itself. The released route-h checkpoint already sitting
on that pair stood in for the download, so what this test reproduces is
everything from the checkpoint to the measured tokens, not the weight
rewrite.

- 37.08 tok/s on the run's own 10-prompt smoke, TPOT median 26.5 ms,
  acceptance 0.6177, zero failed requests.
- Against the 35.09 tok/s release row that is +5.7%, inside the 7% band
  `scripts/verify-result.py` allows for the pair-to-pair offset.
- TTFT median 0.36 s against the release row's 0.32 s — 12.5% slower,
  outside that band. The checker reports TTFT rather than gating on it,
  and this row is why it still gets reported.

What the test does not show: the smoke pass is 10 prompts and the
release row is 64, on a different prompt set, so the +5.7% is not a
like-for-like comparison — a 10-prompt pass carries much more
run-to-run spread than the ruler does. What it does show is that the
published steps, followed on their own from an empty directory, reach a
serving configuration in the same neighbourhood as the one I released.
Getting there took seven fixes to the runbook and the driver, listed
under "What did not work, and when" below; each was found by the clean
run failing, and none of them was visible from inside my own tree.

## Using the served model

`serve/start-head.sh` puts an OpenAI-compatible API on `PORT` (default
8000). Three properties of that API are decided by the serve line and
cannot be guessed from the outside, so they are written down here.

**The model id is `GLM-5.3-Flash-NVFP4-Wabi`.** The script passes
`--served-model-name`, so that string — not the checkpoint path — is
what a request has to carry; any other id comes back 404. `/v1/models`
returns it together with the serving window:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

A complete request, from the head node:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "GLM-5.3-Flash-NVFP4-Wabi",
       "messages": [{"role": "user", "content": "Hello."}],
       "max_tokens": 128}'
```

An OpenAI-compatible client needs three settings and nothing else: base
URL `http://127.0.0.1:8000/v1` (the head's address in place of
localhost when the client runs elsewhere), model
`GLM-5.3-Flash-NVFP4-Wabi`, and any string as the API key — the server
does not check one. I drive this server from a coding agent set up
exactly that way.

**Tool calls are on.** The serve line carries
`--enable-auto-tool-choice --tool-call-parser glm45`, so a client may
send `tools` and gets structured `tool_calls` back. Without those two
flags every request carrying `tools` is refused with HTTP 400:

```
"auto" tool choice requires --enable-auto-tool-choice and
--tool-call-parser to be set
```

**Read `reasoning` as well as `content`, and do not try to switch
thinking off.** `--reasoning-parser glm45` moves the model's thinking
into a separate `reasoning` field on the message and leaves `content`
holding the answer. The knob that looks like an off switch is not one:

| what the client sends | `reasoning` | `content` |
|---|---|---|
| nothing — the default | the thinking | the answer |
| `chat_template_kwargs: {"enable_thinking": false}` | empty | thinking and answer, run together |
| an empty assistant turn with `continue_final_message` | the whole reply | empty |

`enable_thinking: false` stops the parser, not the model: it keeps
thinking, and the thinking lands in `content`, where the reader sees
it. The empty-assistant continuation — the idiom `bench/measure.py`
uses to skip thinking on the ruler — is the same mechanism from the
other side: the template closes the thinking block itself, the model
never emits the closing tag, and the parser leaves the entire reply in
`reasoning`. Neither case is fixable with a server-side setting. Send
neither, read both fields, and a stock OpenAI-compatible client
behaves.

**Check the host headroom at boot, every boot.** `serve/check-headroom.sh`,
run on both nodes before the first request. This is the one operational
step that the speed table cannot warn you about, so it gets its own
paragraph.

GB10 is unified memory: the GPU allocates out of host RAM, and about
100 GB of that reservation shows up in no standard kernel counter —
`nvidia-smi` reports FB Memory Usage as N/A on this part — so
`MemAvailable` is the only honest gauge you have. vLLM sizes its budget
from whatever happened to be free when it profiled, which means the
margin is decided by the boot rather than by the flags. The same
unmodified script on the same node reached READY with 5470, 9728 and
10270 MiB free on three different runs: a 4.8 GiB spread across boots
that differ in nothing I can see.

Then about 4.7 GiB more is spent *after* READY. Sampling `MemAvailable`
once a second on both nodes while climbing 21k → 32k → 64k → 128k →
197,485-token prompts, the best boot fell from 10270 MiB (8.24%) to
5579 MiB (4.48%) and stayed there. It is a one-time high-water mark, not
a leak — repeating a length costs nothing more — but it is charged the
first time each larger prefill shape arrives, which is exactly what a
coding agent does as its context grows. That boot served the whole
ladder, including the full declared window, with zero failures.

The 5470 MiB boot did not. It ran for six hours and was killed partway
through a 20,915-token agent prompt, with the driver logging
`NV_ERR_NO_MEMORY` first. It never had room for its own warm-up.

So the floor is warm-up plus whatever reaps you, and the check is at
boot: a node under it should be restarted, not tuned. Lowering
`--gpu-memory-utilization` is not the lever it looks like — at 204800
the limiting rank gets about 3.5 GiB of KV and 0.01 of utilization is
1.2 GiB, so two steps down and the engine can no longer open the window
it advertises. And one warning specific to reproducing this: **DGX OS
ships neither `earlyoom` nor `systemd-oomd`.** On our nodes an OOM
daemon we had installed ourselves turned this into a clean process kill.
Without one, the same pressure on this hardware is a hung node.

## Speed results

Fixed ruler: 64 Japanese prose prompts, `temperature=0`,
`max_tokens=512`, thinking skipped via an empty assistant continuation.
Unless noted, C=1 streams all 64 prompts sequentially with per-prompt
TTFT/TPOT. The shipped `serve/` scripts pin `--max-model-len 204800`,
`--max-num-seqs 20`, `--gpu-memory-utilization 0.85`,
`--enable-prefix-caching`, FP8 KV cache and
`RAY_memory_usage_threshold=0.99` on both nodes; the
measured rows below ran the measurement rig instead — almost all of them
at `--max-model-len 16384` with `--max-num-seqs 20`, the 2026-09-13
stock baseline at `--max-model-len 131072` with 32 slots and the pair-2
stock RDMA row at 16384 with 32 slots; the c1pair and route-h rows ran
`--gpu-memory-utilization 0.86`, and expert parallel was off on the two
stock + MTP rows marked `ep` = `off` (17.84 and 19.05 tok/s) and on
everywhere else. Each row's exact flags live in the file its
`source_log` entry in `results/results.tsv` points to under
`results/logs/`.

**Why the shipped window is 204800 and the table is not.** 16384 was the
rig value — the length I first got MTP up on — and it stayed pinned
through every comparison above. Re-measured on 2026-09-17 at the shipped
window, with expert parallel on as everywhere else in this table, the
ruler reads 34.78 tok/s — TPOT median 28.1 ms, TTFT median 0.313 s, 0 of
the 64 requests failed — against 35.09 at 16384 / 20 / 0.86. That is a
0.9% difference across a 12.5x longer window, inside the run-to-run
spread of about 2% on identical configurations.

**The shipped script is faster than that row, because it does not enable
expert parallel.** Nothing in `serve/start-head.sh` passes
`--enable-expert-parallel`, while every row in this table was measured
with it on. Run from the published script exactly as it ships, the same
64-prompt ruler reads **37.33 tok/s** — TPOT median 26.4 ms, TTFT median
0.302 s, 0 of 64 failed, acceptance 0.6168 — on a freshly booted pair,
and 36.95 on a repeat. Adding the two EP flags back to that same script
drops it to 35.05 (TPOT 27.9 ms, acceptance 0.6223), which reproduces the
34.78 row to within 0.8%. So the gap is expert parallel and nothing else.
EP also costs host memory here: it raised the post-READY high-water mark
from 4691 MiB to 6014 MiB and pushed the ladder's low water from 4.48% to
3.38%. It is slower and hungrier on this pair, so the scripts do not ship
it — and the table above keeps its own measured condition rather than
borrowing the faster number.

The longer window does not cost concurrency
either: 20 sequence slots do come up at 204800 (909 s to READY, then the
same 0-failure ruler), which they did not at 307200. Utilization is the
one flag that moved down rather than up — 0.89 has been refused at boot
on this pair, and at 0.88 the head node ran with about 2.1% of host RAM
free — close enough to exhaustion that it does not matter what reaps
the worker first — so the scripts ship 0.85 and leave 8.6%. The requant
is part of why the window fits: at 0.88 the stock checkpoint tops out at
156672 tokens (vLLM prints that ceiling when it refuses to start), while
route h boots at 204800. What the window is worth on prompts that
actually use it is measured in "Long context, measured" below.

How I count:

- **tok/s** — total completion tokens / total wall time of the pass,
  including TTFT. Whole-window throughput.
- **TPOT med** — median across prompts of
  `(wall - TTFT) / (tokens - 1)`.
- **weighted TPOT** — `sum(wall - TTFT) / sum(tokens - 1)` across
  prompts; a token-weighted mean. Five shipped JSONs carry this as an
  explicit recorded field (the four release re-runs plus the route-h
  no-draft RDMA row). For nine more rows the same quantity is
  computable from each run's per-prompt records in the pre-sanitisation
  factory log (not part of this repository); I did that arithmetic
  myself and added the result to the row's shipped `results/logs/`
  entry, noting the derivation there. The 2026-09-13 baseline has no
  per-prompt records anywhere I could find — its own recorded figure
  (92.5 ms) is a plain mean over prompts, a different statistic — so
  that row, and the remaining rows I have not derived this for, read
  `n/m` rather than mixing three definitions in one column. The
  reciprocal of TPOT med is also a
  different statistic from tok/s and is not expected to match it
  (51.5 ms is 19.42 tok/s by arithmetic, against the 18.98 measured on
  that run).
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
| stock | 2 | sockets | - | on | 10.75 † | 95.3 † | n/m | 0.361 † | - | ref | — |
| stock + MTP | 2 | sockets | 2 | on | 17.71 | 55.5 | 55.73 | 0.447 | 0.6128 | ref | — |
| stock + MTP | 2 | sockets | 2 | off | 17.84 | 55.2 | n/m | 0.441 | 0.6183 | ref | — |
| stock + MTP | 1 | sockets | 2 | off | 19.05 | 51.9 | 51.77 | 0.430 | 0.6175 | ref | — |
| stock | 2 | NCCL/IB (RDMA) | - | on | 14.32 † | 69.3 † | 69.29 | 0.349 † | - | ref | — |
| stock | 1 | NCCL/IB (RDMA) | - | on | 14.39 | 69.0 | n/m | 0.293 | - | ref | — |
| stock + MTP | 2 | NCCL/IB (RDMA) | 2 | on | 23.93 | 41.0 | 41.17 | 0.366 | 0.6137 | ref | — |
| route h | 1 | sockets | - | on | 18.98 | 51.5 | 52.12 | 0.330 | - | PASS (PPL x1.051) | verified on sockets |
| route h | 1 | NCCL/IB (RDMA) | - | on | 27.15 | 35.7 | 36.4 | 0.269 | - | PASS (PPL x1.051) | verified on sockets; not re-run under RDMA |
| route h + MTP | 1 | sockets | 2 | on | 24.01 | 40.7 | 40.96 | 0.389 | 0.6221 | PASS (PPL x1.051) | verified on sockets |
| **route h + MTP** | **1** | **NCCL/IB (RDMA)** | **2** | **on** | **35.09** | **27.9** | **27.92** | **0.320** | **0.6221** | **PASS (PPL x1.051)** | **verified on sockets; not re-run under RDMA** |
| route g + MTP | 2 | NCCL/IB (RDMA) | 2 | on | 29.46 | 33.3 | 33.35 | 0.340 | 0.6182 | PASS (PPL x0.999) | verified on sockets; not re-run under RDMA |
| route h + MTP | 1 | NCCL/IB (RDMA) | 1 | on | 34.28 | 28.6 | n/m | 0.304 | 0.7979 | PASS (PPL x1.051) | verified on sockets; not re-run under RDMA |
| route h + MTP | 1 | NCCL/IB (RDMA) | 3 | on | 31.98 | 30.8 | n/m | 0.331 | 0.483 | PASS (PPL x1.051) | verified on sockets; not re-run under RDMA |
| route h + MTP | 2 | NCCL/IB (RDMA) | 2 | on | 34.47 | 28.2 | 28.43 | 0.313 | 0.6221 | PASS (PPL x1.051) | verified on sockets; not re-run under RDMA |
| route h + MTP | 1 | NCCL/IB (RDMA) | 2 | on | 34.48 | 28.3 | 28.4 | 0.317 | 0.6193 | PASS (PPL x1.051) | verified on sockets; not re-run under RDMA |
| route h + MTP | 1 | sockets | 2 | on | 23.75 | 41.2 | 41.4 | 0.394 | 0.6213 | PASS (PPL x1.051) | verified on sockets |
| route h + MTP | 1 | NCCL/IB (RDMA) | 2 | on | 35.12 | 27.9 | 27.9 | 0.313 | 0.6193 | PASS (PPL x1.051) | verified on sockets; not re-run under RDMA |
| route h + MTP | 1 | sockets | 2 | on | 24.35 | 40.4 | 40.4 | 0.39 | 0.6174 | PASS (PPL x1.051) | verified on sockets |
| route c + MTP | 1 | sockets | 2 | on | 19.83 | 49.6 | n/m | 0.435 | 0.6218 | PASS | verified on sockets |
| route c + MTP | 1 | sockets | 3 | on | 19.39 | 50.9 | n/m | 0.461 | 0.4925 | PASS | verified on sockets |
| route c + MTP | 1 | sockets | 4 | on | 17.44 | 56.7 | n/m | 0.446 | 0.3925 | PASS | verified on sockets |
| route h + MTP | 2 | NCCL/IB (RDMA) | 2 | off | 37.33 | 26.4 | n/m | 0.302 | 0.6168 | — |  |
| route h + MTP | 2 | NCCL/IB (RDMA) | 2 | on | 35.05 | 27.9 | n/m | 0.304 | 0.6223 | — |  |

<!-- results:end -->

† I ran these rows with C=1 over the first 8 prompts of the ruler, not
all 64.

The route-h no-draft leg of the decomposition landed on 2026-09-16
(27.15 tok/s) and the stock no-draft leg on 2026-09-17 (14.39 tok/s),
both on pair 1 over RDMA on the same 64-prompt ruler, so the
denominator a speedup factor for the requant needs is measured:
27.15 / 14.39 = 1.89.

There is no "vs stock" column in this table on purpose: a column would
invite a ratio on every row, and most of the stock rows here were taken
on a different pair, a different transport or 8 prompts instead of 64,
so those ratios would fold three condition changes into one number. The
two pairs that do differ in a single condition are named in the
paragraph above and in "What each change buys".

K is not settled either. On pair 1 over RDMA with the 64-prompt ruler,
K=1 measured 34.28 tok/s and K=2 measured 34.48 / 35.09 / 35.12 across
three passes; run-to-run spread on identical configs is about 2%, so
this experiment does not separate K=1 from K=2. K=3 measured 31.98 on
the same pair and transport, and acceptance falls from 0.6193 at K=2
to 0.483 at K=3. I ship K=2 because it is the best of the values I
measured, not because K=1 was ruled out.

Repeat measurements (both landed 2026-09-16):

- **Second pair** — the same headline configuration on the other node
  pair read 34.47 tok/s, inside the roughly 5-7% pair-to-pair offset I
  see on repeated runs of identical configs.
- **Publication re-measurement** — an interleaved TCP/RDMA re-run of the
  headline config on a fresh, non-overlapping 64-prompt set with
  prefix caching disabled and warm-up counted separately: 34.48 and
  35.12 tok/s over RDMA, 23.75 and 24.35 tok/s over TCP. The RDMA
  re-runs sit within ~2% of the original 35.09.
- Per-pass GPU telemetry, sampled with `nvidia-smi` every 2 s on both
  nodes for the duration of each pass (523–772 samples per node). SM
  clock was 2190 MHz in every sample of all four passes. Power stayed
  in a 20–27 W band and GPU temperature reached at most 68 °C across
  the four passes; I did not keep a per-pass breakdown of where in
  that band each pass sat. These per-sample CSVs are not part of this
  repository; the four re-run JSONs under `results/logs/` are.

The headline run (pair 1, 2026-09-16) produced 32407 completion tokens
over 923.43 s of wall time with zero failed requests; the shipped log
records totals only, not per-request finish reasons, but I counted
them from the pre-sanitisation factory log and the headline run's own
split is 61 `length` / 3 `stop` over its 64 prompts. The four release
re-runs record the same field directly and land on the identical split
— i.e. most prompts ran into the 512-token cap rather than stopping on
their own. On the stock checkpoint the C=32 aggregate was
95.08 tok/s over sockets (pair 2, 64 prompts) and 109.03 tok/s over
RDMA (pair 2, 64 prompts); both of those passes ran with 32 sequence
slots, which the shipped `--max-num-seqs 20` cannot reproduce, so read
them as rig numbers rather than as what the released scripts do. No
C=32 row exists for route h yet.

### By prompt kind

The 64-prompt ruler is prose. To see how much the kind of text moves the
number I ran separate 32-prompt sets on the released configuration: pair
1, NCCL/IB (RDMA), route h + MTP K=2, `temperature=0`, expert parallel
on, each set measured C=1 as its own pass. These sets are not the
64-prompt ruler and not subsets of it — different prompts, 32 instead of
64 — so they stay out of the table above.

- prose, `max_tokens=512`: 34.87 tok/s, TPOT median 28.1 ms, TTFT
  median 0.319 s.
- prose, `max_tokens=128`: 32.80 tok/s, TPOT median 28.9 ms, TTFT
  median 0.293 s.
- code, `max_tokens=512`: 38.20 tok/s, TPOT median 25.6 ms, TTFT
  median 0.342 s.
- structured output and JSON: measured too; their readings are in the
  paragraph below, without TTFT medians.

Structured prompts ran at 34.84 tok/s at 512 tokens and 33.51 at 128 (TPOT median 27.4 and 27.6 ms); JSON-shaped prompts ran at 35.35 and 33.30 (25.0 and 25.2 ms); code at 128 tokens ran at 36.08 (25.1 ms). All four kinds sit between 32.8 and 38.2 tok/s on this configuration, with zero failed prompts in every pass.

The prose reading, 34.87 tok/s, sits on the 35.09 of the 64-prompt ruler
under the same serving configuration — the agreement I would want
between two different prose sets. An earlier pass over this same
32-prompt prose set, same configuration, read 28.88 tok/s: it ran inside
a slow window, with weighted TPOT at 34.0 ms against 27.9 ms on the
headline pass while acceptance stayed put (0.6189 against 0.6221), so
the difference is time per decode cycle and not the draft. Both readings
stay on record. The spread between them is what one 32-prompt pass can
do on this machine when something else is touching it, and it is wider
than the ~2% the 64-prompt ruler shows across re-runs.

## Long context, measured

The probe is `bench/longctx.py`: Japanese filler grown to a target
length, ten facts planted at head, middle and tail token depths, one
question each, `temperature=0`, graded by exact code match rather than
by a judge. The set is `bench/longctx-probe.jsonl`, and every document
is the shared prefix of its ten questions, so one long prefill is paid
per length. One pass on the shipped configuration —
`--max-model-len 204800`, `--max-num-seqs 20`,
`--gpu-memory-utilization 0.85`, `--enable-prefix-caching`, route h with
MTP at K=2 over RDMA:

| prompt tokens | needles found | first token, cold | first token, cache warm | prefill |
|---:|---|---:|---:|---:|
| 16,345 | 10/10 | 10.02 s | 4.27 s | 1630.6 tok/s |
| 65,545 | 10/10 | 39.95 s | 2.91 s | 1640.8 tok/s |
| 130,990 | 10/10 | 79.86 s | 3.15 s | 1640.2 tok/s |
| 194,544 | 10/10 | 120.04 s | 4.74 s | 1620.6 tok/s |

40 of 40, no depth weaker than another: head 12/12, middle 16/16, tail
12/12. One reading from a different configuration, kept out of the table
on purpose: at `--max-model-len 307200` with utilization at 0.88 — not
what the scripts ship — a single 204,767-token document answered 10/10.
It needs a window the released scripts do not open, so it is not a row
above.

This is retrieval, not quality. It says the model still finds a planted
fact at 194,544 tokens; it says nothing about whether its prose or its
reasoning hold up there.

Three limits come with the long window.

- **A prompt the size of the window does not fit.** The probe's top
  stage aims at 190k rather than 200k for a measured reason: a
  204,754-token document plus `max_tokens=64` is 204,818 against a
  204,800 limit, and the server returns `HTTP 400`. The engine survives
  — the next request is answered normally — but that document cannot be
  asked about. The usable ceiling is the window minus the generation
  budget minus the chat template.
- **The first token on a long prompt is slow.** Prefill holds between
  1620 and 1641 tok/s across all four lengths, so the 194,544-token
  prompt takes 120.04 s before the first token appears. Nothing is
  stuck; that is prefill. `--enable-prefix-caching` is in the shipped
  flags for this reason: pin one long document as the prefix and vary
  only the question, and every request after the first is an order of
  magnitude faster — at 130,990 tokens, 79.86 s cold against 3.15 s
  warm.
- **Past about 259,000 tokens nothing gets through on this hardware.**
  Not with a longer `--max-model-len`, not with a lower
  `--gpu-memory-utilization`, not with a smaller
  `--max-num-batched-tokens`: host memory runs out and the node's OOM
  reaper takes the vLLM worker. The `peak activation` figure vLLM prints
  at boot is measured on an 8192-token dummy run, and the scratch space
  the sparse-attention indexer wants during a long prefill sits outside
  the `--gpu-memory-utilization` budget entirely. Dropping utilization
  from 0.88 to 0.85 moved the death from 62 s into the prefill to 155 s;
  it did not make the prompt fit.

`RAY_memory_usage_threshold=0.99` is exported on both nodes by
`serve/start-head.sh` and `serve/start-worker.sh`, and it is not
optional. Unified memory means `--gpu-memory-utilization` is taken out
of host RAM, so Ray's own OOM monitor sees the node above its default
0.95 and kills the largest actor it can find, which is the vLLM TP0
worker. What you see is `EngineDeadError` — at boot, or in the middle of
a request — with nothing wrong in the vLLM log; the kill is in the
raylet log. The two scripts bring up two separate raylets, each with its
own monitor, so the variable has to be exported on both.

## What did not work, and when

Measured on the same ruler; listed so the search space is on record.

<!-- failures:start -->

| item | what | cause | date |
|---|---|---|---|
| **moe-dsl-kernel-overlay** | Three successive failures. First, the patched kernel path was dead code: the backend is gated to device family 100 and never instantiates on sm_121, so an apparent -4 ms TPOT delta was run-to-run noise. Second, after re-hooking the class the deployment selects, every call failed a static eligibility check (the model's SwiGLU clamp limit) and silently fell back to stock: 256 calls, zero dispatches. Third, a host-side cute.make_layout call raised a TypeError at first use. | Instrumented, then fixed or abandoned each time; the overlay is not in the released config. | 2026-09-15 |
| **fp8-dense** | FP8-quantized dense linear variants ran slower than W4A16 NVFP4 on this stack (pair 1, sockets, 64-prompt ruler): no draft 9.3 tok/s / TPOT 107.1 ms against W4A16's 18.98 / 51.5; with the MTP K=2 draft 16.49 tok/s / TPOT 59.7 ms against W4A16's 24.01 / 40.7. | Measured slower; not adopted. | 2026-09-15 |
| **mtp-k-ge-3** | K=3 gave 19.39 tok/s and K=4 gave 17.44 tok/s vs K=2's 19.83 on an earlier requant route (sockets, pair 1); on stock (pair 2, sockets, expert parallel on), K=4 gave 15.2 and K=5 gave 13.1. | Per-position acceptance drops more quickly than the extra draft tokens save steps. K=2 was the best of the K values measured on this route; on route h + RDMA a first sweep-A pass measured K=1 at 34.28 and K=3 at 31.98 tok/s; the reverse-order pass was run and failed to boot at every K including K=0 (only 99.88 GiB was free on the device at startup against the 104.65 GiB the utilization target asks for), which is a startup-memory limit rather than a K effect, so the full same-checkpoint sweep in both K orders has still not produced a complete set of numbers. | 2026-09-16 |
| **ep-off** | Removing --enable-expert-parallel gained ~0.7% (pair 2, sockets, MTP K=2, 64-prompt ruler: 17.71 -> 17.84 tok/s). | Superseded. That reading was pair 2 over sockets at a 16384 window. On RDMA at 204800 the same removal is worth 6.5% (35.05 -> 37.33 tok/s on the same script) and lowers the post-READY high-water mark from 6014 to 4691 MiB, so the scripts now ship expert parallel off. | 2026-09-15 |
| **eager-mode** | --enforce-eager speeds up the stock single-stream step by 7.0 ms on the 256-step attribution totals (86.95 vs 93.98 ms) and by 13.4 ms on the C=1 TPOT median (77.5 vs 90.9 ms), but the gain mostly does not carry into MTP decode. | The step it speeds up is not the step speculation runs. Not adopted. | 2026-09-15 |
| **k4-k5-boot-oom** | On the STOCK checkpoint at these memory settings (128 GB unified memory per node, max seq len 16384, gpu-memory-utilization 0.86) the engine refused to start at K=5 twice: after the speculator's CUDA-graph capture (103 s and 3.03 GiB on one rank) the KV cache left was smaller than one max-seq-len request needs (0.89 GiB available against 0.92 GiB needed). One K=4 attempt died at the same capture step. Restricting the CUDA-graph capture sizes to the two sizes the C=1 measurement needs made both boot (READY 879 s and 901 s) and produced the K=4 15.2 and K=5 13.1 rows above, so this is a capture-size configuration limit, not a hard K ceiling. On route h + RDMA, K=1, K=3 and K=4 all booted (READY 885 s, 906 s and 918 s); the K=4 arm was stopped by the operator before it measured, so route h has no K=4 number, and K=5 was never run on route h at all. | Speculator graph capture grows with K and is taken out of the KV budget: supported for stock K=5 by the capture size and the KV figures above, and worked around by capturing fewer sizes. Two unrelated events are kept out of this column: an earlier K=4 attempt was killed by an orphaned pipeline's cleanup, and a later reverse-order route-h sweep failed to boot at every K including K=0 because only 99.88 GiB was free on the device at startup against the 104.65 GiB the utilization target asks for. | 2026-09-15 |
| **requant-loader-write** | A fused-KDA write missing the NVFP4 global scale produced KeyError ...weight_scale_2 at load. | Writer bug; now covered by verify.py config and the kda-quant overlay. | 2026-09-15 |
| **nccl-ll-symm-ar** | NCCL low-latency envs and symmetric-memory all-reduce booted but returned empty response bodies or died under concurrency. | Broken output path; unsafe, not adopted. | 2026-09-15 |
| **ev-ordering** | The expensive drafter-training data chain (extraction, two generation passes, conversion; ~5 h of wall time on the 2-GPU pair, ~10 GPU-hours) was queued ahead of the cheap K sweep; the trained drafter's holdout top-1 came out ~0.08, and the sweep showed K=2 was already the best of the measured values. | Sequenced by pipeline momentum instead of expected information per GPU-hour. | 2026-09-15 |

<!-- failures:end -->

Seven more failures came from the clean-room run above, all in the
runbook and the driver rather than in the model, the transport or the
checkpoint — the kind that only shows up when the instructions are
followed literally from an empty directory. Five of them: a `mktemp`
template the runbook's own shell rejected (fewer than three trailing
`X`); a serve config the driver wrote but never sourced; files an
in-image helper created as root, into a directory the next step then
could not write; the case where the operator node and the serving head
are the same machine, which the script handled by trying to SSH to
itself; and shell quoting that ate the quotes around a JSON argument.
Each is fixed in the files published here, not in the run.

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
0.9217 -> 0.9211, SciCode 0.5621 -> 0.5769, MMMU Pro 0.7688 -> 0.7630,
AA-LCR 0.7100 -> 0.7106, IFBench 0.6130 -> 0.6054, Terminal-Bench 2.1
0.8258 -> 0.8315. That makes it a trustworthy base. My route h only
extends the same 4-bit treatment to the layers the official release left
in BF16, and I keep the gate numbers next to the speed numbers so the
trade-off stays visible. Those card figures come from standard public
benchmarks and are not comparable with the eval-200 totals above, which
are scored by this repository's own strict grader.

## What is left for a v2

This is a list of what I know is unfinished, not a roadmap. I am not
predicting how much any of it is worth; I will publish a number when I
have measured one.

**Speed, still open**

- My own NVFP4 MoE kernel. It has never actually run in a served
  request: three attempts hooked a class the GB10 build does not
  instantiate, then the wrong class, then a host-side call that threw
  and silently disabled the overlay. Until it runs with a proof-of-life
  line in the engine log, there is nothing to measure.
- All-reduce. Cross-node peer wait was 20.8 ms of an ~87 ms decode step
  in the one step attribution I ran (eager, stock, single stream, over
  sockets). I have not repeated that attribution under RDMA — an
  attempt to do so died at boot and produced no step data — so I do
  not know how much of that wait carries over onto the fast transport;
  the higher tok/s the RDMA rows show is a whole-pass measurement, not
  an attribution to this bucket. I have not tried to reduce the number
  of collectives itself either.
- A trained draft model. The checkpoint's own MTP head accepts 0.62 of
  the tokens it proposes. A draft trained on this model's own outputs
  could accept more, which shortens every cycle. My first attempt is
  data-starved — 594 training records, about 301k tokens against a
  ~3M-token recipe target, holdout per-slot hit 0.03-0.08 — and needs
  roughly ten times the data before it is worth measuring.
- Speculation depth. K=1 (34.28) and K=2 (35.09) are not separated by
  this experiment's run-to-run spread, and K=3 is slower. A
  better draft would change where that optimum sits.

**Quality, not yet measured**

- English and other languages. Every probe here is Japanese.
- Code correctness. Code prompts appear in the speed table and nowhere
  in the quality table.
- Multi-turn conversations, tool calling, instruction following,
  long-context quality — the needle probe reaches 194,544 tokens at
  40/40, but that is retrieval and nothing in the quality table ran on a
  prompt longer than about 2k — safety behaviour, and the model with
  thinking enabled, which is how this family is normally used.
- Perplexity on a larger corpus, with a confidence interval. The current
  probe is eight sentences and the code does not keep the per-token
  values, so the ratio has no interval attached to it.
- A paired test on eval-200. The grader discards per-item results, so
  the same 150 items cannot be tested as pairs, which is what they are.

**Reproduction, not yet complete**

- The clean-room run skipped the download and the requant themselves:
  the weights were already on the node. The full path, from an empty
  disk through the ~204 GB download and the ~30-minute requant, has not
  been run end to end by anyone.
- That run used 10 prompts, not the 64 the reference numbers come from.

**What I already know does not work.** FP8 for the dense linears, K of 3
or more on prose, and forcing eager mode. The numbers and the conditions
they were measured under are in the failure table above; that is the
place to look before retrying any of them.

## The part that does not end

Making the model fast took a day of experiments. Making the result
publishable took longer, and it is the part I was not ready for.

Every layer of checking I added found something, and each finding was
real. A machine check on wording and leaks passed on the first run, so I
added a check that traced every number in the text back to a log —
fourteen of them could not be traced. I fixed those, then had a reader
with no context read the whole thing, and the headline comparison turned
out to divide two numbers measured on different hardware, over a
different transport, on a different number of prompts. I fixed that,
then checked the shipped evidence against the tables and found a
throughput figure that was the wrong row of its own log. I fixed that,
then audited the quantization script against what the text claimed, and
found a helper that would silently build a broken draft if you pointed
it at the wrong checkpoint. I fixed that, then had the quality claims
read adversarially, and learned that the grader for one of the four
categories rewards a model for hedging — which is the direction my
numbers had moved.

None of those were careless. Each one needed a different kind of
looking. And the pattern was always the same: add a check, find
something, fix it, and the fix is small. The checking is not what costs;
the discovery that there is always one more layer is what costs.

At some point the honest move is to stop, not because the work is
finished but because the next layer is worth less than shipping. I
stopped here. The things I know are not measured are listed above, in
their own section, and the things I have not thought to check are the
reason this is v1 and not final. If you find one, that is the system
working, and I would rather hear it than not.

## About the name

Wabi (侘び) is the Japanese sense of accepting what is imperfect or
plain and finding richness in it; this release is a work in progress
that I publish as it is, improvements included when they are measured.

## What comes next

The rest of the per-kind sets, then the full K sweep in both K orders;
if a configuration beats 35.09 tok/s on this ruler and passes the same
gate, it goes out as v2.

## Files

```
AGENTS.md                   reproduction runbook (prereqs, commands, times)
CONTRIBUTORS.md             who worked on this
LICENSE / NOTICE            Apache-2.0 and attribution
setup.env.example           per-node paths and image tags (copy to setup.env)
requant/requant.py          weight-only NVFP4 quantization, targets a-e/g/h
requant/verify.py           config verifier + gate v2 (capture/check)
requant/build-mtp-draft.py  BF16 MTP draft dir from the STOCK checkpoint
overlays/                   image-source patchers (fail-closed anchors):
  patch-kda.py              restore quant_config in fused KDA members
  patch-mla.py              pass quant_config into Glm5NextMLAAttention
  patch-mtp.py              build the MTP draft subtree unquantized
  apply-step-attr-patch.py + step_attr.py   CUDA-event step attribution
  fetch-image-file.sh       snapshot a file out of the image
  build-overlays.sh         fetch + patch everything into _build/
docker/Dockerfile.nccl-ib   derived image: questing rdma-core on noble
serve/                      start-head.sh / start-worker.sh / nccl-ib.sh
                            + check-headroom.sh + serve.env.example
scripts/agent-run.sh        one-shot driver for the whole runbook
scripts/verify-result.py    checks a measured row against the expected one
bench/                      measure.py + prompts-64.jsonl + eval-200.jsonl
                            + README.md (what the sets are)
results/                    results.tsv, quality.tsv, failures.tsv and the
                            sanitised per-run logs the tables cite
tests/                      run-tests.sh (offline smoke) +
                            check-md-invariants.py (wording-only edit check)
```

Two `eval-200.jsonl` prompts were re-quoted for publication (ASCII-safe
quoting / reworded instruction); expected answers and grading are
unchanged. `bench/prompts-64.jsonl` ships exactly as measured — see
[bench/README.md](bench/README.md).

## Who did what

The design decisions, the acceptance criteria and every measurement in
this repository are mine. The implementation was carried out with the
AI seats listed below and in [CONTRIBUTORS.md](CONTRIBUTORS.md); the
gate, the runs and the numbers were checked by me before release.

## Contributors

- tenhkspark (the maintainer) — the one who worried, watched, and
  said go.
- Claude Fable 5.1 (Anthropic) — direction, experiment design,
  review and acceptance.
- Claude Opus 5 (Anthropic) — pre-release review.
- Astra (OpenAI) — adversarial review of the measurements and the
  plan.
- Devin SWE-2 (Cognition) — implementation, diagnostics, running the
  experiment queue, repository drafting.
- GLM-5.3 (Z.ai) — implementation seat.
- GLM-5.3-Flash (Z.ai) — implementation seat and summarisation.

## License

The code in this repository is Apache-2.0, see LICENSE. The weights are
a separate matter: they are not part of this repository, and the derived
checkpoint published on Hugging Face carries the upstream MIT license it
inherits from `zai-org/GLM-5.3-Flash` rather than Apache-2.0. That
Hugging Face repository ships the MIT text verbatim alongside a NOTICE
that names the upstream model, the NVIDIA Model Optimizer base
quantization and the contributors. To build the weights yourself instead,
requantize the NVIDIA checkpoint with `requant/requant.py`.
