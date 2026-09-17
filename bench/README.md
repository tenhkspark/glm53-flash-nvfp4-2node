# bench/ — the fixed sets

All three sets are frozen inputs. They are published exactly as they
were when the numbers in the top-level [README](../README.md) were
measured: nothing in them was reworded, reordered or cleaned up
afterwards, so any awkward phrasing inside an individual prompt is part
of the measured input and stays.

## prompts-64.jsonl — the speed ruler

64 Japanese prose prompts, one JSON object per line with a `prompt`
field and a `seed_id`. Every speed row in the top-level README ran this
file at `temperature=0`, `max_tokens=512`, thinking skipped via an
empty assistant continuation (`continue_final_message` with an empty
assistant message). Rows marked † used only the first 8 lines; every
other row used all 64.

The set is fixed so that two runs are comparable at all. Changing a
prompt changes the token counts and the acceptance statistics, which is
why the file is shipped as measured rather than tidied.

Two lines read oddly on close inspection for the same reason: line 23
kept a leftover artifact from an earlier text-substitution pass, and
line 38 uses a word from the author's own personal-workflow vocabulary
rather than a general term. Both stayed as measured.

```bash
# the C=1, 64-prompt pass every single-stream row in the README used
python3 bench/measure.py --url http://127.0.0.1:8000 --label myrun \
    --prompts bench/prompts-64.jsonl --levels 1:64 --outdir results
```

Without `--levels` the script walks its default ladder — C=1 over the
first 8 prompts, C=4 over 16, C=16 over 32, C=32 over all 64 — which is
where the concurrency figures in the README come from.

## eval-200.jsonl — the quality set

200 Japanese single-turn items, 50 each of `reason`, `trap`, `tool` and
`longread`, each with the expected answer and the rubric used to grade
it. `requant/verify.py` answers every item greedily on the served
endpoint and grades it deterministically; the top-level README explains
the grading and which 150 items the gate actually scores.

Two prompts in this file were re-quoted for publication (ASCII-safe
quoting, one reworded instruction); their expected answers and grading
are unchanged.

## longctx-probe.jsonl — the long-context probe

40 Japanese needle-in-a-haystack questions over 4 documents: one
document per prompt length (16k, 64k, 128k, 190k tokens) with ten
needles buried in it, at token depths 6/10/14% (head), 44/48/52/56%
(middle) and 86/90/94% (tail). The middle band is the point of the set
-- losing the middle of a long context is the typical failure mode, and
the short single-turn items in `eval-200.jsonl` cannot see it. Three or
four points per band means a band is not one lucky position.

**The top stage is 190k, not 200k, because a 204800 window cannot serve
a 200k prompt.** The set used to end at a document of 204,754 tokens;
adding `max_tokens=64` makes the request 204,818 against the 204,800
limit and the server answers `HTTP 400` (measured 2026-09-17 — the
engine survives, the request does not). 190k leaves about 10k of
headroom for the answer and the chat template.

**The document is the shared prefix of its ten prompts.** Only the
trailing question changes, so the server's prefix cache answers
questions 2..10 without re-reading the document and the long prefill is
paid once per length rather than once per question. That is what makes a
four-length pass a matter of minutes; it also splits TTFT into two
numbers that mean different things, and `run` records both per length:

- `ttft_first_s` — question 1, a cold prefill, roughly proportional to
  `prompt_tokens` (`prefill_tok_s` is the implied prefill rate)
- `ttft_cached_median_s` — median of questions 2..10, which is what a
  user of a fixed long document actually waits

If the two are the same, the endpoint is not serving the prefix from
cache and the pass will take about ten times as long; check that before
blaming the model.

The file is a recipe rather than a text dump: each line records the
number of filler lines, each needle's line index, id and expected
answer, and `bench/longctx.py` rebuilds the exact prompt from
`prompts-64.jsonl` and `eval-200.jsonl` (trap items excluded). The
filler is therefore this repository's own frozen text, cycled, one
numbered record per line, and the set costs 8 KB on disk instead of the
~1.4 MB the four documents would take.

Token counts are measured with the checkpoint's own `tokenizer.json`,
not estimated from character counts; every prompt lands within 0.4% of
its target (`prompt_tokens` on each needle is the measured value, and
`meta.tokenizer` names what measured it).

Grading is mechanical. The needle is
`観測点 <id> の基準値は <code> である。`, the code is four digits, a
hyphen and two letters, and each of the ten codes occurs exactly once in
the whole prompt; an answer passes iff the code appears in it after NFKC
folding plus whitespace and dash normalisation. No LLM judge. `exact` is
also recorded (the answer was the bare code) but does not decide the
score.

```bash
# offline: rebuild every prompt, assert each needle and its code occur
# exactly once, print the length/depth table -- no server needed
python3 bench/longctx.py check

# does the endpoint really tokenise the way the set was built? one
# /tokenize call, no prefill
python3 bench/longctx.py calibrate --url http://127.0.0.1:8000

# the same question exactly, four /tokenize calls: re-measure the
# recorded token counts against the served checkpoint
python3 bench/longctx.py check --url http://127.0.0.1:8000

# the measurement: greedy, temperature 0, serial and in order (the
# prefix cache only helps if a document's ten questions follow each
# other)
python3 bench/longctx.py run --url http://127.0.0.1:8000 --label myrun \
    --outdir results

# a smoke subset first: it measures the endpoint's real prefill rate and
# proves the prefix cache is working, in a couple of minutes
python3 bench/longctx.py run --url http://127.0.0.1:8000 --label smoke \
    --lengths 16k,64k --outdir results
```

A full pass sends 407,372 prompt tokens cold plus 36 cache hits. Almost
all of the wall time is the four cold prefills, so the run is as long as
the endpoint's prefill throughput makes it; `--lengths` cuts the set
down when that is too much. Rebuilding the set (`longctx.py build
--tokenizer ...`) needs the `tokenizers` package and the checkpoint's
`tokenizer.json`; running, checking and calibrating need neither.
