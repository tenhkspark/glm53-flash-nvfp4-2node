#!/usr/bin/env python3
"""Fixed-ruler throughput / spec-decode-acceptance measurement for an
OpenAI-compatible server.

The ruler: a frozen prompt set (JSONL, one {"prompt": "..."} per line),
max_tokens 512, temperature 0, thinking skipped via an empty assistant
continuation (continue_final_message). Levels are (client concurrency,
number of prompts from the set): C=1 uses the first 8 prompts with
streaming (TTFT/TPOT per request), C=4 the first 16, C=16 the first 32,
C=32 all 64. A semaphore caps in-flight requests at c, so actual
concurrency is c even though n prompts run.

Aggregate tok/s = sum(completion_tokens) / wall time. Acceptance comes
from the server's /metrics spec-decode counters read before and after
the whole pass.

Usage: python3 measure.py --url http://127.0.0.1:8888 --label control \
         --prompts prompts-64.jsonl --outdir results
"""
import argparse
import json
import re
import threading
import time
import urllib.request

SEMA = None

# (client concurrency, number of prompts from the frozen set)
LEVELS = [(1, 8), (4, 16), (16, 32), (32, 64)]


def cap(c):
    global SEMA
    SEMA = threading.Semaphore(c)


def post(url, payload, timeout=900):
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def payload_for(model, prompt, max_tokens):
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ""},
        ],
        "continue_final_message": True,
        "add_generation_prompt": False,
        "max_tokens": max_tokens,
        "temperature": 0,
    }


def get_metrics(url):
    with urllib.request.urlopen(url + "/metrics", timeout=30) as r:
        return r.read().decode()


def spec_counters(text):
    """Sum vLLM counters, including their Prometheus labels and _total suffix.

    vLLM creates Counter names without _total; prometheus_client exports
    them with that suffix and labels.
    """
    out = {}
    pat = re.compile(
        r"(vllm:spec_decode_(?:num_drafts|num_draft_tokens|num_accepted_tokens))"
        r"(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+]+)")
    for line in text.splitlines():
        m = pat.match(line.strip())
        if m:
            name = m.group(1) + "_total"
            out[name] = out.get(name, 0.0) + float(m.group(2))
    return out


def run_stream_one(url, model, prompt, max_tokens):
    """Single streaming request -> (completion_tokens, ttft_s, tpot_s, wall_s,
    finish_reason)."""
    body = payload_for(model, prompt, max_tokens)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    tokens = 0
    finish = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                tokens = int(chunk["usage"].get("completion_tokens", tokens))
            for ch in chunk.get("choices", []):
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                delta = ch.get("delta", {})
                if delta.get("content") or delta.get("reasoning"):
                    if ttft is None:
                        ttft = time.time() - t0
                    tokens += 1
    wall = time.time() - t0
    # usage-based count wins when present; TPOT excludes the first token
    tpot = (wall - ttft) / (tokens - 1) if ttft and tokens > 1 else None
    return tokens, ttft, tpot, wall, finish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--model", default="GLM-5.3-Flash-NVFP4")
    ap.add_argument("--label", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--levels", default="")
    ap.add_argument("--stream-all-levels", action="store_true",
                    help="collect TTFT/TPOT per request at every concurrency")
    args = ap.parse_args()

    levels = LEVELS
    if args.levels:
        levels = [tuple(int(x) for x in lv.split(":")) for lv in args.levels.split(",")]

    prompts = []
    with open(args.prompts, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["prompt"])

    metrics0 = spec_counters(get_metrics(args.url))
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(f"MEASURE label={args.label} start={started} levels={levels} "
          f"max_tokens={args.max_tokens}", flush=True)

    results = []
    for c, n in levels:
        use = prompts[:n]
        if c == 1:
            toks, ttfts, tpots = [], [], []
            per_prompt = []
            finish_counts = {}
            fails = 0
            for p in use:
                try:
                    tk, tf, tp, wl, fr = run_stream_one(
                        args.url, args.model, p, args.max_tokens)
                except Exception as e:  # noqa: BLE001
                    fails += 1
                    per_prompt.append({"error": repr(e)})
                    print(f"  C=1 prompt FAIL {e!r}", flush=True)
                    continue
                toks.append(tk)
                if tf is not None:
                    ttfts.append(tf)
                if tp is not None:
                    tpots.append(tp)
                finish_counts[fr] = finish_counts.get(fr, 0) + 1
                per_prompt.append({"tokens": tk,
                                   "ttft_s": round(tf, 3) if tf is not None else None,
                                   "tpot_ms": round(tp * 1000, 1) if tp is not None else None,
                                   "wall_s": round(wl, 2),
                                   "finish_reason": fr})
                print(f"  C=1 prompt done tokens={tk} ttft={tf:.2f}s tpot="
                      f"{tp*1000 if tp else 0:.1f}ms wall={wl:.2f}s "
                      f"finish={fr}", flush=True)
            wall = sum(pp["wall_s"] for pp in per_prompt if "wall_s" in pp)
            total = sum(toks)
            rec = {
                "c": c, "prompts": n, "total_completion_tokens": total,
                "wall_s": round(wall, 2), "agg_tok_s": round(total / wall, 2) if wall else None,
                "ttft_median_s": round(sorted(ttfts)[len(ttfts) // 2], 3) if ttfts else None,
                "tpot_median_ms": round(sorted(tpots)[len(tpots) // 2] * 1000, 1) if tpots else None,
                "tpot_mean_ms": round(sum(tpots) / len(tpots) * 1000, 1) if tpots else None,
                "fails": fails, "finish_reasons": finish_counts,
                "per_prompt": per_prompt,
            }
        else:
            results_c = [None] * n
            errs = []

            timing_c = [None] * n

            def worker(i, p):
                with SEMA:
                    try:
                        if args.stream_all_levels:
                            results_c[i], tf, tp, _, _ = run_stream_one(
                                args.url, args.model, p, args.max_tokens)
                            timing_c[i] = (tf, tp)
                        else:
                            with post(args.url, payload_for(args.model, p, args.max_tokens)) as r:
                                u = json.loads(r.read()).get("usage", {})
                                results_c[i] = int(u.get("completion_tokens", 0))
                    except Exception as e:  # noqa: BLE001
                        errs.append(repr(e))

            t0 = time.time()
            cap(c)
            threads = [threading.Thread(target=worker, args=(i, p)) for i, p in enumerate(use)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            wall = time.time() - t0
            total = sum(x for x in results_c if x)
            ttfts = [x[0] for x in timing_c if x and x[0] is not None]
            tpots = [x[1] for x in timing_c if x and x[1] is not None]
            rec = {
                "c": c, "prompts": n, "total_completion_tokens": total,
                "wall_s": round(wall, 2), "agg_tok_s": round(total / wall, 2) if wall else None,
                "ttft_median_s": round(sorted(ttfts)[len(ttfts) // 2], 3) if ttfts else None,
                "tpot_median_ms": round(sorted(tpots)[len(tpots) // 2] * 1000, 1) if tpots else None,
                "fails": len(errs), "errors": errs,
            }
        results.append(rec)
        print(f"LEVEL label={args.label} C={rec['c']} prompts={rec['prompts']} "
              f"tok/s={rec['agg_tok_s']} tokens={rec['total_completion_tokens']} "
              f"wall={rec['wall_s']}s fails={rec['fails']}"
              + (f" tpot_med_ms={rec['tpot_median_ms']} ttft_med_s={rec['ttft_median_s']}"
                 if rec.get("tpot_median_ms") is not None else ""), flush=True)

    metrics1 = spec_counters(get_metrics(args.url))
    d_acc = metrics1.get("vllm:spec_decode_num_accepted_tokens_total", 0) - \
        metrics0.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    d_dra_tok = metrics1.get("vllm:spec_decode_num_draft_tokens_total", 0) - \
        metrics0.get("vllm:spec_decode_num_draft_tokens_total", 0)
    d_dra = metrics1.get("vllm:spec_decode_num_drafts_total", 0) - \
        metrics0.get("vllm:spec_decode_num_drafts_total", 0)
    accept = round(d_acc / d_dra_tok, 4) if d_dra_tok else None
    mean_len = round(1 + d_acc / d_dra, 4) if d_dra else None
    print(f"ACCEPT label={args.label} d_accepted={d_acc} d_draft_tokens={d_dra_tok} "
          f"d_drafts={d_dra} acceptance={accept} mean_accepted_len={mean_len}", flush=True)

    out = {
        "label": args.label, "url": args.url, "started": started,
        "ended": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "max_tokens": args.max_tokens, "levels": results,
        "acceptance": {
            "d_accepted_tokens": d_acc, "d_draft_tokens": d_dra_tok,
            "d_drafts": d_dra, "acceptance": accept, "mean_accepted_len": mean_len,
        },
        "spec_counters_before": metrics0, "spec_counters_after": metrics1,
    }
    path = f"{args.outdir}/measure-{args.label}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"WROTE {path}", flush=True)


if __name__ == "__main__":
    main()
