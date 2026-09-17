#!/usr/bin/env python3
"""Compare a ruler result against the README reference table.

Reads a results/measure-<label>.json written by bench/measure.py and
checks the C=1 row against the matching README "Speed results" row:

  --result FILE   path to measure-<label>.json
  --ref NAME      h-rdma-mtp | h-sockets-mtp | h-nospec (default h-rdma-mtp)
  --tol FRAC      pass band: |measured-ref|/ref <= tol (default 0.07)
  --prompts N     prompts the C=1 level must have completed (default 64)

Tolerance rationale: two identical node pairs measured 7% apart on the
same ruler (pair-to-pair offset); the same pair re-measured is ~4%.
A reproduction on different hardware is compared at the 7% band.

Gated metrics: C=1 aggregate tok/s, C=1 median TPOT, and coverage —
zero failed prompts and all 64 of the frozen set completed. Without the
coverage gate a run where most prompts died still lands in the speed
band, since tok/s is computed over the prompts that finished. TTFT and
MTP acceptance are printed for context but not gated.

Exit 0 = VERIFY PASS, 1 = VERIFY FAIL, 2 = bad input.
"""
import argparse
import json
import sys

# Mirrors the README "Speed results" table (route h rows, C=1).
REF = {
    # default path: NCCL over RDMA (NCCL_IB=1), route h, official MTP K=2
    "h-rdma-mtp":    {"tok_s": 35.09, "tpot_ms": 27.9,
                      "ttft_s": 0.320, "accept": 0.6221},
    # same checkpoint and K, but NCCL over TCP sockets (NCCL_IB=0)
    "h-sockets-mtp": {"tok_s": 24.01, "tpot_ms": 40.7,
                      "ttft_s": 0.389, "accept": 0.6221},
    "h-nospec":      {"tok_s": 18.98, "tpot_ms": 51.5,
                      "ttft_s": 0.330, "accept": None},
}


def line(name, measured, ref, tol, gated=True):
    if measured is None:
        print(f"VERIFY {name}: missing in result")
        return False if gated else True
    if ref is None:
        print(f"VERIFY {name}: measured={measured} (no reference)")
        return True
    delta = (measured - ref) / ref
    ok = abs(delta) <= tol
    tag = "ok" if ok else ("OUT" if gated else "info")
    gate = "" if gated else " (report only)"
    print(f"VERIFY {name}: measured={measured} ref={ref} "
          f"delta={delta * 100:+.1f}% tol={tol * 100:.0f}% {tag}{gate}")
    return ok if gated else True


def coverage(c1, want):
    """Gate the C=1 level on a full, failure-free pass.

    measure.py records "prompts" (how many of the frozen set the level
    ran), "fails" (prompts that raised), and for C=1 a per_prompt list
    whose failed entries carry "error" instead of timings.
    """
    n, fails = c1.get("prompts"), c1.get("fails")
    pp = c1.get("per_prompt")
    if pp is not None:
        done = len([r for r in pp if "error" not in r])
    elif isinstance(n, int) and isinstance(fails, int):
        done = n - fails
    else:
        done = None
    ok = True
    if fails is None:
        print("VERIFY c1_fails: missing in result OUT")
        ok = False
    elif fails != 0:
        print(f"VERIFY c1_fails: {fails} prompt(s) failed, want 0 OUT")
        ok = False
    else:
        print("VERIFY c1_fails: 0 ok")
    if n != want or done != want:
        print(f"VERIFY c1_coverage: prompts={n} completed={done} "
              f"want={want} OUT (partial runs still land in the band)")
        ok = False
    else:
        print(f"VERIFY c1_coverage: {done}/{want} prompts completed ok")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--ref", choices=sorted(REF), default="h-rdma-mtp")
    ap.add_argument("--tol", type=float, default=0.07)
    ap.add_argument("--prompts", type=int, default=64,
                    help="prompts the C=1 level must have completed")
    args = ap.parse_args()

    try:
        data = json.load(open(args.result))
    except OSError as e:
        print(f"VERIFY FAIL: cannot read {args.result}: {e}")
        return 2
    c1 = next((lv for lv in data.get("levels", []) if lv.get("c") == 1), None)
    if c1 is None:
        print("VERIFY FAIL: result has no C=1 level")
        return 2

    ref = REF[args.ref]
    print(f"VERIFY ref={args.ref} tol={args.tol} "
          f"(pair-to-pair ~7%, run-to-run ~4%; see README)")
    ok = True
    ok &= coverage(c1, args.prompts)
    ok &= line("c1_tok_s", c1.get("agg_tok_s"), ref["tok_s"], args.tol)
    ok &= line("c1_tpot_ms", c1.get("tpot_median_ms"), ref["tpot_ms"],
               args.tol)
    ok &= line("c1_ttft_s", c1.get("ttft_median_s"), ref["ttft_s"],
               args.tol, gated=False)
    acc = (data.get("acceptance") or {}).get("acceptance")
    ok &= line("acceptance", acc, ref["accept"], args.tol, gated=False)
    print("VERIFY PASS" if ok else "VERIFY FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
