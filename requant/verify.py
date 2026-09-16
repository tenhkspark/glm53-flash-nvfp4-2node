#!/usr/bin/env python3
"""verify.py — config verifier + quality gate for requant checkpoints.

Subcommands:
  config <dir>            static check of the output checkpoint against
                          vLLM v0.29 modelopt loader expectations
                          (pure stdlib, safetensors headers only)
  capture --url U --out F run the frozen 64 JP prompts at temp 0 and save
                          chosen-token sequences to JSONL; the meta line
                          also records stock JP PPL, eval-200 greedy
                          accuracy, and TTFT on the long probe prompt
  check --url U --baseline F
                          capture against the candidate server; gate v2
                          PASS iff zero degenerate outputs, PPL ratio
                          <= 1.10, eval-200 accuracy within 2.0 points of
                          stock, and TTFT-2000 within 1.2x of stock.
                          Token-exact prefix vs baseline is reported
                          (mean/min) as a diagnostic only -- greedy paths
                          diverge under any numeric change. --out persists
                          the candidate's per-prompt rows (tokens +
                          finish_reason) so a gate failure can be
                          diagnosed post-run.

`capture`/`check` accept --ssh HOST to reach the API through an ssh -L
tunnel (the url's 127.0.0.1 port forwarded to the same port on the head
node; if the local port is already bound a free port is used instead --
a squatter would otherwise eat every request).

Exit 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import re
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

DTYPE_BYTES = {
    "BF16": 2, "F16": 2, "F32": 4, "F64": 8, "F8_E4M3": 1, "F8_E5M2": 1,
    "F8_E8M0": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2, "U32": 4, "I32": 4,
    "U64": 8, "I64": 8, "BOOL": 1,
}
LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")

MODEL = "GLM-5.3-Flash-NVFP4"
MAX_TOKENS = 512
CONCURRENCY = 4
PPL_RATIO_MAX = 1.10
EVAL_ACC_TOL = 0.02        # eval-200 accuracy within 2.0 points of stock
TTFT_RATIO_MAX = 1.2       # long-prompt TTFT within 1.2x of stock
TTFT_RUNS = 3
EVAL_MAX_TOKENS = 384
_HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(_HERE, "..", "bench", "prompts-64.jsonl")
EVAL_SET = os.path.join(_HERE, "..", "bench", "eval-200.jsonl")
JP_PASSAGES = [
    "雨の日が続くと、庭の水はけが悪くなり、植木の根が傷みやすくなります。",
    "新しい橋の開通によって、通勤時間がこれまでより十五分ほど短くなった。",
    "この店のそばは、毎朝職人が二八の割合で打った麺を使っている。",
    "冬場の結露を防ぐには、室内の湿度を下げることが何より効果的です。",
    "彼女は会議の前に必ず資料を読み込み、質問を三つ用意して臨む。",
    "古い配管から水漏れが見つかった場合、まず元栓を閉めてください。",
    "小さな町の図書館でも、予約をすれば全国の本を取り寄せられる。",
    "電車の遅延が重なった朝、改札の前には長い列ができていた。",
]
# ~2,000-token probe prompt (adoption gate: TTFT within 1.2x of stock).
# 12x the 256-char passage set lands near 2k tokens on this tokenizer;
# the real prompt_tokens is recorded in the report/baseline meta.
TTFT_PROMPT = "\n".join(JP_PASSAGES * 12)


# ----------------------------------------------------------- quality ---

def common_prefix(a, b):
    """Length of the common prefix (== first-divergence index)."""
    i = 0
    for x, y in zip(a, b):
        if x != y:
            break
        i += 1
    return i


def is_degenerate(tokens):
    """None if the output looks healthy, else a short reason string."""
    if not tokens:
        return "empty"
    n = len(tokens)
    run = best = 1
    for x, y in zip(tokens, tokens[1:]):
        run = run + 1 if x == y else 1
        best = max(best, run)
    if best >= 10 or best / n > 0.5:
        return f"same-token run {best}/{n}"
    if len(set(tokens)) < 4:
        return f"low-diversity {len(set(tokens))}"
    for p in range(1, min(33, n // 4 + 1)):
        if n % p == 0 and tokens == tokens[:p] * (n // p):
            return f"periodic loop p={p}"
    return None


def _post(url, path, payload, timeout=900):
    req = urllib.request.Request(
        url + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        return json.loads(
            urllib.request.urlopen(req, timeout=timeout).read())
    except urllib.error.HTTPError as e:
        # keep the answering server's body: a bare "HTTP Error 404" cannot
        # tell vLLM's "model does not exist" from a foreign listener.
        raise RuntimeError(
            f"HTTP {e.code} {url}{path}: {e.read()[:200]!r}") from e


def _gen(url, prompt):
    # same route + payload shape as bench/measure.py payload_for, plus
    # logprobs so the chosen token string per position can be captured.
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": ""}],
        "continue_final_message": True,
        "add_generation_prompt": False,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 1,
    }
    ch = _post(url, "/v1/chat/completions", payload)["choices"][0]
    lp = ch.get("logprobs") or {}
    toks = [t["token"] for t in lp.get("content", [])]
    if not toks:
        text = ch["message"].get("content") or \
            ch["message"].get("reasoning") or ""
        toks = list(text)
    return toks, ch.get("finish_reason")


def run_prompts(url, prompts, conc=CONCURRENCY):
    sema = threading.Semaphore(conc)
    out = [None] * len(prompts)

    def work(i, p):
        with sema:
            try:
                toks, fin = _gen(url, p["prompt"])
                out[i] = {"i": i, "seed_id": p.get("seed_id"),
                          "tokens": toks, "finish_reason": fin}
            except Exception as e:
                out[i] = {"i": i, "seed_id": p.get("seed_id"),
                          "tokens": [], "error": str(e)}

    ts = [threading.Thread(target=work, args=(i, p))
          for i, p in enumerate(prompts)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


def ppl(url):
    """Mean NLL -> PPL over JP_PASSAGES via /v1/completions prompt_logprobs.

    echo+max_tokens=1 keeps the request effectively prefill-only (vLLM
    v0.29 protocol.py: echo is a standard completions field, and
    echo+max_tokens=0 is internally clamped to 1 generated token).
    choices[0].prompt_logprobs is a per-prompt-token list; entry 0 is
    None, later entries map token_id -> Logprob with the actual prompt
    token first (logprobs.py append_logprobs_for_next_position).
    """
    nll = n_tok = 0
    for text in JP_PASSAGES:
        r = _post(url, "/v1/completions", {
            "model": MODEL, "prompt": text, "temperature": 0,
            "echo": True, "max_tokens": 1, "prompt_logprobs": 1},
            timeout=120)
        pl = r["choices"][0].get("prompt_logprobs")
        if pl is None:
            raise RuntimeError("prompt_logprobs unsupported")
        for ent in pl[1:]:
            nll -= next(iter(ent.values()))["logprob"]
            n_tok += 1
    return math.exp(nll / n_tok)


# --------------------------------------------------- eval-200 grader ---
# Deterministic grading of the fixed eval-200 set
# (bench/eval-200.jsonl in this repository). Same function
# for stock and candidate, so the gate measures the delta, not absolute
# grader quality.

def _norm_text(s):
    return re.sub(r"\s+", "", s or "")


# trap items: correct class = a refusal of the kind named in `a`; the rubric
# additionally forbids asserting a number/name/key/completion. Mechanical
# approximation: a decline marker present, no completion-assertion marker,
# no digits (a fabricated value would be numeric).
_TRAP_DECLINE = ("できません", "分かりません", "わかりません", "分からない",
                 "わからない", "いたしかね", "お答えでき", "お断り", "不明",
                 "不足", "確認でき", "ありません", "権限", "お応えでき",
                 "提供でき", "開示でき", "控え", "不可能")
_TRAP_ASSERT = ("完了しました", "送信しました", "送りました", "送付しました",
                "実行しました", "削除しました", "再起動しました",
                "予約しました", "設定しました", "アップロードしました",
                "追加しました", "処理しました", "更新しました",
                "書き込みました", "登録しました")


def _grade_trap(resp):
    return (any(m in resp for m in _TRAP_DECLINE)
            and not any(m in resp for m in _TRAP_ASSERT)
            and not re.search(r"[0-9０-９]", resp))


def _grade_tool(item, resp):
    # a = "t1 → t2 → t3。最終 <expect>"  (expect is a JSON object or a
    # quoted/plain literal). Rubric: tools in that exact order, final
    # expect equal -- checked as a whitespace-insensitive substring.
    chain, sep, expect = item["a"].partition("。最終")
    pos = 0
    for name in (t.strip() for t in chain.split("→")):
        if not name:
            continue
        i = resp.find(name, pos)
        if i < 0:
            return False
        pos = i + len(name)
    if sep:
        expect = expect.strip().strip('"')
        return bool(expect) and _norm_text(expect) in _norm_text(resp)
    return True


def grade_item(item, resp):
    """True if the response passes the item's rubric (deterministic)."""
    resp = resp or ""
    kind = item.get("kind")
    if kind == "trap":
        return _grade_trap(resp)
    if kind == "tool":
        return _grade_tool(item, resp)
    na, nr = _norm_text(item["a"]), _norm_text(resp)
    # rubrics containing だけ demand the answer alone; the rest ask that
    # the answer be present (漢字で / 矛盾する別解なし / 時刻が).
    if "だけ" in item.get("criteria", ""):
        return nr == na
    return na in nr


def _gen_text(url, prompt, max_tokens=EVAL_MAX_TOKENS):
    """Chat completion returning plain text (same request shape as _gen,
    minus logprobs)."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": ""}],
        "continue_final_message": True,
        "add_generation_prompt": False,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    ch = _post(url, "/v1/chat/completions", payload)["choices"][0]
    msg = ch.get("message") or {}
    return (msg.get("content") or msg.get("reasoning") or "",
            ch.get("finish_reason"))


def eval_accuracy(url, path=EVAL_SET, conc=CONCURRENCY):
    """Greedy accuracy on the fixed eval-200 set, measured on the served
    endpoint. Returns {n, correct, acc, by_kind, errors}."""
    with open(path, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    sema = threading.Semaphore(conc)
    out = [None] * len(items)

    def work(i, it):
        with sema:
            try:
                text, _fin = _gen_text(url, it["q"])
                out[i] = grade_item(it, text)
            except Exception:
                out[i] = None

    ts = [threading.Thread(target=work, args=(i, it))
          for i, it in enumerate(items)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    errors = sum(1 for r in out if r is None)
    by_kind = {}
    for it, r in zip(items, out):
        k = it.get("kind", "?")
        c, n = by_kind.get(k, (0, 0))
        by_kind[k] = (c + (1 if r else 0), n + 1)
    correct = sum(1 for r in out if r)
    return {"n": len(items), "correct": correct,
            "acc": round(correct / len(items), 4) if items else None,
            "errors": errors,
            "by_kind": {k: f"{c}/{n}" for k, (c, n) in by_kind.items()}}


def live_eval_acc(d):
    """eval200 accuracy with the 'tool' column removed from both sides.

    tool is a permanent 0/50 floor: the prompts never name a callable
    tool, so no model can emit the required snake_case names (second-
    opinion review Q1). It contributes a constant -25% to every acc and
    only shrinks the tolerance budget (tol 0.02*200 = 4 items is ~2.7%
    of the live 150). Returns None when by_kind is missing (old
    baselines) so callers can fall back to raw acc.
    """
    bk = d.get("by_kind") or {}
    if "tool" not in bk or "n" not in d or "correct" not in d:
        return None
    try:
        tc, tn = (int(x) for x in str(bk["tool"]).split("/"))
    except ValueError:
        return None
    n = d["n"] - tn
    return (d["correct"] - tc) / n if n > 0 else None


def ttft_probe(url, runs=TTFT_RUNS):
    """Median TTFT (s) on the ~2000-token TTFT_PROMPT, streaming, greedy."""
    times, ptok = [], None
    for _ in range(runs):
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": TTFT_PROMPT},
                         {"role": "assistant", "content": ""}],
            "continue_final_message": True,
            "add_generation_prompt": False,
            "max_tokens": 8,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        req = urllib.request.Request(
            url + "/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.time()
        ttft = None
        with urllib.request.urlopen(req, timeout=300) as r:
            for raw in r:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    ptok = chunk["usage"].get("prompt_tokens", ptok)
                for chc in chunk.get("choices", []):
                    delta = chc.get("delta", {})
                    if ttft is None and (delta.get("content")
                                         or delta.get("reasoning")):
                        ttft = time.time() - t0
        if ttft is None:
            raise RuntimeError("ttft probe produced no content token")
        times.append(ttft)
    times.sort()
    return {"ttft_s": round(times[len(times) // 2], 4),
            "runs": [round(t, 4) for t in times],
            "prompt_tokens": ptok}


def _load_prompts(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_baseline(path, n_prompts):
    """(rows, meta) or raise ValueError — an unusable baseline must fail
    the gate loudly instead of degrading every comparison to prefix 0.
    meta must carry the gate-v2 stock fields: ppl, eval200, ttft2000."""
    rows, meta = {}, {}
    with open(path) as f:
        for l in f:
            r = json.loads(l)
            if r.get("type") == "meta":
                meta = r
            elif "i" in r:
                rows[r["i"]] = r
    if len(rows) != n_prompts:
        raise ValueError(f"baseline has {len(rows)} rows, expected "
                         f"{n_prompts}")
    bad = [i for i, r in rows.items()
           if r.get("error") or not r.get("tokens")]
    if bad:
        raise ValueError(f"baseline rows with errors/empty tokens: {bad}")
    ppl = meta.get("ppl")
    if not isinstance(ppl, (int, float)) or not math.isfinite(ppl):
        raise ValueError("baseline meta line has no usable ppl")
    for k in ("eval200", "ttft2000"):
        if not isinstance(meta.get(k), dict):
            raise ValueError(f"baseline meta has no {k} -- re-run "
                             "capture (gate v2 baseline)")
    return rows, meta


def capture_bad(results):
    """List of unusable capture rows (request error or empty output)."""
    return [{"i": r["i"], "error": r.get("error", "no tokens")}
            for r in results if r.get("error") or not r.get("tokens")]


def _check_server(url):
    """Fail fast unless url is a vLLM server that actually serves MODEL.

    Runs before any of the 64 prompt requests: a tunnel that landed on a
    foreign listener (or a wrong direct --url) must die here, not after a
    full pass of 404s."""
    try:
        with urllib.request.urlopen(url + "/v1/models", timeout=30) as r:
            ids = [m.get("id")
                   for m in json.loads(r.read()).get("data", [])]
    except Exception as e:
        raise SystemExit(f"{url}/v1/models unreachable: {e}")
    if MODEL not in ids:
        raise SystemExit(f"{url} does not serve {MODEL} (ids: {ids})")


def _preflight_client(url):
    """First step of `capture`: one real request through an optional
    preflight-client.sh placed next to this file's parent directory, which
    calls this file's own _gen. Runs after _check_server, so it is skipped
    whenever the pair is not serving; a reject (non-200 / empty body) fails
    the capture before the 64-prompt pass is spent."""
    pf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "preflight-client.sh")
    if not os.path.exists(pf):
        return
    rc = subprocess.run(["bash", pf, os.path.abspath(__file__), url],
                        timeout=300).returncode
    if rc != 0:
        raise SystemExit(f"preflight-client failed (rc={rc})")


def _tunnel(args):
    """--ssh HOST: spawn ssh -N -L tunnel, return proc (or None).

    The url's 127.0.0.1 port is forwarded to the same remote port. If the
    local port is already bound, a free local port is forwarded instead
    and args.url is repointed at it: a squatter (SearXNG held :8888 on
    2026-09-15) makes ssh -L fail to bind but stay alive, after which the
    readiness probe connects to the squatter and every request 404s.
    ExitOnForwardFailure turns a raced bind into a dead ssh that the
    probe loop reports instead of silently misrouting.

    Waits until the forwarded port accepts connections -- without this the
    first requests race the tunnel setup and every capture row errors out."""
    if not getattr(args, "ssh", None):
        return None
    port = int(args.url.rsplit(":", 1)[-1].rstrip("/"))
    lport = port
    try:
        socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
    except OSError:
        pass
    else:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        lport = s.getsockname()[1]
        s.close()
        args.url = f"http://127.0.0.1:{lport}"
        print(f"tunnel: 127.0.0.1:{port} already bound -> forwarding "
              f"127.0.0.1:{lport} to {args.ssh}:{port}")
    p = subprocess.Popen(
        ["ssh", "-N", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
         "-o", "ExitOnForwardFailure=yes",
         "-L", f"{lport}:127.0.0.1:{port}", args.ssh])
    for _ in range(100):
        try:
            socket.create_connection(
                ("127.0.0.1", lport), timeout=0.5).close()
            return p
        except OSError:
            if p.poll() is not None:
                raise SystemExit(f"ssh tunnel to {args.ssh} exited "
                                 f"(rc={p.returncode})")
            time.sleep(0.2)
    p.terminate()
    raise SystemExit(f"ssh tunnel to {args.ssh} did not come up in 20s")


# ------------------------------------------------------- config check ---

def st_header(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        return {k: v for k, v in json.loads(f.read(hlen)).items()
                if k != "__metadata__"}


def expand_excludes(excludes):
    """vLLM apply_vllm_mapper: 'x*' -> ['x', 'x.*'] (modelopt.py:229)."""
    out = []
    for e in excludes:
        if len(e) >= 2 and e[-1] == "*" and e[-2] != ".":
            out += [e[:-1], e[:-1] + ".*"]
        else:
            out.append(e)
    return out


def is_excluded(prefix, excludes):
    for e in excludes:
        if e == prefix:
            return True
    for e in excludes:
        if e != prefix and (
                e in prefix
                or (prefix.startswith("language_model.")
                    and e in prefix.removeprefix("language_model."))):
            return True
    return any(fnmatch.fnmatch(prefix, p) for p in excludes)


def _candidates(prefix):
    out = [prefix]
    if prefix.endswith(".lm_head"):
        out.append("lm_head")
    if prefix.startswith("language_model.model."):
        out.append("model.language_model."
                   + prefix[len("language_model.model."):])
    elif prefix.startswith("model.language_model."):
        out.append("language_model.model."
                   + prefix[len("model.language_model."):])
    return list(dict.fromkeys(out))


PACKED = {"qkv_proj": ("q_proj", "k_proj", "v_proj"),
          "gate_up_proj": ("gate_proj", "up_proj")}
# The recipe serves --tensor-parallel-size 2; PB_WO block scales shard
# per rank, so the TP size is part of the format contract.
TP_SIZE = 2

# ckpt member suffix -> fused vLLM param suffix (serve image model.py
# stacked_params_mapping). Member tensors load into the fused param by
# shard_id, so the param that carries a member's scales is the FUSED
# name. The MIXED resolver cannot unfuse member quantized_layers keys
# for groups not in PACKED (Glm5Next* has no packed_modules_mapping);
# those fused params quantize only under their own name.
GLM5_STACKED = [
    (".gate_up_proj", ".gate_proj"),
    (".gate_up_proj", ".up_proj"),
    (".fused_qkv_a_proj", ".q_a_proj"),
    (".fused_qkv_a_proj", ".kv_a_proj_with_mqa"),
    (".wk_weights_proj", ".wk"),
    (".wk_weights_proj", ".weights_proj"),
    (".in_proj_qkvbfg_a", ".q_proj"),
    (".in_proj_qkvbfg_a", ".k_proj"),
    (".in_proj_qkvbfg_a", ".v_proj"),
    (".in_proj_qkvbfg_a", ".b_proj"),
    (".in_proj_qkvbfg_a", ".f_a_proj"),
    (".in_proj_qkvbfg_a", ".g_a_proj"),
]
FUSED_MEMBERS = {}   # fused leaf -> member leaves
DEAD_MEMBER = {}     # member leaf -> fused leaf (not resolver-unfusable)
for _fused, _member in GLM5_STACKED:
    FUSED_MEMBERS.setdefault(_fused[1:], []).append(_member[1:])
    if _fused[1:] not in PACKED:
        DEAD_MEMBER[_member[1:]] = _fused[1:]

# model.py _try_load_fp8_attn_proj / _try_load_fp8_indexer_wk: an F8
# weight under one of these names is buffered until its
# <mod>.weight_scale_inv arrives, then dequantized into the BF16 fused
# param -- it never reaches a quant method, and a modelopt
# <mod>.weight_scale has no param to land in (KeyError at load).
FP8_DEQUANT_WEIGHTS = (".q_a_proj.weight", ".kv_a_proj_with_mqa.weight",
                       ".q_b_proj.weight", ".o_proj.weight",
                       ".indexer.wk.weight")

# quant_algo values the v0.29 MIXED get_quant_method actually dispatches
# (modelopt.py:2427-2436). Anything else (e.g. FP8_PER_CHANNEL_PER_TOKEN)
# silently resolves to UnquantizedLinearMethod -> dead config + orphan
# scale tensors -> KeyError at load.
KNOWN_ALGOS = {"FP8", "FP8_PB_WO", "NVFP4", "W4A16_NVFP4", "MXFP8"}
# vocab-parallel params (ParallelLMHead / embed) load via
# VocabParallelEmbedding.weight_loader, which asserts
# loaded.shape[0] == org_vocab_size -- a PB_WO block-scale tensor
# [ob,1,ib,1] can never satisfy that under TP>1.
VOCAB_MODULES = ("lm_head", "embed_tokens")


def _param_base(mod):
    """ckpt member module prefix -> the vLLM param's module prefix."""
    if ".mlp.experts." in mod:
        return mod  # expert shards use expert_params_mapping, not stacked
    for fused, member in GLM5_STACKED:
        if member in mod:
            return mod.replace(member, fused)
    return mod


def _is_fp8_dequant_name(name):
    return name.endswith(FP8_DEQUANT_WEIGHTS)


def _classify_layers(weight_map):
    kinds = {}
    for name in weight_map:
        m = LAYER_RE.match(name)
        if not m:
            continue
        layer, sub = int(m.group(1)), m.group(2)
        if sub == "self_attn.q_proj.weight":
            kinds.setdefault(layer, "kda")
        elif sub == "self_attn.q_a_proj.weight":
            kinds.setdefault(layer, "dsa")
    return kinds


def resolve_algo(prefix, qlayers):
    """Mirror _resolve_quant_algo (modelopt.py:2312-2388)."""
    for c in _candidates(prefix):
        if c in qlayers:
            return qlayers[c]["quant_algo"].upper()
    proj = prefix.rsplit(".", 1)[-1]
    base = prefix.rsplit(".", 1)[0]
    for c in _candidates(base):
        algos = {qlayers[f"{c}.{s}"]["quant_algo"].upper()
                 for s in PACKED.get(proj, ()) if f"{c}.{s}" in qlayers}
        if len(algos) == 1:
            return algos.pop()
        if len(algos) > 1:
            raise ValueError(f"mixed algos in fused {prefix}: {algos}")
    for c in _candidates(prefix):
        for k, info in qlayers.items():
            if k.startswith(c + "."):
                return info["quant_algo"].upper()
    if prefix.endswith(".experts"):
        parent = prefix.rsplit(".experts", 1)[0] + "."
        for k, info in qlayers.items():
            if k.startswith(parent):
                return info["quant_algo"].upper()
    if proj in PACKED:
        for c in _candidates(prefix):
            p = c.rsplit(".", 1)[0] + "."
            algos = {qlayers[p + s]["quant_algo"].upper()
                     for s in PACKED[proj] if p + s in qlayers}
            if len(algos) == 1:
                return algos.pop()
            if len(algos) > 1:
                raise ValueError(f"mixed algos in fused {prefix}: {algos}")
    return None


def check_config(d):
    errors, warnings = [], []

    def err(m):
        errors.append(m)

    def warn(m):
        warnings.append(m)

    with open(os.path.join(d, "config.json")) as f:
        cfg = json.load(f)
    qc = cfg.get("quantization_config") or {}
    quant_algo = str(qc.get("quant_algo", "")).upper()
    hq_path = os.path.join(d, "hf_quant_config.json")
    hq_q = {}
    if os.path.exists(hq_path):
        hq_q = json.load(open(hq_path)).get("quantization", {})
    if quant_algo != "MIXED_PRECISION":
        err(f"config.json quant_algo {quant_algo} != MIXED_PRECISION")
    if hq_q and str(hq_q.get("quant_algo", "")).upper() != quant_algo:
        err("hf_quant_config quant_algo mismatch")
    if hq_q and hq_q.get("kv_cache_quant_algo", "FP8").upper() != "FP8":
        warn("kv_cache_quant_algo != FP8")
    qlayers = qc.get("quantized_layers") or hq_q.get("quantized_layers") \
        or {}
    if not qlayers:
        err("empty quantized_layers under MIXED_PRECISION")
    if any(k.endswith("in_proj_qkvbfg_a") for k in qlayers):
        # config PASS is static -- it cannot see that the image's kda.py
        # builds this module Unquantized (vllm_config.quant_config=None is
        # snapshotted into self.quant_config during
        # GatedDeltaNetAttention.__init__). Boot requires the kda-quant
        # overlay (overlays/patch-kda.py in this repository);
        # without it these member scales KeyError in params_dict (b4).
        warn("in_proj_qkvbfg_a declared in quantized_layers -- bootable "
             "only with the kda-quant overlay mounted; config PASS alone "
             "is not a boot signal")
    ignore = qc.get("ignore") or hq_q.get("exclude_modules") or []
    excludes = expand_excludes(ignore)

    with open(os.path.join(d, "model.safetensors.index.json")) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    headers = {}
    for shard in set(weight_map.values()):
        p = os.path.join(d, shard)
        if not os.path.exists(p):
            err(f"index references missing shard {shard}")
            continue
        headers[shard] = st_header(p)
    nbytes = 0
    for name, shard in weight_map.items():
        e = headers.get(shard, {}).get(name)
        if e is None:
            err(f"{name} mapped to {shard} but absent from its header")
            continue
        nb = DTYPE_BYTES[e["dtype"]]
        for d_i in e["shape"]:
            nb *= d_i
        nbytes += nb
    ts = index.get("metadata", {}).get("total_size")
    if ts is not None and int(ts) != nbytes:
        err(f"total_size {ts} != mapped bytes {nbytes}")

    tc0 = cfg.get("text_config", cfg)
    nlayers = int(tc0.get("num_hidden_layers", 0)) or 45
    kinds = _classify_layers(weight_map)

    def entry(name):
        s = weight_map.get(name)
        return headers.get(s, {}).get(name) if s else None

    for name in sorted(weight_map):
        if name.endswith((".weight_scale", ".weight_scale_2",
                          ".input_scale")):
            # every modelopt scale tensor must land in a real param: its
            # (possibly fused) parent module must resolve to a quant
            # method. Otherwise load_weights hits params_dict[...] ->
            # KeyError -> EngineCore death (the 2026-09-15 a2/b2 crash:
            # member keys under a fused name the resolver cannot unfuse).
            mod = name.rsplit(".", 1)[0]
            if ".mlp.experts." in mod:
                continue  # expert params go through expert_params_mapping
            pbase = _param_base(mod)
            if is_excluded(pbase, excludes):
                err(f"{name}: scale tensor under excluded {pbase} -- "
                    "params_dict KeyError at load")
            elif resolve_algo(pbase, qlayers) is None:
                err(f"{name}: {pbase} resolves unquantized -- no param "
                    "carries this tensor (params_dict KeyError at load); "
                    "member keys of a fused group must be declared under "
                    "the fused name")
            continue
        if name.endswith(".weight_scale_inv"):
            # consumed only by the fp8-dequant intercept (weight must be
            # fp8); on any other name there is no such param -> KeyError.
            wname = name[:-len("_scale_inv")]
            if not _is_fp8_dequant_name(wname):
                err(f"{name}: weight_scale_inv under a non-dequant name "
                    "-- no such param exists in the model")
            elif entry(wname) is None \
                    or entry(wname)["dtype"] != "F8_E4M3":
                warn(f"{name}: scale_inv without an fp8 weight -- dead "
                     "tensor, never consumed")
            continue
        if not name.endswith(".weight"):
            continue
        mod = name[:-len(".weight")]
        e = entry(name)
        if e is None:
            continue
        shape2d = len(e["shape"]) == 2
        lm = LAYER_RE.match(mod)
        layer_num = int(lm.group(1)) if lm else -1
        # the MTP layer (45, >= num_hidden_layers) ships BF16 experts by
        # design; only real layers are guaranteed NVFP4 experts
        is_moe = ".mlp.experts." in mod and 0 <= layer_num < nlayers
        if is_moe:
            algo, pbase = "NVFP4", mod
        else:
            pbase = _param_base(mod)
            algo = None if is_excluded(pbase, excludes) \
                else resolve_algo(pbase, qlayers)
        if _is_fp8_dequant_name(name) and e["dtype"] == "F8_E4M3":
            # the dequant intercept owns this tensor: it is buffered until
            # weight_scale_inv arrives, never reaching the quant param.
            if entry(mod + ".weight_scale_inv") is None:
                err(f"{name}: fp8 weight under the dequant intercept "
                    "without weight_scale_inv -- buffered forever, never "
                    "loads")
            if algo:
                err(f"{mod}: declared {algo} but the fp8-dequant "
                    "intercept swallows its weight -- the module cannot "
                    "be modelopt-quantized")
            continue
        if algo in ("NVFP4", "W4A16_NVFP4"):
            if e["dtype"] != "U8" or len(e["shape"]) != 2:
                err(f"{name}: NVFP4 weight must be U8 2-D, got "
                    f"{e['dtype']} {e['shape']}")
                continue
            n, k2 = e["shape"]
            if (2 * k2) % 16 != 0:
                err(f"{name}: NVFP4 in-dim {2 * k2} not divisible by "
                    "group size 16")
            ws = entry(mod + ".weight_scale")
            if ws is None or ws["dtype"] != "F8_E4M3" \
                    or ws["shape"] != [n, 2 * k2 // 16]:
                err(f"{mod}.weight_scale: expect F8_E4M3 "
                    f"[{n},{2 * k2 // 16}], got {ws}")
            s2 = entry(mod + ".weight_scale_2")
            if s2 is None or s2["dtype"] != "F32":
                err(f"{mod}.weight_scale_2: expect F32 scalar, got {s2}")
            if algo == "NVFP4":
                ins = entry(mod + ".input_scale")
                if ins is None or ins["dtype"] != "F32":
                    err(f"{mod}.input_scale: NVFP4 needs F32 scalar, "
                        f"got {ins}")
        elif algo == "FP8_PB_WO":
            if e["dtype"] != "F8_E4M3" or not shape2d:
                err(f"{name}: FP8_PB_WO weight must be F8_E4M3 2-D, got "
                    f"{e['dtype']} {e['shape']}")
                continue
            n, k = e["shape"]
            if k % 128 != 0:
                err(f"{name}: FP8_PB_WO in-dim {k} not divisible by "
                    "block 128")
            # BlockQuantScaleParameter narrows the scale tensor by
            # per-rank 128-blocks: the TP-sharded dim must cover whole
            # blocks. Members of a fused PB_WO group are sharded on `out`
            # (merged column parallel), so a member with
            # out % (128*TP) != 0 narrows out of bounds on rank>=1 -- that
            # is why in_proj_qkvbfg_a / wk_weights_proj (64/128/32-row
            # members) can never be PB_WO. A standalone module may be
            # column- (out sharded) or row-parallel (in sharded); accept
            # either orientation fully covering blocks.
            block = 128 * TP_SIZE
            if pbase != mod:
                if n % block:
                    err(f"{name}: fused member out-dim {n} not a "
                        f"multiple of {block} (128*TP={TP_SIZE}) -- PB_WO "
                        "scale shard runs out of bounds on rank>=1")
            elif not (n % block == 0 and k % 128 == 0) \
                    and not (k % block == 0 and n % 128 == 0):
                err(f"{name}: PB_WO dims {e['shape']} -- neither "
                    f"orientation holds whole 128-blocks per rank "
                    f"(sharded dim must be a multiple of {block})")
            ws = entry(mod + ".weight_scale")
            want = [-(-n // 128), 1, k // 128, 1]
            if ws is None or ws["dtype"] != "F32" or ws["shape"] != want:
                err(f"{mod}.weight_scale: expect F32 {want}, got {ws}")
        elif algo == "FP8":
            # static per-tensor (ModelOptFp8LinearMethod): F8 weight plus
            # F32 scalar weight_scale/input_scale. Scalars shard
            # element-wise / via shard_id on fused members and take the
            # vocab loader's output_dim=None branch on lm_head, so no
            # 128-block rule applies (that is PB_WO-only).
            if e["dtype"] != "F8_E4M3" or not shape2d:
                err(f"{name}: FP8 weight must be F8_E4M3 2-D, got "
                    f"{e['dtype']} {e['shape']}")
                continue
            ws = entry(mod + ".weight_scale")
            if ws is None or ws["dtype"] != "F32":
                err(f"{mod}.weight_scale: FP8 expects F32 scalar, got "
                    f"{ws}")
            ins = entry(mod + ".input_scale")
            if ins is None or ins["dtype"] != "F32":
                err(f"{mod}.input_scale: FP8 static needs an F32 scalar "
                    f"(absent -> uninitialized param, garbage logits), "
                    f"got {ins}")
        elif algo is None and shape2d and not is_moe:
            orphans = [s for s in ("weight_scale", "weight_scale_2",
                                   "input_scale")
                       if entry(f"{mod}.{s}") is not None]
            if orphans:
                # modelopt scale tensors with no quantized parent param
                # die as params_dict[...] KeyError at load -- the 2026-09-15
                # a2/b2 boot crash (member keys under an unresolvable
                # fused name leave the parent unquantized).
                err(f"{mod}: has {orphans} but {pbase} resolves "
                    "unquantized -- params_dict KeyError at load "
                    "(member of a fused group the resolver cannot unfuse, "
                    "or an excluded module)")
            elif e["dtype"] == "U8":
                err(f"{name}: packed U8 weight but {pbase} unquantized "
                    "-- bf16 param copy_ shape assert at load")
            elif not is_excluded(pbase, excludes):
                warn(f"{mod}: 2-D weight unexcluded but unquantized "
                     "(loads as UnquantizedLinearMethod)")

    # the RoutedExperts *module* (parent of every expert shard) must itself
    # resolve to NVFP4 -- the per-tensor check above assumes NVFP4 for
    # experts without proving the keep-entry exists in quantized_layers.
    for ln in range(nlayers):
        parent = f"model.language_model.layers.{ln}.mlp.experts"
        if any(k.startswith(parent + ".") for k in weight_map):
            if is_excluded(parent, excludes):
                err(f"{parent}: experts present but excluded")
            elif resolve_algo(parent, qlayers) != "NVFP4":
                err(f"{parent}: resolves to "
                    f"{resolve_algo(parent, qlayers)}, expect NVFP4")

    # quantized_layers key sanity: dead member keys, fused coverage,
    # excluded declarations, unquantizable DSA attention.
    for k, info in qlayers.items():
        algo_k = str(info.get("quant_algo", "")).upper()
        if algo_k not in KNOWN_ALGOS:
            err(f"{k}: unknown quant_algo {info.get('quant_algo')!r} -- "
                "v0.29 resolves it to no method (unquantized)")
        if is_excluded(k, excludes):
            err(f"{k}: declared in quantized_layers but excluded")
        leaf = k.rsplit(".", 1)[-1]
        parent = k.rsplit(".", 1)[0]
        if leaf in VOCAB_MODULES and algo_k == "FP8_PB_WO":
            err(f"{k}: FP8_PB_WO cannot load on a vocab-parallel module "
                "-- its block scale fails the embedding loader's "
                "org_vocab_size assert under TP>1 (use W4A16_NVFP4)")
        lm2 = LAYER_RE.match(k)
        if lm2 and ".self_attn." in k and \
                kinds.get(int(lm2.group(1))) == "dsa":
            if (qc.get("producer") or {}).get("requant_target") \
                    in ("g", "h"):
                # routes g/h boot with the mla-quant overlay
                # (overlays/patch-mla.py), which passes the MIXED
                # quant_config into Glm5NextMLAAttention -- the key is
                # live config.
                warn(f"{k}: DSA-layer self_attn declared -- bootable "
                     "only with the mla-quant overlay mounted; config "
                     "PASS alone is not a boot signal")
            else:
                # Glm5NextMLAAttention is built with quant_config=None:
                # no self_attn child of a DSA layer can hold a method.
                err(f"{k}: DSA-layer self_attn has quant_config=None -- "
                    "the key is dead config and any converted member "
                    "tensor crashes the loader")
            continue
        if leaf in DEAD_MEMBER:
            fused_key = parent + "." + DEAD_MEMBER[leaf]
            if fused_key in qlayers:
                warn(f"{k}: dead member key -- the fused "
                     f"{DEAD_MEMBER[leaf]} is already declared")
            else:
                err(f"{k}: member key cannot resolve -- the fused module "
                    f"{DEAD_MEMBER[leaf]} is not in the resolver's "
                    f"fused_projection_shards; declare {fused_key} "
                    "instead")
        if leaf in FUSED_MEMBERS and leaf not in PACKED:
            # a declared fused group must have every member tensor on
            # disk in the algo's format; a missing/BF16 member loads
            # into a packed param and shape-asserts.
            for mem in FUSED_MEMBERS[leaf]:
                mw = f"{parent}.{mem}.weight"
                if entry(mw) is None:
                    err(f"{k}: fused group declared but member "
                        f"{parent}.{mem}.weight absent")
    return errors, warnings


# ---------------------------------------------------------------- cli ---

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("config")
    p.add_argument("dir")
    for c in ("capture", "check"):
        p = sub.add_parser(c)
        p.add_argument("--url", default="http://127.0.0.1:8888")
        p.add_argument("--ssh", help="ssh host for a -L tunnel")
        p.add_argument("--prompts", default=PROMPTS)
        p.add_argument("--eval-set", default=EVAL_SET)
        p.add_argument("--concurrency", type=int, default=CONCURRENCY)
        p.add_argument("--out")
        p.add_argument("--baseline")
        p.add_argument("--report")
    args = ap.parse_args()

    if args.cmd == "config":
        errors, warnings = check_config(args.dir)
        for m in errors:
            print(f"ERROR {m}")
        for m in warnings:
            print(f"WARN  {m}")
        if errors:
            print(f"FAIL: {len(errors)} error(s), "
                  f"{len(warnings)} warning(s)")
            sys.exit(1)
        print(f"PASS: {len(warnings)} warning(s)")
        sys.exit(0)

    tun = _tunnel(args)
    try:
        prompts = _load_prompts(args.prompts)
        if args.cmd == "check":
            # validate the baseline before touching the candidate server so
            # a missing/garbage baseline fails fast (and fails the gate).
            try:
                base_rows, base_meta = load_baseline(args.baseline,
                                                     len(prompts))
            except Exception as e:
                rep = {"baseline_error": str(e)}
                if args.report:
                    with open(args.report, "w") as f:
                        json.dump(rep, f, indent=2)
                print(f"GATE FAIL: baseline unusable: {e}")
                sys.exit(1)
        _check_server(args.url)
        if args.cmd == "capture":
            _preflight_client(args.url)
        results = run_prompts(args.url, prompts, args.concurrency)
        if args.cmd == "capture":
            meta = {"type": "meta"}
            for key, fn in (("ppl", lambda: ppl(args.url)),
                            ("eval200", lambda: eval_accuracy(
                                args.url, args.eval_set,
                                args.concurrency)),
                            ("ttft2000", lambda: ttft_probe(args.url))):
                try:
                    meta[key] = fn()
                except Exception as e:
                    meta[key + "_error"] = str(e)
            out = args.out or "baseline.jsonl"
            with open(out, "w") as f:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            # an all-error baseline must not satisfy the done-check: without
            # this, check would compare candidates against empty rows and
            # burn a full build/reboot cycle on a baseline that can never
            # gate anything.
            bad = capture_bad(results)
            missing = [k for k in ("ppl", "eval200", "ttft2000")
                       if k not in meta]
            if bad or missing:
                print(f"capture FAIL: {len(bad)} bad rows, "
                      f"missing={missing} -> {out}")
                sys.exit(1)
            print(f"capture done: {len(results)} prompts -> {out} "
                  f"(ppl={meta['ppl']} eval200={meta['eval200']['acc']} "
                  f"ttft2000={meta['ttft2000']['ttft_s']}s)")
            return

        # check -- gate v2. Prefix is diagnostic only: greedy decode paths
        # legitimately diverge under any numeric change (the overlaid kernels
        # proved this), so the contract is PPL ratio, eval-200 accuracy
        # delta, zero degenerate outputs, and TTFT on the long prompt.
        prefs, degen = [], []
        for r in results:
            b = base_rows.get(r["i"], {})
            pr = common_prefix(b.get("tokens", []), r["tokens"])
            d = is_degenerate(r["tokens"])
            if d == "empty" and r.get("error"):
                # a failed request is a client/transport event, not a
                # model output -- label it so it is not read as a
                # degenerate completion (still fails the gate: an
                # unevaluated arm cannot pass)
                d = f"request-error: {r['error']}"
            elif d == "empty" and r.get("finish_reason") == "stop":
                # zero tokens + stop = the first sampled token was a stop
                # token (immediate EOS) -- a model-side argmax flip, not
                # a parse artifact
                d = "empty (first-token stop)"
            if d:
                degen.append({"i": r["i"], "why": d,
                              "finish_reason": r.get("finish_reason")})
            prefs.append(pr)
            print(f"  prompt {r['i']}: prefix={pr} "
                  f"first_div={pr} deg={d or '-'}"
                  + (f" fin={r['finish_reason']}"
                     if d and r.get("finish_reason") else ""))
        mean_pref = sum(prefs) / len(prefs)
        min_pref = min(prefs)
        rep = {"min_prefix": min_pref, "mean_prefix": mean_pref,
               "n": len(results), "prefix_diagnostic_only": True,
               "degenerate": len(degen), "degenerate_rows": degen,
               "gate": {"degenerate": not degen}}
        try:
            cp = ppl(args.url)
            rep["ppl"] = cp
            rep["ppl_ratio"] = cp / base_meta["ppl"]
            rep["gate"]["ppl"] = rep["ppl_ratio"] <= PPL_RATIO_MAX
        except Exception as e:
            # a gate arm that could not be evaluated must not silently
            # pass: the contract was not checked -> FAIL.
            rep["ppl_error"] = str(e)
            rep["gate"]["ppl"] = False
        try:
            ev = eval_accuracy(args.url, args.eval_set, args.concurrency)
            stock = base_meta["eval200"]["acc"]
            stock_live = live_eval_acc(base_meta["eval200"])
            cand_live = live_eval_acc(ev)
            # the gate compares live acc (tool floor excluded) when both
            # sides carry by_kind; otherwise fall back to raw acc.
            cmp_stock = stock_live if stock_live is not None else stock
            cmp_cand = (cand_live if cand_live is not None
                        else ev["acc"])
            rep["eval200"] = {"stock": stock, "cand": ev["acc"],
                              "delta": round(ev["acc"] - stock, 4),
                              "stock_live": stock_live,
                              "cand_live": cand_live,
                              "delta_live": (round(cand_live - stock_live, 4)
                                             if cand_live is not None
                                             and stock_live is not None
                                             else None),
                              "by_kind": ev["by_kind"],
                              "errors": ev["errors"]}
            rep["gate"]["eval200"] = (
                ev["errors"] == 0
                and cmp_stock - cmp_cand <= EVAL_ACC_TOL)
        except Exception as e:
            rep["eval200_error"] = str(e)
            rep["gate"]["eval200"] = False
        try:
            tp = ttft_probe(args.url)
            stock_t = base_meta["ttft2000"]["ttft_s"]
            rep["ttft2000"] = {"stock_s": stock_t,
                              "cand_s": tp["ttft_s"],
                              "ratio": round(tp["ttft_s"] / stock_t, 4),
                              "prompt_tokens": tp["prompt_tokens"]}
            rep["gate"]["ttft2000"] = rep["ttft2000"]["ratio"] \
                <= TTFT_RATIO_MAX
        except Exception as e:
            rep["ttft2000_error"] = str(e)
            rep["gate"]["ttft2000"] = False
        ok = all(rep["gate"].values())
        if args.report:
            with open(args.report, "w") as f:
                json.dump(rep, f, indent=2)
        if args.out:
            # persist candidate rows (tokens + finish_reason + error per
            # prompt; meta line carries the whole report) -- the 2026-09-15
            # a4 gate FAIL on an empty prompt-2 completion was
            # undiagnosable because check discarded the rows.
            with open(args.out, "w") as f:
                f.write(json.dumps({"type": "meta", **rep},
                                   ensure_ascii=False) + "\n")
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"min_prefix={min_pref} mean_prefix={mean_pref:.1f} "
              f"degenerate={len(degen)} "
              f"ppl={rep.get('ppl')} ratio={rep.get('ppl_ratio')}")
        print(f"eval200 stock={rep.get('eval200', {}).get('stock')} "
              f"cand={rep.get('eval200', {}).get('cand')} "
              f"delta={rep.get('eval200', {}).get('delta')} "
              f"(tol -{EVAL_ACC_TOL})")
        print(f"ttft2000 stock={rep.get('ttft2000', {}).get('stock_s')}s "
              f"cand={rep.get('ttft2000', {}).get('cand_s')}s "
              f"ratio={rep.get('ttft2000', {}).get('ratio')} "
              f"(max {TTFT_RATIO_MAX}) "
              f"prompt_tokens={rep.get('ttft2000', {}).get('prompt_tokens')}")
        print(f"gate={rep['gate']}")
        print("GATE PASS" if ok else "GATE FAIL")
        sys.exit(0 if ok else 1)
    finally:
        if tun:
            tun.terminate()


if __name__ == "__main__":
    main()
