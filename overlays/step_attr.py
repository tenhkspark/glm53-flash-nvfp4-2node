# SPDX-License-Identifier: Apache-2.0
"""Per-decode-step GPU time attribution for GLM-5.3-Flash.

Mounted at dist-packages root as ``step_attr.py`` and imported by the
patched model.py / model_runner.py / cuda_communicator.py produced by
apply-step-attr-patch.py. Active only when the container env has
STEP_ATTR=1; every hook is a no-op otherwise.

Method: each instrumented call site records a CUDA event pair into a
bucket (attn / moe / ar / sampler); model_runner records one step
begin/end pair per engine step. Events are queued on the compute stream
and accumulated host-side -- no sync per step. Every EVERY steps
(default 256) the host synchronizes once, aggregates elapsed_time, and prints one
line to stderr:

    [step-attr] steps=256 total=.. attn=.. moe=.. ar=.. sampler=.. other=..

Values are ms per step, mean over the window. Bucket times are
exclusive: a span nested inside another is subtracted from its parent
(so TP all-reduces inside attention/MoE blocks land in ``ar``, not
double-counted in attn/moe). ``other = total - sum(buckets)`` collects
input prep, MTP draft, mHC ops, engine-visible gaps.

Cudagraph safety: a span is skipped while a stream is capturing or
while dynamo is compiling, and inside a replayed graph Python does not
run at all, so captured regions contribute 0 to attn/moe/ar under
graphs -- their time lands in ``other`` (compare with the eager arm).
step_begin/step_end live in model_runner.py, outside any captured
region, so ``total``/``sampler`` stay valid in graph mode.
"""

import os
import sys

import torch

_ENABLED = os.environ.get("STEP_ATTR") == "1"
_EVERY = int(os.environ.get("STEP_ATTR_EVERY", "256"))

_BUCKETS = ("attn", "moe", "ar", "sampler")

_spans = []  # (bucket, parent_bucket|None, start_ev, end_ev)
_stack = []  # buckets of currently open spans
_steps = []  # (start_ev, end_ev)
_step_start = None
_recording = False
_pending = 0  # steps since last flush
_broken = False  # latched off on first unexpected error


def _warn_once(msg):
    print("[step-attr] disabled: %s" % msg, file=sys.stderr, flush=True)


class span:
    """Record a CUDA event pair into ``bucket`` while a step is recording."""

    __slots__ = ("bucket", "_parent", "_ev")

    def __init__(self, bucket):
        self.bucket = bucket
        self._parent = None
        self._ev = None

    def __enter__(self):
        global _broken
        if _broken or not _ENABLED or not _recording:
            return self
        try:
            if (
                torch.cuda.is_current_stream_capturing()
                or torch.compiler.is_compiling()
            ):
                return self
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
        except Exception as e:  # never let instrumentation kill serving
            _broken = True
            _warn_once("span enter: %r" % e)
            return self
        self._parent = _stack[-1] if _stack else None
        _stack.append(self.bucket)
        self._ev = ev
        return self

    def __exit__(self, *exc):
        global _broken
        if self._ev is None:
            return False
        try:
            _stack.pop()
            end = torch.cuda.Event(enable_timing=True)
            end.record()
        except Exception as e:
            _broken = True
            _warn_once("span exit: %r" % e)
            self._ev = None
            return False
        _spans.append((self.bucket, self._parent, self._ev, end))
        self._ev = None
        return False


def step_begin():
    """Record the step-start event on the current stream (worker side)."""
    global _step_start, _recording, _broken
    if _broken or not _ENABLED:
        return
    try:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
    except Exception as e:
        _broken = True
        _warn_once("step_begin: %r" % e)
        return
    _step_start = ev
    _recording = True


def step_end():
    """Close the current step and flush every EVERY steps."""
    global _step_start, _recording, _pending, _broken
    if _broken or not _ENABLED:
        return
    _recording = False
    if _step_start is None:
        return
    try:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
    except Exception as e:
        _broken = True
        _warn_once("step_end: %r" % e)
        _step_start = None
        return
    _steps.append((_step_start, ev))
    _step_start = None
    _pending += 1
    if _pending >= _EVERY:
        try:
            flush()
        except Exception as e:  # instrumentation must never kill serving
            _broken = True
            _warn_once("flush: %r" % e)
            _pending = 0
            _spans.clear()
            _steps.clear()


def flush():
    """Sync once, aggregate elapsed ms, print one line, reset."""
    global _pending
    if not _steps:
        _pending = 0
        return
    torch.cuda.synchronize()
    raw = dict.fromkeys(_BUCKETS, 0.0)
    nested = dict.fromkeys(_BUCKETS, 0.0)
    for bucket, parent, s, e in _spans:
        ms = s.elapsed_time(e)
        if bucket in raw:
            raw[bucket] += ms
        if parent in nested:
            nested[parent] += ms
    total = sum(s.elapsed_time(e) for s, e in _steps)
    n = len(_steps)
    per = {b: (raw[b] - nested[b]) / n for b in _BUCKETS}
    other = total / n - sum(per.values())
    print(
        "[step-attr] steps=%d total=%.2f attn=%.2f moe=%.2f ar=%.2f "
        "sampler=%.2f other=%.2f"
        % (n, total / n, per["attn"], per["moe"], per["ar"],
           per["sampler"], other),
        file=sys.stderr,
        flush=True,
    )
    _spans.clear()
    _steps.clear()
    _pending = 0


if _ENABLED:
    print("[step-attr] enabled every=%d" % _EVERY, file=sys.stderr, flush=True)
