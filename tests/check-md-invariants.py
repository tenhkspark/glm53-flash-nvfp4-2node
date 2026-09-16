#!/usr/bin/env python3
"""check-md-invariants.py BEFORE.md AFTER.md

Verify that an edit pass over a markdown file (e.g. a language polish)
changed wording only:

- fenced code blocks: byte-identical, in order
- inline `code` spans: byte-identical, in order
- every number token in the document: identical sequence

Exits 0 when all three hold, 1 otherwise, printing the first
differences. Whitespace-only changes inside prose are allowed; table
contents count through the number check.
"""
import re
import sys

FENCE = re.compile(r"^```.*?$(.*?)^```", re.M | re.S)
INLINE = re.compile(r"`([^`\n]+)`")
# numbers incl. decimals and versions: 1, 1.051, 50.0, 2.30.7+cuda13.3,
# 0.6221, 16384, 10.75 ... also units stuck on: 190GB, 35.09x
NUM = re.compile(r"\d+(?:[.:-][\w+]+)*")


def extract(text):
    fenced = [m.group(1) for m in FENCE.finditer(text)]
    inline = INLINE.findall(FENCE.sub("", text))
    nums = NUM.findall(text)
    return fenced, inline, nums


def diff_seq(name, a, b):
    if a == b:
        return True
    print(f"FAIL: {name} differ ({len(a)} before, {len(b)} after)")
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            print(f"  first diff at #{i}:")
            print(f"    before: {x!r}")
            print(f"    after : {y!r}")
            break
    if len(a) != len(b):
        print(f"  length differs: {len(a)} -> {len(b)}")
    return False


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    before = open(sys.argv[1], encoding="utf-8").read()
    after = open(sys.argv[2], encoding="utf-8").read()
    fb, ib, nb = extract(before)
    fa, ia, na = extract(after)
    ok = diff_seq("fenced code blocks", fb, fa)
    ok &= diff_seq("inline code spans", ib, ia)
    ok &= diff_seq("number tokens", nb, na)
    print("PASS: code and numbers identical" if ok else "INVARIANTS BROKEN")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
