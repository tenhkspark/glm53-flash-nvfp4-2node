#!/bin/bash
# run-tests.sh — offline smoke tests for the repo (no GPU, no serving).
# Covers: python compile, shell syntax, JSONL sanity, patcher
# fail-closed behavior, and CLI entry points.
set -u
cd "$(dirname "$0")/.."
fail=0
say() { printf '%-56s %s\n' "$1" "$2"; }

# 1. every python file compiles
for f in $(find . -name '*.py' -not -path './overlays/_*/*'); do
  if python3 -m py_compile "$f" 2>/tmp/rtc.err; then
    say "py_compile $f" ok
  else
    say "py_compile $f" FAIL; sed 's/^/    /' /tmp/rtc.err | head -3; fail=1
  fi
done

# 2. every shell file parses
for f in $(find . -name '*.sh' -not -path './overlays/_*/*'); do
  if bash -n "$f" 2>/tmp/rtc.err; then
    say "bash -n $f" ok
  else
    say "bash -n $f" FAIL; sed 's/^/    /' /tmp/rtc.err | head -3; fail=1
  fi
done

# 3. JSONL sanity: parse + expected counts
python3 - <<'EOF' && say "jsonl parse + counts" ok || { say "jsonl" FAIL; fail=1; }
import json, sys
p = [json.loads(l) for l in open("bench/prompts-64.jsonl")]
e = [json.loads(l) for l in open("bench/eval-200.jsonl")]
assert len(p) == 64 and all("prompt" in r for r in p), "prompts-64"
assert len(e) == 200 and all({"q", "a", "criteria"} <= set(r) for r in e), \
    "eval-200"
EOF

# 4. patchers fail closed on a non-matching input
tmp=$(mktemp -d)
echo "print('not the image file')" > "$tmp/x.py"
for p in patch-kda patch-mla patch-mtp; do
  if python3 "overlays/$p.py" "$tmp/x.py" "$tmp/out.py" >/dev/null 2>&1; then
    say "$p fail-closed" "FAIL (accepted bad input)"; fail=1
  else
    say "$p fail-closed" ok
  fi
done
# patchers must also refuse already-patched input
for p in patch-kda patch-mla patch-mtp; do
  marker=$(grep -oE 'MARKER = "[^"]+"' "overlays/$p.py" | cut -d'"' -f2)
  printf 'class A:\n    pass\n# %s\n' "$marker" > "$tmp/tagged.py"
  if python3 "overlays/$p.py" "$tmp/tagged.py" "$tmp/out.py" >/dev/null 2>&1; then
    say "$p re-patch refusal" "FAIL (accepted tagged input)"; fail=1
  else
    say "$p re-patch refusal" ok
  fi
done
rm -rf "$tmp"

# 5. CLI entry points respond to -h
for f in requant/verify.py bench/measure.py overlays/apply-step-attr-patch.py; do
  python3 "$f" -h >/dev/null 2>&1
  rc=$?
  # argparse -h exits 0; scripts without -h exit 2 — accept either, a
  # traceback means a real import break.
  if python3 "$f" -h 2>&1 | grep -q Traceback; then
    say "$f -h" "FAIL (traceback)"; fail=1
  else
    say "$f -h" "ok (rc=$rc)"
  fi
done

# 6. requant.py needs torch+safetensors; import-check when available
if python3 -c "import torch, safetensors" 2>/dev/null; then
  for f in requant/requant.py requant/build-mtp-draft.py; do
    if python3 "$f" -h >/dev/null 2>&1; then
      say "$f -h (torch present)" ok
    else
      say "$f -h" FAIL; fail=1
    fi
  done
else
  say "requant scripts import check" "skip (no torch)"
fi

# 7. agent-run.sh: env validation + --dry-run
tmp=$(mktemp -d)
# 7a. missing env file must fail at STEP 1
if scripts/agent-run.sh --env "$tmp/nope.env" >/dev/null 2>&1; then
  say "agent-run missing env" "FAIL (rc=0)"; fail=1
else
  scripts/agent-run.sh --env "$tmp/nope.env" 2>&1 | grep -q "STEP 1 FAIL" \
    && say "agent-run missing env" ok \
    || { say "agent-run missing env" "FAIL (no STEP 1 FAIL line)"; fail=1; }
fi
# 7b. CHANGEME values must fail, naming the var
sed 's/\$WEIGHTS_ROOT//g' setup.env.example > "$tmp/bad.env"
if scripts/agent-run.sh --env "$tmp/bad.env" 2>&1 \
    | grep -q "STEP 1 FAIL: set HEAD_HOST"; then
  say "agent-run CHANGEME env" ok
else
  say "agent-run CHANGEME env" "FAIL"; fail=1
fi
# 7c. a complete env + --dry-run: prints every STEP, every command is a
#     DRY$ line, rc=0, and nothing is executed (no ssh/rsync leaves this
#     machine — dry-run must not fail on unreachable hosts)
cat > "$tmp/good.env" <<'EOF'
HEAD_HOST=headnode
WORK_HOST=worknode
HEAD_IP=192.0.2.1
HEAD_IF=enp1s0f0np0
WORK_IP=192.0.2.2
WORK_IF=enp1s0f1np1
WEIGHTS_ROOT=/data/weights
EOF
out=$(scripts/agent-run.sh --env "$tmp/good.env" --dry-run 2>&1); rc=$?
if [ $rc -eq 0 ] \
  && grep -q "STEP 1 OK" <<<"$out" \
  && grep -q "DRY\$ ssh headnode" <<<"$out" \
  && grep -q "DRY-RUN done" <<<"$out" \
  && ! grep -q "STEP .* FAIL" <<<"$out"; then
  say "agent-run --dry-run" ok
else
  say "agent-run --dry-run" "FAIL (rc=$rc)"; fail=1
  printf '%s\n' "$out" | tail -10 | sed 's/^/    /'
fi
# 7d. MTP_K>=4 refused at env validation
sed 's/$/\nMTP_K=4/' "$tmp/good.env" > "$tmp/k4.env"
if scripts/agent-run.sh --env "$tmp/k4.env" --dry-run 2>&1 \
    | grep -q "STEP 1 FAIL: MTP_K=4"; then
  say "agent-run MTP_K=4 refused" ok
else
  say "agent-run MTP_K=4 refused" "FAIL"; fail=1
fi
# 7f. local operator: HEAD_HOST=localhost / $(hostname) runs head steps
#     without ssh (the operator may be the head node itself); the worker
#     still goes over ssh. Dry-run only — nothing executes.
cat > "$tmp/local.env" <<'EOF'
HEAD_HOST=localhost
WORK_HOST=worknode
HEAD_IP=192.0.2.1
HEAD_IF=enp1s0f0np0
WORK_IP=192.0.2.2
WORK_IF=enp1s0f1np1
WEIGHTS_ROOT=/data/weights
EOF
out=$(scripts/agent-run.sh --env "$tmp/local.env" --dry-run 2>&1); rc=$?
if [ $rc -eq 0 ] \
  && grep -q 'DRY\$ local ' <<<"$out" \
  && grep -q 'DRY\$ ssh worknode ' <<<"$out" \
  && ! grep -q 'DRY\$ ssh localhost' <<<"$out" \
  && ! grep -q 'localhost:' <<<"$out"; then
  say "agent-run local-operator (localhost)" ok
else
  say "agent-run local-operator (localhost)" "FAIL (rc=$rc)"; fail=1
  printf '%s\n' "$out" | tail -10 | sed 's/^/    /'
fi
# 7g. HEAD_HOST=$(hostname) selects the same local path
sed "s/^HEAD_HOST=.*/HEAD_HOST=$(hostname)/" "$tmp/local.env" > "$tmp/host.env"
out=$(scripts/agent-run.sh --env "$tmp/host.env" --dry-run 2>&1); rc=$?
if [ $rc -eq 0 ] \
  && grep -q 'DRY\$ local ' <<<"$out" \
  && ! grep -q "DRY\$ ssh $(hostname)" <<<"$out"; then
  say "agent-run local-operator (hostname)" ok
else
  say "agent-run local-operator (hostname)" "FAIL (rc=$rc)"; fail=1
fi
# 7h. STEP 2 weights-present skip: a complete checkpoint under STOCK_DIR
#     must end prereqs with "STEP 2 OK (weights present)" and never
#     touch pip on the host. A fake ssh answers the prereq probes; the
#     weights completeness check itself runs the real local python3
#     against a stub checkpoint dir.
mkdir -p "$tmp/bin" "$tmp/stock"
cat > "$tmp/stock/model.safetensors.index.json" <<'EOF'
{"metadata": {"total_size": 10},
 "weight_map": {"a": "shard-1.safetensors", "b": "shard-2.safetensors"}}
EOF
printf '123456' > "$tmp/stock/shard-1.safetensors"
printf '123456' > "$tmp/stock/shard-2.safetensors"
cat > "$tmp/bin/ssh" <<'EOF'
#!/bin/bash
cmd="${!#}"
case "$cmd" in
  *model.safetensors.index.json*) exec bash -c "$cmd" ;;  # real check
  *"uname -m"*) echo aarch64 ;;
  *nvidia-smi*) echo "NVIDIA GB10" ;;
  *ibdev2netdev*|*"rdma link"*)
    echo "rocep1s0f0 port 1 ==> enp1s0f0np0 (Up)"
    echo "rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)" ;;
  *dgx-release*|*lsb_release*|*"uname -srm"*) echo "DGX Spark" ;;
  *"docker info"*|*infiniband*|*"ssh -o BatchMode"*) exit 0 ;;
  *"df -BG"*) echo 900 ;;
  *pip*|*"command -v hf"*) echo "stub ssh: host pip/hf invoked" >&2; exit 3 ;;
  *) exit 0 ;;
esac
EOF
chmod +x "$tmp/bin/ssh"
printf '#!/bin/bash\nexit 0\n' > "$tmp/bin/rsync"
chmod +x "$tmp/bin/rsync"
cat > "$tmp/w.env" <<EOF
HEAD_HOST=fake-head
WORK_HOST=fake-work
HEAD_IP=192.0.2.1
HEAD_IF=enp1s0f0np0
WORK_IP=192.0.2.2
WORK_IF=enp1s0f1np1
WEIGHTS_ROOT=$tmp
STOCK_DIR=$tmp/stock
EOF
out=$(PATH="$tmp/bin:$PATH" scripts/agent-run.sh --env "$tmp/w.env" \
      prereqs 2>&1); rc=$?
if [ $rc -eq 0 ] && grep -q 'STEP 2 OK (weights present)' <<<"$out"; then
  say "agent-run step2 weights-present skip" ok
else
  say "agent-run step2 weights-present skip" "FAIL (rc=$rc)"; fail=1
  printf '%s\n' "$out" | tail -10 | sed 's/^/    /'
fi
# an incomplete checkpoint must NOT claim the skip — and still must not
# reach for pip (STEP 4 downloads inside the image instead)
rm "$tmp/stock/shard-2.safetensors"
out=$(PATH="$tmp/bin:$PATH" scripts/agent-run.sh --env "$tmp/w.env" \
      prereqs 2>&1); rc=$?
if [ $rc -eq 0 ] && grep -q '^STEP 2 OK$' <<<"$out" \
  && ! grep -q 'weights present' <<<"$out"; then
  say "agent-run step2 incomplete-weights" ok
else
  say "agent-run step2 incomplete-weights" "FAIL (rc=$rc)"; fail=1
  printf '%s\n' "$out" | tail -10 | sed 's/^/    /'
fi
# 7i. STEP 8's in-image helper must bypass the image ENTRYPOINT: the
#     image enters via the vllm CLI, which needs a GPU just to build its
#     serve parser and would eat the helper argv (the cleanroom-h-boot-i
#     failure, 2026-09-16). The fake ssh records the remote commands; no
#     draft index under MTP_DIR makes the build branch run for real.
cat > "$tmp/bin/ssh" <<'EOF'
#!/bin/bash
cmd="${!#}"
printf '%s\n' "$cmd" >> "$SSH_LOG"
case "$cmd" in
  *"MTP-bf16/model.safetensors.index.json"*) exit 1 ;;  # rdone: draft absent
  *"&& pwd"*) echo /fake/remote ;;                      # absdir
  *) exit 0 ;;
esac
EOF
chmod +x "$tmp/bin/ssh"
: > "$tmp/ssh.log"
out=$(SSH_LOG="$tmp/ssh.log" PATH="$tmp/bin:$PATH" \
      scripts/agent-run.sh --env "$tmp/w.env" stage 2>&1); rc=$?
drun=$(grep "docker run" "$tmp/ssh.log")
if [ $rc -eq 0 ] && grep -q 'STEP 8 OK' <<<"$out" \
  && grep -q -- "--entrypoint python3" <<<"$drun" \
  && grep -qF -- '--user $(id -u):$(id -g)' <<<"$drun" \
  && grep -q "' /repo/requant/build-mtp-draft.py" <<<"$drun"; then
  say "agent-run step8 bypasses vllm entrypoint" ok
else
  say "agent-run step8 bypasses vllm entrypoint" "FAIL (rc=$rc)"; fail=1
  printf '%s\n' "$out" | tail -10 | sed 's/^/    /'
  printf '%s\n' "$drun" | sed 's/^/    /'
fi
# and every in-image helper run in the script must carry the flag
if [ "$(grep -c 'docker run --rm' scripts/agent-run.sh)" \
     = "$(grep -c '^[[:space:]]*--entrypoint python3' scripts/agent-run.sh)" ]; then
  say "agent-run helper entrypoints" ok
else
  say "agent-run helper entrypoints" "FAIL (a docker run lacks --entrypoint)"; fail=1
fi
# every helper must also run as the remote uid, not root: root-owned
# output shards fail the follow-up rsync "Permission denied" (the
# cleanroom-h-boot-j failure, 2026-09-16)
if [ "$(grep -c 'docker run --rm' scripts/agent-run.sh)" \
     = "$(grep -cF -- '--user \$(id -u):\$(id -g)' scripts/agent-run.sh)" ]; then
  say "agent-run helper --user" ok
else
  say "agent-run helper --user" "FAIL (a docker run lacks --user)"; fail=1
fi
# 7e. verify-result.py: inside band -> PASS, outside -> FAIL
cat > "$tmp/measure-x.json" <<'EOF'
{"label": "t", "levels": [{"c": 1, "agg_tok_s": 24.6, "tpot_median_ms": 41.0,
  "ttft_median_s": 0.39, "fails": 0}], "acceptance": {"acceptance": 0.62}}
EOF
if python3 scripts/verify-result.py --result "$tmp/measure-x.json" \
    --ref h-rdma-mtp 2>&1 | grep -q "VERIFY PASS"; then
  say "verify-result in-band" ok
else
  say "verify-result in-band" "FAIL"; fail=1
fi
python3 - "$tmp/measure-x.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1])); d["levels"][0]["agg_tok_s"] = 10.0
json.dump(d, open(sys.argv[1], "w"))
EOF
if python3 scripts/verify-result.py --result "$tmp/measure-x.json" \
    --ref h-rdma-mtp >/dev/null 2>&1; then
  say "verify-result out-of-band" "FAIL (rc=0)"; fail=1
else
  say "verify-result out-of-band" ok
fi
rm -rf "$tmp"

# 8. check-md-invariants.py: identical input passes, a changed number fails
tmp=$(mktemp -d)
if python3 tests/check-md-invariants.py README.md README.md >/dev/null 2>&1; then
  say "md-invariants self-check" ok
else
  say "md-invariants self-check" FAIL; fail=1
fi
sed 's/35\.09/36.09/' README.md > "$tmp/mut.md"
if python3 tests/check-md-invariants.py README.md "$tmp/mut.md" >/dev/null 2>&1; then
  say "md-invariants detects change" "FAIL (accepted mutation)"; fail=1
else
  say "md-invariants detects change" ok
fi
rm -rf "$tmp"

echo
[ "$fail" = 0 ] && { echo "ALL TESTS PASS"; exit 0; }
echo "TESTS FAILED"; exit 1
