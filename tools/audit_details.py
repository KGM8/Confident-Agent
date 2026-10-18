"""
audit_details.py -- look inside results/critic_audit.jsonl: what did the critic actually say?

    python tools/audit_details.py
    python tools/audit_details.py results/critic_audit.jsonl

Prints (all text is shown with non-ASCII characters escaped, so it is safe in the Windows console):
  1. how each reply was laid out, for both readers
  2. a cross-tab of the robust verdict against whether the saved answer's content was correct
  3. EVERY reply to a content-correct answer (the crux: why were right answers not released?)
  4. examples of unparseable replies, and any reply that said PASS
"""
import collections
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/critic_audit.jsonl"
recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def show(r, n=170):
    raw = (r.get("critic_raw") or "").replace("\n", " | ")
    return f"[{r['kind']} #{r['id']}] {r['robust_verdict']}/{r['robust_status']}  " + ascii(raw[:n])


print("1. reply layout")
print("   legacy reader status:", dict(collections.Counter(r["legacy_status"] for r in recs)))
print("   robust reader status:", dict(collections.Counter(r["robust_status"] for r in recs)))

print("\n2. robust verdict x content correctness (rows = was the saved answer's content correct?)")
verdicts = ["PASS", "FAIL", "CONFLICT", "UNPARSEABLE"]
print(f"   {'':16s}" + "".join(f"{v:>13s}" for v in verdicts) + f"{'total':>8s}")
for label, flag in (("content-correct", True), ("content-wrong", False)):
    sub = [r for r in recs if r["content_correct"] == flag]
    c = collections.Counter(r["robust_verdict"] for r in sub)
    print(f"   {label:16s}" + "".join(f"{c.get(v, 0):>13d}" for v in verdicts) + f"{len(sub):>8d}")

print("\n3. every critic reply to a content-correct answer")
for r in recs:
    if r["content_correct"]:
        print("   " + show(r, 200))

print("\n4a. unparseable replies (first 8)")
for r in [r for r in recs if r["robust_verdict"] == "UNPARSEABLE"][:8]:
    print("   " + show(r))
print("\n4b. replies that said PASS")
for r in [r for r in recs if r["robust_verdict"] == "PASS"]:
    print("   " + show(r, 300) + f"   | answer content correct: {r['content_correct']}")
