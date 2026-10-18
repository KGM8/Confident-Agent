"""
final_checks.py -- three validity checks on the saved results (no model re-run).

  A. BASELINE PARSER AUDIT (MCQ). run_eval.py falls back to the first A-D
     character found anywhere in the output, so non-answers such as
     "Sure, I'm ready to help!" are recorded as option A. This audit separates
     CLEAN answers (output starts with a standalone letter A-D) from outputs
     where the letter was scavenged from prose, and re-scores accuracy strictly.

  B. OPEN-ENDED SCORING SYMMETRY. The baseline script scores with bidirectional
     containment (gold in pred OR pred in gold, first line only, stop-words
     removed); the agent evaluation uses gold-in-pred only. Both arms are
     re-scored under BOTH rules.

  C. SENSITIVITY ANALYSIS. Repeats the baseline-vs-agent comparison after
     dropping items found to be ill-posed or off-domain, with exact McNemar tests.

Run from the project root:
    python tools/final_checks.py
Defaults assume the file names used in this project; override with the flags.
"""
import argparse
import json
import re
import string
from math import comb

STOP = {"a", "an", "the"}
CLEAN_LETTER = re.compile(r"^\s*[\(\[]?([A-Da-d])[\)\]\.:,]?(\s|$)")


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return {json.loads(l)["id"]: json.loads(l) for l in f if l.strip()}


def clean_letter(raw):
    m = CLEAN_LETTER.match(raw or "")
    return m.group(1).upper() if m else None


def norm_alnum(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())


def norm_baseline(t):
    t = t.lower().strip().translate(str.maketrans("", "", string.punctuation))
    return " ".join(w for w in t.split() if w not in STOP)


def scorer_baseline(gold, pred):          # run_eval.py's rule (bidirectional, first line)
    g, p = norm_baseline(gold), norm_baseline((pred or "").strip().split("\n")[0])
    return bool(g) and bool(p) and (g in p or p in g)


def scorer_agent(gold, pred):             # evaluation/benchmark.py's rule (gold in pred)
    g = norm_alnum(gold)
    return bool(g) and g in norm_alnum(pred or "")


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, sum(comb(n, i) for i in range(k + 1)) / 2 ** (n - 1))


def compare(label, ids, a_correct, b_correct, a_name="baseline", b_name="agent"):
    both = a_only = b_only = neither = 0
    for i in ids:
        a, b = a_correct[i], b_correct[i]
        both += a and b; a_only += a and not b; b_only += b and not a; neither += (not a) and (not b)
    n = len(ids)
    print(f"  {label}: n={n} | {a_name} {sum(a_correct[i] for i in ids)}/{n} "
          f"({sum(a_correct[i] for i in ids)/n:.1%}) | {b_name} {sum(b_correct[i] for i in ids)}/{n} "
          f"({sum(b_correct[i] for i in ids)/n:.1%}) | both right {both}, {a_name}-only {a_only}, "
          f"{b_name}-only {b_only}, both wrong {neither} | exact McNemar p = {mcnemar_exact(a_only, b_only):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-mcq", default="baseline_mcq_sample30_results.jsonl")
    ap.add_argument("--baseline-open", default="baseline_open_sample30_results.jsonl")
    ap.add_argument("--agent-mcq", default="results/mcq_agent_results.jsonl")
    ap.add_argument("--agent-open", default="results/open_agent_results.jsonl")
    ap.add_argument("--exclude-mcq", default="13,15,45,111,112,216,229,259")
    ap.add_argument("--exclude-open", default="18")
    args = ap.parse_args()
    ex_mcq = {int(x) for x in args.exclude_mcq.split(",") if x}
    ex_open = {int(x) for x in args.exclude_open.split(",") if x}

    # ------------------------------------------------------------------ A
    bm, am = load(args.baseline_mcq), load(args.agent_mcq)
    ids = sorted(set(bm) & set(am))
    print("=" * 78)
    print("A. BASELINE MCQ PARSER AUDIT")
    print("=" * 78)
    clean = {i: clean_letter(bm[i]["raw_output"]) for i in ids}
    n_clean = sum(1 for i in ids if clean[i])
    lenient_ok = {i: bool(bm[i]["correct"]) for i in ids}
    strict_ok = {i: clean[i] is not None and clean[i] == bm[i]["gold"] for i in ids}
    spurious = [i for i in ids if lenient_ok[i] and clean[i] is None]
    print(f"  outputs that are a clean letter answer:         {n_clean}/{len(ids)}")
    print(f"  outputs where the letter was scavenged:         {len(ids) - n_clean}/{len(ids)}")
    print(f"  lenient accuracy (as originally reported):      {sum(lenient_ok.values())}/{len(ids)}")
    print(f"  strict accuracy (clean answers only):           {sum(strict_ok.values())}/{len(ids)}")
    print(f"  credited as correct but NOT a clean answer:     {len(spurious)}")
    print("  scavenged-letter rows (id | raw output | lenient prediction | gold | counted correct):")
    for i in ids:
        if clean[i] is None:
            print(f"    {i:>4} | {bm[i]['raw_output'][:60]!r} | {bm[i]['predicted']} | {bm[i]['gold']} | {lenient_ok[i]}")

    # ------------------------------------------------------------------ B
    bo, ao = load(args.baseline_open), load(args.agent_open)
    oids = sorted(set(bo) & set(ao))
    print("\n" + "=" * 78)
    print("B. OPEN-ENDED SCORING SYMMETRY (both arms under both rules)")
    print("=" * 78)
    for rule_name, rule in (("baseline rule (bidirectional)", scorer_baseline), ("agent rule (gold-in-pred)", scorer_agent)):
        b_ok = {i: rule(bo[i]["gold"], bo[i]["raw_output"]) for i in oids}
        a_ok = {i: rule(ao[i]["gold"], ao[i]["raw_output"]) for i in oids}
        compare(rule_name, oids, b_ok, a_ok)
    print("  baseline answers credited under the bidirectional rule (gold | prediction):")
    for i in oids:
        if scorer_baseline(bo[i]["gold"], bo[i]["raw_output"]):
            print(f"    {i:>4} | {bo[i]['gold'][:40]!r} | {bo[i]['raw_output'][:60]!r}")

    # ------------------------------------------------------------------ C
    print("\n" + "=" * 78)
    print(f"C. SENSITIVITY ANALYSIS (excluding MCQ ids {sorted(ex_mcq)} and open ids {sorted(ex_open)})")
    print("=" * 78)
    agent_mcq_ok = {i: am[i]["predicted"] is not None and am[i]["predicted"] == am[i]["gold"] for i in ids}
    keep = [i for i in ids if i not in ex_mcq]
    print("  MCQ (agent = content-correct, ignoring abstention):")
    compare("all 30, lenient baseline   ", ids, lenient_ok, agent_mcq_ok)
    compare("all 30, strict baseline    ", ids, strict_ok, agent_mcq_ok)
    compare("retained, lenient baseline ", keep, lenient_ok, agent_mcq_ok)
    compare("retained, strict baseline  ", keep, strict_ok, agent_mcq_ok)
    b_ok = {i: scorer_agent(bo[i]["gold"], bo[i]["raw_output"]) for i in oids}
    a_ok = {i: scorer_agent(ao[i]["gold"], ao[i]["raw_output"]) for i in oids}
    okeep = [i for i in oids if i not in ex_open]
    print("  Open-ended (both arms scored with the same gold-in-pred rule):")
    compare("all 30                     ", oids, b_ok, a_ok)
    compare("retained                   ", okeep, b_ok, a_ok)
    print("\nPaste this whole output back so the report numbers can be finalised.")


if __name__ == "__main__":
    main()
