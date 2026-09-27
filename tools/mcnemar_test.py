"""
Paired significance test (exact McNemar's test) comparing baseline vs agent
on the SAME questions, matched by id. This is the right test here because
every question was answered by both systems (paired binary outcomes:
correct/incorrect), not independent samples — exactly the "paired prompts,
two conditions" case.

For the agent's side, correctness is computed on CONTENT regardless of
decision (ABSTAIN/ANSWER) — i.e. "would this have been right if the agent
had been confident enough to commit" — since that's the fair like-for-like
comparison against a baseline that never abstains.

Usage:
    python tools/mcnemar_test.py --type mcq --baseline baseline_mcq_sample30_results.jsonl --agent results/mcq_agent_results.jsonl
    python tools/mcnemar_test.py --type open --baseline baseline_open_sample30_results.jsonl --agent results/open_agent_results.jsonl
"""
import argparse
import json
import re
from math import comb


def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return {json.loads(l)["id"]: json.loads(l) for l in f if l.strip()}


def exact_mcnemar_p(b, c):
    """Two-sided exact McNemar p-value via the binomial distribution."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** (n - 1))
    return min(p, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["mcq", "open"], required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--agent", required=True)
    args = parser.parse_args()

    baseline = load(args.baseline)
    agent = load(args.agent)
    shared_ids = sorted(set(baseline) & set(agent))
    if len(shared_ids) < len(baseline) or len(shared_ids) < len(agent):
        print(f"WARNING: only {len(shared_ids)} question ids appear in BOTH files "
              f"(baseline has {len(baseline)}, agent has {len(agent)}). "
              f"Double-check both were run on the exact same sample file.")

    both_correct = base_only = agent_only = both_wrong = 0
    for qid in shared_ids:
        b_row, a_row = baseline[qid], agent[qid]
        b_correct = bool(b_row["correct"])

        if args.type == "mcq":
            a_correct = a_row["predicted"] == a_row["gold"]
        else:
            a_correct = normalize(a_row["gold"]) in normalize(a_row["predicted"])

        if b_correct and a_correct:
            both_correct += 1
        elif b_correct and not a_correct:
            base_only += 1
        elif not b_correct and a_correct:
            agent_only += 1
        else:
            both_wrong += 1

    n = len(shared_ids)
    print(f"\nPaired comparison on {n} matched questions ({args.type}):")
    print(f"  Both correct:            {both_correct}")
    print(f"  Baseline right, agent wrong (b): {base_only}")
    print(f"  Agent right, baseline wrong (c): {agent_only}")
    print(f"  Both wrong:              {both_wrong}")
    print(f"\n  Baseline accuracy: {(both_correct+base_only)/n:.1%}")
    print(f"  Agent accuracy (content, ignoring abstention): {(both_correct+agent_only)/n:.1%}")

    p = exact_mcnemar_p(base_only, agent_only)
    print(f"\n  Exact McNemar's test p-value: {p:.4f}")
    if p < 0.05:
        print("  -> Statistically significant difference at alpha=0.05.")
    else:
        print("  -> NOT statistically significant at alpha=0.05 "
              "(with n=30 and few discordant pairs, this is expected even for a real effect — "
              "note this as a sample-size limitation in Threats to Validity, don't overclaim).")


if __name__ == "__main__":
    main()