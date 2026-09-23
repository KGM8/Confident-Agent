"""
Rebalances an MCQ benchmark's option ordering so the gold answer isn't
overwhelmingly "A" (in the group's current benchmark_300.jsonl, 286/300 = 95.3%
of gold answers are A — see chat discussion). Shuffles each question's four
option values into a new A/B/C/D order (deterministically, via a seeded RNG, so
the run is reproducible) and updates the gold letter to match. Question text and
the actual answer content are never changed — only which letter it sits behind.

Usage:
    python tools/fix_mcq_bias.py path/to/mcq_benchmark_300.jsonl path/to/mcq_benchmark_300_fixed.jsonl
"""
import argparse
import json
import random


def rebalance(in_path: str, out_path: str, seed: int = 42):
    rng = random.Random(seed)
    letters = ["A", "B", "C", "D"]

    fixed_rows = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            choices = row["choices"]
            gold_letter = row["answer"]
            gold_text = choices[gold_letter]

            values = list(choices.values())
            rng.shuffle(values)
            new_choices = dict(zip(letters, values))
            new_gold_letter = next(l for l, v in new_choices.items() if v == gold_text)

            row["choices"] = new_choices
            row["answer"] = new_gold_letter
            fixed_rows.append(row)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in fixed_rows:
            f.write(json.dumps(row) + "\n")

    from collections import Counter
    dist = Counter(r["answer"] for r in fixed_rows)
    print(f"Wrote {len(fixed_rows)} questions to {out_path}")
    print(f"New answer letter distribution: {dict(dist)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path")
    parser.add_argument("out_path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rebalance(args.in_path, args.out_path, seed=args.seed)
