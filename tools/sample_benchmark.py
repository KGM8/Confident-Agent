"""
Creates a reproducible random sample of N questions from a benchmark file —
used when the full 300-question set isn't feasible to run through the agent
in a reasonable time on local hardware (a documented compute-budget decision,
not a shortcut — see your report's Threats to Validity section).

Usage:
    python tools/sample_benchmark.py benchmark_300_fixed.jsonl mcq_sample30.jsonl 30
    python tools/sample_benchmark.py open_benchmark_300.jsonl open_sample30.jsonl 30
"""
import argparse
import json
import random


def sample(in_path: str, out_path: str, n: int, seed: int = 42):
    with open(in_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    rng = random.Random(seed)
    chosen = rng.sample(rows, min(n, len(rows)))
    chosen.sort(key=lambda r: r["id"])  # keep output in a stable, readable order

    with open(out_path, "w", encoding="utf-8") as f:
        for row in chosen:
            f.write(json.dumps(row) + "\n")

    ids = [r["id"] for r in chosen]
    print(f"Wrote {len(chosen)} questions to {out_path}")
    print(f"Sampled question ids: {ids}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path")
    parser.add_argument("out_path")
    parser.add_argument("n", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sample(args.in_path, args.out_path, args.n, seed=args.seed)