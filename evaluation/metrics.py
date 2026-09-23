"""
Metrics beyond plain accuracy — this is what Phase 5 (§4.6) needs: the brief is
explicit that an intervention which prevents hallucinations by refusing to
answer anything isn't automatically good, so we need both:
  - accuracy on the questions the agent chose to answer
  - coverage: what fraction of questions it was willing to answer at all
  - false-accept rate: answered, but wrong (this is the "unacceptable
    hallucination" the whole project is about)
  - false-reject rate (proxy): abstained on a question, unknown whether it would
    have gotten it right — reported as abstention rate, flagged as a limitation
    since we don't have a "would have been correct" oracle for abstained items
    without also running the un-gated answerer on them.
"""
import argparse
import json


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_metrics(results):
    total = len(results)
    answered = [r for r in results if r["decision"] == "ANSWER"]
    abstained = [r for r in results if r["decision"] == "ABSTAIN"]

    correct_answered = sum(1 for r in answered if r["correct"])
    wrong_answered = len(answered) - correct_answered

    accuracy_on_answered = correct_answered / len(answered) if answered else 0.0
    overall_accuracy = correct_answered / total if total else 0.0
    coverage = len(answered) / total if total else 0.0
    abstention_rate = len(abstained) / total if total else 0.0
    false_accept_rate = wrong_answered / total if total else 0.0  # answered wrongly, out of all Qs

    avg_conf_correct = (sum(r["confidence"] for r in answered if r["correct"])
                         / correct_answered) if correct_answered else 0.0
    avg_conf_wrong = (sum(r["confidence"] for r in answered if not r["correct"])
                       / wrong_answered) if wrong_answered else 0.0

    return {
        "total_questions": total,
        "answered": len(answered),
        "abstained": len(abstained),
        "coverage": round(coverage, 4),
        "abstention_rate": round(abstention_rate, 4),
        "accuracy_on_answered": round(accuracy_on_answered, 4),
        "overall_accuracy": round(overall_accuracy, 4),
        "false_accept_rate": round(false_accept_rate, 4),
        "avg_confidence_when_correct": round(avg_conf_correct, 4),
        "avg_confidence_when_wrong": round(avg_conf_wrong, 4),
        # a well-calibrated agent should have avg_confidence_when_correct
        # noticeably higher than avg_confidence_when_wrong
        "confidence_gap": round(avg_conf_correct - avg_conf_wrong, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path")
    parser.add_argument("--type", choices=["mcq", "open"], required=True)
    args = parser.parse_args()

    results = load_results(args.results_path)
    metrics = compute_metrics(results)
    print(json.dumps({"benchmark_type": args.type, **metrics}, indent=2))


if __name__ == "__main__":
    main()
