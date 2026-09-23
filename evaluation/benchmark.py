"""
Runs the Confident Agent over a benchmark file and writes results in a format
that's a superset of the baseline's (id, question, gold, predicted, raw_output,
correct) — so existing baseline-analysis code still works — plus the extra
fields (confidence, decision, citations, retrieval_verdict, critic_verdict,
revision_count) needed for Phase 5/6 comparison (accept/reject rates,
calibration, not just raw accuracy).
"""
import json
import re
from tqdm import tqdm

from pipeline.confident_agent import ConfidentAgent


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _open_answer_correct(predicted: str, gold: str) -> bool:
    return _normalize(gold) in _normalize(predicted)


def run_mcq_benchmark(agent: ConfidentAgent, benchmark_path: str, out_path: str):
    with open(benchmark_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    results = []
    for row in tqdm(rows, desc="MCQ benchmark"):
        resp = agent.answer(row["question"], mcq_choices=row["choices"])
        predicted = resp.predicted_letter
        correct = (predicted == row["answer"]) if predicted else False
        results.append({
            "id": row["id"],
            "question": row["question"],
            "gold": row["answer"],
            "predicted": predicted,
            "raw_output": resp.answer,
            "correct": correct,
            "decision": resp.decision,
            "confidence": resp.confidence,
            "retrieval_verdict": resp.retrieval_verdict,
            "critic_verdict": resp.critic_verdict,
            "revision_count": resp.revision_count,
            "citations": resp.citations,
            "no_context": resp.no_context,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return results


def run_open_benchmark(agent: ConfidentAgent, benchmark_path: str, out_path: str):
    with open(benchmark_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    results = []
    for row in tqdm(rows, desc="Open benchmark"):
        resp = agent.answer(row["question"])
        correct = (resp.decision == "ANSWER"
                   and _open_answer_correct(resp.answer, row["answer"]))
        results.append({
            "id": row["id"],
            "question": row["question"],
            "gold": row["answer"],
            "predicted": resp.answer,
            "raw_output": resp.answer,
            "correct": correct,
            "decision": resp.decision,
            "confidence": resp.confidence,
            "retrieval_verdict": resp.retrieval_verdict,
            "critic_verdict": resp.critic_verdict,
            "revision_count": resp.revision_count,
            "citations": resp.citations,
            "no_context": resp.no_context,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return results
