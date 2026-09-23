"""
Critic Agent: an independent grounding check. Given the same context and the
Answer Agent's proposed answer, checks whether the answer is actually supported
by the excerpts — this is deliberately a *separate* LLM call from the one that
generated the answer, rather than asking the answerer to self-certify, since
self-certification is weak evidence (a model that hallucinated the answer is
not a reliable judge of its own hallucination).
"""
from typing import List

from retrieval.vector_store import RetrievedChunk
from agents.answerer import _format_context

SYSTEM_PROMPT = (
    "You are a strict fact-checker. You will be shown source excerpts and a "
    "proposed answer. Decide whether the proposed answer is fully and "
    "specifically supported by the excerpts. "
    "Respond in exactly this format on one line: "
    "VERDICT: PASS or VERDICT: FAIL, followed by a short reason on the next line. "
    "FAIL if the answer adds any fact not present in the excerpts, contradicts "
    "the excerpts, or the excerpts don't actually address the question."
)


def critique_answer(llm, question: str, chunks: List[RetrievedChunk],
                     answer: str, max_tokens: int = 120):
    context = _format_context(chunks)
    user_prompt = (
        f"Source excerpts:\n{context}\n\nQuestion: {question}\n\n"
        f"Proposed answer: {answer}"
    )
    raw = llm.complete(SYSTEM_PROMPT, user_prompt, temperature=0.0,
                        max_tokens=max_tokens)

    verdict = "FAIL"
    reason = raw.strip()
    for line in raw.splitlines():
        if line.strip().upper().startswith("VERDICT:"):
            verdict = "PASS" if "PASS" in line.upper() else "FAIL"
            break
    reason_lines = [l for l in raw.splitlines() if not l.strip().upper().startswith("VERDICT:")]
    if reason_lines:
        reason = " ".join(reason_lines).strip()

    return verdict, reason
