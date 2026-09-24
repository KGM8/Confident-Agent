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
    "You are a strict fact-checker. You will be shown several numbered source "
    "excerpts and a proposed answer. The excerpts were retrieved by a search "
    "system and are NOT all guaranteed to be relevant — some may be irrelevant "
    "to the question. "
    "Respond in exactly this format on one line: "
    "VERDICT: PASS or VERDICT: FAIL or VERDICT: CONFLICT, followed by a short "
    "reason on the next line.\n"
    "PASS: the answer is fully and specifically supported by at least one "
    "excerpt, and does not contradict any excerpt.\n"
    "CONFLICT: different excerpts genuinely give different answers to the "
    "question (e.g. different years for the same event), AND the proposed "
    "answer correctly reports that disagreement rather than picking one "
    "silently. This is not a failure — do not treat disagreement between "
    "excerpts as the answer's fault.\n"
    "FAIL: the answer adds a fact not present in any excerpt, contradicts an "
    "excerpt, no excerpt addresses the question at all, or excerpts disagree "
    "but the answer picked one side without saying so."
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
        upper = line.strip().upper()
        if upper.startswith("VERDICT:"):
            if "CONFLICT" in upper:
                verdict = "CONFLICT"
            elif "PASS" in upper:
                verdict = "PASS"
            else:
                verdict = "FAIL"
            break

    # Small local models frequently get the free-text reasoning right but
    # mislabel the verdict line itself (e.g. reasoning clearly describes a
    # source disagreement, but still writes "VERDICT: FAIL"). Since the
    # reasoning is the more reliable signal here, fall back to keyword
    # detection rather than trusting a verdict label that contradicts its
    # own stated reasoning.
    if verdict == "FAIL":
        reason_lower = raw.lower()
        conflict_signals = ("conflict", "disagree", "different sources",
                             "different excerpts", "sources differ")
        if any(sig in reason_lower for sig in conflict_signals):
            verdict = "CONFLICT"
    
    reason_lines = [l for l in raw.splitlines() if not l.strip().upper().startswith("VERDICT:")]
    if reason_lines:
        reason = " ".join(reason_lines).strip()

    return verdict, reason
