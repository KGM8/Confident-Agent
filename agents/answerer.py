"""
Answer Agent: generates an answer strictly grounded in the retrieved chunks and
required to cite which chunk(s) it used. This is the main hallucination-prevention
lever — instructing the model to say "not found in context" rather than invent an
answer when the context doesn't contain it.
"""
from typing import List

from retrieval.vector_store import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a factual question-answering assistant for South African history. "
    "You are shown several numbered source excerpts, retrieved by a search "
    "system. Some excerpts may be irrelevant to the question — ignore those and "
    "use only the ones that actually answer it. "
    "You must answer ONLY using information found in these excerpts. "
    "Never use outside knowledge, even if you believe you know the answer. "
    "If different excerpts give DIFFERENT answers to the same question (e.g. "
    "one says 1980, another says 1984), do NOT silently pick one. Instead say "
    "exactly which excerpts disagree and what each one says, e.g.: "
    "'Sources disagree: [SRC:2] states 1980; [SRC:1] states 1984 and 1985.' "
    "If, and only if, NONE of the excerpts contain the answer, respond with "
    "exactly this and nothing else: NOT_FOUND_IN_CONTEXT "
    "Otherwise, give a short factual answer, then end it with the citation tag "
    "of the excerpt(s) you actually used, in the form [SRC:n]. "
    "Do not combine NOT_FOUND_IN_CONTEXT with an answer or citations."
)


def _format_context(chunks: List[RetrievedChunk]) -> str:
    lines = []
    for i, rc in enumerate(chunks, start=1):
        lines.append(f"[{i}] ({rc.chunk.source_id}) {rc.chunk.text.strip()}")
    return "\n\n".join(lines)


def generate_answer(llm, question: str, chunks: List[RetrievedChunk],
                     mcq_choices=None, temperature: float = 0.2,
                     max_tokens: int = 200, feedback: str = None) -> str:
    context = _format_context(chunks)

    if mcq_choices:
        choice_lines = "\n".join(f"{k}. {v}" for k, v in mcq_choices.items())
        task = (
            f"Question: {question}\n\nOptions:\n{choice_lines}\n\n"
            "Answer with ONLY the single letter of the correct option "
            "(A, B, C, or D), based strictly on the source excerpts. "
            "If the excerpts don't support any option confidently, answer "
            "NOT_FOUND_IN_CONTEXT instead of guessing."
        )
    else:
        task = f"Question: {question}"

    user_prompt = f"Source excerpts:\n{context}\n\n{task}"
    if feedback:
        user_prompt += (
            f"\n\nA reviewer rejected your previous answer for this reason: "
            f"\"{feedback}\". Revise your answer accordingly, staying strictly "
            "within the source excerpts."
        )

    raw = llm.complete(SYSTEM_PROMPT, user_prompt, temperature=temperature,
                        max_tokens=max_tokens)
    return _clean_answer(raw)


def _clean_answer(raw: str) -> str:
    """
    Small local models sometimes tack NOT_FOUND_IN_CONTEXT onto the end of an
    otherwise substantive answer, contradicting their own instructions. If
    there's real content before it, the model clearly did find something —
    strip the contradictory trailing marker rather than let it confuse the
    critic/confidence stage downstream.
    """
    marker = "NOT_FOUND_IN_CONTEXT"
    if marker in raw:
        before = raw.split(marker)[0].strip()
        if len(before) > 10:  # there's real content before the marker
            return before
    return raw