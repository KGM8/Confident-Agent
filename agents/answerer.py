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
    "You must answer ONLY using the numbered source excerpts provided below. "
    "If the excerpts do not contain the answer, respond exactly with: "
    "NOT_FOUND_IN_CONTEXT. "
    "Never use outside knowledge, even if you believe you know the answer. "
    "Keep answers short and factual. End your answer with the citation tag of the "
    "excerpt(s) you used, in the form [SRC:n]."
)


def _format_context(chunks: List[RetrievedChunk]) -> str:
    lines = []
    for i, rc in enumerate(chunks, start=1):
        lines.append(f"[{i}] ({rc.chunk.id}) {rc.chunk.text.strip()[:4000]}") #Initally 800 
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

    return llm.complete(SYSTEM_PROMPT, user_prompt, temperature=temperature,
                         max_tokens=max_tokens)
