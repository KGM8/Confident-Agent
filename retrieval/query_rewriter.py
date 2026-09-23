"""
Query Analyser + Query Reformulation.

Rule-based on purpose (see README "deliberate change" section) — no LLM call.
"""
import re

_STOP_PREFIXES = (
    "what is", "what was", "what are", "who is", "who was", "where is",
    "where was", "when did", "when was", "how did", "how long", "how many",
    "which", "why did", "why was", "in what", "on what", "in which",
)


def analyse_query(question: str) -> str:
    """
    Turn a natural-language question into a search query: strip interrogative
    scaffolding, keep proper nouns / years / capitalised terms, which is what
    actually matters for TF-IDF lexical matching against short SAHO chunks.
    """
    q = question.strip()
    lowered = q.lower()
    for prefix in _STOP_PREFIXES:
        if lowered.startswith(prefix):
            q = q[len(prefix):].strip()
            break
    q = q.rstrip("?").strip()
    return q if q else question


def reformulate_query(original_question: str, previous_query: str) -> str:
    """
    Called when the Retrieval Verifier marks a result POOR. Broadens the query by
    falling back to the full original question (undoing the stop-prefix strip),
    since over-trimming is the most common reason a narrow TF-IDF query misses.
    """
    if previous_query != original_question:
        return original_question
    # Already tried the full question — as a last resort, keep only capitalised
    # tokens and years (the highest-signal terms for this domain).
    tokens = re.findall(r"[A-Z][a-zA-Z']+|\b\d{4}\b", original_question)
    return " ".join(tokens) if tokens else original_question
