"""
Critic Agent: an independent grounding check. Given the same context and the
Answer Agent's proposed answer, checks whether the answer is actually supported
by the excerpts — this is deliberately a *separate* LLM call from the one that
generated the answer, rather than asking the answerer to self-certify, since
self-certification is weak evidence (a model that hallucinated the answer is
not a reliable judge of its own hallucination).
"""
import re
from typing import List

from retrieval.vector_store import RetrievedChunk
from agents.answerer import _format_context

_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")


def find_ungrounded_years(answer: str, chunks: List[RetrievedChunk]) -> List[str]:
    """
    Deterministic sanity check, no LLM involved: every 4-digit year the answer
    states must appear verbatim somewhere in the retrieved excerpts. This
    can't hallucinate the way an LLM critic can (see the Sheila Cussons case —
    the critic accepted a fabricated "1984 and 1985" that appeared in none of
    the retrieved text). Catches numeric hallucinations before they ever reach
    the LLM critic.
    """
    years_in_answer = set(_YEAR_RE.findall(answer))
    if not years_in_answer:
        return []
    corpus_text = " ".join(rc.chunk.text for rc in chunks)
    return sorted(y for y in years_in_answer if y not in corpus_text)


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


def build_critic_prompt(question: str, chunks: List[RetrievedChunk], answer: str) -> str:
    """The exact user prompt sent to the critic (shared by the pipeline and the audit tool)."""
    context = _format_context(chunks)
    return (
        f"Source excerpts:\n{context}\n\nQuestion: {question}\n\n"
        f"Proposed answer: {answer}"
    )


# --------------------------------------------------------------------------
# Verdict parsing.
#
# LEGACY (the original experimental condition): a verdict is recognised only on
# a line that STARTS with "VERDICT:". Anything else defaults to FAIL. Llama-2
# habitually prefixes replies with an emoji or markdown ("🔍 VERDICT: PASS",
# "**VERDICT: PASS**"), so a PASS written that way was silently read as FAIL.
#
# ROBUST (the corrected condition): finds the labelled verdict anywhere in the
# reply, tolerating emoji, markdown and chatty prefixes, and reports an explicit
# parse status instead of silently defaulting. No keyword-based CONFLICT guess.
# --------------------------------------------------------------------------
_LABELLED = re.compile(r"VERDICT\s*[:=\-\u2013\u2014]?\s*[^A-Za-z0-9\n]*\s*(PASS|FAIL|CONFLICT)\b", re.IGNORECASE)
_TOKEN = re.compile(r"\b(PASS|FAIL|CONFLICT)\b", re.IGNORECASE)


def parse_verdict_legacy(raw: str):
    """Exact behaviour of the original parser. Returns (verdict, parse_status)."""
    verdict = "FAIL"
    status = "legacy_default_fail"
    for line in raw.splitlines():
        upper = line.strip().upper()
        if upper.startswith("VERDICT:"):
            status = "legacy_line_start"
            if "CONFLICT" in upper:
                verdict = "CONFLICT"
            elif "PASS" in upper:
                verdict = "PASS"
            else:
                verdict = "FAIL"
            break
    if verdict == "FAIL":
        reason_lower = raw.lower()
        conflict_signals = ("conflict", "disagree", "different sources",
                             "different excerpts", "sources differ")
        if any(sig in reason_lower for sig in conflict_signals):
            verdict = "CONFLICT"
            status = "legacy_keyword_conflict"
    return verdict, status


def parse_verdict_robust(raw: str):
    """Returns (verdict, parse_status). verdict is PASS / FAIL / CONFLICT or UNPARSEABLE.
    parse_status: strict (reply starts with VERDICT:), tolerant (label found after a
    prefix), loose (bare token near the start), ambiguous (several different tokens on
    the verdict line), unparseable (no verdict found)."""
    text = raw.replace("*", " ").replace("`", " ").replace("_", " ")
    m = _LABELLED.search(text)
    if m:
        line_end = text.find("\n", m.start())
        verdict_line = text[m.start(): line_end if line_end != -1 else len(text)]
        tokens = {t.upper() for t in _TOKEN.findall(verdict_line)}
        if len(tokens) > 1:
            return "UNPARSEABLE", "ambiguous"
        status = "strict" if text.strip().upper().startswith("VERDICT") else "tolerant"
        return m.group(1).upper(), status
    head_tokens = {t.upper() for t in _TOKEN.findall(text[:80])}
    if len(head_tokens) > 1:
        # e.g. the model echoing the instruction: "PASS, CONFLICT, or FAIL" -- never a verdict
        return "UNPARSEABLE", "ambiguous"
    if len(head_tokens) == 1:
        return next(iter(head_tokens)), "loose"
    return "UNPARSEABLE", "unparseable"


def _reason_from_raw(raw: str) -> str:
    reason = raw.strip()
    reason_lines = [l for l in raw.splitlines() if not l.strip().upper().startswith("VERDICT:")]
    if reason_lines:
        reason = " ".join(reason_lines).strip()
    return reason


def critique_answer_detailed(llm, question: str, chunks: List[RetrievedChunk],
                              answer: str, max_tokens: int = 120, parse_mode: str = "legacy"):
    """One critic call. Returns a dict with the raw reply, verdict, reason and parse status."""
    raw = llm.complete(SYSTEM_PROMPT, build_critic_prompt(question, chunks, answer),
                        temperature=0.0, max_tokens=max_tokens)
    if parse_mode == "robust":
        verdict, status = parse_verdict_robust(raw)
    elif parse_mode == "legacy":
        verdict, status = parse_verdict_legacy(raw)
    else:
        raise ValueError(f"unknown critic parse_mode: {parse_mode!r}")
    return {"raw": raw, "verdict": verdict, "reason": _reason_from_raw(raw), "parse_status": status}


def critique_answer(llm, question: str, chunks: List[RetrievedChunk],
                     answer: str, max_tokens: int = 120):
    """Original interface (legacy parsing): returns (verdict, reason)."""
    d = critique_answer_detailed(llm, question, chunks, answer, max_tokens, parse_mode="legacy")
    return d["verdict"], d["reason"]
