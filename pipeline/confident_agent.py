"""
Orchestrates the full pipeline exactly as drawn:

QUESTION -> RETRIEVAL (+ reformulation loop) -> ANSWER AGENT -> CRITIC (+ revision
loop) -> CONFIDENCE CALCULATOR -> FINAL RESPONSE
"""
import re
from dataclasses import dataclass, field
from typing import Optional, Dict

from retrieval.retriever import Retriever
from agents.answerer import generate_answer
from agents.verifier import critique_answer, find_ungrounded_years
from agents.judge import calculate_confidence, ConfidenceResult

_MCQ_LETTER_RE = re.compile(r"^\s*[\W_]*([A-Da-d])[\W_]*\s*$")


@dataclass
class AgentResponse:
    question: str
    answer: str
    predicted_letter: Optional[str]
    citations: list
    confidence: float
    decision: str            # ANSWER or ABSTAIN
    retrieval_verdict: str
    critic_verdict: str
    critic_reason: str
    revision_count: int
    retrieval_query_used: str
    ungrounded_years: list = field(default_factory=list)
    no_context: bool = False


class ConfidentAgent:
    def __init__(self, llm, retriever: Retriever, cfg: dict):
        self.llm = llm
        self.retriever = retriever
        self.cfg = cfg

    def answer(self, question: str, mcq_choices: Optional[Dict[str, str]] = None) -> AgentResponse:
        llm_cfg = self.cfg["llm"]
        critic_cfg = self.cfg["critic"]
        conf_cfg = self.cfg["confidence"]

        # 1. Retrieval Agent + Retrieval Verifier (+ reformulation loop)
        retrieval = self.retriever.retrieve(question)

        if not retrieval.chunks:
            # Nothing in the corpus at all (e.g. data/chunks/ is still empty) —
            # abstain immediately rather than let the answerer hallucinate with
            # no grounding whatsoever.
            return AgentResponse(
                question=question, answer="NOT_FOUND_IN_CONTEXT",
                predicted_letter=None, citations=[], confidence=0.0,
                decision="ABSTAIN", retrieval_verdict="POOR",
                critic_verdict="N/A", critic_reason="No chunks retrieved.",
                revision_count=0, retrieval_query_used=retrieval.query_used,
                no_context=True,
            )

        max_tokens = llm_cfg["max_tokens_mcq"] if mcq_choices else llm_cfg["max_tokens_answer"]

        # 2. Answer Agent
        answer_text = generate_answer(
            self.llm, question, retrieval.chunks, mcq_choices=mcq_choices,
            temperature=llm_cfg["temperature"], max_tokens=max_tokens,
        )

        # 3. Critic Agent (+ revision loop). Two independent checks run every
        # time: the LLM critic, and a deterministic year-grounding check that
        # can't hallucinate the way the LLM critic can (see the Sheila Cussons
        # case — the critic confidently PASSed a citation to a year that
        # appeared in none of the retrieved text). If they disagree, the
        # deterministic check wins.
        def run_critic(text):
            llm_verdict, llm_reason = critique_answer(
                self.llm, question, retrieval.chunks, text,
                max_tokens=llm_cfg["max_tokens_critic"],
            )
            ungrounded = find_ungrounded_years(text, retrieval.chunks)
            if ungrounded:
                return "FAIL", (
                    f"{llm_reason} [Overridden: deterministic grounding check "
                    f"found year(s) {', '.join(ungrounded)} that do not appear "
                    f"verbatim in any retrieved excerpt — likely fabricated, "
                    f"regardless of the critic's own verdict.]"
                ).strip()
            return llm_verdict, llm_reason

        revision_count = 0
        critic_verdict, critic_reason = run_critic(answer_text)
        max_revisions = 0 if mcq_choices else critic_cfg["max_revisions"]
        while critic_verdict == "FAIL" and revision_count < max_revisions:
            answer_text = generate_answer(
                self.llm, question, retrieval.chunks, mcq_choices=mcq_choices,
                temperature=llm_cfg["temperature"], max_tokens=max_tokens,
                feedback=critic_reason,
            )
            revision_count += 1
            critic_verdict, critic_reason = run_critic(answer_text)

        # 4. Confidence Calculator
        conf: ConfidenceResult = calculate_confidence(
            retrieval_top_score=retrieval.top_score,
            critic_passed=(critic_verdict == "PASS"),
            revision_count=revision_count,
            weight_retrieval=conf_cfg["weight_retrieval"],
            weight_critic=conf_cfg["weight_critic"],
            weight_revision_penalty=conf_cfg["weight_revision_penalty"],
            abstain_threshold=conf_cfg["abstain_threshold"],
        )

        # Deterministic numeric-grounding check already ran inside run_critic()
        # above (before every LLM critic call, including on revisions) — a
        # FAIL from it already flows through critic_verdict/critic_reason and
        # into the confidence calculator normally. Recompute it here just for
        # reporting on the final answer that was actually returned.
        ungrounded_years = find_ungrounded_years(answer_text, retrieval.chunks)

        predicted_letter = None
        if mcq_choices:
            m = _MCQ_LETTER_RE.match(answer_text.strip())
            if m:
                predicted_letter = m.group(1).upper()
            else:
                # Recovery: local models frequently answer with the option's
                # actual CONTENT instead of its letter label, despite explicit
                # instructions (e.g. answering "1984" instead of "D"). If the
                # output exactly matches, or clearly contains, exactly one
                # option's text, credit that as the answer rather than
                # discarding a response that got the content right.
                normalized = answer_text.strip().lower().rstrip(". ")
                exact = [letter for letter, text in mcq_choices.items()
                         if text.strip().lower() == normalized]
                if exact:
                    predicted_letter = exact[0]
                else:
                    contains = [letter for letter, text in mcq_choices.items()
                                if len(text.strip()) >= 3
                                and text.strip().lower() in normalized]
                    if len(contains) == 1:
                        predicted_letter = contains[0]
            if predicted_letter is None:
                conf.decision = "ABSTAIN"

        citations = sorted(set(rc.chunk.source_id for rc in retrieval.chunks))

        return AgentResponse(
            question=question, answer=answer_text, predicted_letter=predicted_letter,
            citations=citations, confidence=conf.score, decision=conf.decision,
            retrieval_verdict=retrieval.verdict, critic_verdict=critic_verdict,
            critic_reason=critic_reason, revision_count=revision_count,
            retrieval_query_used=retrieval.query_used, ungrounded_years=ungrounded_years,
        )
