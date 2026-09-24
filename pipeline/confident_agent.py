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
from agents.verifier import critique_answer
from agents.judge import calculate_confidence, ConfidenceResult

_MCQ_LETTER_RE = re.compile(r"^\s*\(?([ABCD])\)?\b")


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

        # 3. Critic Agent (+ revision loop)
        revision_count = 0
        critic_verdict, critic_reason = critique_answer(
            self.llm, question, retrieval.chunks, answer_text,
            max_tokens=llm_cfg["max_tokens_critic"],
        )
        while critic_verdict == "FAIL" and revision_count < critic_cfg["max_revisions"]:
            answer_text = generate_answer(
                self.llm, question, retrieval.chunks, mcq_choices=mcq_choices,
                temperature=llm_cfg["temperature"], max_tokens=max_tokens,
                feedback=critic_reason,
            )
            revision_count += 1
            critic_verdict, critic_reason = critique_answer(
                self.llm, question, retrieval.chunks, answer_text,
                max_tokens=llm_cfg["max_tokens_critic"],
            )

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

        predicted_letter = None
        if mcq_choices:
            m = _MCQ_LETTER_RE.match(answer_text)
            predicted_letter = m.group(1) if m else None
            # Unparseable MCQ output is a NO-ANSWER, not a silent default — this
            # is exactly the bug found in the baseline's evaluation script.
            if predicted_letter is None:
                conf.decision = "ABSTAIN"

        # citations = [rc.chunk.id for rc in retrieval.chunks]
        citations = sorted(set(rc.chunk.source_id for rc in retrieval.chunks))

        return AgentResponse(
            question=question, answer=answer_text, predicted_letter=predicted_letter,
            citations=citations, confidence=conf.score, decision=conf.decision,
            retrieval_verdict=retrieval.verdict, critic_verdict=critic_verdict,
            critic_reason=critic_reason, revision_count=revision_count,
            retrieval_query_used=retrieval.query_used,
        )
