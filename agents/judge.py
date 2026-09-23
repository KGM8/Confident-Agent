"""
Confidence Calculator: combines retrieval quality, critic verdict, and how many
revision rounds were needed into a single 0-1 confidence score, plus a decision
(ANSWER vs ABSTAIN). No LLM call — this step needs to be deterministic and
auditable, not another source of hallucination.
"""
from dataclasses import dataclass


@dataclass
class ConfidenceResult:
    score: float
    decision: str  # "ANSWER" or "ABSTAIN"


def calculate_confidence(retrieval_top_score: float, critic_passed: bool,
                          revision_count: int, weight_retrieval: float,
                          weight_critic: float, weight_revision_penalty: float,
                          abstain_threshold: float) -> ConfidenceResult:
    # retrieval_top_score is a TF-IDF cosine similarity in [0,1]; clip defensively
    retrieval_component = max(0.0, min(1.0, retrieval_top_score))
    critic_component = 1.0 if critic_passed else 0.0
    penalty = weight_revision_penalty * revision_count

    score = (weight_retrieval * retrieval_component
             + weight_critic * critic_component
             - penalty)
    score = max(0.0, min(1.0, score))

    # A failed critic verdict is disqualifying regardless of retrieval strength —
    # confidence should never round-trip a known-unsupported answer up to ANSWER.
    if not critic_passed:
        decision = "ABSTAIN"
    else:
        decision = "ANSWER" if score >= abstain_threshold else "ABSTAIN"

    return ConfidenceResult(score=round(score, 3), decision=decision)
