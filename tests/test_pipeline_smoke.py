"""
Smoke test with no network/model dependency: uses FakeLLMClient and a tiny
synthetic corpus (deliberately NOT real South African history — generic
placeholder facts, so the test doesn't assert anything about real content).
Run with: python -m tests.test_pipeline_smoke
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_client import FakeLLMClient
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from pipeline.confident_agent import ConfidentAgent

DEFAULT_CFG = {
    "llm": {"temperature": 0.2, "max_tokens_answer": 200, "max_tokens_critic": 120, "max_tokens_mcq": 10},
    "critic": {"max_revisions": 1},
    "confidence": {
        "weight_retrieval": 0.4, "weight_critic": 0.5,
        "weight_revision_penalty": 0.1, "abstain_threshold": 0.5,
    },
}


def build_demo_store(tmpdir):
    docs = {
        "doc-alpha": "Zorgon City was founded in the year 1850 by the explorer Milo Tenn. "
                     "It became the capital of the Free Republic of Vestland in 1901.",
        "doc-beta": "The Great Bridge of Kavara was completed in 1932 after nine years of "
                    "construction, connecting the eastern and western districts.",
    }
    for doc_id, text in docs.items():
        with open(os.path.join(tmpdir, f"{doc_id}.md"), "w") as f:
            f.write(text)
    return VectorStore(tmpdir)


def test_grounded_answer_passes_and_is_confident():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = build_demo_store(tmpdir)
        retriever = Retriever(store, top_k=2, good_threshold=0.05, max_reformulations=1)
        llm = FakeLLMClient(responses=[
            "Zorgon City was founded in 1850. [SRC:1]",   # answerer
            "VERDICT: PASS\nThe answer matches the excerpt exactly.",  # critic
        ])
        agent = ConfidentAgent(llm, retriever, DEFAULT_CFG)

        resp = agent.answer("In what year was Zorgon City founded?")
        assert resp.decision == "ANSWER", resp
        assert resp.critic_verdict == "PASS"
        assert resp.confidence > 0.5
        assert "doc-alpha" in resp.citations
        print("PASS: grounded answer -> ANSWER, confidence", resp.confidence)


def test_ungrounded_answer_triggers_revision_then_abstain():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = build_demo_store(tmpdir)
        retriever = Retriever(store, top_k=2, good_threshold=0.05, max_reformulations=1)
        # First answer hallucinates a fact not in context -> critic FAILs twice
        # (once initially, once after the single allowed revision) -> abstain.
        llm = FakeLLMClient(responses=[
            "Zorgon City was founded in 1972 by Queen Sarela.",           # answerer (wrong)
            "VERDICT: FAIL\nThe excerpt says 1850, not 1972; Queen Sarela is not mentioned.",  # critic
            "Zorgon City was founded in 1972.",                          # answerer revision (still wrong)
            "VERDICT: FAIL\nStill contradicts the excerpt, which says 1850.",  # critic again
        ])
        agent = ConfidentAgent(llm, retriever, DEFAULT_CFG)

        resp = agent.answer("In what year was Zorgon City founded?")
        assert resp.revision_count == 1
        assert resp.critic_verdict == "FAIL"
        assert resp.decision == "ABSTAIN"
        print("PASS: ungrounded answer -> revision -> still FAIL -> ABSTAIN")


def test_empty_corpus_abstains_immediately():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(tmpdir)  # empty dir
        retriever = Retriever(store, top_k=2, good_threshold=0.05, max_reformulations=1)
        llm = FakeLLMClient(responses=[])
        agent = ConfidentAgent(llm, retriever, DEFAULT_CFG)

        resp = agent.answer("Any question at all?")
        assert resp.no_context is True
        assert resp.decision == "ABSTAIN"
        assert len(llm.calls) == 0  # never even called the LLM
        print("PASS: empty corpus -> immediate abstain, no LLM call wasted")


def test_mcq_unparseable_output_abstains_not_defaults_to_a():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = build_demo_store(tmpdir)
        retriever = Retriever(store, top_k=2, good_threshold=0.05, max_reformulations=1)
        llm = FakeLLMClient(responses=[
            "Sure, I'm ready to help!",       # unparseable answerer output (the exact baseline bug)
            "VERDICT: FAIL\nNo letter given.",  # critic correctly fails it
            "Sure, I'm ready to help!",       # revision still unparseable
            "VERDICT: FAIL\nStill no letter given.",
        ])
        agent = ConfidentAgent(llm, retriever, DEFAULT_CFG)

        resp = agent.answer("Which year?", mcq_choices={"A": "1850", "B": "1900", "C": "1901", "D": "1972"})
        assert resp.predicted_letter is None
        assert resp.decision == "ABSTAIN"
        print("PASS: unparseable MCQ output -> abstain, not silently scored as A")


if __name__ == "__main__":
    test_grounded_answer_passes_and_is_confident()
    test_ungrounded_answer_triggers_revision_then_abstain()
    test_empty_corpus_abstains_immediately()
    test_mcq_unparseable_output_abstains_not_defaults_to_a()
    print("\nAll smoke tests passed.")
