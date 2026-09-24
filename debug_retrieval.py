"""
Debug helper: shows exactly what the retriever pulls for a question, against
your real data/chunks/ corpus, so we can see what the Answer Agent actually saw.

Usage: python debug_retrieval.py "your question here"
"""
import sys
import yaml

from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever

question = sys.argv[1] if len(sys.argv) > 1 else "In which year did massive national school boycotts erupt in the townships?"

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

store = VectorStore(cfg["retrieval"]["chunks_dir"])
print(f"Loaded {len(store.chunks)} sub-chunks from {len(set(c.source_id for c in store.chunks))} documents\n")

retriever = Retriever(
    store, top_k=cfg["retrieval"]["top_k"],
    good_threshold=cfg["retrieval"]["good_retrieval_threshold"],
    max_reformulations=cfg["retrieval"]["max_reformulations"],
)
result = retriever.retrieve(question)
print(f"Query used: {result.query_used!r}")
print(f"Verdict: {result.verdict}\n")
for i, rc in enumerate(result.chunks, 1):
    print(f"--- [{i}] score={rc.score:.3f} source={rc.chunk.source_id} ---")
    print(rc.chunk.text)
    print()