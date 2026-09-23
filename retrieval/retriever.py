from dataclasses import dataclass
from typing import List

from retrieval.vector_store import VectorStore, RetrievedChunk
from retrieval.query_rewriter import analyse_query, reformulate_query


@dataclass
class RetrievalResult:
    query_used: str
    chunks: List[RetrievedChunk]
    top_score: float
    verdict: str          # "GOOD" or "POOR"
    reformulated: bool


class Retriever:
    def __init__(self, vector_store: VectorStore, top_k: int = 4,
                 good_threshold: float = 0.15, max_reformulations: int = 1):
        self.store = vector_store
        self.top_k = top_k
        self.good_threshold = good_threshold
        self.max_reformulations = max_reformulations

    def _search_once(self, query: str) -> List[RetrievedChunk]:
        return self.store.search(query, top_k=self.top_k)

    def _verdict(self, chunks: List[RetrievedChunk]) -> str:
        if not chunks or chunks[0].score < self.good_threshold:
            return "POOR"
        return "GOOD"

    def retrieve(self, question: str) -> RetrievalResult:
        query = analyse_query(question)
        chunks = self._search_once(query)
        verdict = self._verdict(chunks)
        reformulated = False

        attempts = 0
        while verdict == "POOR" and attempts < self.max_reformulations:
            new_query = reformulate_query(question, query)
            if new_query == query:
                break
            query = new_query
            chunks = self._search_once(query)
            verdict = self._verdict(chunks)
            reformulated = True
            attempts += 1

        top_score = chunks[0].score if chunks else 0.0
        return RetrievalResult(
            query_used=query, chunks=chunks, top_score=top_score,
            verdict=verdict, reformulated=reformulated,
        )
