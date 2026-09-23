"""
TF-IDF vector store over the local SAHO chunk corpus.

Deliberately not using neural embeddings: this needs to run fully offline on a
student laptop, with no model download step, no GPU dependency, and no extra
moving part that can silently fail. TF-IDF + cosine similarity is a legitimate,
easily-justified retrieval baseline for a project whose contribution is the
*validation architecture*, not the retriever itself. Swap this module out if you
want to experiment with embeddings later — the interface (`search`) is what the
rest of the pipeline depends on.
"""
import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    id: str
    text: str


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, chunks_dir: str):
        self.chunks_dir = chunks_dir
        self.chunks: List[Chunk] = self._load_chunks()
        self.vectorizer = None
        self.matrix = None
        if self.chunks:
            self._build_index()

    def _load_chunks(self) -> List[Chunk]:
        jsonl_path = os.path.join(self.chunks_dir, "chunks.jsonl")
        if os.path.exists(jsonl_path):
            chunks = []
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    chunks.append(Chunk(id=obj["id"], text=obj["text"]))
            return chunks

        chunks = []
        if os.path.isdir(self.chunks_dir):
            for fname in sorted(os.listdir(self.chunks_dir)):
                if fname.lower().endswith((".md", ".txt")):
                    path = os.path.join(self.chunks_dir, fname)
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    chunk_id = os.path.splitext(fname)[0]
                    chunks.append(Chunk(id=chunk_id, text=text))
        return chunks

    def _build_index(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_df=0.9, min_df=1, ngram_range=(1, 2)
        )
        self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])

    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    def search(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        if self.is_empty():
            return []
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [RetrievedChunk(chunk=self.chunks[i], score=float(sims[i])) for i in top_idx]
