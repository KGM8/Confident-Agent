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
import re
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    id: str            # unique sub-chunk id, e.g. "biography-0102-biography-102#3"
    source_id: str      # original document id — this is what citations use
    text: str


def _split_into_subchunks(text: str, max_chars: int = 900) -> List[str]:
    """
    Splits a long document into paragraph-sized pieces (~max_chars each) so
    every piece is small enough to hand an LLM in full, with no truncation
    guesswork about where the relevant sentence might land. Splits on blank
    lines first; if a single paragraph is still too long, splits at sentence
    boundaries (never mid-sentence) so a fact never gets orphaned across a
    chunk boundary the way whole-document truncation used to.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text] if text.strip() else []

    def split_long_paragraph(p: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", p)
        pieces, cur = [], ""
        for s in sentences:
            if len(s) > max_chars:
                # a single "sentence" is itself too long (e.g. no punctuation) —
                # only now fall back to a hard character split
                if cur:
                    pieces.append(cur)
                    cur = ""
                for start in range(0, len(s), max_chars):
                    pieces.append(s[start:start + max_chars])
                continue
            if len(cur) + len(s) + 1 <= max_chars:
                cur = (cur + " " + s).strip()
            else:
                if cur:
                    pieces.append(cur)
                cur = s
        if cur:
            pieces.append(cur)
        return pieces

    subchunks, current = [], ""
    for p in paragraphs:
        if len(p) > max_chars:
            if current:
                subchunks.append(current)
                current = ""
            subchunks.extend(split_long_paragraph(p))
            continue
        if len(current) + len(p) + 1 <= max_chars:
            current = (current + "\n" + p).strip()
        else:
            if current:
                subchunks.append(current)
            current = p
    if current:
        subchunks.append(current)
    return subchunks


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
                    for i, sub in enumerate(_split_into_subchunks(obj["text"])):
                        chunks.append(Chunk(id=f"{obj['id']}#{i}", source_id=obj["id"], text=sub))
            return chunks

        chunks = []
        if os.path.isdir(self.chunks_dir):
            for fname in sorted(os.listdir(self.chunks_dir)):
                if fname.lower().endswith((".md", ".txt")):
                    path = os.path.join(self.chunks_dir, fname)
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    doc_id = os.path.splitext(fname)[0]
                    for i, sub in enumerate(_split_into_subchunks(text)):
                        chunks.append(Chunk(id=f"{doc_id}#{i}", source_id=doc_id, text=sub))
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
