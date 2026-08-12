import math
import re
from typing import Any, Dict, List
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    source_id: str
    text: str
    embedding: List[float] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    chunk: DocumentChunk
    score: float
    provenance: str


class SimpleVectorStore:
    def __init__(self):
        self.chunks: List[DocumentChunk] = []

    def add_chunks(self, chunks: List[DocumentChunk]):
        self.chunks.extend(chunks)

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[RetrievalResult]:
        results = []
        for chunk in self.chunks:
            if not chunk.embedding:
                continue
            score = self._cosine_similarity(query_embedding, chunk.embedding)
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    provenance=f"doc:{chunk.source_id}#chunk:{chunk.chunk_id}"
                )
            )
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


class LMLMRAGPipeline:
    def __init__(self, vector_store: SimpleVectorStore):
        self.store = vector_store

    def ingest(self, source_id: str, text: str):
        sentences = re.split(r'(?<=[.?!])\s+', text)
        chunks = []
        for idx, sentence in enumerate(sentences):
            if sentence.strip():
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"c_{idx}",
                        source_id=source_id,
                        text=sentence,
                        embedding=self._mock_embed(sentence)
                    )
                )
        self.store.add_chunks(chunks)

    def retrieve_context(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        return self.store.search(self._mock_embed(query), top_k=top_k)

    @staticmethod
    def _mock_embed(text: str) -> List[float]:
        val = sum(ord(c) for c in text[:10]) % 100 / 100.0
        return [val, 1.0 - val, (val * 2) % 1.0]
