"""Hybrid retriever: BM25 (sparse) + semantic (dense) with RRF fusion over Qdrant."""

import asyncio
import logging
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from app.config import settings
from app.rag.embedder import embed_single
from app.rag.sparse_encoder import encode_sparse_query

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict


class HybridRetriever:
    def __init__(self):
        self._client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Hybrid search: dense + sparse with RRF fusion."""
        top_k = top_k or settings.rag_final_top_k

        query_vector = embed_single(query)
        sparse_vector = encode_sparse_query(query)

        results = self._client.query_points(
            collection_name=settings.qdrant_collection,
            prefetch=[
                models.Prefetch(
                    query=query_vector,
                    using="dense",
                    limit=settings.rag_semantic_top_k,
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=settings.rag_bm25_top_k,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
        )

        output = []
        for point in results.points:
            payload = point.payload or {}
            output.append(SearchResult(
                text=payload.get("text", ""),
                score=point.score,
                metadata={
                    k: v for k, v in payload.items() if k != "text"
                },
            ))
        return output

    async def asearch(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Async wrapper for hybrid search."""
        return await asyncio.to_thread(self.search, query, top_k)

    def search_simple(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Semantic-only search (fallback)."""
        top_k = top_k or settings.rag_final_top_k
        query_vector = embed_single(query)

        results = self._client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            using="dense",
            limit=top_k,
            score_threshold=settings.rag_score_threshold,
        )

        output = []
        for point in results.points:
            payload = point.payload or {}
            output.append(SearchResult(
                text=payload.get("text", ""),
                score=point.score,
                metadata={
                    k: v for k, v in payload.items() if k != "text"
                },
            ))
        return output

    def collection_exists(self) -> bool:
        try:
            self._client.get_collection(settings.qdrant_collection)
            return True
        except Exception:
            return False
