"""RAG search tool for CrewAI agents — hybrid search over Qdrant knowledge base."""

import logging

from crewai.tools import tool

logger = logging.getLogger(__name__)

_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from app.rag.retriever import HybridRetriever
        _retriever = HybridRetriever()
    return _retriever


@tool("knowledge_base_search")
def knowledge_base_search(query: str) -> str:
    """Search the local knowledge base of scientific publications.
    Uses hybrid search (BM25 + semantic) for accurate retrieval.
    Returns relevant text passages from indexed documents.

    Args:
        query: Search query — a question or topic to find information about.
    """
    retriever = _get_retriever()

    if not retriever.collection_exists():
        return "Knowledge base is empty. No documents have been indexed yet."

    try:
        results = retriever.search(query)
    except Exception as e:
        logger.error("RAG search failed: %s", e)
        return f"Knowledge base search failed: {e}"

    if not results:
        return "No relevant passages found in the knowledge base."

    output = []
    for i, r in enumerate(results, 1):
        source = r.metadata.get("filename", "unknown")
        chunk_idx = r.metadata.get("chunk_index", "?")
        output.append(
            f"[Source {i}: {source}, chunk {chunk_idx}, score={r.score:.3f}]\n{r.text}"
        )

    return "\n\n---\n\n".join(output)
