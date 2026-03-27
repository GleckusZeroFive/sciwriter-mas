"""Document indexer: reads text files, chunks them, and uploads to Qdrant."""

import logging
import uuid
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

from app.config import settings
from app.rag.embedder import embed_documents
from app.rag.sparse_encoder import encode_sparse

logger = logging.getLogger(__name__)


def _create_collection(client: QdrantClient) -> None:
    """Create Qdrant collection with dense + sparse vectors."""
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={
            "dense": models.VectorParams(
                size=settings.embedding_dim,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            ),
        },
    )
    logger.info("Created collection: %s", settings.qdrant_collection)


def _read_text_file(path: Path) -> str:
    """Read a text file with encoding fallback."""
    for encoding in ("utf-8", "cp1251", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, ValueError):
            continue
    raise ValueError(f"Cannot read {path} with any known encoding")


def index_directory(directory: str | Path, recreate: bool = False) -> int:
    """Index all .txt and .md files from a directory into Qdrant.

    Returns number of chunks indexed.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    # Recreate collection if requested
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection in existing:
        if recreate:
            client.delete_collection(settings.qdrant_collection)
            logger.info("Deleted existing collection: %s", settings.qdrant_collection)
        else:
            logger.info("Collection exists, appending documents")

    if settings.qdrant_collection not in [
        c.name for c in client.get_collections().collections
    ]:
        _create_collection(client)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    files = sorted(directory.glob("*.txt")) + sorted(directory.glob("*.md"))
    if not files:
        logger.warning("No .txt or .md files found in %s", directory)
        return 0

    total_chunks = 0

    for filepath in files:
        logger.info("Indexing: %s", filepath.name)
        text = _read_text_file(filepath)
        chunks = splitter.split_text(text)

        if not chunks:
            continue

        # Embed
        dense_vectors = embed_documents(chunks)
        sparse_vectors = [encode_sparse(chunk) for chunk in chunks]

        # Upload
        points = []
        for i, (chunk, dense, sparse) in enumerate(
            zip(chunks, dense_vectors, sparse_vectors)
        ):
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense,
                    "sparse": sparse,
                },
                payload={
                    "text": chunk,
                    "filename": filepath.name,
                    "chunk_index": i,
                },
            ))

        client.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
        )
        logger.info("  -> %d chunks indexed from %s", len(chunks), filepath.name)
        total_chunks += len(chunks)

    logger.info("Total: %d chunks from %d files", total_chunks, len(files))
    return total_chunks


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    path = sys.argv[1] if len(sys.argv) > 1 else str(settings.data_dir / "knowledge_base")
    recreate = "--recreate" in sys.argv
    count = index_directory(path, recreate=recreate)
    print(f"Indexed {count} chunks")
