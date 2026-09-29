from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import settings
from app.logger import setup_logger
import uuid

logger = setup_logger(__name__)

_embedder = None


def get_embedder() -> HuggingFaceEmbeddings:
    """Ленивая инициализация локальной embedding-модели."""
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _embedder = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedder


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        check_compatibility=False,
    )


def ensure_collection(client: QdrantClient):
    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in collections:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Created collection: {settings.qdrant_collection}")
    else:
        logger.info(f"Collection {settings.qdrant_collection} already exists")


def index_chunks(chunks: List[str], source: str):
    client = get_qdrant_client()
    embedder = get_embedder()
    ensure_collection(client)

    points = []
    for i, chunk in enumerate(chunks):
        vector = embedder.embed_query(chunk)
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"content": chunk, "source": source, "chunk_index": i},
            )
        )

    client.upsert(collection_name=settings.qdrant_collection, points=points)
    logger.info(f"Indexed {len(points)} chunks from {source}")
    return len(points)