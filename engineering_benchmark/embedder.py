"""
engineering_benchmark/embedder.py
===================================
Embeds SOPChunk objects and upserts them into Qdrant.

Reads all configuration from config/settings.py — no hardcoded values.

What it does
------------
1. Loads e5-mistral-7b-instruct (or configured model) on GPU/CPU
2. Embeds chunks in batches of embedding.embed_batch_size
3. Upserts to Qdrant in batches of embedding.qdrant_upsert_batch
4. Writes chunk metadata to Postgres sop_chunks table
5. Returns an EmbedResult with counts and any errors

Model note
----------
e5-mistral-7b-instruct requires a query prefix for retrieval:
    "Instruct: Retrieve relevant SOP sections\nQuery: {text}"
For document embedding (what we do here) no prefix is needed.

Public API
----------
    embed_chunks(chunks, pg_conn)  -> EmbedResult
    EmbedResult                    — dataclass with counts and errors
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from engineering_benchmark.sop_chunker import SOPChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EmbedResult
# ---------------------------------------------------------------------------

@dataclass
class EmbedResult:
    """Result of an embed_chunks() call."""

    total:    int = 0
    upserted: int = 0
    skipped:  int = 0
    errors:   list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"EmbedResult: total={self.total} upserted={self.upserted} "
            f"skipped={self.skipped} errors={len(self.errors)}"
        )


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None


def _load_model():
    """
    Load the embedding model and tokenizer. Cached after first call.
    Uses settings.embedding.model and settings.embedding.device.
    """
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    settings = get_settings()
    model_name = settings.embedding.model
    device = settings.embedding.device

    logger.info("embedder: loading model '%s' on device '%s'", model_name, device)

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device=device)
        _model = model
        _tokenizer = None  # SentenceTransformer handles tokenization internally
        logger.info("embedder: model loaded via SentenceTransformer")
        return _model, _tokenizer

    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. "
            "Install with: pip install -e '.[training]'"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model '{model_name}': {e}") from e


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

# e5-mistral-7b-instruct is an asymmetric retrieval model: documents are
# embedded raw, but queries MUST be prefixed with a task instruction.
# Using the wrong (or no) prefix at query time measurably degrades recall.
# See: https://huggingface.co/intfloat/e5-mistral-7b-instruct
E5_QUERY_INSTRUCTION: str = (
    "Instruct: Retrieve relevant SOP sections for an audit workpaper task.\n"
    "Query: "
)


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts. Returns list of float vectors.
    Vector length matches settings.embedding.dimensions.
    """
    model, _ = _load_model()
    settings = get_settings()

    try:
        vectors = model.encode(
            texts,
            batch_size=settings.embedding.embed_batch_size,
            normalize_embeddings=True,   # cosine similarity requires normalized vectors
            show_progress_bar=False,
        )
        return vectors.tolist()
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}") from e


def embed_query(query_text: str) -> list[float]:
    """Embed a retrieval query with the e5-mistral instruction prefix.

    Documents and queries embed differently for e5-mistral — documents
    use no prefix, queries must be prefixed with E5_QUERY_INSTRUCTION.
    Always use this function (NOT _embed_batch) to embed retrieval
    queries; otherwise vector similarity is computed in mismatched
    spaces and recall drops noticeably.

    Returns a single vector matching settings.embedding.dimensions.
    """
    if not query_text or not query_text.strip():
        raise ValueError("embed_query: query_text is empty")
    prefixed = E5_QUERY_INSTRUCTION + query_text.strip()
    vectors = _embed_batch([prefixed])
    return vectors[0]


# ---------------------------------------------------------------------------
# Qdrant upsert
# ---------------------------------------------------------------------------

def _upsert_to_qdrant(
    chunks: list[SOPChunk],
    vectors: list[list[float]],
) -> int:
    """
    Upsert chunk vectors and payloads to Qdrant.
    Returns number of points upserted.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    settings = get_settings()
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    client = QdrantClient(host=host, port=port, timeout=30)

    collection = settings.qdrant.collection_sop
    upsert_batch = settings.embedding.qdrant_upsert_batch
    total_upserted = 0

    # Create collection if it doesn't exist yet
    # (happens after a wipe or on first run)
    from qdrant_client.models import Distance, VectorParams
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        dimensions = settings.embedding.dimensions
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE),
        )
        logger.info("embedder: created Qdrant collection '%s' (dim=%d)", collection, dimensions)

    # Build points
    points = []
    for chunk, vector in zip(chunks, vectors):
        payload = {
            "source_doc":     chunk.source_doc,
            "sop_version":    chunk.sop_version,
            "section_prefix": chunk.section_prefix,
            "content":        chunk.content,
            "char_start":     chunk.char_start,
            "char_end":       chunk.char_end,
            "token_count":    chunk.token_count,
            "workpaper_type": chunk.workpaper_type,
            "workpaper_ids":  chunk.workpaper_ids,   # specific workpaper IDs (Option C)
            "is_rollforward": chunk.metadata.get("is_rollforward", False),
            "is_table":       chunk.is_table,
            "chunks_hash":    chunk.chunk_id,
        }
        import uuid
        # Convert sha256 hex to UUID by taking first 32 chars
        point_uuid = str(uuid.UUID(chunk.chunk_id[:32]))
        points.append(PointStruct(
            id=point_uuid,
            vector=vector,
            payload=payload,
        ))

    # Upsert in batches
    for i in range(0, len(points), upsert_batch):
        batch = points[i: i + upsert_batch]
        client.upsert(collection_name=collection, points=batch)
        total_upserted += len(batch)
        logger.debug(
            "embedder: upserted batch %d-%d to Qdrant",
            i, i + len(batch),
        )

    return total_upserted


# ---------------------------------------------------------------------------
# Postgres insert
# ---------------------------------------------------------------------------

def _insert_to_postgres(chunks: list[SOPChunk], pg_conn) -> None:
    """
    Insert chunk metadata into the sop_chunks Postgres table.
    Skips chunks whose qdrant_point_id already exists (ON CONFLICT DO NOTHING).

    Parameters
    ----------
    chunks : list[SOPChunk]
    pg_conn : psycopg2 connection
        Caller provides the connection — embedder does not manage it.
    """
    if pg_conn is None:
        logger.debug("embedder: no pg_conn provided — skipping Postgres insert")
        return

    insert_sql = """
        INSERT INTO sop_chunks (
            qdrant_point_id, source_doc, sop_version, chunks_version,
            chunks_hash, section_prefix, content,
            char_start, char_end, token_count,
            workpaper_type, is_rollforward, embedded_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
        )
        ON CONFLICT (qdrant_point_id) DO NOTHING;
    """

    with pg_conn.cursor() as cur:
        for chunk in chunks:
            cur.execute(insert_sql, (
                chunk.chunk_id,
                chunk.source_doc,
                chunk.sop_version,
                1,                                          # chunks_version
                chunk.chunk_id,                             # chunks_hash = chunk_id
                chunk.section_prefix,
                chunk.content,
                chunk.char_start,
                chunk.char_end,
                chunk.token_count,
                chunk.workpaper_type or None,
                chunk.metadata.get("is_rollforward", False),
            ))
    pg_conn.commit()
    logger.debug("embedder: inserted %d rows into sop_chunks", len(chunks))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_chunks(
    chunks: list[SOPChunk],
    pg_conn=None,
) -> EmbedResult:
    """
    Embed SOPChunk objects and upsert into Qdrant.

    Parameters
    ----------
    chunks : list[SOPChunk]
        Output of sop_chunker.chunk_text() or chunk_file().
    pg_conn : psycopg2 connection | None
        If provided, chunk metadata is also written to Postgres sop_chunks.
        If None, only Qdrant upsert happens.

    Returns
    -------
    EmbedResult
        .upserted — number of points written to Qdrant
        .skipped  — empty/invalid chunks skipped
        .errors   — list of error messages
    """
    settings = get_settings()
    result = EmbedResult(total=len(chunks))

    if not chunks:
        logger.warning("embedder: no chunks to embed")
        return result

    embed_batch_size = settings.embedding.embed_batch_size

    # Filter out empty chunks
    valid_chunks = [c for c in chunks if c.content.strip()]
    result.skipped = len(chunks) - len(valid_chunks)

    if not valid_chunks:
        logger.warning("embedder: all chunks are empty after filtering")
        return result

    logger.info(
        "embedder: embedding %d chunks (batch_size=%d)",
        len(valid_chunks), embed_batch_size,
    )

    # Embed in batches
    all_vectors: list[list[float]] = []

    for i in range(0, len(valid_chunks), embed_batch_size):
        batch = valid_chunks[i: i + embed_batch_size]
        texts = [c.content for c in batch]

        try:
            vectors = _embed_batch(texts)
            all_vectors.extend(vectors)
            logger.debug(
                "embedder: embedded batch %d-%d", i, i + len(batch)
            )
        except Exception as e:
            error_msg = f"Embed batch {i}-{i + len(batch)} failed: {e}"
            logger.error("embedder: %s", error_msg)
            result.errors.append(error_msg)
            # Fill with None placeholders to keep alignment
            all_vectors.extend([None] * len(batch))

    # Filter out failed embeddings
    paired = [
        (chunk, vec)
        for chunk, vec in zip(valid_chunks, all_vectors)
        if vec is not None
    ]

    if not paired:
        logger.error("embedder: all embedding batches failed")
        return result

    upsert_chunks, upsert_vectors = zip(*paired)

    # Upsert to Qdrant
    try:
        upserted = _upsert_to_qdrant(list(upsert_chunks), list(upsert_vectors))
        result.upserted = upserted
    except Exception as e:
        error_msg = f"Qdrant upsert failed: {e}"
        logger.error("embedder: %s", error_msg)
        result.errors.append(error_msg)
        return result

    # Write to Postgres
    try:
        _insert_to_postgres(list(upsert_chunks), pg_conn)
    except Exception as e:
        error_msg = f"Postgres insert failed: {e}"
        logger.error("embedder: %s", error_msg)
        result.errors.append(error_msg)

    logger.info("embedder: %s", result)
    return result