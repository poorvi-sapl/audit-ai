"""
engineering_benchmark/smoke_test.py
=====================================
Mandatory smoke test for Phase 3 infrastructure.

Must pass before any larger embedding run.

What it tests
-------------
1. Qdrant reachable and auditai_sop collection exists
2. Postgres reachable and sop_chunks table exists
3. Embedding model loads and produces correct vector dimensions
4. Full round-trip: chunk → embed → upsert → query → assert top-5 hit

Run with:
    pytest engineering_benchmark/smoke_test.py -v

Or as a script:
    python engineering_benchmark/smoke_test.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known test chunk — deterministic, always findable
# ---------------------------------------------------------------------------

_TEST_CHUNK_TEXT = (
    "SOP §3.1 — Bank Reconciliation: "
    "The auditor must verify that all bank reconciliations are completed "
    "within 30 days of the period end. Outstanding items must be documented "
    "and resolved prior to the issuance of the audit report."
)

_TEST_SOURCE_DOC = "smoke_test_sop.txt"
_TEST_SOP_VERSION = "smoke-test-v1"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qdrant_client():
    from qdrant_client import QdrantClient
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port, timeout=10)


@pytest.fixture(scope="module")
def settings():
    from config.settings import get_settings
    return get_settings()


@pytest.fixture(scope="module")
def pg_conn():
    """Optional Postgres connection — skips Postgres tests if unavailable."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "auditai"),
            user=os.getenv("POSTGRES_USER", "auditai"),
            password=os.getenv("POSTGRES_PASSWORD", "auditai_dev"),
            connect_timeout=5,
        )
        yield conn
        conn.close()
    except Exception as e:
        logger.warning("Postgres not available for smoke test: %s", e)
        yield None


# ---------------------------------------------------------------------------
# Test 1 — Qdrant reachable and collection exists
# ---------------------------------------------------------------------------

def test_qdrant_reachable(qdrant_client, settings):
    """Qdrant is reachable and responds to get_collections()."""
    result = qdrant_client.get_collections()
    assert result is not None


def test_qdrant_collection_exists(qdrant_client, settings):
    """auditai_sop collection exists in Qdrant."""
    collections = [c.name for c in qdrant_client.get_collections().collections]
    assert settings.qdrant.collection_sop in collections, (
        f"Collection '{settings.qdrant.collection_sop}' not found. "
        "Run: python engineering_benchmark/qdrant_setup.py"
    )


def test_qdrant_collection_dimensions(qdrant_client, settings):
    """Collection vector dimensions match stack.yaml embedding.dimensions."""
    info = qdrant_client.get_collection(settings.qdrant.collection_sop)
    actual_dims = info.config.params.vectors.size
    expected_dims = settings.embedding.dimensions
    assert actual_dims == expected_dims, (
        f"Collection has {actual_dims} dims but stack.yaml specifies {expected_dims}. "
        "Delete and recreate the collection: qdrant_setup.py"
    )


def test_qdrant_payload_indexes_exist(qdrant_client, settings):
    """Required payload indexes exist on the collection."""
    info = qdrant_client.get_collection(settings.qdrant.collection_sop)
    indexed_fields = set(info.payload_schema.keys()) if info.payload_schema else set()
    expected_fields = {idx.field for idx in settings.qdrant.payload_indexes}
    missing = expected_fields - indexed_fields
    assert not missing, (
        f"Missing payload indexes: {missing}. "
        "Run: python engineering_benchmark/qdrant_setup.py"
    )


# ---------------------------------------------------------------------------
# Test 2 — Postgres reachable and schema exists
# ---------------------------------------------------------------------------

def test_postgres_sop_chunks_table_exists(pg_conn):
    """sop_chunks table exists in Postgres."""
    if pg_conn is None:
        pytest.skip("Postgres not available")

    with pg_conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'sop_chunks'
            );
        """)
        exists = cur.fetchone()[0]
    assert exists, (
        "sop_chunks table not found. "
        "Run: docker exec -i postgres psql -U auditai -d auditai "
        "< engineering_benchmark/postgres_schema.sql"
    )


def test_postgres_all_tables_exist(pg_conn):
    """All four schema tables exist in Postgres."""
    if pg_conn is None:
        pytest.skip("Postgres not available")

    required_tables = ["engagements", "workpapers", "sop_chunks", "findings"]
    with pg_conn.cursor() as cur:
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public';
        """)
        existing = {row[0] for row in cur.fetchall()}

    missing = [t for t in required_tables if t not in existing]
    assert not missing, (
        f"Missing tables: {missing}. "
        "Run the postgres_schema.sql migration."
    )


# ---------------------------------------------------------------------------
# Test 3 — Embedding model loads and produces correct dimensions
# ---------------------------------------------------------------------------

def test_embedding_model_loads(settings):
    """Embedding model loads without error."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        pytest.skip("sentence-transformers not installed — skipping model load test")

    try:
        model = SentenceTransformer(
            settings.embedding.model,
            device=settings.embedding.device,
        )
        assert model is not None
    except Exception as e:
        pytest.fail(f"Failed to load embedding model: {e}")


def test_embedding_produces_correct_dimensions(settings):
    """Embedding a test sentence produces a vector of the correct size."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        pytest.skip("sentence-transformers not installed")

    try:
        model = SentenceTransformer(
            settings.embedding.model,
            device=settings.embedding.device,
        )
        vector = model.encode(
            ["Test sentence for dimension check."],
            normalize_embeddings=True,
        )
        assert len(vector[0]) == settings.embedding.dimensions, (
            f"Model produced {len(vector[0])} dims, "
            f"expected {settings.embedding.dimensions} from stack.yaml"
        )
    except Exception as e:
        pytest.fail(f"Embedding dimension check failed: {e}")


# ---------------------------------------------------------------------------
# Test 4 — Full round-trip: chunk → embed → upsert → query → top-5 hit
# ---------------------------------------------------------------------------

def test_full_roundtrip_chunk_embed_query(qdrant_client, settings, pg_conn):
    """
    The critical smoke test.
    Upsert a known chunk, query for it, assert it appears in top-5.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        pytest.skip("sentence-transformers not installed — skipping round-trip test")

    from engineering_benchmark.sop_chunker import SOPChunk, _make_chunk_id
    from engineering_benchmark.embedder import embed_chunks

    # Build a known chunk
    chunk_id = _make_chunk_id(_TEST_SOURCE_DOC, 0)
    test_chunk = SOPChunk(
        chunk_id=chunk_id,
        source_doc=_TEST_SOURCE_DOC,
        sop_version=_TEST_SOP_VERSION,
        content=_TEST_CHUNK_TEXT,
        section_prefix="SOP §3.1 — Bank Reconciliation: ",
        char_start=0,
        char_end=len(_TEST_CHUNK_TEXT),
        token_count=50,
        workpaper_type="bank_reconciliation",
        is_table=False,
        metadata={
            "source_doc":    _TEST_SOURCE_DOC,
            "sop_version":   _TEST_SOP_VERSION,
            "workpaper_type": "bank_reconciliation",
            "is_rollforward": False,
        },
    )

    # Embed and upsert
    result = embed_chunks([test_chunk], pg_conn=pg_conn)
    assert result.upserted >= 1, (
        f"Upsert failed: {result.errors}"
    )

    # Query with a semantically similar question
    query_text = "bank reconciliation completion requirements audit period"
    model = SentenceTransformer(
        settings.embedding.model,
        device=settings.embedding.device,
    )
    query_vector = model.encode(
        [query_text], normalize_embeddings=True
    )[0].tolist()

    search_results = qdrant_client.query_points(
        collection_name=settings.qdrant.collection_sop,
        query=query_vector,
        limit=5,
    ).points

    assert len(search_results) > 0, "Qdrant search returned no results"

    result_ids = [str(r.id) for r in search_results]
    import uuid
    chunk_uuid = str(uuid.UUID(chunk_id[:32]))
    assert chunk_uuid in result_ids, (
        f"Known chunk '{chunk_id[:16]}...' not in top-5 results: {result_ids}"
    )

    logger.info(
        "smoke_test: round-trip passed — chunk found at rank %d",
        result_ids.index(chunk_uuid) + 1,
    )


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(pytest.main([__file__, "-v"]))