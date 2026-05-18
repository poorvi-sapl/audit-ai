"""
engineering_benchmark/qdrant_setup.py
=======================================
Creates the Qdrant vector collection for AuditAI SOP chunks.

Reads all configuration from config/settings.py — no hardcoded values.

What it does
------------
1. Connects to Qdrant (localhost:6333 by default, or QDRANT_HOST/PORT env vars)
2. Creates collection `auditai_sop` if it does not exist
3. Creates payload indexes BEFORE any upsert (required by Qdrant for filtering)
4. Verifies the collection is ready with a health check

WARNING: embedding.dimensions in stack.yaml is IMMUTABLE after the first
collection is created. Changing it requires deleting the collection and
re-embedding everything.

Run once before any embedding:
    python engineering_benchmark/qdrant_setup.py

Or import and call setup() from other modules:
    from engineering_benchmark.qdrant_setup import setup
    setup()

Public API
----------
    setup()               -> None   create collection + indexes if not exists
    collection_exists()   -> bool
    get_client()          -> QdrantClient
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PayloadSchemaType,
)

from config.settings import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Distance mapping — stack.yaml string → Qdrant Distance enum
# ---------------------------------------------------------------------------

_DISTANCE_MAP = {
    "Cosine": Distance.COSINE,
    "Dot":    Distance.DOT,
    "Euclid": Distance.EUCLID,
}

# ---------------------------------------------------------------------------
# Payload type mapping — stack.yaml string → Qdrant PayloadSchemaType
# ---------------------------------------------------------------------------

_PAYLOAD_TYPE_MAP = {
    "keyword": PayloadSchemaType.KEYWORD,
    "integer": PayloadSchemaType.INTEGER,
    "bool":    PayloadSchemaType.BOOL,
    "float":   PayloadSchemaType.FLOAT,
    "geo":     PayloadSchemaType.GEO,
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def get_client() -> QdrantClient:
    """
    Return a Qdrant client using QDRANT_HOST / QDRANT_PORT env vars,
    falling back to localhost:6333.
    """
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port, timeout=10)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collection_exists(client: QdrantClient | None = None) -> bool:
    """Return True if the auditai_sop collection already exists."""
    settings = get_settings()
    c = client or get_client()
    existing = [col.name for col in c.get_collections().collections]
    return settings.qdrant.collection_sop in existing


def setup(client: QdrantClient | None = None) -> None:
    """
    Create the auditai_sop Qdrant collection and payload indexes.

    Idempotent — safe to call multiple times. Skips creation if the
    collection already exists.

    Parameters
    ----------
    client : QdrantClient | None
        Optional pre-built client. Creates one from env vars if None.

    Raises
    ------
    Exception
        If Qdrant is unreachable or collection creation fails.
    """
    settings = get_settings()
    c = client or get_client()

    collection_name = settings.qdrant.collection_sop
    dimensions = settings.embedding.dimensions
    distance_str = settings.embedding.distance
    distance = _DISTANCE_MAP.get(distance_str)

    if distance is None:
        raise ValueError(
            f"Unknown distance metric '{distance_str}' in stack.yaml. "
            f"Must be one of: {list(_DISTANCE_MAP.keys())}"
        )

    # ------------------------------------------------------------------
    # 1. Create collection if it does not exist
    # ------------------------------------------------------------------
    if collection_exists(c):
        logger.info(
            "qdrant_setup: collection '%s' already exists — skipping creation",
            collection_name,
        )
    else:
        logger.info(
            "qdrant_setup: creating collection '%s' "
            "(dims=%d, distance=%s, on_disk=True)",
            collection_name, dimensions, distance_str,
        )
        c.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dimensions,
                distance=distance,
                on_disk=True,    # store vectors on disk — required for large collections
            ),
        )
        logger.info("qdrant_setup: collection '%s' created", collection_name)

    # ------------------------------------------------------------------
    # 2. Create payload indexes BEFORE first upsert
    # ------------------------------------------------------------------
    for index_def in settings.qdrant.payload_indexes:
        field = index_def.field
        type_str = index_def.type
        payload_type = _PAYLOAD_TYPE_MAP.get(type_str)

        if payload_type is None:
            logger.warning(
                "qdrant_setup: unknown payload type '%s' for field '%s' — skipping",
                type_str, field,
            )
            continue

        try:
            c.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=payload_type,
            )
            logger.info(
                "qdrant_setup: payload index created — field='%s' type='%s'",
                field, type_str,
            )
        except Exception as e:
            # Index may already exist — log and continue
            logger.debug(
                "qdrant_setup: payload index '%s' may already exist: %s",
                field, e,
            )

    # ------------------------------------------------------------------
    # 3. Verify collection is ready
    # ------------------------------------------------------------------
    info = c.get_collection(collection_name)
    vectors_count = getattr(info, "vectors_count", None) or getattr(info, "points_count", 0)
    logger.info(
        "qdrant_setup: collection '%s' ready — "
        "vectors_count=%s status=%s",
        collection_name,
        vectors_count,
        info.status,
    )
    print(
        f"✓ Qdrant collection '{collection_name}' ready | "
        f"dims={dimensions} | distance={distance_str} | "
        f"vectors={vectors_count}"
    )


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup()