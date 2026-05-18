"""
pipeline/qdrant_retriever.py
==============================
Hybrid SOP retrieval for Phase 4.

Strategy
--------
Primary (Option C — highest precision):
    - Qdrant payload filter: workpaper_type == X
    - Semantic query: workpaper_type + client context string
    - Uses Phase 3 payload indexes directly

Fallback (Option B — when workpaper_type missing):
    - No payload filter
    - Structured query: client_type + audit_type + fiscal_year_end + keywords

Both paths return the same output shape so pipeline.py never
needs to know which strategy ran.

Public API
----------
    retrieve(record, top_k) -> RetrievalResult
        Retrieve relevant SOP chunks for a DocumentRecord.

    RetrievalResult
        .sop_text      str        — joined chunk text, ready for completion_drafter
        .sop_sections  list[str]  — section prefixes found e.g. ["§3.1", "§4.2"]
        .chunks        list[dict] — raw Qdrant payloads for traceability
        .strategy      str        — "filtered" | "fallback"
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from auditai_data_normalization.schema import DocumentRecord

logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Output of retrieve(). Ready to pass directly into pair_builder.build()."""

    sop_text: str
    """Joined SOP chunk content. Pass as sop_text to pair_builder.build()."""

    sop_sections: list[str] = field(default_factory=list)
    """Section prefixes found e.g. ['§3.1', '§4.2']. Pass as sop_sections."""

    chunks: list[dict] = field(default_factory=list)
    """Raw Qdrant payloads — for audit trail and Postgres linkage."""

    strategy: str = "filtered"
    """Which retrieval strategy ran: 'filtered' or 'fallback'."""

    def __str__(self) -> str:
        return (
            f"RetrievalResult: strategy={self.strategy} "
            f"chunks={len(self.chunks)} "
            f"sections={self.sop_sections}"
        )


# ---------------------------------------------------------------------------
# Qdrant client
# ---------------------------------------------------------------------------

def _get_client():
    from qdrant_client import QdrantClient
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port, timeout=10)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is not None:
        return _embed_model

    settings = get_settings()
    try:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(
            settings.embedding.model,
            device=settings.embedding.device,
        )
        logger.info(
            "qdrant_retriever: loaded embedding model '%s'",
            settings.embedding.model,
        )
        return _embed_model
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. "
            "Run: pip install -e '.[training]'"
        )


def _embed_query(query_text: str) -> list[float]:
    """Embed a query string. Uses e5-mistral query prefix for retrieval."""
    model = _get_embed_model()
    # e5-mistral-7b-instruct uses this prefix for retrieval queries
    prefixed = f"Instruct: Retrieve relevant audit SOP sections\nQuery: {query_text}"
    vector = model.encode(
        [prefixed],
        normalize_embeddings=True,
    )[0].tolist()
    return vector


# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------

def _build_filtered_query(record: DocumentRecord) -> tuple[str, str]:
    """
    Build query string and filter value for Option C (filtered retrieval).

    Returns (query_string, workpaper_type_filter)
    """
    wp_type = record.metadata.get("workpaper_type", "") or ""

    # Build context-aware query seed
    confidence_summary = record.metadata.get("confidence_summary", {})
    fields_present = confidence_summary.get("fields_present", [])

    context_parts = [wp_type.replace("_", " ")]

    # Add client type if available
    if "client_type" in fields_present:
        context_parts.append("nonprofit audit" if "NPO" in record.cleaned_text else "audit")

    if "has_single_audit" in fields_present:
        context_parts.append("single audit uniform guidance")

    if "is_gagas" in fields_present:
        context_parts.append("government auditing standards yellow book")

    query = " ".join(p for p in context_parts if p).strip()
    return query, wp_type


def _build_fallback_query(record: DocumentRecord) -> str:
    """
    Build structured query for Option B (fallback — no workpaper_type).

    Combines client_type + audit_type + fiscal_year_end + keywords
    from the record's cleaned_text excerpt.
    """
    parts = []

    # Extract key fields from confidence summary
    confidence_summary = record.metadata.get("confidence_summary", {})
    fields_present = set(confidence_summary.get("fields_present", []))

    if "client_type" in fields_present:
        parts.append("audit")
    if "includes_single_audit" in fields_present:
        parts.append("single audit 2 CFR 200")
    if "includes_gagas" in fields_present:
        parts.append("government auditing standards")
    if "fiscal_year_end" in fields_present:
        parts.append("fiscal year end procedures")

    # Add first 50 words of cleaned text as keyword context
    words = record.cleaned_text.split()[:50]
    parts.append(" ".join(words))

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Result builder
# ---------------------------------------------------------------------------

def _build_result(
    search_results,
    strategy: str,
) -> RetrievalResult:
    """Build a RetrievalResult from Qdrant search results."""
    chunks = []
    sop_sections = []

    for hit in search_results:
        payload = hit.payload or {}
        chunks.append(payload)

        # Extract section prefix e.g. "SOP §3.1 — Reconciliation: "
        prefix = payload.get("section_prefix", "")
        if prefix:
            # Extract just the section number e.g. "§3.1"
            import re
            match = re.search(r"§[\d.]+", prefix)
            if match and match.group(0) not in sop_sections:
                sop_sections.append(match.group(0))

    # Join chunk content with section separators
    sop_text_parts = []
    for chunk in chunks:
        content = chunk.get("content", "")
        if content:
            sop_text_parts.append(content)

    sop_text = "\n\n---\n\n".join(sop_text_parts)

    return RetrievalResult(
        sop_text=sop_text,
        sop_sections=sop_sections,
        chunks=chunks,
        strategy=strategy,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    record: DocumentRecord,
    top_k: int = _DEFAULT_TOP_K,
) -> RetrievalResult:
    """
    Retrieve relevant SOP chunks for a DocumentRecord.

    Uses hybrid strategy:
    - Primary: payload filter on workpaper_type + semantic query (Option C)
    - Fallback: structured query without filter (Option B)

    Parameters
    ----------
    record : DocumentRecord
        Normalized workpaper record from Phase 1.
        Must have cleaned_text and metadata populated.
    top_k : int
        Number of chunks to retrieve. Default 5.

    Returns
    -------
    RetrievalResult
        .sop_text      — pass directly to pair_builder.build()
        .sop_sections  — pass directly to pair_builder.build()
        .strategy      — which strategy was used
    """
    settings = get_settings()
    client = _get_client()
    collection = settings.qdrant.collection_sop

    wp_type = record.metadata.get("workpaper_type", "") or ""

    # ------------------------------------------------------------------
    # Primary — Option C: filtered + semantic
    # ------------------------------------------------------------------
    if wp_type:
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            query_str, filter_value = _build_filtered_query(record)
            query_vector = _embed_query(query_str)

            results = client.query_points(
                collection_name=collection,
                query=query_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="workpaper_type",
                            match=MatchValue(value=filter_value),
                        )
                    ]
                ),
                limit=top_k,
            ).points

            if results:
                logger.info(
                    "qdrant_retriever: filtered retrieval — "
                    "workpaper_type='%s' top_k=%d found=%d",
                    wp_type, top_k, len(results),
                )
                return _build_result(results, strategy="filtered")

            logger.info(
                "qdrant_retriever: filtered retrieval returned 0 results "
                "for workpaper_type='%s' — falling back to unfiltered",
                wp_type,
            )

        except Exception as e:
            logger.warning(
                "qdrant_retriever: filtered retrieval failed: %s — "
                "falling back to unfiltered query",
                e,
            )

    # ------------------------------------------------------------------
    # Fallback — Option B: structured query, no filter
    # ------------------------------------------------------------------
    try:
        query_str = _build_fallback_query(record)
        if not query_str.strip():
            query_str = record.cleaned_text.split()[:30]
            query_str = " ".join(query_str)

        query_vector = _embed_query(query_str)

        results = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
        ).points

        logger.info(
            "qdrant_retriever: fallback retrieval — found=%d",
            len(results),
        )

        if not results:
            logger.warning(
                "qdrant_retriever: no SOP chunks found — "
                "collection may be empty. Run the embedder first."
            )
            return RetrievalResult(
                sop_text="",
                sop_sections=[],
                chunks=[],
                strategy="fallback_empty",
            )

        return _build_result(results, strategy="fallback")

    except Exception as e:
        logger.error("qdrant_retriever: fallback retrieval failed: %s", e)
        return RetrievalResult(
            sop_text="",
            sop_sections=[],
            chunks=[],
            strategy="error",
        )

# ---------------------------------------------------------------------------
# SOP health check
# ---------------------------------------------------------------------------

@dataclass
class SopHealthResult:
    """Result of check_sop_health(). Shown in Streamlit sidebar."""
    healthy:           bool            = False
    collection_exists: bool            = False
    chunk_count:       int             = 0
    test_query_ok:     bool            = False
    issues:            list            = field(default_factory=list)

    def __str__(self) -> str:
        status = "OK" if self.healthy else "DEGRADED"
        return (
            f"SopHealth: {status} | collection={self.collection_exists} "
            f"chunks={self.chunk_count} test_query={self.test_query_ok} "
            f"issues={self.issues}"
        )


def check_sop_health(top_k: int = 3) -> SopHealthResult:
    """
    Verify that the SOP Qdrant collection is reachable and populated.

    Checks:
        1. Qdrant client connects without error
        2. SOP collection exists
        3. Collection has >= min_chunks chunks (from threshold_config)
        4. A test query returns non-empty results

    Returns SopHealthResult. Call from Streamlit sidebar before batch runs.
    """
    import yaml
    from pathlib import Path as _Path

    _thr_path = _Path(__file__).parent.parent / "auditai_data_normalization" / "alias_registry" / "threshold_config.yaml"
    try:
        with open(_thr_path) as _f:
            _cfg = yaml.safe_load(_f) or {}
        _min_chunks = int(_cfg.get("sop_retrieval", {}).get("min_chunks", 1))
    except Exception:
        _min_chunks = 1

    result = SopHealthResult(healthy=False)
    issues = []

    try:
        settings   = get_settings()
        client     = _get_client()
        collection = settings.qdrant_collection_name
    except Exception as e:
        issues.append(f"Qdrant client error: {e}")
        result.issues = issues
        return result

    # Check 1 — collection exists
    try:
        collections = [c.name for c in client.get_collections().collections]
        if collection not in collections:
            issues.append(f"Collection '{collection}' does not exist — run embedder first")
            result.collection_exists = False
            result.issues = issues
            return result
        result.collection_exists = True
    except Exception as e:
        issues.append(f"Could not list collections: {e}")
        result.issues = issues
        return result

    # Check 2 — chunk count
    try:
        info = client.get_collection(collection)
        count = info.points_count or 0
        result.chunk_count = count
        if count < _min_chunks:
            issues.append(
                f"Collection has {count} chunks (min={_min_chunks}) — "
                "SOP may not be embedded yet"
            )
    except Exception as e:
        issues.append(f"Could not get collection info: {e}")

    # Check 3 — test query returns results
    try:
        test_vector = _embed_query("engagement acceptance continuance audit")
        test_results = client.query_points(
            collection_name=collection,
            query=test_vector,
            limit=top_k,
        ).points
        result.test_query_ok = len(test_results) > 0
        if not result.test_query_ok:
            issues.append("Test query returned 0 results — collection may be empty or corrupted")
    except Exception as e:
        issues.append(f"Test query failed: {e}")
        result.test_query_ok = False

    result.healthy = (
        result.collection_exists
        and result.chunk_count >= _min_chunks
        and result.test_query_ok
        and len(issues) == 0
    )
    result.issues = issues

    logger.info("qdrant_retriever: health check — %s", result)
    return result