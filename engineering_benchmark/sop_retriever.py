"""
engineering_benchmark/sop_retriever.py
========================================
SOP retrieval for the workpaper generation pipeline.

Takes a workpaper type (e.g., "NPO-CX-1.1") and a query, embeds the
query with the e5-mistral instruction prefix, filters Qdrant by
workflow + optional workpaper_ids + optional sop_version, and returns
the top-k most relevant SOP chunk contents.

Workpaper-type vocabulary
-------------------------
Phase 1A field-type registry keys workpapers by specific ID
("NPO-CX-1.1"). The SOP chunker tags chunks by coarse workflow
("engagement_acceptance"). This module bridges them via
WORKPAPER_TYPE_TO_WORKFLOW.

A chunk matches a query iff:
    chunk.workpaper_type == workflow(query_workpaper_id)
    AND (
        chunk.workpaper_ids is empty
        OR chunk.workpaper_ids contains query_workpaper_id
    )

This implements Option C from the Phase 1B survey: workflow-level
chunks are the default; specific workpaper IDs narrow the match only
when set on a chunk.

Public API
----------
    retrieve_sop_chunks(workpaper_type, query, top_k=10,
                        sop_version=None) -> list[str]
    workflow_for(workpaper_type) -> str
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from engineering_benchmark.embedder import embed_query

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Workpaper ID → workflow lookup
# ---------------------------------------------------------------------
# Add entries here as new workpaper types are onboarded. The workflow
# names must match the keys in sop_chunker._WORKPAPER_TYPE_KEYWORDS so
# that chunker output and retriever queries share vocabulary.

WORKPAPER_TYPE_TO_WORKFLOW: dict[str, str] = {
    # Engagement acceptance — Phase 1A iteration 1
    "NPO-CX-1.1": "engagement_acceptance",
    "GOV-CX-1.1": "engagement_acceptance",
    "FP-CX-1.1":  "engagement_acceptance",
    "TRB-CX-1.1": "engagement_acceptance",
    # Add other workpaper IDs as iterations expand:
    # "BANK-REC-...": "bank_reconciliation",
    # "TB-...":       "trial_balance",
    # etc.
}


def workflow_for(workpaper_type: str) -> str:
    """Return the workflow name for a workpaper ID.

    Raises KeyError if the workpaper_type is not registered. New
    workpapers need an entry in WORKPAPER_TYPE_TO_WORKFLOW before
    retrieval will work for them.
    """
    if workpaper_type not in WORKPAPER_TYPE_TO_WORKFLOW:
        raise KeyError(
            f"Unknown workpaper_type {workpaper_type!r}. "
            f"Add to WORKPAPER_TYPE_TO_WORKFLOW in sop_retriever.py. "
            f"Known: {sorted(WORKPAPER_TYPE_TO_WORKFLOW)}"
        )
    return WORKPAPER_TYPE_TO_WORKFLOW[workpaper_type]


# ---------------------------------------------------------------------
# Qdrant client construction (testable via dependency injection)
# ---------------------------------------------------------------------

def _get_qdrant_client():
    """Return a Qdrant client from env vars. Imported lazily so tests
    that mock retrieve_sop_chunks at a higher layer don't need Qdrant
    installed."""
    import os
    from qdrant_client import QdrantClient
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port, timeout=15)


# ---------------------------------------------------------------------
# Filter construction
# ---------------------------------------------------------------------

def _build_filter(
    workflow: str,
    workpaper_id: str,
    sop_version: str | None = None,
):
    """Build the Qdrant Filter for workflow + workpaper_id + version.

    Logic:
        must: workpaper_type == workflow
        must: (workpaper_ids is empty) OR (workpaper_ids contains
               workpaper_id) — encoded via `should` clause
        must: sop_version == sop_version (if provided)
    """
    from qdrant_client.models import (
        FieldCondition, Filter, IsEmptyCondition, MatchAny, MatchValue,
        PayloadField,
    )

    must: list = [
        FieldCondition(
            key="workpaper_type",
            match=MatchValue(value=workflow),
        ),
    ]
    if sop_version:
        must.append(
            FieldCondition(
                key="sop_version",
                match=MatchValue(value=sop_version),
            )
        )

    # workpaper_ids is empty OR contains the workpaper_id
    should = [
        IsEmptyCondition(is_empty=PayloadField(key="workpaper_ids")),
        FieldCondition(
            key="workpaper_ids",
            match=MatchAny(any=[workpaper_id]),
        ),
    ]

    return Filter(must=must, should=should)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def retrieve_sop_chunks(
    workpaper_type: str,
    query: str,
    top_k: int = 10,
    sop_version: str | None = None,
    qdrant_client=None,
) -> list[str]:
    """Retrieve top-k relevant SOP chunk contents for a workpaper task.

    Parameters
    ----------
    workpaper_type : str
        Specific workpaper ID, e.g. "NPO-CX-1.1". Must be registered
        in WORKPAPER_TYPE_TO_WORKFLOW.
    query : str
        Natural-language query describing what's being looked up
        (e.g., "engagement acceptance criteria for nonprofits with
        federal funding"). Embedded with the e5-mistral query prefix.
    top_k : int
        Maximum number of chunks to return. Default 10.
    sop_version : str | None
        Optional SOP version filter, e.g. "2024-Q1". If None, all
        versions are eligible (use this only when reproducibility
        against a specific SOP version doesn't matter).
    qdrant_client : QdrantClient | None
        Optional pre-built client (dependency injection for tests).
        If None, a client is constructed from QDRANT_HOST/PORT env vars.

    Returns
    -------
    list[str]
        Ordered list of chunk content strings, highest similarity
        first. Length is at most top_k. Empty list if Qdrant query
        returns nothing.

    Raises
    ------
    KeyError
        If workpaper_type is not in WORKPAPER_TYPE_TO_WORKFLOW.
    """
    workflow = workflow_for(workpaper_type)
    settings = get_settings()
    collection = settings.qdrant.collection_sop

    client = qdrant_client or _get_qdrant_client()

    try:
        query_vector = embed_query(query)
    except Exception as e:
        logger.warning(
            "sop_retriever: embed_query failed (%s) — returning empty.", e,
        )
        return []

    filter_obj = _build_filter(workflow, workpaper_type, sop_version)

    try:
        results = client.search(
            collection_name=collection,
            query_vector=query_vector,
            query_filter=filter_obj,
            limit=top_k,
            with_payload=True,
        )
    except Exception as e:
        logger.warning(
            "sop_retriever: Qdrant search failed (%s) — returning empty.", e,
        )
        return []

    chunks: list[str] = []
    for hit in results:
        payload = hit.payload or {}
        content = payload.get("content", "")
        if content:
            chunks.append(content)

    logger.info(
        "sop_retriever: workpaper=%s workflow=%s version=%s top_k=%d -> %d hits",
        workpaper_type, workflow, sop_version, top_k, len(chunks),
    )
    return chunks
