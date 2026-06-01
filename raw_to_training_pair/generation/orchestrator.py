"""
raw_to_training_pair/generation/orchestrator.py
=================================================
Top-level orchestration for generation training pair production.

In Phase 2.1 (walking skeleton) the orchestrator exists for two
purposes:
  1. Provide a single callable that bundles assembly + retrieval +
     pair building, so callers (tests, batch scripts, the future
     Streamlit integration) hit one entry point.
  2. Be the slot where the gold-label loader plugs in during Phase
     2.2 — right now the gold is passed in directly; in 2.2 it will
     be loaded from a real filled .docx workpaper.

Public API
----------
    build_generation_pair_synthetic(
        gen_input, gold, ...
    ) → dict[messages+metadata]
        Walking-skeleton path: gold is already a GeneratedWorkpaper.

    build_generation_pair_from_extractions(
        workpaper_type, engagement_id, source_extractions, gold,
        sop_query=None, sop_top_k=10, sop_version=None,
        with_sop_retrieval=False,
    ) → dict
        Assembles a GenerationInput from source extractions (with
        optional SOP retrieval), then builds the pair against the
        provided gold.
"""

from __future__ import annotations

import logging
from typing import Any

from auditai_data_normalization.assembly_layer import (
    SourceDocumentExtraction,
    assemble_generation_input,
)
from auditai_data_normalization.generation_contract import GenerationInput
from raw_to_training_pair.generation.pair_builder import build_generation_pair
from raw_to_training_pair.generation.target_schema import GeneratedWorkpaper

logger = logging.getLogger(__name__)


def build_generation_pair_synthetic(
    gen_input: GenerationInput,
    gold: GeneratedWorkpaper,
    block_on_schema_issues: bool = False,
    extra_metadata: dict | None = None,
) -> dict[str, Any]:
    """Synthetic-data entry point — thin pass-through to pair_builder.

    Use when you already have a GenerationInput (from Phase 1A/1B)
    and a GeneratedWorkpaper (synthetic or from a future loader).
    No I/O, no model calls — pure transform.
    """
    return build_generation_pair(
        gen_input=gen_input,
        gold=gold,
        block_on_schema_issues=block_on_schema_issues,
        extra_metadata=extra_metadata,
    )


def build_generation_pair_from_extractions(
    workpaper_type: str,
    engagement_id: str,
    source_extractions: list[SourceDocumentExtraction],
    gold: GeneratedWorkpaper,
    sop_chunks: list[str] | None = None,
    with_sop_retrieval: bool = False,
    sop_query: str | None = None,
    sop_top_k: int = 10,
    sop_version: str | None = None,
    qdrant_client=None,
    block_on_schema_issues: bool = False,
    extra_metadata: dict | None = None,
) -> dict[str, Any]:
    """Convenience: assemble GenerationInput from source extractions
    (optionally with auto SOP retrieval), then build the pair.

    Args:
        workpaper_type, engagement_id, source_extractions:
            Standard Phase 1A assembly inputs.
        gold: The gold-label workpaper (the assistant target).
        sop_chunks: Optional pre-retrieved SOP chunks. Ignored if
            with_sop_retrieval=True.
        with_sop_retrieval: If True, auto-retrieve SOP chunks via
            engineering_benchmark.sop_retriever. Requires Qdrant
            reachable (or qdrant_client mock).
        sop_query, sop_top_k, sop_version, qdrant_client:
            Forwarded to retrieve_sop_chunks() when
            with_sop_retrieval=True.
        block_on_schema_issues, extra_metadata: Forwarded to pair_builder.

    Returns:
        Pair dict ready for JSONL append.
    """
    if with_sop_retrieval:
        # Lazy import — avoids requiring sop_retriever on the import
        # path for callers that don't need retrieval.
        from auditai_data_normalization.assembly_layer import (
            assemble_generation_input_with_sop_retrieval,
        )
        gen_input = assemble_generation_input_with_sop_retrieval(
            workpaper_type=workpaper_type,
            engagement_id=engagement_id,
            source_extractions=source_extractions,
            sop_query=sop_query,
            sop_top_k=sop_top_k,
            sop_version=sop_version,
            qdrant_client=qdrant_client,
        )
    else:
        gen_input = assemble_generation_input(
            workpaper_type=workpaper_type,
            engagement_id=engagement_id,
            source_extractions=source_extractions,
            sop_chunks=sop_chunks,
        )

    return build_generation_pair(
        gen_input=gen_input,
        gold=gold,
        block_on_schema_issues=block_on_schema_issues,
        extra_metadata=extra_metadata,
    )
