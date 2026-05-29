"""
Assembly layer — extraction outputs → GenerationInput.
=======================================================

Takes per-source-document FieldEvidence payloads from the existing
extractor stack, adapts them to ExtractedFacts, merges multi-source
facts for the same field_id, looks up the template field roster
from the field-type registry, and produces a GenerationInput ready
to feed the workpaper generation model.

Scope (intentional):
  - Does NOT re-implement file routing (pdf → text → FieldEvidence).
    Callers run the existing file/structural extractors and pass the
    resulting FieldEvidence dicts here. A higher-level "from raw
    files" wrapper can be added later as ergonomics.
  - Enforces the no-LLM-numbers rule transitively — the adapter
    constructs ExtractedFact via __post_init__, which raises if a
    numeric/date/id field came from llm_extraction.
  - Multi-source merging is deterministic and explicit (see
    merge_facts below).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from auditai_data_normalization.extractors.structural_extractor import (
    FieldEvidence,
)
from auditai_data_normalization.field_evidence_adapter import (
    field_evidence_map_to_facts,
)
from auditai_data_normalization.field_type_registry import load_registry
from auditai_data_normalization.generation_contract import (
    ExtractedFact,
    GenerationInput,
    SourceCitation,
)

logger = logging.getLogger(__name__)

# Confidence penalty applied when multiple sources disagree on a value.
# Disagreement is evidence of uncertainty; we keep the highest-confidence
# value but scale its confidence down to reflect the conflict.
_DISAGREEMENT_CONFIDENCE_PENALTY: float = 0.8


@dataclass
class SourceDocumentExtraction:
    """One source document's extraction payload.

    Carries the document context needed for provenance (path, type)
    alongside the FieldEvidence dict produced by the extractor stack.
    """
    document_path: str
    document_type: str               # 'engagement_letter' | 'audit_report' | etc
    field_evidence: dict[str, FieldEvidence]
    extractor_version: str = ""


# ---------------------------------------------------------------------
# Merge logic — combining facts for the same field_id across documents
# ---------------------------------------------------------------------

def merge_facts(facts: list[ExtractedFact]) -> ExtractedFact:
    """Merge multiple ExtractedFacts for the same field_id.

    Cases:
      1. Single fact         → return as-is
      2. All facts agree on value → combine sources, max confidence,
                                    extractor_method = multi_extractor_agreement
      3. Facts disagree on value  → pick the highest-confidence fact,
                                    apply confidence penalty, flag
                                    CONFLICT in notes

    Raises if the list is empty or facts have mixed field_id/type
    (registry inconsistency, should not happen in normal flow).
    """
    if not facts:
        raise ValueError("merge_facts: cannot merge empty list")
    if len(facts) == 1:
        return facts[0]

    # Sanity: all facts must share field_id and field_type
    field_ids = {f.field_id for f in facts}
    field_types = {f.field_type for f in facts}
    if len(field_ids) != 1:
        raise ValueError(
            f"merge_facts: mixed field_ids {field_ids} — caller must "
            "group facts by field_id before merging."
        )
    if len(field_types) != 1:
        raise ValueError(
            f"merge_facts: mixed field_types {field_types} for "
            f"field_id={field_ids.pop()!r} — registry inconsistency."
        )

    field_id = field_ids.pop()
    field_type = field_types.pop()

    # Filter out facts with no value (None)
    present_facts = [f for f in facts if f.value is not None]
    if not present_facts:
        # All facts agree the field is absent
        return facts[0]

    values = {f.value for f in present_facts}
    all_sources: list[SourceCitation] = []
    for f in present_facts:
        all_sources.extend(f.sources)

    if len(values) == 1:
        # All sources agree — strong evidence
        the_value = values.pop()
        max_conf = max(f.confidence for f in present_facts)
        return ExtractedFact(
            field_id=field_id,
            field_type=field_type,
            value=the_value,
            confidence=min(1.0, max_conf),
            sources=all_sources,
            extractor_method="multi_extractor_agreement",
            notes=f"{len(present_facts)} sources agree on value",
        )

    # Disagreement: pick highest-confidence value, penalize, flag
    best = max(present_facts, key=lambda f: f.confidence)
    penalized = best.confidence * _DISAGREEMENT_CONFIDENCE_PENALTY
    return ExtractedFact(
        field_id=field_id,
        field_type=field_type,
        value=best.value,
        confidence=penalized,
        sources=best.sources,
        extractor_method=best.extractor_method,
        extractor_version=best.extractor_version,
        notes=(
            f"CONFLICT: {len(values)} distinct values across "
            f"{len(present_facts)} sources; chose highest-confidence "
            f"(value={best.value!r}); confidence penalized by "
            f"{1 - _DISAGREEMENT_CONFIDENCE_PENALTY:.0%}"
        ),
    )


# ---------------------------------------------------------------------
# Top-level assembly
# ---------------------------------------------------------------------

def assemble_generation_input(
    workpaper_type: str,
    engagement_id: str,
    source_extractions: list[SourceDocumentExtraction],
    sop_chunks: list[str] | None = None,
) -> GenerationInput:
    """Assemble a GenerationInput from per-document FieldEvidence payloads.

    Args:
        workpaper_type: e.g. "NPO-CX-1.1". Used to look up the registry.
        engagement_id: Internal engagement identifier (passed through).
        source_extractions: One entry per source document, carrying the
                            document context + FieldEvidence dict.
        sop_chunks: Pre-retrieved SOP context. If None, defaults to
                    empty list (SOP retrieval is the caller's concern).

    Returns:
        GenerationInput with merged facts, template field roster from
        the registry, and engagement metadata.

    Raises:
        FileNotFoundError: if the workpaper's registry doesn't exist.
        ValueError: if any extracted fact violates the no-LLM rule
                    (raised transitively from ExtractedFact construction).
    """
    registry = load_registry(workpaper_type)
    template_field_ids = sorted(registry.keys())

    # Step 1: per-document, adapt FieldEvidence → ExtractedFact
    # Group facts by field_id across all source documents.
    facts_by_field: dict[str, list[ExtractedFact]] = {}
    for extraction in source_extractions:
        adapted = field_evidence_map_to_facts(
            evidences=extraction.field_evidence,
            workpaper_type=workpaper_type,
            document_path=extraction.document_path,
            document_type=extraction.document_type,
            extractor_version=extraction.extractor_version,
        )
        for fid, fact in adapted.items():
            facts_by_field.setdefault(fid, []).append(fact)

    # Step 2: merge multi-source facts per field_id
    merged: dict[str, ExtractedFact] = {}
    for fid, facts in facts_by_field.items():
        merged[fid] = merge_facts(facts)

    logger.info(
        "assemble_generation_input: workpaper=%s engagement=%s "
        "source_docs=%d fields_present=%d fields_missing=%d",
        workpaper_type, engagement_id,
        len(source_extractions),
        sum(1 for f in merged.values() if f.is_present),
        len(template_field_ids) - sum(1 for f in merged.values() if f.is_present),
    )

    return GenerationInput(
        workpaper_type=workpaper_type,
        engagement_id=engagement_id,
        sop_chunks=sop_chunks or [],
        extracted_facts=merged,
        template_field_ids=template_field_ids,
    )
