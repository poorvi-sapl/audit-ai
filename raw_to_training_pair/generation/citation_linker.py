"""
raw_to_training_pair/generation/citation_linker.py
====================================================
Auto-link source citations onto a GeneratedWorkpaper using the
provenance carried by Phase 1A ExtractedFacts.

Decision L2 (Phase 2.2): when a gold-loaded field value equals an
extracted fact's value, copy that fact's source citations onto the
gold field. For fields the renderer set without a matching extracted
fact (e.g., SOP-driven boolean defaults), citations stay empty —
which is honest: those values weren't grounded in source documents.

Matching policy
---------------
- Exact value match (case-sensitive for strings, identity for bools).
- For boolean fields, citations are not copied (booleans don't carry
  source citations meaningfully — they're a judgment, not a quote).
  Override via copy_boolean_citations=True.
- For categorical fields, exact value match against the extracted
  fact's value.
- For dates, string comparison after stripping whitespace.

Public API
----------
    auto_link_citations(gold, extracted_facts, copy_boolean_citations=False)
        → GeneratedWorkpaper (new instance, gold is not mutated)
"""

from __future__ import annotations

import logging
from dataclasses import replace

from auditai_data_normalization.generation_contract import ExtractedFact
from auditai_data_normalization.field_type_registry import load_registry
from raw_to_training_pair.generation.target_schema import (
    GeneratedCitation,
    GeneratedFieldValue,
    GeneratedWorkpaper,
)

logger = logging.getLogger(__name__)


def _source_citations_from_fact(
    fact: ExtractedFact,
) -> list[GeneratedCitation]:
    """Convert ExtractedFact.sources (rich Phase 1A SourceCitations)
    into the lighter GeneratedCitation shape used in the assistant
    output."""
    out: list[GeneratedCitation] = []
    for src in fact.sources:
        out.append(GeneratedCitation(
            document=src.document_type or "unknown",
            page=src.page,
            quoted_text=src.quoted_text or "",
        ))
    return out


def _values_match(gold_value, fact_value) -> bool:
    """True iff the gold field value matches the extracted fact value
    for citation-linking purposes."""
    if gold_value is None or fact_value is None:
        return False
    # Boolean comparison is exact (True != "True")
    if isinstance(gold_value, bool) or isinstance(fact_value, bool):
        return gold_value == fact_value
    # Otherwise string compare after stripping whitespace
    return str(gold_value).strip() == str(fact_value).strip()


def auto_link_citations(
    gold: GeneratedWorkpaper,
    extracted_facts: dict[str, ExtractedFact],
    copy_boolean_citations: bool = False,
) -> GeneratedWorkpaper:
    """Return a new GeneratedWorkpaper with citations populated from
    matching ExtractedFact provenance.

    Args:
        gold: The gold workpaper (typically from gold_loader). Not mutated.
        extracted_facts: Dict of field_id → ExtractedFact (from Phase 1A
            assembly_layer.assemble_generation_input output).
        copy_boolean_citations: If True, boolean fields also inherit
            citations from matching extracted facts. Default False
            (booleans are usually judgments, not direct quotes).

    Returns:
        A new GeneratedWorkpaper with the same workpaper_type,
        engagement_id, and field structure, but with citations
        copied onto every gold field that matches an extracted fact.
        Counts the linkages in module-level logger for observability.
    """
    registry = None
    try:
        registry = load_registry(gold.workpaper_type)
    except FileNotFoundError:
        logger.warning(
            "citation_linker: no registry for %s — falling back to "
            "type-agnostic linking",
            gold.workpaper_type,
        )

    linked_count = 0
    new_fields: dict[str, GeneratedFieldValue] = {}

    for fid, gold_fv in gold.fields.items():
        if gold_fv.value is None:
            # Nothing to ground
            new_fields[fid] = gold_fv
            continue

        # Skip booleans by default (judgment-typed)
        if registry is not None and fid in registry:
            ftype = registry[fid].field_type
            if ftype == "boolean" and not copy_boolean_citations:
                new_fields[fid] = gold_fv
                continue

        fact = extracted_facts.get(fid)
        if fact is None or not fact.is_present:
            new_fields[fid] = gold_fv
            continue

        if not _values_match(gold_fv.value, fact.value):
            # Value mismatch — gold differs from extraction. Don't link.
            new_fields[fid] = gold_fv
            continue

        # Link!
        new_citations = _source_citations_from_fact(fact)
        # Preserve any citations already present on the gold field
        # (don't lose reviewer-added citations from prior pipeline runs)
        combined = list(gold_fv.citations) + new_citations
        new_fields[fid] = GeneratedFieldValue(
            value=gold_fv.value,
            citations=combined,
        )
        linked_count += 1

    logger.info(
        "citation_linker: linked %d field(s) for %s/%s",
        linked_count, gold.workpaper_type, gold.engagement_id,
    )

    return GeneratedWorkpaper(
        workpaper_type=gold.workpaper_type,
        engagement_id=gold.engagement_id,
        fields=new_fields,
    )
