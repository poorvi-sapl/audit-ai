"""
FieldEvidence → ExtractedFact adapter.
=======================================

Bridges the existing FieldEvidence dataclass (used by structural_extractor
and related deterministic extractors) to the new ExtractedFact contract.

This is a TRANSITIONAL adapter:
  - char-level offsets are unavailable on legacy FieldEvidence, so
    citations use CHAR_OFFSET_UNAVAILABLE
  - quoted_text is approximated from the existing `anchor` field
  - As individual extractors gain char-level tracking, they should
    produce ExtractedFact directly and skip this adapter

The adapter looks up the field's FieldType from the per-workpaper
field-type registry, then constructs an ExtractedFact with proper
type metadata. It also maps the legacy `method` strings (e.g.,
"salutation_block") to ExtractorMethod literals.
"""

from __future__ import annotations

import logging

from auditai_data_normalization.extractors.structural_extractor import (
    FieldEvidence,
)
from auditai_data_normalization.field_type_registry import get_field_spec
from auditai_data_normalization.generation_contract import (
    CHAR_OFFSET_UNAVAILABLE,
    ExtractedFact,
    ExtractorMethod,
    SourceCitation,
)

logger = logging.getLogger(__name__)


# Map legacy method strings (from structural_extractor.METHOD_* constants
# and related extractors) to ExtractorMethod literals from the contract.
# When new methods are added upstream, update this map.
_METHOD_MAP: dict[str, ExtractorMethod] = {
    "salutation_block":   "structural_heuristic",
    "standalone_date":    "regex_pattern",
    "signoff_date":       "structural_heuristic",
    "prose_flag":         "field_label_proximity",
    "registry":           "structural_heuristic",
}

_DEFAULT_METHOD: ExtractorMethod = "structural_heuristic"


def _map_method(legacy_method: str) -> ExtractorMethod:
    """Map a legacy extractor method string to an ExtractorMethod literal.

    Unknown method strings default to `structural_heuristic` (a safe
    deterministic-extractor category) and emit a one-time warning so
    new methods can be registered explicitly.
    """
    if legacy_method in _METHOD_MAP:
        return _METHOD_MAP[legacy_method]
    logger.warning(
        "field_evidence_adapter: unknown legacy method %r — "
        "defaulting to %r. Add an explicit mapping in _METHOD_MAP.",
        legacy_method, _DEFAULT_METHOD,
    )
    return _DEFAULT_METHOD


def field_evidence_to_extracted_fact(
    evidence: FieldEvidence,
    field_id: str,
    workpaper_type: str,
    document_path: str,
    document_type: str,
    extractor_version: str = "",
) -> ExtractedFact:
    """Convert a legacy FieldEvidence into an ExtractedFact.

    Args:
        evidence: The legacy extractor output.
        field_id: Which field in the workpaper this evidence is for.
        workpaper_type: e.g. "NPO-CX-1.1" — used to look up FieldType.
        document_path: Path or URI of the source document.
        document_type: Logical document type (e.g. "audit_report",
                       "engagement_letter", "prior_year_file").
        extractor_version: Optional version tag for reproducibility.

    Returns:
        ExtractedFact with type from the registry, page-level citation
        (char offsets unavailable), and proper provenance.

    Raises:
        KeyError: if field_id is not in the workpaper's registry.
        ValueError: if the evidence violates the no-LLM rule
                    (raised by ExtractedFact.__post_init__).
    """
    spec = get_field_spec(workpaper_type, field_id)
    method = _map_method(evidence.method)

    # Prefer char-level fields when the extractor populated them; fall back
    # to page-level provenance with `anchor` as quoted_text otherwise.
    has_char = getattr(evidence, "has_char_offsets", False)
    if has_char:
        char_start = evidence.char_start
        char_end = evidence.char_end
        # full_quoted_text was added when char offsets were captured; if for
        # some reason it's empty, fall back to anchor.
        quoted = evidence.full_quoted_text or evidence.anchor or ""
        notes = "char-level provenance via structural extractor"
    else:
        char_start = CHAR_OFFSET_UNAVAILABLE
        char_end = CHAR_OFFSET_UNAVAILABLE
        quoted = evidence.anchor or ""
        notes = (
            "page-level provenance only; char offsets unavailable"
            if quoted
            else "page-level provenance only; no quoted text available"
        )

    citation = SourceCitation(
        document_path=document_path,
        document_type=document_type,
        page=evidence.source_page,
        char_start=char_start,
        char_end=char_end,
        quoted_text=quoted,
    )

    return ExtractedFact(
        field_id=field_id,
        field_type=spec.field_type,
        value=evidence.value,
        confidence=evidence.confidence,
        sources=[citation],
        extractor_method=method,
        extractor_version=extractor_version,
        notes=notes,
    )


def field_evidence_map_to_facts(
    evidences: dict[str, FieldEvidence],
    workpaper_type: str,
    document_path: str,
    document_type: str,
    extractor_version: str = "",
) -> dict[str, ExtractedFact]:
    """Bulk-convert a dict of FieldEvidence into a dict of ExtractedFact.

    Skips field_ids not in the registry (logs a warning) rather than
    raising, since extractors may emit experimental fields that haven't
    been added to the registry yet.
    """
    facts: dict[str, ExtractedFact] = {}
    for fid, ev in evidences.items():
        try:
            facts[fid] = field_evidence_to_extracted_fact(
                evidence=ev,
                field_id=fid,
                workpaper_type=workpaper_type,
                document_path=document_path,
                document_type=document_type,
                extractor_version=extractor_version,
            )
        except KeyError:
            logger.warning(
                "field_evidence_adapter: field %r not in registry for %r — "
                "skipping. Add it to the registry to include.",
                fid, workpaper_type,
            )
    return facts
