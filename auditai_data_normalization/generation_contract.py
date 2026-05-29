"""
Extraction → Generation Contract
=================================

Defines the data structure passed from deterministic extraction
to the workpaper generation model. Every field value carries
provenance: where it was found, by what method, with what
confidence, and the exact source text supporting the value.

This contract makes generated workpapers auditable — peer
reviewers can trace every narrative claim back to a specific
location in a specific source document.

NUMERICAL ACCURACY RULE
Numeric/date/id field types MUST come from deterministic
extractors. LLM extraction is not permitted for these types.
Enforced at construction time via __post_init__.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


# ---------------------------------------------------------------------
# Source citation — points to a specific location in a source document
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SourceCitation:
    """One piece of evidence supporting an extracted value.

    Multiple SourceCitations may support the same value (e.g., the
    client name appears in the engagement letter AND the prior-year
    file). Each citation is granular enough to render as a footnote
    in the generated workpaper.
    """
    document_path: str            # path or URI to source document
    document_type: str            # 'engagement_letter' | 'prior_year_file' | etc
    page: int                     # 1-based page number
    char_start: int               # char offset in extracted text (0-based)
    char_end: int                 # exclusive
    quoted_text: str              # exact source text supporting the value

    def render_citation(self) -> str:
        return f"[{self.document_type} p.{self.page}]"


# ---------------------------------------------------------------------
# Extracted fact — one field value with full provenance
# ---------------------------------------------------------------------

FieldType = Literal[
    "text",         # free-form text (entity names, descriptions)
    "numeric",      # numbers — NEVER LLM
    "date",         # dates — NEVER LLM
    "id",           # identifiers (EIN, doc IDs) — NEVER LLM
    "boolean",      # yes/no/na
    "categorical",  # one of a fixed set
]

ExtractorMethod = Literal[
    "regex_pattern",
    "structural_heuristic",
    "ocr_text_block",
    "table_cell_lookup",
    "field_label_proximity",
    "llm_extraction",                 # disallowed for numeric/date/id
    "multi_extractor_agreement",
]

# Field types where LLM extraction is forbidden — single source of truth
LLM_FORBIDDEN_FIELD_TYPES: frozenset[FieldType] = frozenset({"numeric", "date", "id"})


@dataclass
class ExtractedFact:
    """A single field's extracted value with full provenance.

    The unit of data the generation prompt consumes. Every fact
    carries enough provenance that the generated workpaper can
    cite the source for every claim it makes.
    """
    field_id: str
    field_type: FieldType
    value: str | None                 # extracted value, None if not found
    confidence: float                 # [0.0, 1.0]
    sources: list[SourceCitation]     # ≥1 citation if value is not None
    extractor_method: ExtractorMethod
    extractor_version: str = ""
    extracted_at: datetime = field(default_factory=datetime.utcnow)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.value is not None and not self.sources:
            raise ValueError(
                f"ExtractedFact for {self.field_id} has a value but no sources — "
                "every extracted value must carry provenance."
            )
        if self.field_type in LLM_FORBIDDEN_FIELD_TYPES:
            if self.extractor_method == "llm_extraction":
                raise ValueError(
                    f"Field {self.field_id} is type {self.field_type} but came "
                    f"from llm_extraction — LLM extractors are not permitted "
                    f"to produce {self.field_type} values. Numerical accuracy "
                    "rule violated."
                )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"ExtractedFact for {self.field_id} has confidence "
                f"{self.confidence} outside [0.0, 1.0]."
            )

    @property
    def is_present(self) -> bool:
        return self.value is not None

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.85


# ---------------------------------------------------------------------
# Generation input — the bundle handed to the generation model
# ---------------------------------------------------------------------

@dataclass
class GenerationInput:
    """The complete input to the workpaper generation model.

    The generation prompt is assembled from this bundle. The model
    receives the workpaper_type, the SOP chunks, the extracted facts
    (each with provenance), and the template field schema (which
    fields are required). The model's output is a filled FieldJSON
    where every narrative field cites the SourceCitation(s) supporting
    its content.
    """
    workpaper_type: str                           # e.g. "NPO-CX-1.1"
    engagement_id: str
    sop_chunks: list[str]                         # retrieved SOP context
    extracted_facts: dict[str, ExtractedFact]     # keyed by field_id
    template_field_ids: list[str]                 # all field_ids expected

    def fields_present(self) -> list[str]:
        return [fid for fid, f in self.extracted_facts.items() if f.is_present]

    def fields_missing(self) -> list[str]:
        return [
            fid for fid in self.template_field_ids
            if fid not in self.extracted_facts
            or not self.extracted_facts[fid].is_present
        ]

    def low_confidence_fields(self, threshold: float = 0.70) -> list[str]:
        return [
            fid for fid, f in self.extracted_facts.items()
            if f.is_present and f.confidence < threshold
        ]
