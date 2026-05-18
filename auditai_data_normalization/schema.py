"""
auditai_data_normalization/schema.py
=====================================
The single canonical data structure every raw file becomes after
passing through the ETL pipeline.

Nothing downstream — the LLM extractor, the RAG chunker, the pair
builder, the JSONL exporter — ever touches a raw file. Everything
consumes a DocumentRecord.

Design decisions
----------------
- Plain Python dataclass. No ORM, no Pydantic, no external deps.
  This is the standalone normalization module; production layers
  (SQLAlchemy, MongoDB) are added on top later.
- Completely format-agnostic. Whether the source is a DOCX workpaper,
  a scanned PDF, an Excel trial balance, or an SOP — the output shape
  is identical.
- No domain-specific fields (no engagement_partner, no is_gagas, no
  has_single_audit). Domain labels are applied downstream by the LLM
  extractor or a human reviewer. This pipeline's only job is:
      raw file → clean structured text + metadata + confidence score.
- The `sections` list is the primary content unit. Every downstream
  consumer (chunker, pair builder, RAG embedder) iterates sections,
  not raw_text. raw_text is kept for full-document operations like
  PII scanning and dedup hashing.
- `tables` is separate from `sections` because tables need different
  handling downstream (numeric chunker vs semantic chunker).
- `metadata` is a free dict. Format-specific details (sheet names,
  page counts, OCR confidence per page, heading levels) go here so
  the core schema stays stable as new formats are added.

Schema changelog
----------------
Phase A3 (confidence split):
  + review_confidence     — quality score of the generated completion
  + extraction_gate       — True when extraction_confidence >= 0.50
  + quality_gate          — True when review_confidence >= 0.70
  ~ is_ready_for_training — now checks quality_gate, not raw threshold
  ~ needs_review          — now set when quality_gate is False

Phase B4 (LLM fallback provenance):
  ~ ExtractionMethodStr   — added "llm_fallback" and "hybrid" variants
  + llm_assisted          — True when LLM fallback ran on any field
  + flagged_fields        — fields needing auditor attention
                            (LLM-only extraction, contradicted values,
                             placeholder SOP citations)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Allowed literal types
# ---------------------------------------------------------------------------

FileTypeStr = Literal[
    "docx",
    "pdf_text",    # digital / text-native PDF
    "pdf_scanned", # image-only or low-text PDF, needs OCR
    "xlsx",
    "csv",
    "json",
    "unknown",
]

ExtractionMethodStr = Literal[
    # ── deterministic extractors ──────────────────────────────────────────
    "python_docx",    # python-docx direct extraction
    "pdfplumber",     # pdfplumber primary
    "pdfminer",       # pdfminer.six fallback
    "docling_surya",  # Docling + Surya OCR
    "tesseract",      # Tesseract fallback OCR
    "openpyxl",       # openpyxl / pandas Excel
    "stdlib_json",    # stdlib json.loads()
    # ── LLM extraction modes (Phase B4) ──────────────────────────────────
    "llm",            # LLM as tiebreaker only (original behaviour)
                      #   triggered when two deterministic extractors disagree
    "llm_fallback",   # LLM as primary fallback (Phase B2)
                      #   triggered when extraction_confidence < 0.50
                      #   after deterministic extractors ran
    "hybrid",         # deterministic primary + LLM filled missing fields
                      #   some fields from deterministic, some from llm_fallback
    # ── fallback ─────────────────────────────────────────────────────────
    "unknown",
]


# ---------------------------------------------------------------------------
# Sub-structures  (unchanged)
# ---------------------------------------------------------------------------

@dataclass
class Section:
    """
    One logical unit of content within the document.

    For a DOCX: one heading + its following paragraphs.
    For a PDF:  one page or one detected text block.
    For Excel:  one sheet rendered as text.
    For JSON:   one top-level key block.

    The chunker downstream splits large sections further; small sections
    are passed through as-is. This is the pre-chunked unit.
    """
    index: int

    heading: str = ""
    """
    Section heading or label if detectable.
    DOCX: paragraph style Heading1/2/3.
    PDF:  detected section header (Docling layout analysis).
    Excel: sheet name.
    Empty string if no heading is detectable.
    """

    content: str = ""
    """Full text content of this section, whitespace-normalized."""

    page_or_sheet: str = ""
    """
    Source reference.
    PDF:   "Page 3"
    Excel: "Sheet: Trial Balance"
    DOCX:  empty (DOCX has no page numbers in the DOM)
    """

    token_count: int = 0
    """
    Approximate token count (word-level split as proxy).
    Replaced with Mistral tokenizer count in the chunker step.
    """

    is_table: bool = False
    """True when this section represents a table rendered as text."""


@dataclass
class ExtractedTable:
    """
    A structured table extracted from the document.

    Kept separate from Section because tables feed the numeric chunker
    while text sections feed the semantic chunker.
    """
    index: int

    source: str = ""
    """'Page 3', 'Sheet: Trial Balance', 'Table 2', etc."""

    headers: list[str] = field(default_factory=list)
    """Column header labels. Empty list if no headers detected."""

    rows: list[list[str]] = field(default_factory=list)
    """
    Table data as list of rows, each row a list of cell strings.
    All values are strings — numeric parsing happens downstream.
    Merged cells are forward-filled so every row has the same length.
    """

    raw_text: str = ""
    """
    Human-readable rendering of the table (label: value per line).
    Used for embedding and LLM reading when structured rows aren't needed.
    """


@dataclass
class PIIRedaction:
    """Log entry for one PII redaction action."""

    pii_type: str
    """e.g. 'EIN', 'SSN', 'PERSON', 'CLIENT_ENTITY', 'ACCOUNT_NUM'"""

    replacement: str
    """The token that replaced the PII, e.g. '[EIN]', '[PERSON]'"""

    count: int = 1
    """How many occurrences of this type were replaced in this document."""

    detection_method: str = "presidio"
    """'presidio', 'regex', or 'spacy_ner'"""


# ---------------------------------------------------------------------------
# DocumentRecord — the canonical ETL output
# ---------------------------------------------------------------------------

@dataclass
class DocumentRecord:
    """
    Canonical output of normalize_document(file_path).

    Every raw file — workpaper, financial statement, SOP, form, report —
    produces exactly one DocumentRecord. Nothing downstream touches the
    original file.

    Field groups
    ------------
    1. Identity            source path, file type, content hash
    2. Raw content         full text before any cleaning (for hashing/dedup)
    3. Structured content  sections list, tables list
    4. PII                 scrub log, clean flag
    5. Extraction          method, confidence, timestamps
    6. Review quality      review_confidence, gates          ← A3
    7. LLM provenance      llm_assisted, flagged_fields      ← B4
    8. Metadata            free dict for format-specific details
    9. Training gates      approval flags, reviewer identity
    """

    # ------------------------------------------------------------------
    # 1. Identity
    # ------------------------------------------------------------------

    source_path: str = ""
    file_name: str = ""
    file_type: FileTypeStr = "unknown"
    file_size_bytes: int = 0
    file_hash: str = ""
    """SHA-256 hex digest of raw file bytes. Used for deduplication."""

    # ------------------------------------------------------------------
    # 2. Raw content
    # ------------------------------------------------------------------

    raw_text: str = ""
    """
    Full extracted text as the extractor produced it, before cleaning.
    Never written to training data. Never sent to the LLM.
    Used only for: SHA-256 dedup, MinHash similarity, PII scanning.
    """

    # ------------------------------------------------------------------
    # 3. Structured content
    # ------------------------------------------------------------------

    sections: list[Section] = field(default_factory=list)
    tables: list[ExtractedTable] = field(default_factory=list)
    cleaned_text: str = ""
    """Full text after whitespace normalization and PII stripping."""

    # ------------------------------------------------------------------
    # 4. PII
    # ------------------------------------------------------------------

    pii_scrubbed: bool = False
    pii_redactions: list[PIIRedaction] = field(default_factory=list)

    # ------------------------------------------------------------------
    # 5. Extraction provenance
    # ------------------------------------------------------------------

    extraction_method: ExtractionMethodStr = "unknown"
    """
    Which extractor produced this record.
    Phase B4 values:
      'llm_fallback' — deterministic ran but score < 0.50; LLM filled gaps
      'hybrid'       — deterministic primary + LLM filled specific missing fields
    """

    extraction_confidence: float = 0.0
    """
    Aggregate confidence from confidence.py score_record().
    Phase A2 tier-based formula:
      base = (tier1_found / 8) * 0.70 + (tier2_found / 8) * 0.30
    Floors applied based on tier1_found count.

    Gate meanings (Phase A3):
      >= 0.50  extraction_gate passes — proceed to completion drafter
      >= 0.70  quality_gate passes   — eligible for JSONL
      <  0.50  LLM fallback triggered (Phase B2)
    """

    word_count: int = 0
    page_count: int = 0
    ocr_used: bool = False
    extracted_at: datetime = field(default_factory=datetime.utcnow)
    extraction_error: str = ""
    extraction_status: Literal[
        "success",
        "partial",
        "failed",
        "pending",
    ] = "pending"

    # ------------------------------------------------------------------
    # 6. Review quality  (Phase A3)
    # ------------------------------------------------------------------

    review_confidence: float = 0.0
    """
    Quality score of the AI-generated completion for this record.
    Computed by completion_drafter.py after Gemma generates the output.

    Scoring rubric (Phase C4):
      + 0.30  all 3 required sections present (ENGAGEMENT TYPE, FINDINGS, RECOMMENDATION)
      + 0.20  every finding cites a real SOP section (not a placeholder §X.X)
      + 0.20  severity classification present on every finding
      + 0.20  recommendation is client-type specific (not generic)
      + 0.10  no generic placeholder text detected

    Gate: review_confidence >= 0.70 → quality_gate passes
    Default 0.0 until completion_drafter.py has run.
    """

    extraction_gate: bool = False
    """
    True when extraction_confidence >= 0.50.
    Set by normalize.py after confidence scoring (Phase A4).
    Controls whether this record proceeds to the completion drafter.
    False records trigger LLM fallback extraction (Phase B2).
    """

    quality_gate: bool = False
    """
    True when review_confidence >= 0.70.
    Set by completion_drafter.py after scoring the generated completion.
    Controls JSONL eligibility in quality_gates.py.
    False records go to the review queue for auditor attention —
    they are NOT hard-rejected (Phase A4 split gate logic).
    """

    # ------------------------------------------------------------------
    # 7. LLM provenance  (Phase B4)
    # ------------------------------------------------------------------

    llm_assisted: bool = False
    """
    True when the LLM extractor contributed any field values to this record.
    Covers both:
      - LLM as tiebreaker (extraction_method = 'llm')
      - LLM as fallback   (extraction_method = 'llm_fallback' or 'hybrid')

    Pairs with llm_assisted=True are eligible for training but receive
    closer auditor scrutiny — Tier 1 fields extracted by LLM alone are
    always added to flagged_fields.
    """

    flagged_fields: list[str] = field(default_factory=list)
    """
    Fields needing auditor attention before the training pair is approved.
    Populated by normalize.py and completion_drafter.py.

    A field is added here when:
      - It was extracted by LLM alone with no deterministic corroboration
        (source = 'llm', no matching deterministic value)
      - Two extractors returned contradicting values and LLM broke the tie
        (auditor should verify the LLM chose correctly)
      - A SOP citation in the generated completion is a placeholder (§X.X)
      - The field is Tier 1 and its extraction confidence < 0.70

    Auditors see flagged_fields highlighted in the Streamlit review UI
    (Phase E2). A pair with flagged_fields can still be approved —
    it just requires explicit auditor sign-off on each flagged item.
    """

    # ------------------------------------------------------------------
    # 8. Metadata  (format-specific, free dict)
    # ------------------------------------------------------------------

    metadata: dict[str, Any] = field(default_factory=dict)
    """
    Format-specific details that don't belong in the core schema.

    Common keys by format:
      DOCX:  {heading_count, table_count, has_tracked_changes}
      PDF:   {page_count, is_encrypted, avg_ocr_confidence,
               error_pages: [list of page numbers that failed]}
      Excel: {sheet_names: [...], numeric_sheets: [...],
               merged_cell_count, subtotal_rows_detected}
      JSON:  {top_level_keys: [...], nesting_depth}

    Phase B4 additions written by normalize.py:
      {llm_fallback_fields: [...],   # fields filled by llm_fallback
       llm_tiebreaker_fields: [...], # fields where LLM broke a tie
       tier1_missing: [...],         # Tier 1 fields not found — audit deficiencies
       per_field_scores: {...}}      # raw per-field confidence dict from confidence.py
    """

    # ------------------------------------------------------------------
    # 9. Training gates
    # ------------------------------------------------------------------

    needs_review: bool = False
    """
    Set to True by normalize.py when quality_gate is False.
    Records with needs_review=True are written to review_queue and
    excluded from JSONL until an auditor corrects them.

    Phase A3: previously triggered on extraction_confidence < 0.70.
    Now triggered on quality_gate=False (review_confidence < 0.70)
    so that low-extraction documents can still proceed to the drafter
    and only fail the gate if the generated completion is poor.
    """

    auditor_approved: bool = False
    """
    Hard gate. No record enters training data until this is True.
    Set manually by an auditor in the Streamlit review UI, or
    automatically when extraction_confidence == 1.0 on a known-good
    clean text-native document.
    """

    reviewer_id: str = ""
    """Initials or ID of the auditor who approved this record."""

    review_date: str = ""
    """ISO date when auditor_approved was set."""

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def is_ready_for_training(self) -> bool:
        """
        True when this record meets all gates for training use.

        Phase A3 change: checks quality_gate (review_confidence >= 0.70)
        instead of the old hardcoded extraction_confidence >= 0.70.
        This allows low-extraction documents with good completions to
        enter training, and blocks high-extraction documents whose
        generated completions are poor quality.
        """
        return (
            self.auditor_approved
            and self.pii_scrubbed
            and self.quality_gate                        # review_confidence >= 0.70
            and self.extraction_gate                     # extraction_confidence >= 0.50
            and self.extraction_status in ("success", "partial")
            and bool(self.cleaned_text.strip())
        )

    def is_ready_for_drafting(self) -> bool:
        """
        True when this record can proceed to the completion drafter.
        Lower bar than is_ready_for_training — only needs extraction_gate.

        Phase A4: normalize.py calls this to decide whether to send
        the record to completion_drafter.py or trigger LLM fallback first.
        """
        return (
            self.extraction_gate                         # extraction_confidence >= 0.50
            and self.pii_scrubbed
            and self.extraction_status in ("success", "partial")
            and bool(self.cleaned_text.strip())
        )

    def section_text(self) -> str:
        """Rebuild cleaned_text from sections if it was not set."""
        return "\n\n".join(
            s.content for s in self.sections if s.content.strip()
        )

    def summary(self) -> str:
        """One-line human-readable summary for logging and review queue."""
        llm_tag = " [LLM]" if self.llm_assisted else ""
        flags = f" flags={len(self.flagged_fields)}" if self.flagged_fields else ""
        return (
            f"[{self.file_type}] {self.file_name} | "
            f"sections={len(self.sections)} tables={len(self.tables)} | "
            f"words={self.word_count} | "
            f"ext_conf={self.extraction_confidence:.2f} "
            f"rev_conf={self.review_confidence:.2f} | "
            f"ext_gate={'✓' if self.extraction_gate else '✗'} "
            f"qual_gate={'✓' if self.quality_gate else '✗'} | "
            f"status={self.extraction_status}"
            f"{llm_tag}{flags}"
        )

    def __repr__(self) -> str:
        return f"DocumentRecord({self.summary()})"