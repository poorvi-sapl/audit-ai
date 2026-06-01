"""
raw_to_training_pair/generation/engagement_ingest.py
======================================================
Ingest an engagement folder — multiple source documents — and produce
a list[SourceDocumentExtraction] ready for the Phase 1A assembly
layer.

Per-document flow:
    file → file extractor (pdf_text/docx/ocr/etc.) → DocumentRecord
         → structural_extractor.extract() → dict[str, FieldEvidence]
         → SourceDocumentExtraction(path, doc_type, field_evidence)

Document-type detection is filename-keyword-based for the walking
skeleton — a future iteration can use the existing doc_classifier
module for ML-based classification once it's wired to the generation
path.

Public API
----------
    ingest_engagement_folder(folder, doc_type_hints=None)
        → list[SourceDocumentExtraction]
    classify_document_type(filename) → str
"""

from __future__ import annotations

import logging
from pathlib import Path

from auditai_data_normalization.assembly_layer import (
    SourceDocumentExtraction,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Document-type classification (filename-keyword heuristic)
# ---------------------------------------------------------------------

# Map filename substring → logical document_type used in SourceCitation.
# Order matters: more specific keywords come first. The first matching
# substring (case-insensitive) wins.
_FILENAME_TO_DOC_TYPE: list[tuple[str, str]] = [
    ("engagement_letter",      "engagement_letter"),
    ("engagement-letter",      "engagement_letter"),
    ("engagementletter",       "engagement_letter"),
    ("prior_year",             "prior_year_file"),
    ("prior-year",             "prior_year_file"),
    ("prioryear",              "prior_year_file"),
    ("intake_form",            "client_intake_form"),
    ("intake-form",            "client_intake_form"),
    ("intakeform",             "client_intake_form"),
    ("client_intake",          "client_intake_form"),
    ("financial_statement",    "financial_statements"),
    ("financial-statement",    "financial_statements"),
    ("audit_report",           "audit_report"),
    ("audit-report",           "audit_report"),
    ("board_minutes",          "board_minutes"),
    ("board-minutes",          "board_minutes"),
    ("management_letter",      "management_letter"),
    ("trial_balance",          "trial_balance"),
]

_DEFAULT_DOC_TYPE = "unknown"


def classify_document_type(filename: str) -> str:
    """Heuristic document-type classification from filename.

    Returns one of the known document_type strings, or "unknown" if
    no keyword matches. Caller can override per-file via the
    doc_type_hints argument to ingest_engagement_folder.
    """
    lower = filename.lower()
    for keyword, dtype in _FILENAME_TO_DOC_TYPE:
        if keyword in lower:
            return dtype
    return _DEFAULT_DOC_TYPE


# ---------------------------------------------------------------------
# File routing — file extension → file extractor module
# ---------------------------------------------------------------------

def _file_extractor_for(path: Path):
    """Pick the right file extractor module for a given file extension.

    Returns the module reference (caller invokes module.extract). Returns
    None if the extension isn't supported.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from auditai_data_normalization.extractors import pdf_text_extractor
        return pdf_text_extractor
    if suffix in (".docx", ".doc"):
        from auditai_data_normalization.extractors import docx_extractor
        return docx_extractor
    if suffix == ".csv":
        from auditai_data_normalization.extractors import csv_extractor
        return csv_extractor
    if suffix in (".xlsx", ".xls"):
        from auditai_data_normalization.extractors import xlsx_extractor
        return xlsx_extractor
    if suffix == ".json":
        from auditai_data_normalization.extractors import json_extractor
        return json_extractor
    return None


def _extract_field_evidence(record) -> dict:
    """Run structural_extractor.extract() on a DocumentRecord that has
    `pages_text` / `pages_words`. Returns dict[field_id, FieldEvidence].

    Returns an empty dict if structural extraction is not applicable
    (record lacks pages_text or is not a financial-statement type doc).
    """
    pages_text = getattr(record, "pages_text", None)
    pages_words = getattr(record, "pages_words", None)
    if not pages_text:
        return {}

    try:
        from auditai_data_normalization.extractors.structural_extractor import (
            extract as structural_extract,
        )
        return structural_extract(pages_text, pages_words or [])
    except Exception as e:
        logger.warning(
            "engagement_ingest: structural extraction failed for "
            "DocumentRecord — %s", e,
        )
        return {}


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def ingest_engagement_folder(
    folder: str | Path,
    doc_type_hints: dict[str, str] | None = None,
    extractor_version: str = "",
) -> list[SourceDocumentExtraction]:
    """Ingest every supported file in an engagement folder.

    Parameters
    ----------
    folder : str | Path
        Path to a directory of engagement source documents.
    doc_type_hints : dict[filename → doc_type] | None
        Override the filename-based classification on a per-file basis.
        Filenames are matched case-sensitively against keys.
    extractor_version : str
        Optional version tag stamped onto each SourceDocumentExtraction.

    Returns
    -------
    list[SourceDocumentExtraction]
        One entry per successfully-ingested file. Files with unsupported
        extensions are skipped with a warning; files that fail file-
        extractor invocation are skipped with a warning.

    Raises
    ------
    FileNotFoundError, NotADirectoryError
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"engagement_ingest: {folder_path} not found")
    if not folder_path.is_dir():
        raise NotADirectoryError(
            f"engagement_ingest: {folder_path} is not a directory"
        )

    extractions: list[SourceDocumentExtraction] = []
    hints = doc_type_hints or {}

    for path in sorted(folder_path.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue

        ext_module = _file_extractor_for(path)
        if ext_module is None:
            logger.debug(
                "engagement_ingest: skipping unsupported file %s", path.name,
            )
            continue

        doc_type = hints.get(path.name) or classify_document_type(path.name)

        try:
            record = ext_module.extract(str(path))
        except Exception as e:
            logger.warning(
                "engagement_ingest: file extractor failed on %s — %s",
                path.name, e,
            )
            continue

        field_evidence = _extract_field_evidence(record)

        extractions.append(SourceDocumentExtraction(
            document_path=str(path),
            document_type=doc_type,
            field_evidence=field_evidence,
            extractor_version=extractor_version,
        ))
        logger.info(
            "engagement_ingest: ingested %s (doc_type=%s, %d field_evidence)",
            path.name, doc_type, len(field_evidence),
        )

    logger.info(
        "engagement_ingest: %s → %d source extractions",
        folder_path, len(extractions),
    )
    return extractions
