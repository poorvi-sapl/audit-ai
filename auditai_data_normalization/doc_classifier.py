"""
auditai_data_normalization/doc_classifier.py
=============================================
Phase 2 — Document category detection.

Populates metadata["document_category"] on every DocumentRecord before
normalize.py runs the LLM fallback. This gives llm_extractor.py the
doc_type hint it needs to select the right extraction prompt and label
hints from _DOC_TYPE_CONTEXT.

Without this, extract_all_fields() always receives doc_type=None and
falls through to the "unknown" context — wasting the rich engagement_form
and financial_statement prompt variants that were built in Phase B1.

Categories (matching llm_extractor._DOC_TYPE_CONTEXT keys exactly)
-------------------------------------------------------------------
    "engagement_form"      PPC NPO-CX, engagement acceptance/continuance forms
    "financial_statement"  Audit reports, financial statements, draft financials
    "planning_memo"        Audit planning memos, risk assessments
    "unknown"              Fallback — no strong signal found

Detection strategy
------------------
Two signal sources, applied in order. First strong match wins.

1. Filename signals — fast, O(1), no content access needed.
   Covers the common case where firms follow naming conventions.
   Examples from the actual data folder:
       "NPO-CX-1_1 Engagement Accept and Cont Form.docx" → engagement_form
       "Final Audit Report The Rwanda School Project.pdf" → financial_statement
       "DraftFinancialStatements_2024_IITKF.pdf"         → financial_statement
       "SOP_NPO_CX_1_1_Final_v10.docx"                   → unknown (SOP, not a workpaper)

2. Content signals — section headings and first ~500 chars of cleaned_text.
   Used when filename is ambiguous (e.g. "Client_Workpaper_2024.docx").
   Scans for keyword clusters that uniquely identify each category.

Public API
----------
    detect_category(file_name, sections, cleaned_text) -> str
        Returns one of the four category strings above.
        Called by docx_extractor.extract() and pdf_text_extractor.extract()
        before building the metadata dict.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from auditai_data_normalization.schema import Section


# ---------------------------------------------------------------------------
# Filename signal patterns
# ---------------------------------------------------------------------------
# Ordered — first match wins. More specific patterns listed first.

_FILENAME_PATTERNS: list[tuple[re.Pattern, str]] = [

    # ── engagement_form ──────────────────────────────────────────────────
    # PPC form codes: NPO-CX, GOV-CX, NFP-CX etc.
    (re.compile(r"NPO[-_]CX|GOV[-_]CX|NFP[-_]CX|TRB[-_]CX", re.IGNORECASE),
     "engagement_form"),

    # Explicit "engagement" + "accept" / "cont" / "form" in filename
    (re.compile(r"engagement.{0,20}(accept|cont|form|decision)", re.IGNORECASE),
     "engagement_form"),

    # "accept" or "continuance" alone in filename
    (re.compile(r"(acceptance|continuance|accept\.?cont)", re.IGNORECASE),
    "engagement_form"),

    # ── financial_statement ──────────────────────────────────────────────
    # "Final Audit Report" — most common PDF filename in the data folder
    (re.compile(r"final.{0,10}audit.{0,10}report", re.IGNORECASE),
     "financial_statement"),

    # "Financial Statement" or "Financial Statements" in name
    (re.compile(r"financial.{0,5}statements?", re.IGNORECASE),
     "financial_statement"),

    # "Draft Financial" or "Audited Financial"
    (re.compile(r"(draft|audited).{0,10}financial", re.IGNORECASE),
     "financial_statement"),

    # ── planning_memo ────────────────────────────────────────────────────
    (re.compile(r"planning.{0,10}(memo|memorandum|doc)", re.IGNORECASE),
     "planning_memo"),

    (re.compile(r"risk.{0,10}(assessment|memo|matrix)", re.IGNORECASE),
     "planning_memo"),
]


def _classify_by_filename(file_name: str) -> str | None:
    """
    Return a category string if the filename matches a known pattern,
    or None if no strong filename signal found.
    """
    name = file_name.lower()

    for pattern, category in _FILENAME_PATTERNS:
        if pattern.search(name):
            return category

    # --- fallback safety rules (handles missed keywords) ---
    if any(k in name for k in [
        "engagement",
        "accept",
        "acceptance",
        "continuance",
    ]):
        return "engagement_form"

    return None

    
# ---------------------------------------------------------------------------
# Content signal keyword clusters
# ---------------------------------------------------------------------------
# Each entry: (category, required_keywords, optional_keywords, min_optional)
# A match fires when ALL required keywords AND at least min_optional of
# the optional keywords are found in the content sample.

_CONTENT_SIGNALS: list[tuple[str, list[str], list[str], int]] = [

    # engagement_form — must have acceptance/continuance language + at least
    # one scope-related term
    (
        "engagement_form",
        ["engagement"],                                     # required
        [                                                   # optional
            "accept", "continue", "continuance", "decline",
            "engagement partner", "engagement decision",
            "gagas", "single audit", "yellow book",
            "independence", "npo-cx", "engagement form",
        ],
        3,   # need 3 of the above
    ),

    # financial_statement — auditor report language is the primary signal
    (
        "financial_statement",
        ["financial statement"],                            # required
        [
            "independent auditor", "auditor's report", "opinion",
            "balance sheet", "statement of activities",
            "net assets", "total assets", "total revenue",
            "notes to financial", "management is responsible",
        ],
        2,
    ),

    # planning_memo — planning/risk language
    (
        "planning_memo",
        ["audit plan"],                                     # required
        [
            "risk assessment", "materiality", "planning memo",
            "preliminary", "significant risk", "inherent risk",
            "control risk", "fraud risk",
        ],
        2,
    ),
]


def _classify_by_content(
    sections: "list[Section]",
    cleaned_text: str,
) -> str | None:
    """
    Scan section headings and the first 800 chars of cleaned_text for
    keyword clusters that identify the document category.

    Returns a category string or None if no cluster fires.
    """
    # Build a compact content sample: all headings + first 800 chars of body
    heading_text = " ".join(
        s.heading.lower() for s in (sections or []) if s.heading
    )
    body_sample  = (cleaned_text or "")[:800].lower()
    sample       = heading_text + " " + body_sample

    for category, required, optional, min_opt in _CONTENT_SIGNALS:
        # All required terms must be present
        if not all(kw in sample for kw in required):
            continue
        # Count optional hits
        opt_hits = sum(1 for kw in optional if kw in sample)
        if opt_hits >= min_opt:
            return category

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_category(
    file_name: str,
    sections: "list[Section] | None" = None,
    cleaned_text: str = "",
) -> str:
    """
    Detect the document category for a workpaper.

    Parameters
    ----------
    file_name : str
        The file's base name (not full path). Used for filename signals.
    sections : list[Section] | None
        Extracted sections from the document. Headings are scanned for
        content signals. Pass None or [] if sections are not yet built.
    cleaned_text : str
        The document's cleaned_text string. First 800 chars are scanned.
        Pass "" if not yet available.

    Returns
    -------
    str
        One of: "engagement_form", "financial_statement",
                "planning_memo", "unknown"

    Examples
    --------
    >>> detect_category("NPO-CX-1_1 Engagement Accept and Cont Form.docx")
    'engagement_form'

    >>> detect_category("Final Audit Report The Rwanda School Project FY 2024.pdf")
    'financial_statement'

    >>> detect_category("Client_Workpaper_2024.docx", sections=[...], cleaned_text="...")
    'engagement_form'  # or whatever content signals fire
    """
    # 1. Filename is cheapest — try it first
    category = _classify_by_filename(file_name)
    if category is not None:
        return category

    # 2. Content signals — use when filename is ambiguous
    category = _classify_by_content(sections or [], cleaned_text)
    if category is not None:
        return category

    return "unknown"