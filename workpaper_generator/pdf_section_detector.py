"""
workpaper_generator/pdf_section_detector.py
=============================================
Detects presence/absence of named sections in client audit report PDFs.
Drives Q1(c), Q1(d), and Q2 reference resolution in NPO-CX-1.1.

For NPO-CX-1.1 the lookups are:
  sefa           Schedule of Expenditures of Federal Awards  -> Q1(c)
  supplementary  Supplementary Information                    -> Q1(d)
  compliance     Compliance Section / Report on Compliance    -> Q2 reference

Regex on normalized text is used instead of vector search: these are
stable named-section titles, so deterministic matching gives higher
precision and zero index overhead.

Handles both clean PDF rendering and letter-spaced TOCs with dot
leaders (the 'I.n..d..e..p..e..n..d..e..n..t' style produced by some
audit report templates) via dot-strip + whitespace-collapse
normalization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber

_TOC_PAGE_WINDOW = 5
_MIN_EXTRACTABLE_CHARS = 200
_LOW_QUALITY_CHARS = 2000

SECTION_PATTERNS: dict[str, list[str]] = {
    "sefa": [
        r"schedule\s+of\s+expenditures\s+of\s+federal\s+awards",
        r"\bsefa\b",
        r"schedule\s+of\s+federal\s+awards",
    ],
    "supplementary": [
        r"supplementary\s+information",
        r"supplemental\s+information",
        r"other\s+supplementary\s+information",
    ],
    "compliance": [
        r"report\s+on\s+compliance",
        r"compliance\s+section",
        r"single\s+audit",
        r"uniform\s+guidance",
        r"omb\s+a-?133",
        r"2\s*cfr\s*part\s*200",
    ],
}


@dataclass
class SectionDetectionResult:
    found: bool
    match_text: Optional[str] = None
    location: Optional[str] = None   # 'toc' | 'body'
    page_hint: Optional[int] = None  # 1-indexed page number


@dataclass
class PDFDetectionReport:
    pdf_path: str
    page_count: int
    total_chars_extracted: int
    extraction_quality: str          # 'good' | 'low' | 'failed'
    sections: dict[str, SectionDetectionResult] = field(default_factory=dict)
    header_hints: dict[str, Optional[str]] = field(default_factory=dict)


def _normalize(text: str) -> str:
    text = re.sub(r"\.", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def _extract_pages(pdf_path: Path) -> list[str]:
    with pdfplumber.open(pdf_path) as pdf:
        return [page.extract_text() or "" for page in pdf.pages]


def _find_section(
    target: str,
    toc_text_norm: str,
    pages_norm: list[str],
) -> SectionDetectionResult:
    patterns = SECTION_PATTERNS[target]

    for pattern in patterns:
        m = re.search(pattern, toc_text_norm)
        if m:
            return SectionDetectionResult(
                found=True, match_text=m.group(0), location="toc", page_hint=None
            )

    for pattern in patterns:
        for page_idx, page_norm in enumerate(pages_norm):
            m = re.search(pattern, page_norm)
            if m:
                return SectionDetectionResult(
                    found=True,
                    match_text=m.group(0),
                    location="body",
                    page_hint=page_idx + 1,
                )

    return SectionDetectionResult(found=False)


_FYE_PATTERN = re.compile(
    r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+(\d{4})",
    re.IGNORECASE,
)


_HEADER_SKIP_TOKENS = (
    "AUDITED",
    "FINANCIAL STATEMENTS",
    "REPORT",
    "TABLE OF CONTENTS",
    "INDEPENDENT",
    "INDEPENDENT AUDITOR",
    "PAGE",
)

# PDFs with shifted font encoding (e.g. IMM, OCGI) produce control chars
# like \x03 \x0f \x16. Treat any line containing such bytes as unreadable.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f]")


def _is_readable(line: str) -> bool:
    return not _CONTROL_CHAR_RE.search(line)


def _extract_header_hints(pages_raw: list[str]) -> dict[str, Optional[str]]:
    """
    Suggest header values for the auditor. SOP says these are manual entry,
    so these are hints only — the renderer presents them as pre-fill
    suggestions, not authoritative values.

    Org name is consistent across years and reusable as-is.
    FYE date extracted here is the PRIOR YEAR's date (we are reading the
    PY audit report). The workpaper's 'Statement of Financial Position Date'
    is the CURRENT engagement's FYE — the auditor must enter that manually,
    typically PY + 1 year.

    Scans first 3 pages because some PDFs (IMM, OCGI) render the cover page
    with shifted font encoding; the TOC page (typically page 2) has clean
    text with the org name as the first non-skip line.
    """
    org_name: Optional[str] = None
    py_fye_date: Optional[str] = None

    scan_pages = pages_raw[:3]

    for page_text in scan_pages:
        if org_name and py_fye_date:
            break
        lines = [l.strip() for l in page_text.split("\n") if l.strip()]
        for line in lines[:8]:
            if not _is_readable(line):
                continue
            if any(tok in line.upper() for tok in _HEADER_SKIP_TOKENS):
                continue
            if org_name is None and line.isupper() and 2 <= len(line.split()) <= 10:
                org_name = line.title().rstrip(",")
        if py_fye_date is None:
            for line in lines:
                if not _is_readable(line):
                    continue
                m = _FYE_PATTERN.search(line)
                if m:
                    py_fye_date = m.group(0)
                    break

    return {
        "organization_name": org_name,
        "prior_year_fye_date": py_fye_date,
    }


def _classify_quality(total_chars: int) -> str:
    if total_chars < _MIN_EXTRACTABLE_CHARS:
        return "failed"
    if total_chars < _LOW_QUALITY_CHARS:
        return "low"
    return "good"


def detect(pdf_path: str | Path) -> PDFDetectionReport:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages_raw = _extract_pages(pdf_path)
    pages_norm = [_normalize(p) for p in pages_raw]
    total_chars = sum(len(p) for p in pages_raw)

    toc_text_norm = " ".join(pages_norm[:_TOC_PAGE_WINDOW])
    sections = {
        target: _find_section(target, toc_text_norm, pages_norm)
        for target in SECTION_PATTERNS
    }
    header_hints = _extract_header_hints(pages_raw)

    return PDFDetectionReport(
        pdf_path=str(pdf_path),
        page_count=len(pages_raw),
        total_chars_extracted=total_chars,
        extraction_quality=_classify_quality(total_chars),
        sections=sections,
        header_hints=header_hints,
    )
