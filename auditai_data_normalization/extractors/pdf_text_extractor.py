"""
auditai_data_normalization/extractors/pdf_text_extractor.py
=============================================================
Extracts text and structure from text-native (digital) PDF files
using pdfplumber as the primary engine and pdfminer.six as fallback.

When to use this extractor
---------------------------
Use when the PDF was digitally created (not scanned). Confirmed text-native
if average chars per page > 100. The format router in normalize.py checks
this automatically and routes scanned PDFs to ocr_extractor.py instead.

Tested against
--------------
ECEStep_Final_Financial_Statements_2024_111.pdf — 17 pages, 1958 avg
chars/page, financial statements with tabular numeric data.

What it produces
-----------------
sections  : one Section per page. Pages with a clear header line (all-caps,
            short) get that line as the section heading. Notes pages are
            split further at NOTE boundaries so each note is its own section.
tables    : pdfplumber table extraction per page. Empty/single-cell tables
            are dropped. Column headers inferred from first non-empty row.
metadata  : page_count, avg_chars_per_page, is_encrypted, error_pages,
            fallback_pages (pages that used pdfminer), scanned_page_count

Edge cases handled
------------------
- Encrypted PDFs         → raises PasswordProtectedError
- Scanned pages mixed in → flagged in metadata.error_pages, skipped cleanly
- Per-page timeouts      → page marked as error_page, pipeline continues
- Large PDFs > 100 pages → split into 25-page windows automatically
- pdfplumber empty page  → retried with pdfminer.six fallback
- Dot-leader TOC lines   → stripped (e.g. "Independent Auditor's Report....1")

Public API
----------
    extract(file_path: str | Path) -> DocumentRecord
    is_text_native(file_path: str | Path) -> bool
"""

from __future__ import annotations

import hashlib
import logging
import re
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pypdf import PdfReader

from auditai_data_normalization.schema import (
    DocumentRecord,
    ExtractedTable,
    Section,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Pages with fewer chars than this are likely scanned or image-only
_SCANNED_PAGE_THRESHOLD = 100

# Split large PDFs into windows of this size
_PAGE_WINDOW_SIZE = 25

# Per-page extraction timeout in seconds (Unix only)
_PAGE_TIMEOUT_SECONDS = 90

# Dot-leader pattern in TOC pages  e.g. "Independent Auditor's Report .....1"
_DOT_LEADER_RE = re.compile(r"\.{3,}\s*\d+\s*$", re.MULTILINE)

# All-caps short line — likely a page/section header
_HEADER_LINE_RE = re.compile(r"^[A-Z][A-Z\s\(\),\-\/\.]{3,60}$")

# Note header pattern  e.g. "NOTE 1 - ORGANIZATION" or "NOTE 2 -"
_NOTE_HEADER_RE = re.compile(r"^NOTE\s+\d+\s*[-–]\s*.+", re.IGNORECASE | re.MULTILINE)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class PasswordProtectedError(Exception):
    """Raised when the PDF is encrypted and cannot be opened."""


class PageTimeoutError(Exception):
    """Raised when a single page exceeds the extraction timeout."""


# ---------------------------------------------------------------------------
# Timeout context manager (Unix only)
# ---------------------------------------------------------------------------

@contextmanager
def _page_timeout(seconds: int) -> Generator:
    """Signal-based timeout for a single page extraction (Unix only)."""
    def _handler(signum, frame):
        raise PageTimeoutError(f"Page extraction timed out after {seconds}s")

    try:
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        yield
    finally:
        signal.alarm(0)


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

def _clean_page_text(text: str) -> str:
    """
    Clean raw pdfplumber text:
    - Strip dot-leader TOC lines
    - Collapse excessive whitespace
    - Remove form-feed characters
    """
    if not text:
        return ""
    text = _DOT_LEADER_RE.sub("", text)
    text = text.replace("\f", "\n")
    # Collapse runs of 3+ blank lines to two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _detect_heading(text: str) -> str:
    """
    Try to detect a section heading from the first few lines of page text.
    Returns the heading string or empty string if none detected.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    # First line is all-caps and short → heading
    if _HEADER_LINE_RE.match(lines[0]):
        return lines[0]
    return ""


# ---------------------------------------------------------------------------
# Table extraction helpers
# ---------------------------------------------------------------------------

def _build_table(raw_table: list[list], index: int, page_num: int) -> ExtractedTable | None:
    """
    Convert a pdfplumber raw table (list of rows, each a list of cell strings)
    to an ExtractedTable.

    Returns None for empty or single-cell tables (pdfplumber artefacts).
    """
    if not raw_table:
        return None

    # Clean cells — pdfplumber returns None for empty cells
    cleaned: list[list[str]] = []
    for row in raw_table:
        cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
        # Skip completely empty rows
        if any(c for c in cleaned_row):
            cleaned.append(cleaned_row)

    if not cleaned:
        return None

    # Drop single-cell tables (usually artefacts from pdfplumber layout analysis)
    if len(cleaned) == 1 and len(cleaned[0]) == 1:
        return None

    # First non-empty row as headers
    headers = cleaned[0]
    data_rows = cleaned[1:] if len(cleaned) > 1 else []

    # Build raw_text for embedding
    lines: list[str] = []
    lines.append(" | ".join(h for h in headers if h))
    lines.append("-" * 40)
    for row in data_rows:
        pairs = [
            f"{h}: {v}"
            for h, v in zip(headers, row)
            if h.strip() and v.strip()
        ]
        if pairs:
            lines.append(", ".join(pairs))
        elif any(v.strip() for v in row):
            lines.append(" | ".join(v for v in row if v.strip()))

    return ExtractedTable(
        index=index,
        source=f"Page {page_num}",
        headers=headers,
        rows=data_rows,
        raw_text="\n".join(lines),
    )


# ---------------------------------------------------------------------------
# Notes page splitter
# ---------------------------------------------------------------------------

def _split_notes_page(text: str, base_index: int, page_num: int) -> list[Section]:
    """
    Split a Notes page into one Section per NOTE.
    e.g. "NOTE 1 - ORGANIZATION", "NOTE 2 - SUMMARY OF SIGNIFICANT..."

    Falls back to one section for the whole page if no NOTE headers found.
    """
    sections: list[Section] = []
    current_heading = ""
    current_lines: list[str] = []
    idx = base_index

    def _flush():
        nonlocal idx
        content = "\n".join(current_lines).strip()
        if content or current_heading:
            sections.append(Section(
                index=idx,
                heading=current_heading,
                content=content,
                page_or_sheet=f"Page {page_num}",
                token_count=len(content.split()),
            ))
            idx += 1

    for line in text.splitlines():
        stripped = line.strip()
        if _NOTE_HEADER_RE.match(stripped):
            _flush()
            current_heading = stripped
            current_lines = []
        else:
            if stripped:
                current_lines.append(stripped)

    _flush()
    return sections if sections else [
        Section(
            index=base_index,
            heading=_detect_heading(text),
            content=text.strip(),
            page_or_sheet=f"Page {page_num}",
            token_count=len(text.split()),
        )
    ]


# ---------------------------------------------------------------------------
# pdfminer fallback
# ---------------------------------------------------------------------------

def _pdfminer_page_text(file_path: Path, page_num: int) -> str:
    """
    Extract text from a single page using pdfminer.six as fallback.
    page_num is 1-based.
    """
    try:
        text = pdfminer_extract_text(
            str(file_path),
            page_numbers=[page_num - 1],  # pdfminer uses 0-based
        )
        return _clean_page_text(text or "")
    except Exception as e:
        logger.warning("pdfminer fallback failed on page %d: %s", page_num, e)
        return ""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def is_text_native(file_path: str | Path) -> bool:
    """
    Return True if the PDF is text-native (not scanned).
    Checks average characters per page — below threshold = likely scanned.

    Used by the format router in normalize.py to decide between
    pdf_text_extractor and ocr_extractor.
    """
    path = Path(file_path)
    try:
        with pdfplumber.open(str(path)) as pdf:
            if not pdf.pages:
                return False
            sample = pdf.pages[:min(5, len(pdf.pages))]
            avg = sum(
                len(p.extract_text() or "") for p in sample
            ) / len(sample)
            return avg >= _SCANNED_PAGE_THRESHOLD
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main extraction logic
# ---------------------------------------------------------------------------

def _extract_window(
    file_path: Path,
    start: int,
    end: int,
) -> tuple[list[Section], list[ExtractedTable], list[int], list[int], int]:
    """
    Extract one window of pages [start, end) from the PDF.

    Returns
    -------
    sections, tables, error_pages, fallback_pages, scanned_count
    """
    sections: list[Section] = []
    tables: list[ExtractedTable] = []
    error_pages: list[int] = []
    fallback_pages: list[int] = []
    scanned_count = 0
    section_idx = start  # use page number as base index
    table_idx = 0

    with pdfplumber.open(str(file_path)) as pdf:
        for page_num in range(start + 1, end + 1):  # 1-based
            page = pdf.pages[page_num - 1]

            # --- Extract text with timeout ---
            text = ""
            try:
                with _page_timeout(_PAGE_TIMEOUT_SECONDS):
                    text = page.extract_text() or ""
            except PageTimeoutError:
                logger.warning("Timeout on page %d — skipping", page_num)
                error_pages.append(page_num)
                continue
            except Exception as e:
                logger.warning("pdfplumber failed on page %d: %s", page_num, e)

            # --- Fallback to pdfminer if pdfplumber returned nothing ---
            if not text.strip():
                text = _pdfminer_page_text(file_path, page_num)
                if text.strip():
                    fallback_pages.append(page_num)
                else:
                    # Likely a scanned page
                    scanned_count += 1
                    logger.debug("Page %d appears scanned or empty", page_num)
                    continue

            # --- Detect scanned page by char count ---
            if len(text) < _SCANNED_PAGE_THRESHOLD:
                scanned_count += 1
                logger.debug(
                    "Page %d low char count (%d) — may be scanned",
                    page_num, len(text),
                )

            text = _clean_page_text(text)

            # --- Extract tables ---
            try:
                raw_tables = page.extract_tables() or []
                for rt in raw_tables:
                    tbl = _build_table(rt, table_idx, page_num)
                    if tbl:
                        tables.append(tbl)
                        table_idx += 1
            except Exception as e:
                logger.debug("Table extraction failed on page %d: %s", page_num, e)

            # --- Build sections ---
            # Always strip the page heading first, then check body for NOTEs
            heading = _detect_heading(text)
            body = text
            if heading and text.startswith(heading):
                body = text[len(heading):].strip()

            # Notes pages: split body per NOTE header
            if _NOTE_HEADER_RE.search(body):
                page_sections = _split_notes_page(body, section_idx, page_num)
                # Prepend the page heading to the first note section
                if page_sections and heading:
                    page_sections[0].heading = (
                        f"{heading} — {page_sections[0].heading}"
                        if page_sections[0].heading
                        else heading
                    )
            else:
                page_sections = [Section(
                    index=section_idx,
                    heading=heading,
                    content=body,
                    page_or_sheet=f"Page {page_num}",
                    token_count=len(body.split()),
                )]

            sections.extend(page_sections)
            section_idx += len(page_sections)

    return sections, tables, error_pages, fallback_pages, scanned_count


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract(file_path: str | Path) -> DocumentRecord:
    """
    Extract content from a text-native PDF file.

    Parameters
    ----------
    file_path : str | Path
        Path to a .pdf file. Must exist and be readable.

    Returns
    -------
    DocumentRecord
        Populated with sections, tables, metadata, and provenance fields.
        pii_scrubbed=False — call pii.scrub_record() after this.

    Raises
    ------
    FileNotFoundError
        If file_path does not exist.
    PasswordProtectedError
        If the PDF is encrypted.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Hash and size
    raw_bytes = path.read_bytes()
    file_hash = hashlib.sha256(raw_bytes).hexdigest()
    file_size = len(raw_bytes)

    # Check encryption
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise PasswordProtectedError(
                f"{path.name} is encrypted. Supply password via intake API."
            )
        page_count = len(reader.pages)
    except PasswordProtectedError:
        raise
    except Exception as e:
        return DocumentRecord(
            source_path=str(path),
            file_name=path.name,
            file_type="pdf_text",
            file_size_bytes=file_size,
            file_hash=file_hash,
            extraction_method="pdfplumber",
            extraction_status="failed",
            extraction_error=str(e),
            needs_review=True,
        )

    # --- Process in windows for large PDFs ---
    all_sections: list[Section] = []
    all_tables: list[ExtractedTable] = []
    all_error_pages: list[int] = []
    all_fallback_pages: list[int] = []
    total_scanned = 0

    windows = [
        (i, min(i + _PAGE_WINDOW_SIZE, page_count))
        for i in range(0, page_count, _PAGE_WINDOW_SIZE)
    ]

    for start, end in windows:
        try:
            secs, tbls, errs, falls, scanned = _extract_window(path, start, end)
            all_sections.extend(secs)
            all_tables.extend(tbls)
            all_error_pages.extend(errs)
            all_fallback_pages.extend(falls)
            total_scanned += scanned
        except Exception as e:
            logger.error("Window [%d-%d] failed: %s", start, end, e)
            all_error_pages.extend(range(start + 1, end + 1))

    # --- Determine status ---
    error_ratio = len(all_error_pages) / max(page_count, 1)
    if not all_sections:
        status = "failed"
        error_msg = "No content extracted from any page"
    elif error_ratio > 0.5:
        status = "partial"
        error_msg = f"{len(all_error_pages)} of {page_count} pages failed"
    else:
        status = "success"
        error_msg = ""

    if all_error_pages:
        logger.warning(
            "%s: %d pages failed extraction: %s",
            path.name, len(all_error_pages), all_error_pages[:10],
        )

    # --- Build cleaned_text ---
    cleaned_text = "\n\n".join(
        (f"{s.heading}\n{s.content}" if s.heading else s.content).strip()
        for s in all_sections
        if s.content.strip() or s.heading.strip()
    )

    word_count = len(cleaned_text.split())
    avg_chars = (
        sum(len(s.content) for s in all_sections) / max(len(all_sections), 1)
    )

    # --- Determine file_type: did we end up needing OCR routing? ---
    # If >30% of pages were scanned, flag so format router knows
    needs_ocr = total_scanned / max(page_count, 1) > 0.3
    file_type = "pdf_scanned" if needs_ocr else "pdf_text"

    metadata = {
        "page_count": page_count,
        "avg_chars_per_page": round(avg_chars, 1),
        "is_encrypted": False,
        "error_pages": all_error_pages,
        "fallback_pages": all_fallback_pages,
        "scanned_page_count": total_scanned,
        "needs_ocr_routing": needs_ocr,
        "section_count": len(all_sections),
        "table_count": len(all_tables),
    }

    logger.info(
        "pdf_text_extractor: %s — %d pages, %d sections, %d tables, "
        "%d errors, status=%s",
        path.name, page_count, len(all_sections),
        len(all_tables), len(all_error_pages), status,
    )

    return DocumentRecord(
        source_path=str(path),
        file_name=path.name,
        file_type=file_type,
        file_size_bytes=file_size,
        file_hash=file_hash,
        raw_text="\n\n".join(s.content for s in all_sections),
        cleaned_text=cleaned_text,
        sections=all_sections,
        tables=all_tables,
        extraction_method="pdfplumber",
        extraction_status=status,
        extraction_error=error_msg,
        extraction_confidence=0.0,
        word_count=word_count,
        page_count=page_count,
        ocr_used=False,
        pii_scrubbed=False,
        needs_review=(status != "success"),
        metadata=metadata,
    )