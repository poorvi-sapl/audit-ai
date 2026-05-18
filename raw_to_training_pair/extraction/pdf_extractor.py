"""
raw_to_training_pair/extraction/pdf_extractor.py
=================================================
Phase 2 raw extractor for PDF files — handles both text-native and scanned.

Contract
--------
Returns a plain dict — no DocumentRecord, no normalization.
OCR tier selection mirrors Phase 1 (Docling+Surya → Tesseract) but output
is raw text only, not a structured DocumentRecord.

Output shape
------------
{
    "text"    : str,   # full extracted text, pages joined by double newline
    "tables"  : list,  # [{"headers": [...], "rows": [[...]]}] from pdfplumber
    "metadata": dict   # page_count, ocr_used, extraction_method, error_pages
}

Public API
----------
    extract(file_path: str | Path) -> dict
"""

from __future__ import annotations

import logging
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

logger = logging.getLogger(__name__)

_SCANNED_PAGE_THRESHOLD = 100
_RENDER_DPI = 300


def _is_text_native(path: Path) -> bool:
    """Return True if PDF has readable text (not scanned)."""
    try:
        with pdfplumber.open(str(path)) as pdf:
            if not pdf.pages:
                return False
            sample = pdf.pages[:min(5, len(pdf.pages))]
            avg = sum(len(p.extract_text() or "") for p in sample) / len(sample)
            return avg >= _SCANNED_PAGE_THRESHOLD
    except Exception:
        return False


def _extract_text_native(path: Path) -> tuple[str, list[dict], list[int]]:
    """Extract text and tables from a text-native PDF via pdfplumber."""
    pages_text = []
    tables = []
    error_pages = []

    try:
        reader = PdfReader(str(path))
        page_count = len(reader.pages)
    except Exception as e:
        logger.warning("pypdf failed on %s: %s", path.name, e)
        return "", [], []

    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                text = page.extract_text() or ""
                pages_text.append(text.strip())

                # Extract tables
                raw_tables = page.extract_tables() or []
                for rt in raw_tables:
                    cleaned = [
                        [str(cell).strip() if cell else "" for cell in row]
                        for row in rt
                        if any(cell for cell in row)
                    ]
                    if len(cleaned) >= 2:
                        tables.append({
                            "headers": cleaned[0],
                            "rows": cleaned[1:],
                        })
            except Exception as e:
                logger.warning("pdfplumber failed on page %d: %s", i + 1, e)
                error_pages.append(i + 1)

    return "\n\n".join(t for t in pages_text if t), tables, error_pages


def _extract_scanned(path: Path) -> tuple[str, list[int]]:
    """Extract text from a scanned PDF via OCR (Docling+Surya or Tesseract)."""
    pages_text = []
    error_pages = []

    # Try Docling+Surya first
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(str(path))
        text = result.document.export_to_text()
        logger.info("p2/pdf_extractor: %s — Docling OCR", path.name)
        return text, []
    except ImportError:
        pass  # Docling not installed — fall through to Tesseract
    except Exception as e:
        logger.warning("Docling failed on %s: %s — falling back to Tesseract", path.name, e)

    # Tesseract fallback
    try:
        import pytesseract
        from pdf2image import convert_from_path

        images = convert_from_path(str(path), dpi=_RENDER_DPI)
        for i, image in enumerate(images):
            try:
                text = pytesseract.image_to_string(image, lang="eng")
                pages_text.append(text.strip())
            except Exception as e:
                logger.warning("Tesseract failed on page %d: %s", i + 1, e)
                error_pages.append(i + 1)

        logger.info("p2/pdf_extractor: %s — Tesseract OCR", path.name)
        return "\n\n".join(t for t in pages_text if t), error_pages

    except Exception as e:
        logger.error("All OCR methods failed for %s: %s", path.name, e)
        return "", list(range(1, len(pages_text) + 1))


def extract(file_path: str | Path) -> dict:
    """
    Extract raw text from a PDF file (text-native or scanned).

    Parameters
    ----------
    file_path : str | Path

    Returns
    -------
    dict with keys: text, tables, metadata

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        reader = PdfReader(str(path))
        page_count = len(reader.pages)
    except Exception as e:
        logger.warning("Could not read page count from %s: %s", path.name, e)
        page_count = 0

    is_native = _is_text_native(path)

    if is_native:
        text, tables, error_pages = _extract_text_native(path)
        method = "pdfplumber"
        ocr_used = False
    else:
        text, error_pages = _extract_scanned(path)
        tables = []
        method = "ocr"
        ocr_used = True

    logger.info(
        "p2/pdf_extractor: %s — %s, %d chars, %d error pages",
        path.name, method, len(text), len(error_pages),
    )

    return {
        "text": text,
        "tables": tables,
        "metadata": {
            "page_count": page_count,
            "ocr_used": ocr_used,
            "extraction_method": method,
            "error_pages": error_pages,
        },
    }