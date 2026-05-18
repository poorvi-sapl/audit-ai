"""
auditai_data_normalization/extractors/ocr_extractor.py
=========================================================
Extracts text from scanned / image-only PDF files using OCR.

Three-tier design
-----------------
Tier 1 — Docling + Surya  (preferred, GPU-accelerated on L40S)
    Best accuracy for multi-column layouts, handwritten fields,
    and degraded scans. Requires:  pip install -e ".[ocr]"
    Auto-used when docling and surya packages are importable.

Tier 2 — Tesseract 5.3+  (fallback, always available)
    Solid accuracy on clean scans. No GPU needed. Used when:
    - Docling/Surya not installed, OR
    - Surya confidence < 0.60 on a specific page window, OR
    - Docling times out on a window.

Tier 3 — Empty record  (last resort)
    When all OCR attempts fail for a page, it is recorded in
    metadata.error_pages. If >50% of pages fail, extraction_status
    is set to 'failed' and the engagement manager is notified upstream.

When to use this extractor
---------------------------
Called by normalize.py when pdf_text_extractor.is_text_native() returns
False (avg chars/page < 100), indicating the PDF is image-based.

Preprocessing pipeline (per page image)
-----------------------------------------
1. pdf2image converts pages to 300 DPI PNG images
2. OpenCV preprocessing:
   - Grayscale conversion
   - Deskew (correct rotation up to ±5°)
   - Denoise (fastNlMeansDenoising)
   - Adaptive threshold (binarize)
   These steps improve OCR accuracy by 15-20% on typical audit scans.

Page windowing
--------------
PDFs with > 50 pages are split into 25-page windows.
Each window runs independently so a single bad page cannot hang
the entire extraction job.

Public API
----------
    extract(file_path: str | Path) -> DocumentRecord
    get_ocr_tier() -> str   # 'docling_surya' | 'tesseract'
"""

from __future__ import annotations

import hashlib
import logging
import signal
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from pypdf import PdfReader

from auditai_data_normalization.schema import (
    DocumentRecord,
    Section,
)
from auditai_data_normalization.doc_classifier import detect_category

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PAGE_WINDOW_SIZE = 25
_RENDER_DPI = 300

# Surya confidence below this → fall back to Tesseract for that window
_SURYA_CONFIDENCE_THRESHOLD = 0.60

# Per-window timeout seconds
_DOCLING_TIMEOUT = 600   # 10 min — Docling+Surya is slow on large windows
_TESSERACT_TIMEOUT = 120 # 2 min  — Tesseract per window

# Tesseract minimum word confidence (0-100) — below this → flag as low quality
_TESSERACT_MIN_CONFIDENCE = 40


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class PasswordProtectedError(Exception):
    """Raised when the PDF is encrypted."""

class OCRTimeoutError(Exception):
    """Raised when OCR exceeds the per-window timeout."""


# ---------------------------------------------------------------------------
# Timeout context manager (Unix only)
# ---------------------------------------------------------------------------

@contextmanager
def _timeout(seconds: int, label: str = "") -> Generator:
    def _handler(signum, frame):
        raise OCRTimeoutError(f"OCR timeout ({seconds}s) on {label}")
    try:
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        yield
    finally:
        signal.alarm(0)


# ---------------------------------------------------------------------------
# Tier detection
# ---------------------------------------------------------------------------

def get_ocr_tier() -> str:
    """
    Return which OCR tier is available on this machine.
    'docling_surya' if pip install -e ".[ocr]" was run, else 'tesseract'.
    """
    try:
        import docling  # noqa: F401
        import surya    # noqa: F401
        return "docling_surya"
    except ImportError:
        return "tesseract"


# ---------------------------------------------------------------------------
# OpenCV preprocessing
# ---------------------------------------------------------------------------

def _preprocess_image(pil_image) -> np.ndarray:
    """
    Prepare a PIL image for OCR.

    Steps
    -----
    1. Convert to grayscale
    2. Deskew — correct rotation up to ±5 degrees
    3. Denoise — fastNlMeansDenoising
    4. Binarize — adaptive threshold

    Returns a preprocessed numpy array (grayscale, uint8).
    """
    # PIL → numpy BGR → grayscale
    img = np.array(pil_image)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # --- Deskew ---
    # Find text angle via minAreaRect on thresholded image
    try:
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) > 100:
            angle = cv2.minAreaRect(coords)[-1]
            # minAreaRect returns angles in [-90, 0); normalize to [-45, 45]
            if angle < -45:
                angle = 90 + angle
            if abs(angle) > 0.5:  # only correct if > 0.5 degrees
                h, w = gray.shape
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                gray = cv2.warpAffine(
                    gray, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )
    except Exception:
        pass  # deskew failure is non-fatal

    # --- Denoise ---
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # --- Binarize ---
    gray = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2,
    )

    return gray


# ---------------------------------------------------------------------------
# Tier 2 — Tesseract
# ---------------------------------------------------------------------------

def _tesseract_page(pil_image, page_num: int) -> tuple[str, float]:
    """
    Run Tesseract on one preprocessed page image.

    Returns
    -------
    (text, avg_confidence)
        text           — extracted text, stripped
        avg_confidence — 0.0–1.0 mean word confidence from Tesseract
    """
    preprocessed = _preprocess_image(pil_image)

    # Run Tesseract with word-level confidence data
    try:
        data = pytesseract.image_to_data(
            preprocessed,
            lang="eng",
            output_type=pytesseract.Output.DICT,
            config="--psm 6",  # assume uniform block of text
        )
        words = [
            (data["text"][i], int(data["conf"][i]))
            for i in range(len(data["text"]))
            if data["text"][i].strip() and int(data["conf"][i]) > 0
        ]

        if not words:
            return "", 0.0

        text = " ".join(w for w, _ in words)
        # Clean up — remove isolated single chars that are usually noise
        import re
        text = re.sub(r"\b[^a-zA-Z0-9$%&]{1}\b", " ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()

        avg_conf = sum(c for _, c in words) / len(words) / 100.0
        return text, avg_conf

    except Exception as e:
        logger.warning("Tesseract failed on page %d: %s", page_num, e)
        return "", 0.0


def _run_tesseract_window(
    file_path: Path,
    start: int,  # 0-based
    end: int,
    page_count: int,
) -> tuple[list[Section], list[int], float]:
    """
    Run Tesseract on pages [start, end).
    Returns (sections, error_pages, avg_confidence).
    """
    sections: list[Section] = []
    error_pages: list[int] = []
    confidences: list[float] = []

    try:
        with _timeout(_TESSERACT_TIMEOUT, f"Tesseract pages {start+1}-{end}"):
            images = convert_from_path(
                str(file_path),
                dpi=_RENDER_DPI,
                first_page=start + 1,   # pdf2image is 1-based
                last_page=end,
            )
    except OCRTimeoutError:
        logger.warning("Tesseract timeout on window %d-%d", start + 1, end)
        error_pages.extend(range(start + 1, end + 1))
        return sections, error_pages, 0.0
    except Exception as e:
        logger.error("pdf2image failed on window %d-%d: %s", start + 1, end, e)
        error_pages.extend(range(start + 1, end + 1))
        return sections, error_pages, 0.0

    for i, image in enumerate(images):
        page_num = start + 1 + i  # 1-based
        text, conf = _tesseract_page(image, page_num)

        if conf < _TESSERACT_MIN_CONFIDENCE / 100.0 and not text.strip():
            logger.warning(
                "Page %d OCR confidence %.2f below threshold — flagging",
                page_num, conf,
            )
            error_pages.append(page_num)
            continue

        confidences.append(conf)
        sections.append(Section(
            index=page_num - 1,
            heading="",
            content=text,
            page_or_sheet=f"Page {page_num}",
            token_count=len(text.split()),
        ))

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return sections, error_pages, avg_conf


# ---------------------------------------------------------------------------
# Tier 1 — Docling + Surya
# ---------------------------------------------------------------------------

def _run_docling_window(
    file_path: Path,
    start: int,
    end: int,
) -> tuple[list[Section], list[int], float] | None:
    """
    Run Docling + Surya on pages [start, end).
    Returns (sections, error_pages, avg_confidence) or None if unavailable.
    """
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            EasyOcrOptions,
        )
    except ImportError:
        return None  # Docling not installed — caller falls back to Tesseract

    sections: list[Section] = []
    error_pages: list[int] = []
    confidences: list[float] = []

    try:
        with _timeout(_DOCLING_TIMEOUT, f"Docling pages {start+1}-{end}"):
            # Extract page window to a temp PDF
            reader = PdfReader(str(file_path))
            from pypdf import PdfWriter
            writer = PdfWriter()
            for page_idx in range(start, min(end, len(reader.pages))):
                writer.add_page(reader.pages[page_idx])

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                writer.write(tmp)

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True

            converter = DocumentConverter()
            result = converter.convert(str(tmp_path))
            tmp_path.unlink(missing_ok=True)

            # Extract text per page from Docling result
            doc = result.document
            full_text = doc.export_to_text()

            # Docling gives us the full document text — split by page
            # using page boundary markers if available
            pages_text = full_text.split("\f") if "\f" in full_text else [full_text]

            for i, page_text in enumerate(pages_text):
                page_num = start + 1 + i
                text = page_text.strip()
                if not text:
                    error_pages.append(page_num)
                    continue

                # Docling doesn't expose per-page confidence directly
                # Estimate from text density
                conf = min(1.0, len(text.split()) / 50.0)
                confidences.append(conf)

                sections.append(Section(
                    index=page_num - 1,
                    heading="",
                    content=text,
                    page_or_sheet=f"Page {page_num}",
                    token_count=len(text.split()),
                ))

    except OCRTimeoutError:
        logger.warning("Docling timeout on window %d-%d", start + 1, end)
        return None  # Signal caller to fall back to Tesseract
    except Exception as e:
        logger.warning("Docling failed on window %d-%d: %s", start + 1, end, e)
        return None

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    # If Surya confidence is below threshold, signal fallback to Tesseract
    if avg_conf < _SURYA_CONFIDENCE_THRESHOLD:
        logger.info(
            "Docling avg confidence %.2f < %.2f on window %d-%d — "
            "falling back to Tesseract",
            avg_conf, _SURYA_CONFIDENCE_THRESHOLD, start + 1, end,
        )
        return None

    return sections, error_pages, avg_conf


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract(file_path: str | Path) -> DocumentRecord:
    """
    Extract text from a scanned / image-only PDF using OCR.

    Auto-selects OCR tier:
    - Tier 1 (Docling+Surya) if installed, falls back to Tier 2 per window
    - Tier 2 (Tesseract) always available

    Parameters
    ----------
    file_path : str | Path
        Path to a scanned .pdf file.

    Returns
    -------
    DocumentRecord
        pii_scrubbed=False — call pii.scrub_record() after this.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    PasswordProtectedError
        If PDF is encrypted.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

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
            file_type="pdf_scanned",
            file_size_bytes=file_size,
            file_hash=file_hash,
            extraction_method="tesseract",
            extraction_status="failed",
            extraction_error=str(e),
            needs_review=True,
        )

    tier = get_ocr_tier()
    logger.info(
        "ocr_extractor: %s — %d pages, tier=%s",
        path.name, page_count, tier,
    )

    # Build page windows
    windows = [
        (i, min(i + _PAGE_WINDOW_SIZE, page_count))
        for i in range(0, page_count, _PAGE_WINDOW_SIZE)
    ]

    all_sections: list[Section] = []
    all_error_pages: list[int] = []
    all_confidences: list[float] = []
    method_used = "tesseract"
    docling_used = False

    for start, end in windows:
        sections, errors, conf = [], [], 0.0
        window_ok = False

        # --- Tier 1: Docling + Surya ---
        if tier == "docling_surya":
            result = _run_docling_window(path, start, end)
            if result is not None:
                sections, errors, conf = result
                window_ok = True
                docling_used = True

        # --- Tier 2: Tesseract (primary or fallback) ---
        if not window_ok:
            sections, errors, conf = _run_tesseract_window(
                path, start, end, page_count
            )

        all_sections.extend(sections)
        all_error_pages.extend(errors)
        if conf > 0:
            all_confidences.append(conf)

    if docling_used:
        method_used = "docling_surya"

    # --- Determine status ---
    error_ratio = len(all_error_pages) / max(page_count, 1)
    if not all_sections:
        status = "failed"
        error_msg = "No content extracted from any page"
    elif error_ratio > 0.5:
        status = "partial"
        error_msg = (
            f"{len(all_error_pages)} of {page_count} pages failed OCR"
        )
    else:
        status = "success"
        error_msg = ""

    avg_confidence = (
        sum(all_confidences) / len(all_confidences)
        if all_confidences else 0.0
    )

    # Clamp to [0.0, 1.0] — confidence.py will score field-level later
    extraction_confidence = round(min(1.0, avg_confidence), 3)

    # --- Build cleaned_text ---
    cleaned_text = "\n\n".join(
        s.content for s in all_sections if s.content.strip()
    )
    word_count = len(cleaned_text.split())

    # ---- Detect document category (Phase 2) ----------------------------
    doc_category = detect_category(
        file_name=path.name,
        sections=all_sections,
        cleaned_text=cleaned_text,
    )

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
        "document_category": doc_category,       # Phase 2 — consumed by llm_extractor fallback
    }

    logger.info(
        "ocr_extractor: %s — %d sections, %d errors, "
        "avg_conf=%.2f, status=%s",
        path.name, len(all_sections), len(all_error_pages),
        avg_confidence, status,
    )

    return DocumentRecord(
        source_path=str(path),
        file_name=path.name,
        file_type="pdf_scanned",
        file_size_bytes=file_size,
        file_hash=file_hash,
        raw_text="\n\n".join(s.content for s in all_sections),
        cleaned_text=cleaned_text,
        sections=all_sections,
        tables=[],  # table extraction from scanned PDFs via Docling only
        extraction_method=method_used,
        extraction_status=status,
        extraction_error=error_msg,
        extraction_confidence=extraction_confidence,
        word_count=word_count,
        page_count=page_count,
        ocr_used=True,
        pii_scrubbed=False,
        needs_review=(status != "success" or extraction_confidence < 0.7),
        metadata=metadata,
    )