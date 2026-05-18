"""
auditai_data_normalization/router.py
=====================================
Shared spec-driven router engine. Detector only — never executes extractors.

Loads config/routing.yaml once at startup and resolves the correct extractor
module path for any file, for either phase.

Public API
----------
    route(file_path, phase) -> RouteResult
        Detect file type and return routing info. Never imports or calls
        the extractor — that is the caller's responsibility.

    RouteResult
        .file_type        str   — canonical type key from routing.yaml
        .extractor        str   — full Python module path, or None if skip
        .should_skip      bool  — True for index files, unsupported formats
        .detection_method str   — 'magic_bytes', 'extension', or 'pdf_subtype'

Design constraints
------------------
- Stateless. No phase-specific logic beyond table lookup.
- No extractor imports. No DocumentRecord. No Phase 1 or Phase 2 internals.
- PDF sub-routing (text-native vs scanned) is the only conditional logic
  that cannot live in YAML — it is handled here and nowhere else.
- Raises ValueError for truly unknown files (not in routing.yaml at all).
- Raises FileNotFoundError if file does not exist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "routing.yaml"

# PDF is text-native if average chars/page is at or above this threshold
_PDF_TEXT_NATIVE_THRESHOLD = 100

# Number of pages to sample when checking PDF text density
_PDF_SAMPLE_PAGES = 5

# Magic bytes for known formats (first 4 bytes, uppercase hex)
_MAGIC_ZIP = "504B0304"   # ZIP — .xlsx, .xlsm, .docx, .docm
_MAGIC_OLE = "D0CF11E0"   # OLE2 — legacy .xls, .doc, encrypted Office
_MAGIC_PDF = "25504446"   # %PDF


# ---------------------------------------------------------------------------
# RouteResult
# ---------------------------------------------------------------------------

PhaseStr = Literal["phase1", "phase2"]


@dataclass(frozen=True)
class RouteResult:
    """
    Output of route(). Tells the caller what to do with a file.
    Never contains an imported module or callable — detection only.
    """

    file_type: str
    """
    Canonical type key from routing.yaml.
    e.g. 'docx', 'pdf_text', 'pdf_scanned', 'xlsx', 'csv', 'json'
    """

    extractor: str | None
    """
    Full Python module path to the extract() function.
    e.g. 'auditai_data_normalization.extractors.docx_extractor.extract'
    None when should_skip=True.
    """

    should_skip: bool
    """
    True for index files, unsupported formats, and explicitly skipped types.
    Caller should raise ValueError or log and continue — never attempt extraction.
    """

    detection_method: str
    """
    How the file type was determined.
    'magic_bytes'  — confirmed from first 4 bytes of file
    'extension'    — extension-only (csv, json have no magic bytes)
    'pdf_subtype'  — PDF confirmed by magic bytes, sub-typed by text density
    """

    phase: PhaseStr
    """Which routing table was used."""


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_routing_config() -> dict:
    """
    Load and cache routing.yaml.
    Called once at first route() call — subsequent calls use the cache.
    """
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"routing.yaml not found at {_CONFIG_PATH}. "
            "This file is required for the router to function."
        )
    with open(_CONFIG_PATH) as f:
        config = yaml.safe_load(f) or {}
    logger.debug("Loaded routing config from %s", _CONFIG_PATH)
    return config


def _get_phase_table(phase: PhaseStr) -> dict:
    """Return the routing table for the given phase."""
    config = _load_routing_config()
    table = config.get(phase)
    if table is None:
        raise ValueError(
            f"Phase '{phase}' not found in routing.yaml. "
            f"Available phases: {list(config.keys())}"
        )
    return table


# ---------------------------------------------------------------------------
# Magic byte detection
# ---------------------------------------------------------------------------

def _read_magic(path: Path) -> str:
    """Read first 4 bytes of file and return as uppercase hex string."""
    with open(path, "rb") as f:
        raw = f.read(4)
    return raw.hex().upper()


def _magic_matches(magic: str, expected_hex: str | None) -> bool:
    """True if the file's magic bytes match the expected hex string."""
    if expected_hex is None:
        return False
    return magic.upper() == expected_hex.upper()


# ---------------------------------------------------------------------------
# PDF sub-routing
# ---------------------------------------------------------------------------

def _is_pdf_text_native(path: Path) -> bool:
    """
    Return True if the PDF is text-native (not scanned).

    Samples up to _PDF_SAMPLE_PAGES pages and checks average character count.
    Below _PDF_TEXT_NATIVE_THRESHOLD chars/page → likely scanned.

    Uses pdfplumber which is always available in the Phase 1 environment.
    Gracefully returns False on any error so the router falls back to
    pdf_scanned rather than crashing.
    """
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            if not pdf.pages:
                return False
            sample = pdf.pages[:min(_PDF_SAMPLE_PAGES, len(pdf.pages))]
            avg = sum(
                len(p.extract_text() or "") for p in sample
            ) / len(sample)
            return avg >= _PDF_TEXT_NATIVE_THRESHOLD
    except Exception as e:
        logger.warning(
            "PDF text-native check failed for %s: %s — defaulting to scanned",
            path.name, e,
        )
        return False


# ---------------------------------------------------------------------------
# Core routing logic
# ---------------------------------------------------------------------------

def _find_entry_by_extension(table: dict, suffix: str) -> tuple[str, dict] | None:
    """
    Find the first routing table entry whose extensions list includes suffix.
    Returns (entry_key, entry_dict) or None.
    Skips pdf_text and pdf_scanned — those are handled by PDF sub-routing.
    """
    for key, entry in table.items():
        if key in ("pdf_text", "pdf_scanned"):
            continue
        extensions = entry.get("extensions") or []
        if suffix in extensions:
            return key, entry
    return None


def _find_pdf_entries(table: dict) -> tuple[dict | None, dict | None]:
    """Return (pdf_text_entry, pdf_scanned_entry) from the routing table."""
    return (
        table.get("pdf_text"),
        table.get("pdf_scanned"),
    )


def _build_result(
    file_type: str,
    entry: dict,
    detection_method: str,
    phase: PhaseStr,
) -> RouteResult:
    """Build a RouteResult from a routing table entry."""
    return RouteResult(
        file_type=file_type,
        extractor=entry.get("extractor"),
        should_skip=bool(entry.get("skip", False)),
        detection_method=detection_method,
        phase=phase,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def route(file_path: str | Path, phase: PhaseStr) -> RouteResult:
    """
    Detect file type and return routing info for the given phase.

    Detection order
    ---------------
    1. Read magic bytes (first 4 bytes)
    2. Match magic bytes against routing table entries
    3. For PDFs: sub-route based on text density (text-native vs scanned)
    4. For formats with no magic bytes (csv, json): fall back to extension
    5. If nothing matches: raise ValueError

    Parameters
    ----------
    file_path : str | Path
        Path to the file to route. Must exist.
    phase : 'phase1' | 'phase2'
        Which routing table to use.

    Returns
    -------
    RouteResult
        Detection result. Caller is responsible for importing and calling
        the extractor at result.extractor.

    Raises
    ------
    FileNotFoundError
        If file_path does not exist.
    ValueError
        If file type cannot be determined or is not in routing.yaml,
        or if the phase is not found in routing.yaml.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    table = _get_phase_table(phase)

    # ------------------------------------------------------------------
    # 1. Read magic bytes
    # ------------------------------------------------------------------
    try:
        magic = _read_magic(path)
    except Exception as e:
        logger.warning("Could not read magic bytes from %s: %s", path.name, e)
        magic = ""

    # ------------------------------------------------------------------
    # 2. PDF detection and sub-routing
    # ------------------------------------------------------------------
    if magic == _MAGIC_PDF or suffix == ".pdf":
        pdf_text_entry, pdf_scanned_entry = _find_pdf_entries(table)

        is_text_native = _is_pdf_text_native(path)
        if is_text_native:
            if pdf_text_entry is None:
                raise ValueError(
                    f"pdf_text entry missing from routing.yaml [{phase}] table"
                )
            logger.debug("router: %s → pdf_text (text-native)", path.name)
            return _build_result("pdf_text", pdf_text_entry, "pdf_subtype", phase)
        else:
            if pdf_scanned_entry is None:
                raise ValueError(
                    f"pdf_scanned entry missing from routing.yaml [{phase}] table"
                )
            logger.debug("router: %s → pdf_scanned (scanned/image)", path.name)
            return _build_result("pdf_scanned", pdf_scanned_entry, "pdf_subtype", phase)

    # ------------------------------------------------------------------
    # 3. ZIP-based Office formats (.docx, .xlsx)
    # ------------------------------------------------------------------
    if magic == _MAGIC_ZIP:
        result = _find_entry_by_extension(table, suffix)
        if result:
            key, entry = result
            logger.debug("router: %s → %s (magic+extension)", path.name, key)
            return _build_result(key, entry, "magic_bytes", phase)

        # Ambiguous ZIP — inspect internal structure
        try:
            import zipfile
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
            if any("word/" in n for n in names):
                entry = table.get("docx")
                if entry:
                    return _build_result("docx", entry, "magic_bytes", phase)
            if any("xl/" in n for n in names):
                entry = table.get("xlsx")
                if entry:
                    return _build_result("xlsx", entry, "magic_bytes", phase)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 4. OLE2 binary formats (.xls, .doc, encrypted Office)
    # ------------------------------------------------------------------
    if magic == _MAGIC_OLE:
        result = _find_entry_by_extension(table, suffix)
        if result:
            key, entry = result
            logger.debug("router: %s → %s (OLE magic+extension)", path.name, key)
            return _build_result(key, entry, "magic_bytes", phase)

    # ------------------------------------------------------------------
    # 5. Extension-only formats (csv, json — no magic bytes)
    # ------------------------------------------------------------------
    result = _find_entry_by_extension(table, suffix)
    if result:
        key, entry = result
        logger.debug("router: %s → %s (extension only)", path.name, key)
        return _build_result(key, entry, "extension", phase)

    # ------------------------------------------------------------------
    # 6. Nothing matched
    # ------------------------------------------------------------------
    raise ValueError(
        f"{path.name}: file type could not be determined. "
        f"Extension '{suffix}', magic bytes '{magic}'. "
        f"Add an entry to config/routing.yaml [{phase}] to support this format."
    )


def reload_config() -> None:
    """
    Force reload of routing.yaml on next route() call.
    Useful in tests or when the config file is updated at runtime.
    """
    _load_routing_config.cache_clear()
    logger.info("router: routing config cache cleared")