"""
engineering_benchmark/sop_chunker.py
======================================
Chunks SOP documents into fixed-size token windows for embedding.

Reads all configuration from config/settings.py.

What it does
------------
1. Accepts a SOP document as a string or file path
2. Strips page headers/footers (top 5% + bottom 5% of each page by line count)
3. Splits into chunks of sop_chunker.chunk_size_tokens tokens
   with sop_chunker.chunk_overlap_tokens overlap
4. Keeps tables whole — never splits a table across chunk boundaries
5. Prepends "SOP §X.X — [Section]: " to every chunk if section prefix detected
6. Returns a list of SOPChunk dataclasses ready for embedder.py

Tokenizer
---------
Uses tiktoken (cl100k_base) as specified in stack.yaml sop_chunker.tokenizer.
tiktoken is the correct tokenizer for SOP chunking per the benchmark deck —
it gives consistent token counts regardless of the embedding model used.

Public API
----------
    chunk_text(text, source_doc, sop_version) -> list[SOPChunk]
    chunk_file(file_path, sop_version)        -> list[SOPChunk]
    SOPChunk                                  — dataclass for one chunk
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from config.settings import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SOPChunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class SOPChunk:
    """One chunk of SOP text, ready for embedding and Qdrant upsert."""

    chunk_id: str
    """sha256(source_doc + str(char_start)) — used as Qdrant point ID."""

    source_doc: str
    """Original SOP filename."""

    sop_version: str
    """Version identifier e.g. '2024-Q1', 'v3.2'."""

    content: str
    """Full chunk text including section prefix."""

    section_prefix: str
    """Detected section prefix e.g. 'SOP §3.1 — Reconciliation: '"""

    char_start: int
    """Character offset of this chunk in the original document."""

    char_end: int
    """End character offset."""

    token_count: int
    """Approximate token count via tiktoken."""

    workpaper_type: str = ""
    """Coarse workflow this SOP section applies to (if detectable).

    Examples: 'engagement_acceptance', 'bank_reconciliation',
    'financial_statements'. One workflow may cover multiple specific
    workpaper IDs (e.g., NPO-CX-1.1, GOV-CX-1.1, FP-CX-1.1, TRB-CX-1.1
    all fall under 'engagement_acceptance').
    """

    workpaper_ids: list[str] = field(default_factory=list)
    """Optional list of specific workpaper IDs this chunk applies to.

    Default empty means the chunk is workflow-level and applies to all
    workpaper IDs in its workflow. Populate this only when the chunk
    text is specific to a particular workpaper (e.g., a chunk citing
    501(c)(3) specifically would carry workpaper_ids=['NPO-CX-1.1']).

    Retrieval logic: a chunk matches if its workpaper_type matches the
    query's workflow AND (its workpaper_ids is empty OR contains the
    query's workpaper_id).
    """

    is_table: bool = False
    """True if this chunk contains a table kept whole."""

    metadata: dict = field(default_factory=dict)
    """Additional metadata for Qdrant payload."""


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# SOP section header patterns — e.g. "§3.1", "Section 3.1", "3.1 Reconciliation"
_SECTION_RE = re.compile(
    r"(?:§\s*|Section\s+|SECTION\s+)?(\d+(?:\.\d+)*)\s*[-—–]?\s*([A-Z][^\n]{3,60})",
    re.MULTILINE,
)

# Table detection — markdown-style or pipe-delimited
_TABLE_RE = re.compile(
    r"(\|.+\|[\r\n]+(?:\|[-:]+\|[-:\s|]+[\r\n]+)?(?:\|.+\|[\r\n]*)+)",
    re.MULTILINE,
)

# Workpaper type keywords mapped to canonical types
_WORKPAPER_TYPE_KEYWORDS = {
    "bank reconciliation":    "bank_reconciliation",
    "trial balance":          "trial_balance",
    "financial statement":    "financial_statements",
    "analytical procedure":   "analytical_procedure",
    "engagement acceptance":  "engagement_acceptance",
    "single audit":           "single_audit",
    "compliance":             "compliance",
    "internal control":       "internal_control",
}


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _get_tokenizer():
    """Load tiktoken tokenizer. Uses cl100k_base encoding."""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        logger.warning(
            "tiktoken not installed — using word-count proxy for token counting. "
            "Install with: pip install tiktoken"
        )
        return None


_TOKENIZER = None


def _count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken or word-count proxy."""
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = _get_tokenizer()

    if _TOKENIZER is not None:
        return len(_TOKENIZER.encode(text))
    # Fallback: words * 1.3 as rough token proxy
    return int(len(text.split()) * 1.3)


def _encode(text: str) -> list[int]:
    """Encode text to token IDs."""
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = _get_tokenizer()
    if _TOKENIZER is not None:
        return _TOKENIZER.encode(text)
    return list(range(int(len(text.split()) * 1.3)))


def _decode(token_ids: list[int]) -> str:
    """Decode token IDs back to text."""
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER.decode(token_ids)
    return ""


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def _strip_headers_footers(text: str, strip_fraction: float = 0.05) -> str:
    """
    Strip page headers and footers.

    Splits on form-feed characters (page breaks) and removes the top
    strip_fraction and bottom strip_fraction of lines from each page.
    Pages with fewer than 10 lines are left untouched.
    """
    pages = text.split("\f")
    cleaned_pages = []

    for page in pages:
        lines = page.splitlines()
        if len(lines) < 10:
            cleaned_pages.append(page)
            continue
        n_strip = max(1, int(len(lines) * strip_fraction))
        cleaned = lines[n_strip: len(lines) - n_strip]
        cleaned_pages.append("\n".join(cleaned))

    return "\n".join(cleaned_pages)


def _detect_section_prefix(text: str) -> str:
    """
    Detect SOP section header at the start of a text block.
    Returns formatted prefix e.g. 'SOP §3.1 — Reconciliation: '
    or empty string if no header detected.
    """
    match = _SECTION_RE.match(text.strip())
    if match:
        section_num = match.group(1)
        section_title = match.group(2).strip().rstrip(":")
        return f"SOP §{section_num} — {section_title}: "
    return ""


def _detect_workpaper_type(text: str) -> str:
    """Detect workpaper type from text keywords."""
    lower = text.lower()
    for keyword, wp_type in _WORKPAPER_TYPE_KEYWORDS.items():
        if keyword in lower:
            return wp_type
    return ""


def _make_chunk_id(source_doc: str, char_start: int) -> str:
    """sha256(source_doc + char_start) — deterministic Qdrant point ID."""
    raw = f"{source_doc}:{char_start}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Table-aware splitting
# ---------------------------------------------------------------------------

def _extract_tables(text: str) -> list[tuple[int, int, str]]:
    """
    Find all table spans in text.
    Returns list of (start, end, table_text) tuples.
    """
    tables = []
    for m in _TABLE_RE.finditer(text):
        tables.append((m.start(), m.end(), m.group(0)))
    return tables


def _split_preserving_tables(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[tuple[str, int, int]]:
    """
    Split text into token-bounded chunks while keeping tables whole.

    Returns list of (chunk_text, char_start, char_end) tuples.

    Strategy
    --------
    1. Find all table spans in the text
    2. Split non-table regions using token sliding window
    3. Insert tables as atomic chunks (never split)
    4. If a table exceeds chunk_size, keep it whole with a warning
    """
    settings = get_settings()
    tables = _extract_tables(text)
    table_spans = {(s, e) for s, e, _ in tables}

    chunks: list[tuple[str, int, int]] = []
    pos = 0
    text_len = len(text)

    while pos < text_len:
        # Check if current position is inside a table
        in_table = None
        for t_start, t_end, t_text in tables:
            if t_start <= pos < t_end:
                in_table = (t_start, t_end, t_text)
                break

        if in_table:
            t_start, t_end, t_text = in_table
            # Emit table as atomic chunk
            token_count = _count_tokens(t_text)
            if token_count > chunk_size * 2:
                logger.warning(
                    "sop_chunker: table at char %d has %d tokens > %d*2 — "
                    "keeping whole but this may affect embedding quality",
                    t_start, token_count, chunk_size,
                )
            chunks.append((t_text, t_start, t_end))
            pos = t_end
            continue

        # Find next table start or end of text
        next_table_start = text_len
        for t_start, t_end, _ in tables:
            if t_start > pos:
                next_table_start = min(next_table_start, t_start)
                break

        # Extract non-table segment
        segment = text[pos:next_table_start]
        if not segment.strip():
            pos = next_table_start
            continue

        # Split segment with token sliding window
        tokens = _encode(segment)
        seg_offset = pos
        token_pos = 0

        while token_pos < len(tokens):
            window = tokens[token_pos: token_pos + chunk_size]
            if not window:
                break

            chunk_text = _decode(window)
            # Map token window back to character offsets (approximate)
            char_start = seg_offset + len(_decode(tokens[:token_pos]))
            char_end = char_start + len(chunk_text)

            chunks.append((chunk_text, char_start, min(char_end, text_len)))
            token_pos += chunk_size - overlap

        pos = next_table_start

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    source_doc: str,
    sop_version: str,
) -> list[SOPChunk]:
    """
    Chunk a SOP document string into SOPChunk objects.

    Parameters
    ----------
    text : str
        Full SOP document text.
    source_doc : str
        Source document filename e.g. 'HCLLP_SOP_2024.pdf'.
        Used for chunk_id generation and Postgres linkage.
    sop_version : str
        Version identifier e.g. '2024-Q1'.

    Returns
    -------
    list[SOPChunk]
        Ordered list of chunks ready for embedder.py.
    """
    settings = get_settings()
    chunk_size = settings.sop_chunker.chunk_size_tokens
    overlap = settings.sop_chunker.chunk_overlap_tokens
    prepend_prefix = settings.sop_chunker.prepend_section_prefix

    # Strip headers/footers
    cleaned = _strip_headers_footers(text)

    # Split preserving tables
    raw_chunks = _split_preserving_tables(cleaned, chunk_size, overlap)

    sop_chunks: list[SOPChunk] = []

    for chunk_text_raw, char_start, char_end in raw_chunks:
        if not chunk_text_raw.strip():
            continue

        # Detect section prefix
        section_prefix = _detect_section_prefix(chunk_text_raw) if prepend_prefix else ""

        # Build final content with prefix
        content = (
            section_prefix + chunk_text_raw
            if section_prefix and not chunk_text_raw.startswith(section_prefix)
            else chunk_text_raw
        )

        token_count = _count_tokens(content)
        chunk_id = _make_chunk_id(source_doc, char_start)
        workpaper_type = _detect_workpaper_type(chunk_text_raw)
        is_table = bool(_TABLE_RE.search(chunk_text_raw))

        sop_chunks.append(SOPChunk(
            chunk_id=chunk_id,
            source_doc=source_doc,
            sop_version=sop_version,
            content=content,
            section_prefix=section_prefix,
            char_start=char_start,
            char_end=char_end,
            token_count=token_count,
            workpaper_type=workpaper_type,
            is_table=is_table,
            metadata={
                "source_doc":    source_doc,
                "sop_version":   sop_version,
                "workpaper_type": workpaper_type,
                "is_rollforward": False,
            },
        ))

    logger.info(
        "sop_chunker: %s — %d chunks from %d chars (chunk_size=%d overlap=%d)",
        source_doc, len(sop_chunks), len(text), chunk_size, overlap,
    )

    return sop_chunks


def chunk_file(
    file_path: str | Path,
    sop_version: str,
) -> list[SOPChunk]:
    """
    Chunk a SOP document from a file path.

    Supports .txt, .md, and pre-extracted .pdf text files.
    For PDF binary files use Phase 1 pdf_text_extractor first.

    Parameters
    ----------
    file_path : str | Path
    sop_version : str

    Returns
    -------
    list[SOPChunk]

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"SOP file not found: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")
    return chunk_text(text, source_doc=path.name, sop_version=sop_version)