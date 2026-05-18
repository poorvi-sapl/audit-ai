"""
auditai_data_normalization/extractors/csv_extractor.py
========================================================
Extracts structured data from .csv and .tsv files via pandas.

Output contract
---------------
Returns a DocumentRecord with:
  sections : one Section representing the full table as text
  tables   : one ExtractedTable with headers and rows as strings
  metadata : row_count, column_count

Public API
----------
    extract(file_path: str | Path) -> DocumentRecord
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from auditai_data_normalization.schema import (
    DocumentRecord,
    ExtractedTable,
    Section,
)

logger = logging.getLogger(__name__)


def extract(file_path: str | Path) -> DocumentRecord:
    """
    Extract content from a .csv or .tsv file.

    Parameters
    ----------
    file_path : str | Path
        Path to a .csv or .tsv file.

    Returns
    -------
    DocumentRecord
        pii_scrubbed=False — call pii.scrub_record() after this.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    raw_bytes = path.read_bytes()
    file_hash = hashlib.sha256(raw_bytes).hexdigest()
    sep = "\t" if path.suffix.lower() == ".tsv" else ","

    try:
        df = pd.read_csv(str(path), sep=sep, dtype=str).fillna("")
    except Exception as e:
        return DocumentRecord(
            source_path=str(path),
            file_name=path.name,
            file_type="csv",
            file_size_bytes=len(raw_bytes),
            file_hash=file_hash,
            extraction_method="pandas_csv",
            extraction_status="failed",
            extraction_error=str(e),
            needs_review=True,
        )

    headers = list(df.columns)
    rows = [list(r) for _, r in df.iterrows()]

    lines = [" | ".join(headers), "-" * 40]
    for row in rows[:500]:
        lines.append(" | ".join(str(v) for v in row))
    if len(rows) > 500:
        lines.append(f"[... {len(rows) - 500} more rows truncated]")

    content = "\n".join(lines)

    table = ExtractedTable(
        index=0,
        source=path.name,
        headers=headers,
        rows=rows,
        raw_text=content,
    )
    section = Section(
        index=0,
        heading=path.stem,
        content=content,
        token_count=len(content.split()),
        is_table=True,
    )

    logger.info(
        "csv_extractor: %s — %d rows, %d columns",
        path.name, len(rows), len(headers),
    )

    return DocumentRecord(
        source_path=str(path),
        file_name=path.name,
        file_type="csv",
        file_size_bytes=len(raw_bytes),
        file_hash=file_hash,
        raw_text=content,
        cleaned_text=content,
        sections=[section],
        tables=[table],
        extraction_method="pandas_csv",
        extraction_status="success",
        word_count=len(content.split()),
        pii_scrubbed=False,
        needs_review=False,
        metadata={
            "row_count": len(rows),
            "column_count": len(headers),
        },
    )