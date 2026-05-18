"""
raw_to_training_pair/extraction/csv_extractor.py
=================================================
Phase 2 raw extractor for .csv and .tsv files.

Output shape
------------
{
    "text"    : str,   # table rendered as pipe-delimited text
    "tables"  : list,  # [{"headers": [...], "rows": [[...]]}]
    "metadata": dict   # row_count, column_count
}

Public API
----------
    extract(file_path: str | Path) -> dict
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def extract(file_path: str | Path) -> dict:
    """
    Extract raw rows from a .csv or .tsv file.

    Raises
    ------
    FileNotFoundError, RuntimeError
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    sep = "\t" if path.suffix.lower() == ".tsv" else ","

    try:
        df = pd.read_csv(str(path), sep=sep, dtype=str).fillna("")
    except Exception as e:
        raise RuntimeError(f"pandas failed on {path.name}: {e}") from e

    headers = list(df.columns)
    rows = [list(r) for _, r in df.iterrows()]

    lines = [" | ".join(headers), "-" * 40]
    for row in rows[:500]:
        lines.append(" | ".join(str(v) for v in row))
    if len(rows) > 500:
        lines.append(f"[... {len(rows) - 500} more rows truncated]")

    text = "\n".join(lines)

    logger.info("p2/csv_extractor: %s — %d rows", path.name, len(rows))

    return {
        "text": text,
        "tables": [{"headers": headers, "rows": rows}],
        "metadata": {"row_count": len(rows), "column_count": len(headers)},
    }