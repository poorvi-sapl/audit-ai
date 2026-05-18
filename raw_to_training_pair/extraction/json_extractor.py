"""
raw_to_training_pair/extraction/json_extractor.py
==================================================
Phase 2 raw extractor for .json and .jsonl files.

Output shape
------------
{
    "text"    : str,   # key: value lines joined by newline
    "tables"  : list,  # always empty — JSON has no tabular structure
    "metadata": dict   # key_count or line_count
}

Public API
----------
    extract(file_path: str | Path) -> dict
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract(file_path: str | Path) -> dict:
    """
    Extract raw text from a .json or .jsonl file.

    Raises
    ------
    FileNotFoundError, RuntimeError
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        raise RuntimeError(f"Could not read {path.name}: {e}") from e

    lines = []

    if path.suffix.lower() == ".jsonl":
        count = 0
        for i, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                lines.append(f"Line {i + 1}: {json.dumps(obj)}")
            except json.JSONDecodeError:
                lines.append(f"Line {i + 1}: {line}")
            count += 1
        metadata = {"line_count": count}
    else:
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    lines.append(f"{k}: {json.dumps(v)}")
                metadata = {"key_count": len(obj)}
            else:
                lines.append(json.dumps(obj, indent=2))
                metadata = {"key_count": 1}
        except json.JSONDecodeError as e:
            raise RuntimeError(f"JSON decode error in {path.name}: {e}") from e

    full_text = "\n".join(lines)
    logger.info("p2/json_extractor: %s — %d lines", path.name, len(lines))

    return {"text": full_text, "tables": [], "metadata": metadata}