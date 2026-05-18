"""
auditai_data_normalization/extractors/json_extractor.py
=========================================================
Extracts content from .json and .jsonl files via stdlib json.

Output contract
---------------
Returns a DocumentRecord with:
  sections : one Section per top-level key (JSON) or per line (JSONL)
  tables   : empty — JSON has no tabular structure at this layer
  metadata : line_count (JSONL) or key_count (JSON)

Public API
----------
    extract(file_path: str | Path) -> DocumentRecord
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from auditai_data_normalization.schema import (
    DocumentRecord,
    Section,
)

logger = logging.getLogger(__name__)


def extract(file_path: str | Path) -> DocumentRecord:
    """
    Extract content from a .json or .jsonl file.

    Parameters
    ----------
    file_path : str | Path
        Path to a .json or .jsonl file.

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

    try:
        text = raw_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        return DocumentRecord(
            source_path=str(path),
            file_name=path.name,
            file_type="json",
            file_size_bytes=len(raw_bytes),
            file_hash=file_hash,
            extraction_method="stdlib_json",
            extraction_status="failed",
            extraction_error=str(e),
            needs_review=True,
        )

    sections: list[Section] = []

    # --- JSONL: one section per line ---
    if path.suffix.lower() == ".jsonl":
        for i, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                content = json.dumps(obj, indent=2)
            except json.JSONDecodeError:
                content = line
            sections.append(Section(
                index=i,
                heading=f"Line {i + 1}",
                content=content,
                token_count=len(content.split()),
            ))
        metadata = {"line_count": len(sections)}

    # --- JSON: one section per top-level key ---
    else:
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                for i, (k, v) in enumerate(obj.items()):
                    content = f"{k}: {json.dumps(v)}"
                    sections.append(Section(
                        index=i,
                        heading=str(k),
                        content=content,
                        token_count=len(content.split()),
                    ))
                metadata = {"key_count": len(sections)}
            else:
                content = json.dumps(obj, indent=2)
                sections.append(Section(
                    index=0,
                    heading="",
                    content=content,
                    token_count=len(content.split()),
                ))
                metadata = {"key_count": 1}
        except json.JSONDecodeError as e:
            return DocumentRecord(
                source_path=str(path),
                file_name=path.name,
                file_type="json",
                file_size_bytes=len(raw_bytes),
                file_hash=file_hash,
                extraction_method="stdlib_json",
                extraction_status="failed",
                extraction_error=f"JSON decode error: {e}",
                needs_review=True,
            )

    cleaned_text = "\n\n".join(s.content for s in sections)

    logger.info(
        "json_extractor: %s — %d sections",
        path.name, len(sections),
    )

    return DocumentRecord(
        source_path=str(path),
        file_name=path.name,
        file_type="json",
        file_size_bytes=len(raw_bytes),
        file_hash=file_hash,
        raw_text=text,
        cleaned_text=cleaned_text,
        sections=sections,
        tables=[],
        extraction_method="stdlib_json",
        extraction_status="success",
        word_count=len(cleaned_text.split()),
        pii_scrubbed=False,
        needs_review=False,
        metadata=metadata,
    )