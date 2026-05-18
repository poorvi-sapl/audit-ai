"""
raw_to_training_pair/extraction/
=================================
Phase 2 thin extractors. Raw text and rows only — no normalization,
no PII scrubbing, no DocumentRecord.

All extractors implement the same contract:
    extract(file_path: str | Path) -> dict
    {
        "text"    : str,
        "tables"  : list[dict],  # [{"headers": [...], "rows": [[...]]}]
        "metadata": dict,
    }

Routing to the correct extractor is handled by raw_to_training_pair/router.py
via config/routing.yaml — never import extractors directly in pipeline code.
"""