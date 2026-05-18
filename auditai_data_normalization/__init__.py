"""
auditai_data_normalization
==========================
Turns raw workpapers (any year, format, or scan quality) into a canonical
WorkpaperRecord. This is the foundation consumed by raw_to_training_pair.

Build order within this package:
  schema.py           → WorkpaperRecord dataclass
  field_aliases.yaml  → label variants → canonical field names
  pii.py              → Presidio-based PII stripper (must run first)
  extractors/         → pdfplumber | ocr | llm  (three independent extractors)
  confidence.py       → score_confidence() — all-agree 1.0 / 2-of-3 0.7 / etc.
  normalize.py        → normalize_workpaper() entry point
  review_queue.py     → append-only CSV for records with confidence < 0.7
"""
