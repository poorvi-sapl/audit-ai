"""
raw_to_training_pair
====================
End-to-end pipeline: consumes WorkpaperRecord from auditai_data_normalization
and produces JSONL SFT pairs for Mistral 22B fine-tuning.

Depends on: auditai_data_normalization (Phase 1 must be complete)

Build order within this package:
  router.py             → file-type detection (magic bytes); routes to extractor
  extraction/           → Docling / python-docx / openpyxl / json wrappers
  completion_drafter.py → Gemma 3 12B via Ollama; drafts assistant completions
  pair_builder.py       → 3-message JSONL assembly (system / user / assistant)
  auditor_review.py     → CLI queue; flips auditor_approved = True
  quality_gates.py      → 4 gates: confidence ≥0.7, approved, no dup, stage isolation
  jsonl_writer.py       → json.dumps append-mode, sha256 dedup before write
"""
