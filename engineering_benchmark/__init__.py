"""
engineering_benchmark
=====================
Infrastructure setup and SOP RAG track.
Independent of auditai_data_normalization — can be built in parallel.

Build order within this package:
  postgres_schema.sql → 4 tables: engagements, workpapers, sop_chunks, findings
  qdrant_setup.py     → create auditai_sop collection (cosine, 4096-dim, on-disk)
  sop_chunker.py      → RecursiveCharacterTextSplitter, 400-tok chunks, 50-tok overlap
  embedder.py         → e5-mistral-7b-instruct, batch 32, Qdrant upsert batch 100
  smoke_test.py       → upsert known chunk → query → assert top-5 hit
"""
