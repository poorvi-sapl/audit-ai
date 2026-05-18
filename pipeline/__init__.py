"""
pipeline/
=========
Phase 4 — orchestration layer. Wires Phases 1, 2, and 3 into a single
end-to-end pipeline.

    pipeline.py          — main entry point: file → JSONL pair
    qdrant_retriever.py  — hybrid SOP retrieval (filter + semantic)

Neither Phase 1, 2, nor 3 imports from here.
This package owns all cross-phase wiring.
"""