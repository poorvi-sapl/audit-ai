"""
streamlit_app/app.py
=====================
AuditAI Training Pair Generator — Streamlit UI.

Single-page interface for:
0. SOP Management   — upload SOP documents, embed into Qdrant
1. Upload & Process — upload a workpaper, extract fields, generate pairs
2. Review Queue     — approve / reject pairs across all sessions
3. Export           — write approved pairs to JSONL, download files

UI calls pipeline functions only. Zero business logic here.

Run with:
    streamlit run streamlit_app/app.py
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.pipeline import process_workpaper, write_approved
from raw_to_training_pair.auditor_review import (
    load_pending,
    approve,
    conditional_approve,
    send_for_correction,
    reject,
    stats,
)
from raw_to_training_pair.jsonl_writer import count, get_output_path
from auditai_data_normalization.alias_suggester import (
    load_suggestions,
    approve as alias_approve,
    reject as alias_reject,
    coverage_stats,
    suggest_canonical,
    _load_canonical_fields,
)
from auditai_data_normalization.normalize import reset_alias_cache

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_QUEUE_PATH = Path("data/review_queue.jsonl")
_DATA_DIR   = Path("data")
_CLIENT_TYPES = ["NPO", "Government", "For-Profit", "Tribal"]
_SOP_EXTENSIONS = ["pdf", "txt", "md", "docx"]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AuditAI — Training Pair Generator",
    page_icon="📋",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

defaults = {
    "process_result": None,
    "reviewer_id": "",
    "last_export": None,
    "queue_refresh": 0,
    "sop_embed_result": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _review_score_stats(queue_path: Path) -> dict:
    """
    Read review_confidence from every pair in the queue and return
    aggregated stats split by status.

    Returns dict with keys: all, approved, pending — each a list of floats.
    Returns empty lists if queue file doesn't exist or has no scores.
    """
    import json

    buckets: dict[str, list[float]] = {"all": [], "approved": [], "pending": []}
    if not queue_path.exists():
        return buckets

    with open(queue_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Queue format: {"pair": {..., "metadata": {...}}, "status": "pending", ...}
            pair_data = entry.get("pair", entry)
            score = pair_data.get("metadata", {}).get("review_confidence")
            if score is None:
                continue
            score = float(score)
            status = entry.get("status", "pending")
            buckets["all"].append(score)
            if status in ("approved", "conditional"):
                buckets["approved"].append(score)
            elif status == "pending":
                buckets["pending"].append(score)

    return buckets


def _qdrant_chunk_count(collection_name: str) -> int:
    """
    Return number of vectors in the Qdrant collection.
    Returns:
        >= 0  : connected, value is chunk count (0 if collection not yet created)
        -1    : Qdrant unreachable (connection error)
    """
    try:
        import os
        from qdrant_client import QdrantClient
        client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            timeout=5,
        )
        # Confirm connection by listing collections first
        existing = {c.name for c in client.get_collections().collections}
        if collection_name not in existing:
            return 0   # Connected but collection not yet created — not an error
        info = client.get_collection(collection_name)
        return getattr(info, "vectors_count", None) or getattr(info, "points_count", 0) or 0
    except Exception:
        return -1


def _extract_text_from_sop(file_path: Path) -> str:
    """Extract raw text from SOP file for chunking."""
    suffix = file_path.suffix.lower()
    if suffix in (".txt", ".md"):
        return file_path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".pdf":
        try:
            import pdfplumber
            pages = []
            with pdfplumber.open(str(file_path)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    pages.append(text)
            return "\n\n".join(pages)
        except Exception as e:
            st.error(f"PDF extraction failed: {e}")
            return ""
    elif suffix == ".docx":
        try:
            from docx import Document
            doc = Document(str(file_path))
            parts = []

            # Body paragraphs
            for p in doc.paragraphs:
                if p.text.strip():
                    parts.append(p.text.strip())

            # Tables — bug #15: paragraphs-only missed all table content
            for table in doc.tables:
                for row in table.rows:
                    row_cells = [
                        " ".join(p.text.strip() for p in cell.paragraphs if p.text.strip())
                        for cell in row.cells
                    ]
                    row_text = "\t".join(c for c in row_cells if c)
                    if row_text.strip():
                        parts.append(row_text)

            # Headers and footers from each section
            for section in doc.sections:
                for hf in [section.header, section.footer]:
                    if hf:
                        for p in hf.paragraphs:
                            if p.text.strip():
                                parts.append(p.text.strip())

            return "\n".join(parts)
        except Exception as e:
            st.error(f"DOCX extraction failed: {e}")
            return ""
    return ""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Settings")

    use_mock = st.checkbox(
        "Use mock completions",
        value=False,
        help=(
            "ON  → semi-deterministic mock (fast, no Ollama needed)\n"
            "OFF → real Gemma 3 12B via Ollama (requires ollama serve)"
        ),
    )

    if use_mock:
        st.info("Mock mode — completions are simulated")
    else:
        st.warning("Real mode — Ollama must be running")

    st.divider()

    # Qdrant SOP chunk count — wrapped so a missing Qdrant never blocks page load
    try:
        from config.settings import get_settings
        settings = get_settings()
        qdrant_count = _qdrant_chunk_count(settings.qdrant.collection_sop)
        if qdrant_count >= 0:
            st.metric("SOP chunks in Qdrant", qdrant_count)
        else:
            st.caption("⚠️ Qdrant unreachable")
    except Exception:
        st.caption("⚠️ Qdrant unreachable")

    st.divider()
    st.subheader("📊 Queue stats")

    queue_stats = stats(_QUEUE_PATH)
    col1, col2 = st.columns(2)
    col1.metric("Total",       queue_stats["total"])
    col1.metric("Pending",     queue_stats["pending"])
    col2.metric("Approved",    queue_stats["approved"]
                               + queue_stats.get("conditional", 0))
    col2.metric("Needs rework",queue_stats.get("correction", 0))
    col1.metric("Rejected",    queue_stats["rejected"])

    if st.button("Show avg review scores", key="score_stats_btn", use_container_width=True):
        _scores = _review_score_stats(_QUEUE_PATH)

        def _fmt(vals: list[float]) -> str:
            if not vals:
                return "—"
            return f"{sum(vals) / len(vals):.3f}"

        s_col1, s_col2, s_col3 = st.columns(3)
        s_col1.metric("All",      _fmt(_scores["all"]),      help=f"{len(_scores['all'])} pairs")
        s_col2.metric("Approved", _fmt(_scores["approved"]), help=f"{len(_scores['approved'])} pairs")
        s_col3.metric("Pending",  _fmt(_scores["pending"]),  help=f"{len(_scores['pending'])} pairs")

        if _scores["all"]:
            import pandas as pd
            _df = pd.DataFrame({"score": _scores["all"]})
            _bins = [0.0, 0.5, 0.7, 0.85, 1.01]
            _labels = ["0–0.5", "0.5–0.7", "0.7–0.85", "0.85–1.0"]
            _df["bucket"] = pd.cut(_df["score"], bins=_bins, labels=_labels, right=False)
            _dist = _df["bucket"].value_counts().reindex(_labels, fill_value=0)
            st.bar_chart(_dist, height=120)
        else:
            st.caption("No scored pairs in queue yet.")

    st.divider()
    st.subheader("🏷️ Alias coverage")

    alias_csv = _DATA_DIR / "suggested_aliases.csv"
    alias_stats = coverage_stats(alias_csv)

    if alias_stats.total_seen == 0:
        st.caption("No unknown labels seen yet.")
    else:
        st.metric(
            "Coverage",
            f"{alias_stats.coverage_pct:.1f}%",
            help=f"{alias_stats.approved} of {alias_stats.total_seen} unknown labels mapped",
        )
        acol1, acol2 = st.columns(2)
        acol1.metric(
            "Pending", alias_stats.pending,
            delta=f"{alias_stats.high_conf} high-conf" if alias_stats.high_conf else None,
        )
        acol2.metric("Approved", alias_stats.approved)

    st.divider()
    st.caption("AuditAI — HCLLP Internal Tool")

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------

st.title("📋 AuditAI Training Pair Generator")
st.caption("Manage SOPs → Upload workpapers → Generate pairs → Review → Export JSONL")

st.divider()

# ===========================================================================
# SECTION 0 — SOP Management
# ===========================================================================

st.header("0 · SOP Management")
st.caption("Upload SOP documents to embed into Qdrant. Do this once before processing workpapers.")

sop_col1, sop_col2 = st.columns([2, 1])

with sop_col1:
    uploaded_sop = st.file_uploader(
        "Upload a SOP document",
        type=_SOP_EXTENSIONS,
        help="Supported: .pdf, .txt, .md, .docx",
        key="sop_uploader",
    )

with sop_col2:
    sop_version = st.text_input(
        "SOP version",
        value="2024-Q1",
        placeholder="e.g. 2024-Q1, v3.2",
        help="Version tag for this SOP — used for versioning in Qdrant and Postgres",
    )

embed_btn = st.button(
    "🔵 Embed SOP into Qdrant",
    disabled=(uploaded_sop is None or not sop_version.strip()),
    type="primary",
    key="embed_sop_btn",
)

if embed_btn and uploaded_sop and sop_version.strip():
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_sop_path = _DATA_DIR / uploaded_sop.name

    with open(tmp_sop_path, "wb") as f:
        f.write(uploaded_sop.getbuffer())

    with st.spinner(f"Embedding {uploaded_sop.name} — loading model and chunking..."):
        try:
            from engineering_benchmark.sop_chunker import chunk_text
            from engineering_benchmark.embedder import embed_chunks

            # Extract text
            sop_text = _extract_text_from_sop(tmp_sop_path)

            if not sop_text.strip():
                st.error("Could not extract text from this SOP file.")
            else:
                # Chunk
                chunks = chunk_text(
                    text=sop_text,
                    source_doc=uploaded_sop.name,
                    sop_version=sop_version.strip(),
                )

                # Embed and upsert
                embed_result = embed_chunks(chunks, pg_conn=None)
                st.session_state.sop_embed_result = {
                    "file_name":  uploaded_sop.name,
                    "sop_version": sop_version.strip(),
                    "chunks_created": len(chunks),
                    "upserted": embed_result.upserted,
                    "skipped": embed_result.skipped,
                    "errors": embed_result.errors,
                }

        except ImportError as e:
            st.error(
                f"sentence-transformers not installed: {e}\n"
                "Run: pip install -e '.[training]'"
            )
        except Exception as e:
            st.error(f"Embedding failed: {e}")

# Show SOP embed result
if st.session_state.sop_embed_result is not None:
    res = st.session_state.sop_embed_result
    if res["errors"]:
        st.warning(
            f"⚠️ Embedded with errors — "
            f"chunks: {res['chunks_created']} | "
            f"upserted: {res['upserted']} | "
            f"errors: {len(res['errors'])}"
        )
        with st.expander("Embedding errors"):
            for err in res["errors"]:
                st.error(err)
    else:
        st.success(
            f"✅ **{res['file_name']}** embedded successfully — "
            f"{res['chunks_created']} chunks created, "
            f"{res['upserted']} upserted to Qdrant "
            f"(version: {res['sop_version']})"
        )

st.divider()

# ===========================================================================
# SECTION 0.5 — Alias Suggestions
# ===========================================================================

st.header("0.5 · Alias Suggestions")
st.caption(
    "Unknown field labels found during extraction. "
    "Approve to add them to field_aliases.yaml. "
    "Reject to stop suggesting. Takes ~10 min per batch."
)

_alias_csv = _DATA_DIR / "suggested_aliases.csv"
_alias_pending = [
    r for r in load_suggestions(_alias_csv)
    if r.get("status") == "pending"
]

if not _alias_pending:
    st.info("✅ No pending alias suggestions — all labels are mapped.")
else:
    # ── Batch approve high-confidence ─────────────────────────────────────
    _high_conf = [
        s for s in _alias_pending
        if s.get("llm_confidence") == "high" and s.get("suggested_canonical")
    ]
    if _high_conf:
        st.markdown(f"**{len(_high_conf)} high-confidence suggestions** ready for batch approval")
        if st.button(
            f"✅ Batch approve {len(_high_conf)} high-confidence suggestions",
            disabled=not st.session_state.reviewer_id,
            key="batch_approve_btn",
        ):
            _n = sum(
                1 for s in _high_conf
                if alias_approve(s["raw_label"], s["suggested_canonical"],
                                 st.session_state.reviewer_id, _alias_csv)
            )
            reset_alias_cache()
            st.success(f"Approved {_n} aliases — field_aliases.yaml updated.")
            st.rerun()
        if not st.session_state.reviewer_id:
            st.caption("⚠️ Enter your reviewer ID in Section 2 to approve")
        st.divider()

    # ── Per-row review table ───────────────────────────────────────────────
    st.markdown(f"**{len(_alias_pending)} pending suggestions**")
    _canonical_fields = _load_canonical_fields()

    for _i, _s in enumerate(_alias_pending):
        _raw        = _s.get("raw_label", "")
        _sample     = _s.get("extracted_value", "")
        _src        = _s.get("source_file", "")
        _suggested  = _s.get("suggested_canonical", "")
        _llm_conf   = _s.get("llm_confidence", "pending")
        _seen       = _s.get("seen_count", "1")

        _conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴",
                      "none": "⚫", "pending": "⏳"}.get(_llm_conf, "⏳")

        with st.expander(
            f"{_conf_icon} `{_raw}` — seen {_seen}× "
            f"| suggestion: `{_suggested or 'none'}` ({_llm_conf})",
            expanded=(_llm_conf == "high"),
        ):
            _dc1, _dc2 = st.columns([2, 2])
            with _dc1:
                st.markdown(f"**Raw label:** `{_raw}`")
                st.markdown(f"**Sample value:** `{_sample or '—'}`")
                st.markdown(f"**Source:** `{_src or '—'}`")
                st.markdown(f"**Seen:** {_seen}×")

            with _dc2:
                _sel_key = f"alias_select_{_i}_{_raw[:10]}"
                _def_idx = 0
                if _suggested and _suggested in _canonical_fields:
                    try:
                        _def_idx = _canonical_fields.index(_suggested)
                    except ValueError:
                        _def_idx = 0

                _selected = st.selectbox(
                    "Map to canonical field",
                    options=[""] + _canonical_fields,
                    index=_def_idx + 1 if _suggested else 0,
                    key=_sel_key,
                )

                if not _suggested and st.button(
                    "🤖 Ask Gemma",
                    key=f"llm_btn_{_i}_{_raw[:8]}",
                    disabled=not st.session_state.reviewer_id,
                ):
                    with st.spinner("Asking Gemma..."):
                        _res = suggest_canonical(_raw, _canonical_fields)
                    if _res.has_suggestion:
                        st.info(
                            f"Gemma suggests: `{_res.suggested_canonical}` "
                            f"(confidence: {_res.llm_confidence})\n\n"
                            f"Reason: {_res.reasoning}"
                        )
                    else:
                        st.warning("Gemma found no confident match.")

            _bc1, _bc2, _ = st.columns([1, 1, 4])
            with _bc1:
                if st.button(
                    "✅ Approve",
                    key=f"alias_approve_{_i}_{_raw[:8]}",
                    disabled=(not st.session_state.reviewer_id or not _selected),
                ):
                    if alias_approve(_raw, _selected, st.session_state.reviewer_id, _alias_csv):
                        reset_alias_cache()
                        st.success(f"Mapped `{_raw}` → `{_selected}`")
                        st.rerun()
                    else:
                        st.error("Approval failed.")
            with _bc2:
                if st.button(
                    "❌ Reject",
                    key=f"alias_reject_{_i}_{_raw[:8]}",
                    disabled=not st.session_state.reviewer_id,
                ):
                    if alias_reject(_raw, st.session_state.reviewer_id, _alias_csv):
                        st.warning(f"Rejected `{_raw}`")
                        st.rerun()
                    else:
                        st.error("Rejection failed.")

st.divider()

# ===========================================================================
# SECTION 1 — Upload & Process
# ===========================================================================

st.header("1 · Upload & Process")

col_upload, col_options = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload a workpaper",
        type=["docx", "xlsx", "xls", "pdf", "csv", "json"],
        help="Supported: .docx, .xlsx, .xls, .pdf, .csv, .json",
    )

with col_options:
    st.markdown("**Client types to generate**")
    selected_client_types = []
    for ct in _CLIENT_TYPES:
        if st.checkbox(ct, value=(ct == "NPO"), key=f"ct_{ct}"):
            selected_client_types.append(ct)

    if not selected_client_types:
        st.warning("Select at least one client type")

process_btn = st.button(
    "⚡ Process workpaper",
    disabled=(uploaded_file is None or not selected_client_types),
    type="primary",
)

if process_btn and uploaded_file and selected_client_types:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = _DATA_DIR / uploaded_file.name

    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner(f"Processing {uploaded_file.name}..."):
        result = process_workpaper(
            file_path=str(tmp_path),
            queue_path=_QUEUE_PATH,
            data_dir=_DATA_DIR,
            run_parallel=False,
            client_types=selected_client_types,
            use_mock=use_mock,
        )

    st.session_state.process_result = result
    st.session_state.queue_refresh += 1

# Show results
if st.session_state.process_result is not None:
    result = st.session_state.process_result

    if result.skipped:
        st.error(f"⚠️ Skipped: {result.skip_reason}")
    else:
        st.success(
            f"✅ Processed **{result.file_name}** — "
            f"{result.pairs_queued} pairs queued, "
            f"{result.pairs_failed} failed"
        )

        st.subheader("Extraction summary")
        record = result.record

        ecol1, ecol2, ecol3 = st.columns(3)
        ecol1.metric(
            "Confidence",
            f"{record.extraction_confidence:.2f}",
            delta="✓ pass" if record.extraction_confidence >= 0.7 else "✗ low",
        )
        ecol2.metric("Word count", record.word_count)
        ecol3.metric("File type", record.file_type)

        conf_summary = record.metadata.get("confidence_summary", {})
        fields_present = conf_summary.get("fields_present", [])
        fields_missing = conf_summary.get("fields_missing", [])
        pii_redactions = record.pii_redactions

        fcol1, fcol2 = st.columns(2)

        with fcol1:
            st.markdown(f"**Fields found ({len(fields_present)})**")
            if fields_present:
                st.code("\n".join(fields_present))
            else:
                st.caption("None detected")

        with fcol2:
            st.markdown(f"**Fields missing ({len(fields_missing)})**")
            if fields_missing:
                st.code("\n".join(fields_missing))
            else:
                st.caption("None — all fields found ✓")

        if pii_redactions:
            st.markdown("**PII redacted**")
            pii_summary = ", ".join(
                f"{r.pii_type}: {r.count}" for r in pii_redactions
            )
            st.info(f"🔒 {pii_summary}")

        if result.gate_failures:
            with st.expander(f"⚠️ Gate failures ({len(result.gate_failures)})"):
                for gf in result.gate_failures:
                    st.warning(gf)

st.divider()

# ===========================================================================
# SECTION 2 — Review Queue
# ===========================================================================

st.header("2 · Review Queue")

reviewer_id = st.text_input(
    "Your reviewer ID",
    value=st.session_state.reviewer_id,
    placeholder="e.g. SH, MS1, JD",
    help="Required to approve or reject pairs",
    key="reviewer_id_input",
)
st.session_state.reviewer_id = reviewer_id

pending_entries = load_pending(_QUEUE_PATH)

if not pending_entries:
    st.info("No pairs pending review.")
else:
    st.markdown(f"**{len(pending_entries)} pairs awaiting review**")

    for i, entry in enumerate(pending_entries):
        pair = entry.get("pair", {})
        meta = pair.get("metadata", {})
        messages = pair.get("messages", [])

        pair_hash = meta.get("pair_hash", "")
        label = (
            f"{meta.get('file_name', 'unknown')} · "
            f"{meta.get('client_type', '')} · "
            f"{meta.get('pair_type', '')} · "
            f"stage={meta.get('stage', '')} · "
            f"conf={meta.get('extraction_confidence', 0):.2f}"
        )

        with st.expander(f"#{i + 1} — {label}"):

            # ── E2 — Uncertainty summary banner ───────────────────────────
            uncertain       = meta.get("uncertain_sections", []) or []
            flagged_fields  = meta.get("flagged_fields", []) or []
            fields_missing  = meta.get("fields_missing", []) or []
            rev_conf        = meta.get("review_confidence", 0.0)
            llm_asst        = meta.get("llm_assisted", False)
            struct_evidence = meta.get("structural_evidence", {}) or {}

            # Collect all fields needing attention for the banner
            _all_attention = sorted(set(
                list(uncertain) + list(flagged_fields) + list(fields_missing)
            ))

            if _all_attention:
                st.warning(
                    f"⚠️ **{len(_all_attention)} field(s) need attention:** "
                    + ", ".join(f"`{f}`" for f in _all_attention)
                )
            if llm_asst:
                st.info("🤖 LLM-assisted extraction — verify flagged fields before approving")

            # Review confidence + extraction confidence row
            _rc_col, _ec_col = st.columns(2)
            if rev_conf > 0:
                _rc_col.caption(
                    f"Review confidence: **{rev_conf:.2f}** "
                    f"{'✓' if rev_conf >= 0.70 else '⚠️ below threshold'}"
                )
            _ext_conf = meta.get("extraction_confidence", 0.0)
            if _ext_conf > 0:
                _ec_col.caption(
                    f"Extraction confidence: **{_ext_conf:.2f}** "
                    f"{'✓' if _ext_conf >= 0.50 else '⚠️ low'}"
                )

            # ── E2 — Flagged fields detail panel ──────────────────────────
            if flagged_fields or fields_missing or struct_evidence:
                with st.expander(
                    f"🔍 Field detail — "
                    f"{len(flagged_fields)} flagged · "
                    f"{len(fields_missing)} missing · "
                    f"{len(struct_evidence)} structural"
                ):
                    if fields_missing:
                        st.markdown("**Missing fields** *(not found by any extractor)*")
                        for _f in fields_missing:
                            st.markdown(f"- ❌ `{_f}`")

                    if flagged_fields:
                        st.markdown("**Flagged fields** *(low confidence or LLM-only)*")
                        for _f in flagged_fields:
                            _tag = "🤖" if llm_asst else "⚠️"
                            st.markdown(f"- {_tag} `{_f}`")

                    if struct_evidence:
                        st.markdown("**Structural evidence** *(Phase 4 heuristic extraction)*")
                        for _fname, _ev in struct_evidence.items():
                            st.markdown(
                                f"- `{_fname}`: **{_ev.get('value', '')}** "
                                f"— conf={_ev.get('confidence', 0):.2f} "
                                f"p{_ev.get('source_page', '?')} "
                                f"via `{_ev.get('method', '?')}`"
                            )

            tab_sys, tab_user, tab_asst = st.tabs(["System", "User", "Assistant"])

            with tab_sys:
                sys_content = next(
                    (m["content"] for m in messages if m["role"] == "system"), ""
                )
                st.text_area("System prompt", sys_content, height=150,
                             disabled=True, key=f"sys_{i}")

            with tab_user:
                user_content = next(
                    (m["content"] for m in messages if m["role"] == "user"), ""
                )
                st.text_area("User message", user_content, height=250,
                             disabled=True, key=f"user_{i}")

            with tab_asst:
                asst_content = next(
                    (m["content"] for m in messages if m["role"] == "assistant"), ""
                )

                # ── E2 — Highlight placeholder SOP citations ──────────────
                import re as _re
                _placeholder_re = _re.compile(r"§X\.X", _re.IGNORECASE)
                _placeholder_count = len(_placeholder_re.findall(asst_content))

                if _placeholder_count > 0:
                    st.warning(
                        f"⚠️ **{_placeholder_count} placeholder SOP citation(s)** — "
                        "findings marked `§X.X` need real SOP sections before approving"
                    )
                    _annotated = _placeholder_re.sub("**[⚠️ §X.X — needs citation]**", asst_content)
                    st.markdown("**Completion preview** *(placeholder citations highlighted)*")
                    st.markdown(
                        f"<div style='background:#fff8e1;border-left:3px solid #f9a825;"
                        f"padding:10px;font-family:monospace;font-size:0.85em;"
                        f"white-space:pre-wrap;'>{_annotated}</div>",
                        unsafe_allow_html=True,
                    )
                    st.divider()

                # ── E2 — Inline edit (always available) ───────────────────
                edited_key = f"edit_{i}_{pair_hash[:8]}"
                if edited_key not in st.session_state:
                    st.session_state[edited_key] = asst_content
                edited_completion = st.text_area(
                    "Assistant completion (editable — changes apply on approve)",
                    value=st.session_state[edited_key],
                    height=250,
                    key=edited_key,
                )
                if edited_completion != asst_content:
                    st.caption("✏️ Edited — will save edited version on approve")

            with st.expander("Metadata"):
                st.json(meta)

            notes_key = f"notes_{i}_{pair_hash[:8]}"
            if notes_key not in st.session_state:
                st.session_state[notes_key] = ""

            notes = st.text_input(
                "Notes (optional)",
                key=notes_key,
                placeholder="Reviewer notes...",
            )

            # E3 — four-tier approval buttons
            _hint_key = f"hint_{i}_{pair_hash[:8]}"
            if _hint_key not in st.session_state:
                st.session_state[_hint_key] = ""

            _bc1, _bc2, _bc3, _bc4 = st.columns([1, 1.4, 1.4, 1])
            _disabled = not st.session_state.reviewer_id

            with _bc1:
                if st.button(
                    "✅ Approve",
                    key=f"approve_{i}_{pair_hash[:8]}",
                    disabled=_disabled,
                    help="Full approve — goes straight to JSONL",
                ):
                    # Save edited completion back into pair before approving
                    _edited = st.session_state.get(f"edit_{i}_{pair_hash[:8]}", "")
                    _entry_msgs = entry.get("pair", {}).get("messages", [])
                    for _m in _entry_msgs:
                        if _m["role"] == "assistant" and _edited:
                            _m["content"] = _edited
                    if approve(pair_hash, st.session_state.reviewer_id, notes, _QUEUE_PATH):
                        st.success("Approved!")
                        st.session_state.queue_refresh += 1
                        st.rerun()
                    else:
                        st.error("Approve failed")

            with _bc2:
                if st.button(
                    "✅ Conditional",
                    key=f"cond_{i}_{pair_hash[:8]}",
                    disabled=(_disabled or not notes),
                    help="Approve with caveat — note required",
                ):
                    if conditional_approve(pair_hash, st.session_state.reviewer_id,
                                           notes, _QUEUE_PATH):
                        st.success("Conditionally approved")
                        st.session_state.queue_refresh += 1
                        st.rerun()
                    else:
                        st.error("Conditional approve failed")
                if not notes and not _disabled:
                    st.caption("Add a note to use conditional approve")

            with _bc3:
                st.text_input(
                    "Correction hint",
                    key=_hint_key,
                    placeholder="e.g. Finding 1 should cite SOP §3.2",
                )
                if st.button(
                    "🔄 Send for correction",
                    key=f"correct_{i}_{pair_hash[:8]}",
                    disabled=(_disabled or not st.session_state[_hint_key]),
                    help="Gemma re-runs with your hint",
                ):
                    if send_for_correction(pair_hash, st.session_state.reviewer_id,
                                           st.session_state[_hint_key], _QUEUE_PATH):
                        st.info("Sent for correction")
                        st.session_state.queue_refresh += 1
                        st.rerun()
                    else:
                        st.error("Send for correction failed")

            with _bc4:
                if st.button(
                    "❌ Reject",
                    key=f"reject_{i}_{pair_hash[:8]}",
                    disabled=_disabled,
                    help="Discard — not written to JSONL",
                ):
                    if reject(pair_hash, st.session_state.reviewer_id, notes, _QUEUE_PATH):
                        st.warning("Rejected.")
                        st.session_state.queue_refresh += 1
                        st.rerun()
                    else:
                        st.error("Reject failed")

            if not reviewer_id:
                st.caption("⚠️ Enter your reviewer ID above to use review actions")

st.divider()

# ===========================================================================
# SECTION 3 — Export
# ===========================================================================

st.header("3 · Export")

export_col1, export_col2 = st.columns([1, 2])

with export_col1:
    if st.button("📤 Write approved pairs", type="primary"):
        with st.spinner("Writing approved pairs to JSONL..."):
            export_result = write_approved(
                queue_path=_QUEUE_PATH,
                data_dir=_DATA_DIR,
            )
        st.session_state.last_export = export_result

if st.session_state.last_export is not None:
    exp = st.session_state.last_export
    st.success(
        f"Written: **{exp['written']}** · "
        f"Skipped (duplicates): {exp['skipped']} · "
        f"Errors: {exp['errors']}"
    )

st.subheader("Output files")

for stage, label in [("stage2", "Stage 2 — Domain"), ("stage3", "Stage 3 — Firm")]:
    try:
        fpath = get_output_path(stage, _DATA_DIR)
        n = count(fpath)
        dcol1, dcol2 = st.columns([3, 1])
        dcol1.markdown(f"**{label}** — `{fpath.name}` — {n} pairs")

        if fpath.exists() and n > 0:
            with open(fpath, "rb") as f:
                dcol2.download_button(
                    label="⬇️ Download",
                    data=f.read(),
                    file_name=fpath.name,
                    mime="application/jsonl",
                    key=f"dl_{stage}",
                )
        else:
            dcol2.caption("No data yet")
    except Exception:
        pass