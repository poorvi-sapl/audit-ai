"""
streamlit_app/generation_pair_renderer.py
===========================================
Render generation-task pairs in the Streamlit review queue.

The existing review-task render path (System / User / Assistant tabs
with edit-and-approve flow) doesn't fit generation pairs — reviewers
need a side-by-side facts-vs-gold comparison to verify "did the AI
fill the workpaper correctly?" not "did the AI find the deficiencies?"

This module is invoked from app.py's queue loop when
metadata.pair_type == "generation". Kept standalone so the heavy
review-task render code in app.py doesn't get further tangled.

Public API
----------
    render_generation_pair(pair, index)
        Renders one generation pair inside the current st container.
        Returns nothing; side effects only.
"""

from __future__ import annotations

import json
from typing import Any

# streamlit is imported lazily inside the render functions below — the
# pure helpers in this module (_compare_status, _build_comparison_rows,
# _summary_counts, _value_for_display) don't need it, and keeping the
# top-level import-free lets the helpers be unit-tested in environments
# without streamlit installed.

# Visual status indicators for the comparison column
_MATCH = "✓"
_MISMATCH = "✗"
_GOLD_ONLY = "⊕"      # gold has value, no matching fact
_FACT_ONLY = "⊖"      # fact has value, gold left null
_BOTH_NULL = "·"


def _value_for_display(val: Any) -> str:
    """Compact display representation for a field value."""
    if val is None:
        return "—"
    if isinstance(val, bool):
        return "Yes" if val else "No"
    s = str(val)
    return s if len(s) <= 80 else s[:77] + "..."


def _compare_status(fact_value: Any, gold_value: Any) -> str:
    """Return one of the indicator constants."""
    fact_present = fact_value is not None
    gold_present = gold_value is not None and gold_value != ""
    if not fact_present and not gold_present:
        return _BOTH_NULL
    if fact_present and not gold_present:
        return _FACT_ONLY
    if gold_present and not fact_present:
        return _GOLD_ONLY
    # Both present — compare
    if isinstance(fact_value, bool) or isinstance(gold_value, bool):
        return _MATCH if fact_value == gold_value else _MISMATCH
    return _MATCH if str(fact_value).strip() == str(gold_value).strip() else _MISMATCH


def _summary_counts(rows: list[dict]) -> dict[str, int]:
    counts = {_MATCH: 0, _MISMATCH: 0, _GOLD_ONLY: 0, _FACT_ONLY: 0, _BOTH_NULL: 0}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return counts


def _build_comparison_rows(
    facts_summary: dict, gold_fields: dict,
) -> list[dict]:
    """Build one row per field_id (union of facts and gold keys),
    each carrying the comparison data. Sorted by field_id."""
    all_ids = sorted(set(facts_summary.keys()) | set(gold_fields.keys()))
    rows: list[dict] = []
    for fid in all_ids:
        fact_entry = facts_summary.get(fid) or {}
        gold_entry = gold_fields.get(fid) or {}
        fact_value = fact_entry.get("value")
        gold_value = gold_entry.get("value")
        sources = fact_entry.get("sources", [])
        citations = gold_entry.get("citations", [])
        rows.append({
            "field_id": fid,
            "status": _compare_status(fact_value, gold_value),
            "fact_value_display": _value_for_display(fact_value),
            "gold_value_display": _value_for_display(gold_value),
            "fact_value_raw": fact_value,
            "gold_value_raw": gold_value,
            "fact_confidence": fact_entry.get("confidence"),
            "fact_extractor_method": fact_entry.get("extractor_method", ""),
            "sources_count": len(sources),
            "first_source": sources[0] if sources else None,
            "citations_count": len(citations),
        })
    return rows


def _render_summary_banner(rows: list[dict]) -> None:
    import streamlit as st
    counts = _summary_counts(rows)
    total = len(rows)
    cols = st.columns(5)
    cols[0].metric("Total fields", total)
    cols[1].metric(f"{_MATCH} Match", counts.get(_MATCH, 0))
    cols[2].metric(f"{_MISMATCH} Mismatch", counts.get(_MISMATCH, 0))
    cols[3].metric(f"{_GOLD_ONLY} Gold only", counts.get(_GOLD_ONLY, 0))
    cols[4].metric(f"{_FACT_ONLY} Fact only", counts.get(_FACT_ONLY, 0))

    mismatches = counts.get(_MISMATCH, 0)
    if mismatches > 0:
        st.warning(
            f"⚠️ {mismatches} field(s) where the gold value disagrees "
            "with the extracted fact — these warrant reviewer attention "
            "(extraction was wrong, gold was wrong, or values are both "
            "valid representations)."
        )
    elif counts.get(_GOLD_ONLY, 0) > 0:
        st.info(
            f"ℹ️ {counts.get(_GOLD_ONLY, 0)} field(s) filled in gold "
            "without matching extracted fact — these are SOP-driven "
            "defaults or auditor-inferred values, not grounded in source docs."
        )


def _render_comparison_table(rows: list[dict]) -> None:
    """Render the side-by-side facts vs gold table.

    Uses st.dataframe for sortable / filterable display rather than
    a custom HTML table — gives reviewers built-in sorting and
    column resizing for free.
    """
    import streamlit as st
    table_data = [
        {
            "Field": r["field_id"],
            "Status": r["status"],
            "Extracted fact": r["fact_value_display"],
            "Gold value": r["gold_value_display"],
            "Fact conf": (
                f"{r['fact_confidence']:.2f}"
                if r["fact_confidence"] is not None else "—"
            ),
            "Method": r["fact_extractor_method"] or "—",
            "Sources": r["sources_count"],
            "Citations": r["citations_count"],
        }
        for r in rows
    ]
    st.dataframe(table_data, use_container_width=True, hide_index=True)


def _render_mismatches_detail(rows: list[dict]) -> None:
    """Expand: for every MISMATCH row, show the fact's first source
    text alongside the gold value. Lets the reviewer judge which
    side is right with one click."""
    import streamlit as st
    mismatches = [r for r in rows if r["status"] == _MISMATCH]
    if not mismatches:
        return
    with st.expander(
        f"🔍 Mismatch detail ({len(mismatches)} field(s) need reviewer judgment)",
        expanded=False,
    ):
        for r in mismatches:
            st.markdown(f"**`{r['field_id']}`**")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("Extracted fact:")
                st.code(r["fact_value_display"], language=None)
                if r["first_source"]:
                    src = r["first_source"]
                    st.caption(
                        f"From {src.get('document_type', '?')} "
                        f"p.{src.get('page', '?')}"
                    )
                    quoted = (src.get("quoted_text") or "").strip()
                    if quoted:
                        st.caption(f"Quoted: \"{quoted[:200]}\"")
            with c2:
                st.markdown("Gold value:")
                st.code(r["gold_value_display"], language=None)
            st.divider()


def _render_metadata_summary(meta: dict) -> None:
    """Compact metadata view — workpaper, engagement, key counts,
    schema/PII issues."""
    import streamlit as st
    c1, c2, c3, c4 = st.columns(4)
    c1.caption(f"Workpaper: **{meta.get('workpaper_type', '?')}**")
    c2.caption(f"Engagement: **{meta.get('engagement_id', '?')}**")
    c3.caption(f"SOP chunks: **{meta.get('sop_chunks_count', 0)}**")
    c4.caption(f"Task: **{meta.get('task', '?')}**")

    schema_issues = meta.get("schema_issues") or []
    if schema_issues:
        with st.expander(
            f"⚠️ Schema issues ({len(schema_issues)})", expanded=False,
        ):
            for issue in schema_issues:
                st.markdown(f"- {issue}")

    pii = meta.get("pii_issues") or {}
    total_pii = (pii.get("user_pii_count") or 0) + (pii.get("assistant_pii_count") or 0)
    if total_pii > 0:
        st.error(
            f"🛑 PII detected — user: {pii.get('user_pii_count')} "
            f"({pii.get('user_pii_types')}), assistant: "
            f"{pii.get('assistant_pii_count')} "
            f"({pii.get('assistant_pii_types')}). DO NOT APPROVE until "
            "the PII is scrubbed."
        )


def _render_gold_json(messages: list[dict]) -> None:
    """Raw gold JSON for reviewers who want to inspect the full
    assistant target as the model will see it."""
    import streamlit as st
    assistant = next(
        (m["content"] for m in messages if m.get("role") == "assistant"),
        "",
    )
    with st.expander("📄 Raw gold (assistant message)", expanded=False):
        try:
            parsed = json.loads(assistant)
            st.json(parsed)
        except json.JSONDecodeError:
            st.code(assistant, language="json")


def _render_user_message(messages: list[dict]) -> None:
    """User message — what the fine-tuned model will receive as input."""
    import streamlit as st
    user = next(
        (m["content"] for m in messages if m.get("role") == "user"), "",
    )
    with st.expander("📝 User message (model input)", expanded=False):
        st.text(user)


def render_generation_pair(pair: dict, index: int) -> None:
    """Render one generation pair in the review queue.

    Layout:
        1. Metadata summary (workpaper, engagement, counts)
        2. Comparison summary banner (match counts)
        3. Side-by-side facts vs gold table
        4. Mismatch detail expander (if any)
        5. Raw gold JSON (expander)
        6. User message (expander)

    Action buttons (Approve / Reject / etc.) are handled by the
    caller in app.py so this module stays UI-only and avoids
    coupling to the review queue's state-management code.
    """
    import streamlit as st
    meta = pair.get("metadata") or {}
    messages = pair.get("messages") or []

    _render_metadata_summary(meta)

    facts_summary = meta.get("extracted_facts_summary") or {}
    assistant = next(
        (m.get("content", "") for m in messages if m.get("role") == "assistant"),
        "",
    )
    try:
        gold_payload = json.loads(assistant)
        gold_fields = gold_payload.get("fields", {})
    except json.JSONDecodeError:
        st.error(
            "Gold JSON failed to parse — the model target is malformed. "
            "Cannot render side-by-side view; falling back to raw display."
        )
        gold_fields = {}

    rows = _build_comparison_rows(facts_summary, gold_fields)

    if not rows:
        st.info("No fields to compare (empty facts and empty gold).")
    else:
        _render_summary_banner(rows)
        _render_comparison_table(rows)
        _render_mismatches_detail(rows)

    _render_gold_json(messages)
    _render_user_message(messages)
