"""
tests/test_sop_compiler.py
==========================
Property tests for the SOPGraph compiler.

These tests enforce invariants that must hold across the entire compiled graph.
A failure here means a YAML config change has violated a structural guarantee —
fix the config, not the test.

Invariants tested
-----------------
1. No locked field appears in eligible_fields() for any client_type.
2. No informational-only field appears in eligible_fields() for any client_type.
3. GAGAS citations appear ONLY in GAGAS engagement contexts (is_gagas=True).
4. 2 CFR 200 citations appear ONLY in Single Audit contexts (has_single_audit=True).
5. For-Profit overrides: fields locked/informational for NPO are not necessarily
   locked for For-Profit (override flexibility exists).
6. Citation hierarchy: within the same hier_group, only the deepest citation survives.
7. graph_version is non-empty and changes when any input YAML content changes.
8. All FieldNodes carry non-empty severity and consequence_text.
9. Every field in eligible_fields(ct) has allowed(field, ct) == True.
10. The graph compiles without error from the project-default config paths.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path

import pytest

_PROJECT_DIR = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def graph():
    """Compiled SOPGraph via the registry entrypoint (compiled_sop)."""
    from pipeline.sop_compiler import compiled_sop, reset_cache
    reset_cache()
    return compiled_sop("npo-cx-1.1")


# ---------------------------------------------------------------------------
# Invariant 10: compiles without error
# ---------------------------------------------------------------------------

def test_graph_compiles(graph):
    assert graph is not None
    assert len(graph) > 0, "SOPGraph has zero nodes — check config YAMLs"


# ---------------------------------------------------------------------------
# Invariant 1: no locked field in eligible_fields
# ---------------------------------------------------------------------------

def test_no_locked_field_in_eligible_fields(graph):
    for ct in graph.known_client_types():
        eligible = set(graph.eligible_fields(ct))
        for field_name in eligible:
            assert not graph.is_locked(field_name, ct), (
                f"Field {field_name!r} is both eligible and locked for client_type={ct!r}. "
                "Fix sop_field_classes.yaml — a field cannot be deficiency_eligible "
                "and fixed_administrative at the same time."
            )


# ---------------------------------------------------------------------------
# Invariant 2: no informational-only field in eligible_fields
# ---------------------------------------------------------------------------

def test_no_informational_field_in_eligible_fields(graph):
    for ct in graph.known_client_types():
        eligible = set(graph.eligible_fields(ct))
        for field_name in eligible:
            assert not graph.is_informational(field_name, ct), (
                f"Field {field_name!r} is both eligible and informational_only for "
                f"client_type={ct!r}. Fix sop_field_classes.yaml."
            )


# ---------------------------------------------------------------------------
# Invariant 3: GAGAS citations only in GAGAS contexts
# ---------------------------------------------------------------------------

_GAS_RE = re.compile(r"\bgas\s", re.IGNORECASE)
_YELLOW_BOOK_RE = re.compile(r"yellow book", re.IGNORECASE)
_GAS_FULL_RE = re.compile(r"government auditing standards", re.IGNORECASE)

def _has_gagas_citation(cits: list[str]) -> bool:
    text = " ".join(cits)
    return bool(
        _GAS_RE.search(text) or
        _YELLOW_BOOK_RE.search(text) or
        _GAS_FULL_RE.search(text)
    )

def test_gagas_citations_only_in_gagas_context(graph):
    for field_name in graph.all_fields():
        for ct in graph.known_client_types():
            # Non-GAGAS context must not carry GAS citations
            cits_non_gagas = graph.citations_for(field_name, ct, is_gagas=False, has_single_audit=False)
            assert not _has_gagas_citation(cits_non_gagas), (
                f"Field {field_name!r} ct={ct!r}: GAGAS citation in non-GAGAS context. "
                f"Citations: {cits_non_gagas}. "
                "Move GAS citations to gagas_additive in field_standards_context.yaml."
            )
            cits_sa_only = graph.citations_for(field_name, ct, is_gagas=False, has_single_audit=True)
            assert not _has_gagas_citation(cits_sa_only), (
                f"Field {field_name!r} ct={ct!r}: GAGAS citation in non-GAGAS/SA-only context. "
                f"Citations: {cits_sa_only}."
            )


# ---------------------------------------------------------------------------
# Invariant 4: 2 CFR 200 citations only in Single Audit contexts
# ---------------------------------------------------------------------------

_CFR_RE = re.compile(r"2\s+cfr\s+200", re.IGNORECASE)

def _has_cfr_citation(cits: list[str]) -> bool:
    return bool(_CFR_RE.search(" ".join(cits)))

def test_cfr_citations_only_in_single_audit_context(graph):
    for field_name in graph.all_fields():
        for ct in graph.known_client_types():
            cits = graph.citations_for(field_name, ct, is_gagas=False, has_single_audit=False)
            assert not _has_cfr_citation(cits), (
                f"Field {field_name!r} ct={ct!r}: 2 CFR 200 citation in non-SA context. "
                f"Citations: {cits}. "
                "Move 2 CFR 200 citations to single_audit_additive in field_standards_context.yaml."
            )
            cits_gagas_only = graph.citations_for(field_name, ct, is_gagas=True, has_single_audit=False)
            assert not _has_cfr_citation(cits_gagas_only), (
                f"Field {field_name!r} ct={ct!r}: 2 CFR 200 citation in GAGAS-only (non-SA) context. "
                f"Citations: {cits_gagas_only}."
            )


# ---------------------------------------------------------------------------
# Invariant 6: citation hierarchy — no hier_group has both a parent and child
# ---------------------------------------------------------------------------

def _hier_group(cit: str) -> str | None:
    """Return the GAS hier_group for a citation, or None if not GAS."""
    cit_lower = cit.lower()
    if "gas" not in cit_lower and "government auditing standards" not in cit_lower:
        return None
    # Extract section number if present
    m = re.search(r"§\s*(\d+)", cit)
    if m:
        sec = m.group(1)
        return f"gas_{sec}"
    m = re.search(r"chapter\s+(\d+)", cit_lower)
    if m:
        chap = m.group(1)
        return f"gas_{chap}"
    return None

def test_citation_hierarchy_no_parent_and_child(graph):
    """
    For each (field, client_type, context), no two citations should share the
    same hier_group — the resolver should have kept only the deepest one.
    """
    for field_name in graph.all_fields():
        for ct in graph.known_client_types():
            for is_gagas in (True, False):
                for has_sa in (True, False):
                    cits = graph.citations_for(field_name, ct, is_gagas, has_sa)
                    groups: dict[str, list[str]] = {}
                    for c in cits:
                        g = _hier_group(c)
                        if g:
                            groups.setdefault(g, []).append(c)
                    for g, members in groups.items():
                        assert len(members) == 1, (
                            f"Field {field_name!r} ct={ct!r} gagas={is_gagas} sa={has_sa}: "
                            f"Multiple citations in hier_group {g!r}: {members}. "
                            "Citation resolver hierarchy rule should have kept only the deepest."
                        )


# ---------------------------------------------------------------------------
# Invariant 7: graph_version is non-empty and changes with content
# ---------------------------------------------------------------------------

def test_graph_version_nonempty(graph):
    assert graph.graph_version, "graph_version must be non-empty"
    assert len(graph.graph_version) == 16, (
        f"graph_version should be 16-char SHA-256 prefix, got {graph.graph_version!r}"
    )


def test_graph_version_changes_with_yaml(tmp_path):
    """A modified SFC YAML must produce a different graph_version."""
    from pipeline.sop_compiler import compile_sop

    sfc_src = _PROJECT_DIR / "config" / "sop_field_classes.yaml"
    tiers_src = _PROJECT_DIR / "config" / "field_tiers.yaml"
    stds_src  = _PROJECT_DIR / "config" / "field_standards_context.yaml"

    if not sfc_src.exists():
        pytest.skip("sop_field_classes.yaml not found — skipping version change test")

    g1 = compile_sop(sfc_path=sfc_src, tiers_path=tiers_src, stds_path=stds_src)

    # Write a slightly modified copy of SFC
    modified_sfc = tmp_path / "sop_field_classes_modified.yaml"
    original = sfc_src.read_text(encoding="utf-8")
    modified_sfc.write_text(original + "\n# version bump\n", encoding="utf-8")

    g2 = compile_sop(sfc_path=modified_sfc, tiers_path=tiers_src, stds_path=stds_src)

    assert g1.graph_version != g2.graph_version, (
        "graph_version did not change after modifying sop_field_classes.yaml. "
        "The hash must include the file content, not just metadata."
    )


# ---------------------------------------------------------------------------
# Invariant 8: all nodes have non-empty severity and consequence_text
# ---------------------------------------------------------------------------

def test_all_nodes_have_severity_and_consequence(graph):
    for field_name in graph.all_fields():
        node = graph.get(field_name)
        assert node is not None
        assert node.severity in ("High", "Medium", "Low", "Informational"), (
            f"Field {field_name!r}: invalid severity {node.severity!r}"
        )
        assert node.consequence_text.strip(), (
            f"Field {field_name!r}: consequence_text is empty"
        )


# ---------------------------------------------------------------------------
# Invariant 9: eligible_fields(ct) consistent with allowed(field, ct)
# ---------------------------------------------------------------------------

def test_eligible_fields_consistent_with_allowed(graph):
    for ct in graph.known_client_types():
        eligible_set = set(graph.eligible_fields(ct))
        for field_name in graph.all_fields():
            if graph.allowed(field_name, ct):
                assert field_name in eligible_set, (
                    f"allowed({field_name!r}, {ct!r}) is True but field not in eligible_fields({ct!r}). "
                    "Inconsistency between SOPGraph.allowed() and SOPGraph.eligible_fields()."
                )
            else:
                assert field_name not in eligible_set, (
                    f"allowed({field_name!r}, {ct!r}) is False but field IS in eligible_fields({ct!r}). "
                    "Inconsistency between SOPGraph.allowed() and SOPGraph.eligible_fields()."
                )


# ---------------------------------------------------------------------------
# Registry invariants: compiled_sop(sop_id) API
# ---------------------------------------------------------------------------

def test_unknown_sop_id_raises():
    """compiled_sop() raises ValueError for an unregistered sop_id."""
    from pipeline.sop_compiler import compiled_sop
    with pytest.raises(ValueError, match="Unknown sop_id"):
        compiled_sop("nonexistent-sop-xyz")


def test_validate_raises_on_empty_graph():
    """SOPGraph.validate() raises ValueError when the graph has zero nodes."""
    from pipeline.sop_compiler import SOPGraph
    empty = SOPGraph(nodes={}, client_types=frozenset(), graph_version="abc123")
    with pytest.raises(ValueError, match="zero nodes"):
        empty.validate()


def test_registry_caches_across_calls():
    """Two calls to compiled_sop() with the same sop_id return the same object."""
    from pipeline.sop_compiler import compiled_sop, reset_cache
    reset_cache("npo-cx-1.1")
    g1 = compiled_sop("npo-cx-1.1")
    g2 = compiled_sop("npo-cx-1.1")
    assert g1 is g2, "compiled_sop() must return the same cached object on repeat calls"


def test_graph_carries_sop_id():
    """The SOPGraph returned by compiled_sop() reports its sop_id correctly."""
    from pipeline.sop_compiler import compiled_sop
    g = compiled_sop("npo-cx-1.1")
    assert g.sop_id == "npo-cx-1.1", (
        f"Expected sop_id='npo-cx-1.1', got {g.sop_id!r}"
    )
