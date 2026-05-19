"""
pipeline/sop_compiler.py
=========================
SOP rule graph compiler: converts config YAMLs into an executable SOPGraph.

Design principle
----------------
The SOP is NOT data. After compilation, it becomes an executable rule graph
where each node represents a canonical field and carries its complete policy:
which client types allow deficiency sampling, what citation applies, what
standards are layered on top, what severity to assign.

This is what allows the same SOP to serve multiple clients correctly without
rewriting any logic. New client type → update sop_field_classes.yaml →
recompile → correct behaviour everywhere.

SOPGraph structure
------------------
Each node in the graph is a FieldNode:

    FieldNode:
        field_name              canonical field name
        allowed_deficiency_for  set of client_types where deficiency_allowed
        locked_for              set of client_types where field is locked
        sop_section             internal SOP section (from FieldAuthority)
        severity                High | Medium | Low per SOP
        citations               {client_type: resolved list[str]}
                                pre-resolved citation sets per engagement context
        consequence_text        pure consequence statement (no citations embedded)
        finding_label           audit finding headline

The graph exposes:
    allowed(field, client_type)           → bool
    citations_for(field, client_type,
                  is_gagas, has_sa)       → list[str]
    finding_text(field, client_type,
                 is_gagas, has_sa)        → str (consequence + standards)
    eligible_fields(client_type)         → list[str]

Public API
----------
    SOPGraph                  — compiled rule graph
    FieldNode                 — one node in the graph
    compile_sop()             — compile all config YAMLs → SOPGraph (uncached, pathable)
    compiled_sop(sop_id)      — return validated, cached SOPGraph for the given SOP
    reset_cache([sop_id])     — evict one or all entries from the compiled SOPGraph registry
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_DIR = Path(__file__).parent.parent

# Engagement context keys used for pre-resolved citation sets.
# Each combination is pre-compiled at graph build time.
_ENGAGEMENT_CONTEXTS: list[tuple[bool, bool]] = [
    (False, False),   # GAAS only
    (True,  False),   # GAGAS
    (False, True),    # Single Audit (rare, but possible)
    (True,  True),    # GAGAS + Single Audit
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FieldNode:
    """
    One node in the SOPGraph — the complete compiled policy for one field.

    All values are pre-resolved at compile time. No YAML parsing or
    dynamic resolution happens at query time.
    """
    field_name:            str
    allowed_deficiency_for: frozenset[str]   # client_types where deficiency_allowed
    locked_for:            frozenset[str]    # client_types where is_locked
    informational_for:     frozenset[str]    # client_types where is_informational_only
    locked_in_base:        bool              # True if locked regardless of client_type
    sampling_weight:       float | None      # base weight; None → caller uses tier default
    sop_section:           str | None        # internal SOP §section
    severity:              str               # High | Medium | Low
    consequence_text:      str               # pure consequence (no citations)
    finding_label:         str               # audit finding headline

    # Pre-resolved citation lists keyed by (client_type, is_gagas, has_single_audit)
    # Populated for all _ENGAGEMENT_CONTEXTS × known client_types at compile time.
    _citation_cache: dict[tuple[str, bool, bool], list[str]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def allowed(self, client_type: str) -> bool:
        """True iff deficiency sampling is allowed for this client_type."""
        return client_type in self.allowed_deficiency_for

    def is_locked(self, client_type: str) -> bool:
        """True iff this field is fixed_administrative for this client_type."""
        return client_type in self.locked_for

    def is_informational(self, client_type: str) -> bool:
        """True iff this field is informational_only for this client_type."""
        return client_type in self.informational_for

    def citations_for(
        self,
        client_type:      str,
        is_gagas:         bool,
        has_single_audit: bool,
    ) -> list[str]:
        """
        Pre-resolved standards citations for this field in the given context.
        Returns empty list if none configured.
        """
        return self._citation_cache.get((client_type, is_gagas, has_single_audit), [])

    def finding_text(
        self,
        client_type:      str,
        is_gagas:         bool,
        has_single_audit: bool,
    ) -> str:
        """
        Full risk text for a finding: consequence + 'Applicable standards: ...'
        Returns consequence alone if no citations are configured.
        """
        cits = self.citations_for(client_type, is_gagas, has_single_audit)
        if cits:
            return f"{self.consequence_text} Applicable standards: {'; '.join(cits)}."
        return self.consequence_text


class SOPGraph:
    """
    Compiled SOP rule graph.

    Immutable after construction. Thread-safe for reads.
    Query via allowed(), eligible_fields(), citations_for(), finding_text().
    """

    def __init__(
        self,
        nodes:        dict[str, FieldNode],
        client_types: frozenset[str],
        graph_version: str = "",
        sop_id:       str = "",
    ) -> None:
        self._nodes        = nodes
        self._client_types = client_types
        self._graph_version = graph_version
        self._sop_id       = sop_id

    @property
    def graph_version(self) -> str:
        """
        SHA-256 hex digest (first 16 chars) of the combined content of all
        input YAMLs: sop_field_classes, field_tiers, field_standards_context.
        Changes whenever any of those files changes. Recorded in pair metadata
        so training pairs can be traced back to the compiler state that produced them.
        """
        return self._graph_version

    @property
    def sop_id(self) -> str:
        """SOP identifier this graph was compiled for."""
        return self._sop_id

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Raise ValueError if the graph violates any structural invariant.

        Called automatically by compiled_sop() before the graph enters
        the registry.  An invalid graph is never cached — the exception
        propagates to the caller and the next call retries compilation.
        """
        if not self._nodes:
            raise ValueError("SOPGraph has zero nodes — check config YAMLs")
        if not self._graph_version:
            raise ValueError("SOPGraph graph_version is empty")
        for field_name, node in self._nodes.items():
            if node.severity not in ("High", "Medium", "Low", "Informational"):
                raise ValueError(
                    f"SOPGraph invariant violated: field {field_name!r} has "
                    f"invalid severity {node.severity!r}"
                )
            if not node.consequence_text.strip():
                raise ValueError(
                    f"SOPGraph invariant violated: field {field_name!r} has "
                    "empty consequence_text"
                )
            for ct in self._client_types:
                if node.allowed(ct) and node.is_locked(ct):
                    raise ValueError(
                        f"SOPGraph invariant violated: field {field_name!r} is "
                        f"both eligible and locked for client_type={ct!r}. "
                        "Fix sop_field_classes.yaml."
                    )
                if node.allowed(ct) and node.is_informational(ct):
                    raise ValueError(
                        f"SOPGraph invariant violated: field {field_name!r} is "
                        f"both eligible and informational for client_type={ct!r}. "
                        "Fix sop_field_classes.yaml."
                    )

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get(self, field_name: str) -> FieldNode | None:
        """Return the FieldNode for a canonical field, or None if not compiled."""
        return self._nodes.get(field_name)

    def allowed(self, field_name: str, client_type: str) -> bool:
        """True iff deficiency sampling is allowed for field × client_type."""
        node = self._nodes.get(field_name)
        return node is not None and node.allowed(client_type)

    def is_locked(self, field_name: str, client_type: str) -> bool:
        """True iff field is fixed_administrative for client_type."""
        node = self._nodes.get(field_name)
        return node is not None and node.is_locked(client_type)

    def is_informational(self, field_name: str, client_type: str) -> bool:
        """True iff field is informational_only for client_type."""
        node = self._nodes.get(field_name)
        return node is not None and node.is_informational(client_type)

    def sampling_weight(self, field_name: str) -> float | None:
        """Base sampling weight for this field; None → caller uses tier default."""
        node = self._nodes.get(field_name)
        return node.sampling_weight if node else None

    def eligible_fields(self, client_type: str) -> list[str]:
        """All fields where deficiency sampling is allowed for client_type."""
        return [f for f, n in self._nodes.items() if n.allowed(client_type)]

    def citations_for(
        self,
        field_name:       str,
        client_type:      str,
        is_gagas:         bool,
        has_single_audit: bool,
    ) -> list[str]:
        """Pre-resolved citation list for a field in the given context."""
        node = self._nodes.get(field_name)
        if node is None:
            return []
        return node.citations_for(client_type, is_gagas, has_single_audit)

    def finding_text(
        self,
        field_name:       str,
        client_type:      str,
        is_gagas:         bool,
        has_single_audit: bool,
    ) -> str | None:
        """
        Full risk text for a finding in the given context.
        Returns None if the field has no node in the graph.
        """
        node = self._nodes.get(field_name)
        if node is None:
            return None
        return node.finding_text(client_type, is_gagas, has_single_audit)

    def all_fields(self) -> list[str]:
        return list(self._nodes.keys())

    def known_client_types(self) -> frozenset[str]:
        return self._client_types

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        return (
            f"SOPGraph(sop_id={self._sop_id!r}, fields={len(self)}, "
            f"client_types={sorted(self._client_types)}, "
            f"version={self._graph_version!r})"
        )


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------

def compile_sop(
    sfc_path:    Path | None = None,
    tiers_path:  Path | None = None,
    stds_path:   Path | None = None,
    sop_id:      str = "",
) -> SOPGraph:
    """
    Compile sop_field_classes.yaml + field_tiers.yaml + field_standards_context.yaml
    into a SOPGraph.

    All citation resolution, consequence assembly, and label lookup happen here.
    The resulting SOPGraph is queried directly at render time — no YAML I/O,
    no string parsing, no dynamic dispatch.

    Uncached and pathable — call compiled_sop(sop_id) for the cached registry
    path.  Use this function directly in tests with custom config paths.
    """
    from pipeline.field_authority import _build_authority_table
    from raw_to_training_pair.citation_resolver import resolve as _resolve_cits
    from raw_to_training_pair.completion_renderer import (
        _FIELD_CONSEQUENCE_TEXT,
        _DEFAULT_CONSEQUENCE_TEXT,
        _FIELD_FINDING_LABEL,
        _load_standards_ctx,
    )

    import yaml as _yaml

    _sfc   = sfc_path   or (_PROJECT_DIR / "config" / "sop_field_classes.yaml")
    _tiers = tiers_path or (_PROJECT_DIR / "config" / "field_tiers.yaml")
    _stds  = stds_path  or (_PROJECT_DIR / "config" / "field_standards_context.yaml")

    # Compute a version hash from all three input files so any config change
    # produces a new graph_version that can be recorded in pair metadata.
    _h = hashlib.sha256()
    for _p in (_sfc, _tiers, _stds):
        if _p.exists():
            _h.update(_p.read_bytes())
    _graph_version = _h.hexdigest()[:16]

    authority = _build_authority_table(_sfc, _tiers)
    stds_ctx  = _load_standards_ctx()

    # Collect all known client types from SFC overrides + defaults
    _known_client_types: set[str] = {"NPO", "Government", "Tribal", "For-Profit"}
    for f in authority.all_fields():
        for ct in ("NPO", "Government", "Tribal", "For-Profit"):
            pass   # defaults already in set; SFC overrides surfaced via authority.get()

    nodes: dict[str, FieldNode] = {}

    for field_name in authority.all_fields():
        # Base authority (client_type="" — no override)
        base_auth = authority.get(field_name, "")
        if base_auth is None:
            continue

        # Pre-resolve per client_type allowed/locked/informational sets
        allowed_for:     set[str] = set()
        locked_for:      set[str] = set()
        informational_for: set[str] = set()
        for ct in _known_client_types:
            auth = authority.get(field_name, ct) or base_auth
            if auth.deficiency_allowed:
                allowed_for.add(ct)
            if auth.is_locked:
                locked_for.add(ct)
            if auth.is_informational:
                informational_for.add(ct)

        # locked_in_base: True when the BASE authority (no client_type override)
        # is locked — meaning no client_type can unlock it via an override.
        locked_in_base = base_auth.is_locked

        # Consequence text and finding label from renderer registries
        consequence   = _FIELD_CONSEQUENCE_TEXT.get(field_name, _DEFAULT_CONSEQUENCE_TEXT)
        finding_label = _FIELD_FINDING_LABEL.get(field_name, field_name.replace("_", " ").title())

        # Pre-resolve citation cache for all contexts × client_types
        citation_cache: dict[tuple[str, bool, bool], list[str]] = {}
        field_stds = stds_ctx.get(field_name, {})

        for ct in _known_client_types:
            for is_gagas, has_sa in _ENGAGEMENT_CONTEXTS:
                cits: list[str] = list(field_stds.get("base") or [])
                if is_gagas:
                    cits.extend(field_stds.get("gagas_additive") or [])
                if has_sa:
                    cits.extend(field_stds.get("single_audit_additive") or [])
                citation_cache[(ct, is_gagas, has_sa)] = _resolve_cits(cits)

        node = FieldNode(
            field_name             = field_name,
            allowed_deficiency_for = frozenset(allowed_for),
            locked_for             = frozenset(locked_for),
            informational_for      = frozenset(informational_for),
            locked_in_base         = locked_in_base,
            sampling_weight        = base_auth.sampling_weight,
            sop_section            = base_auth.sop_section,
            severity               = base_auth.severity,
            consequence_text       = consequence,
            finding_label          = finding_label,
            _citation_cache        = citation_cache,
        )
        nodes[field_name] = node

        logger.debug(
            "sop_compiler: compiled '%s' — allowed_for=%s locked_for=%s sop=%s",
            field_name, sorted(allowed_for), sorted(locked_for), base_auth.sop_section,
        )

    graph = SOPGraph(
        nodes=nodes,
        client_types=frozenset(_known_client_types),
        graph_version=_graph_version,
        sop_id=sop_id,
    )
    logger.info(
        "sop_compiler: compiled SOPGraph — %d fields, %d client_types, version=%s",
        len(graph), len(_known_client_types), _graph_version,
    )
    return graph


# Known SOP identifiers. All currently compile from the same default config
# paths (one YAML set covers the initial NPO-CX-1.1 scope).
# Future: map each sop_id to its own (sfc_path, tiers_path, stds_path) tuple
# so genuinely independent SOPs can carry different field policies.
_SOP_CONFIG_REGISTRY: frozenset[str] = frozenset({
    "npo-cx-1.1",
    "ad-5001",
})

_registry: dict[str, SOPGraph] = {}


def compiled_sop(sop_id: str) -> SOPGraph:
    """
    Return the compiled, validated SOPGraph for sop_id.

    First call for a given sop_id compiles from config YAMLs, calls
    graph.validate(), then caches the result.  Only graphs that pass
    validate() enter the registry — if validate() raises, the graph is
    never cached and the exception propagates so the caller gets a clean
    traceback pointing to the broken config.

    Parameters
    ----------
    sop_id : str
        Must be a key in _SOP_CONFIG_REGISTRY (e.g. "npo-cx-1.1").

    Raises
    ------
    ValueError
        If sop_id is not in _SOP_CONFIG_REGISTRY, or the compiled graph
        fails structural validation.
    """
    if sop_id not in _SOP_CONFIG_REGISTRY:
        raise ValueError(
            f"Unknown sop_id {sop_id!r}. "
            f"Registered SOPs: {sorted(_SOP_CONFIG_REGISTRY)}. "
            "Add the sop_id to _SOP_CONFIG_REGISTRY in sop_compiler.py."
        )
    if sop_id not in _registry:
        graph = compile_sop(sop_id=sop_id)
        graph.validate()
        _registry[sop_id] = graph
        logger.info(
            "sop_compiler: loaded SOPGraph sop_id=%r version=%s fields=%d client_types=%s",
            sop_id, graph.graph_version, len(graph), sorted(graph.known_client_types()),
        )
    return _registry[sop_id]


def list_loaded() -> list[str]:
    """Return the sop_ids currently held in the compiled registry."""
    return list(_registry.keys())


def describe_registry() -> dict[str, dict]:
    """
    Snapshot of all loaded SOPGraphs keyed by sop_id.

    Returns {sop_id: {graph_version, fields, client_types}} for each entry.
    Useful for health endpoints, CLI diagnostics, and pre-flight checks in
    multi-SOP deployments.  Reads the live registry without triggering new
    compilations.
    """
    return {
        sid: {
            "graph_version": g.graph_version,
            "fields":        len(g),
            "client_types":  sorted(g.known_client_types()),
        }
        for sid, g in _registry.items()
    }


def reset_cache(sop_id: str | None = None) -> None:
    """
    Evict compiled SOPGraph(s) from the registry.

    Parameters
    ----------
    sop_id : str | None
        If given, evicts only that entry.
        If None (default), clears the entire registry.
    Useful in tests and after YAML hot-reloads.
    """
    if sop_id is None:
        _registry.clear()
    else:
        _registry.pop(sop_id, None)
