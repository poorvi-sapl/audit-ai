"""
pipeline/field_authority.py
============================
SOP field authority compiler: internal implementation only.

FieldAuthorityTable is a compiler-internal type used exclusively by
sop_compiler.py to build the SOPGraph. Do not import or use this module
directly in production code — query the SOPGraph via compiled_sop(sop_id) instead.

Public API (compiler-internal only)
------------------------------------
    FieldAuthority              — immutable policy record for one (field, client_type)
    FieldAuthorityTable         — compiled table, queryable by field + client_type
    _build_authority_table()    — compile SFC YAML + tiers → FieldAuthorityTable
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_PROJECT_DIR = Path(__file__).parent.parent
_SFC_PATH    = _PROJECT_DIR / "config" / "sop_field_classes.yaml"
_TIERS_PATH  = _PROJECT_DIR / "config" / "field_tiers.yaml"

# Class precedence: higher index = more restrictive. When two slugs map to
# the same canonical field with conflicting classes, the more restrictive wins.
_CLASS_RANK: dict[str, int] = {
    "deficiency_eligible":  0,
    "fixed_administrative": 1,
    "informational_only":   2,
}

_SEVERITY_TIER: dict[str, str] = {"tier1": "High", "tier2": "Medium"}
_SEVERITY_VALID = frozenset({"High", "Medium", "Low", "Informational"})

# Slug → §Section (duplicated from completion_renderer to avoid cross-package dep)
_SLUG_RE = re.compile(r"^(?:(part_[ivx]+)_)?q(\d+[a-z]?)$", re.IGNORECASE)


def _slug_to_section(slug: str) -> str | None:
    m = _SLUG_RE.match(slug.strip().lower())
    if not m:
        return None
    part_prefix = m.group(1)
    q_body      = m.group(2)
    num_match   = re.match(r"(\d+)([a-z]?)$", q_body)
    if not num_match:
        return None
    num, letter = num_match.group(1), num_match.group(2)
    section = f"§Q{num}" + (f"({letter})" if letter else "")
    if part_prefix:
        part_label = part_prefix.replace("_", " ").title()
        section    = f"§{part_label} Q{num}" + (f"({letter})" if letter else "")
    return section


def _coerce_severity(raw: str | None, tier_key: str) -> str:
    if raw and str(raw).title() in _SEVERITY_VALID:
        return str(raw).title()
    return _SEVERITY_TIER.get(tier_key, "Medium")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldAuthority:
    """
    Immutable compiled authority record for one (field_name, client_type) pair.

    All downstream policy decisions derive from this record. Never re-read
    the YAML to answer a policy question — query the FieldAuthorityTable instead.
    """
    field_name:         str
    deficiency_allowed: bool          # True iff effective class == "deficiency_eligible"
    is_locked:          bool          # True iff effective class == "fixed_administrative"
    is_informational:   bool          # True iff effective class == "informational_only"
    sampling_weight:    float | None  # None → caller uses tier-based default
    sop_section:        str | None    # §section for citation; None → sop_unverified
    severity:           str           # High | Medium | Low | Informational
    client_type:        str           # "" = base; "NPO"/"Government"/etc. = override applied


class FieldAuthorityTable:
    """
    Compiled, queryable field authority table.

    Compiled once from sop_field_classes.yaml + field_tiers.yaml at startup.
    Shared across sampler, hard gate, sop_compiler, and quality gate.
    Thread-safe for reads (immutable after construction).

    Internal structure
    ------------------
    _entries: {canonical_field: {"": base_authority, "NPO": override, ...}}

    The "" key holds the base authority (no client_type override applied).
    Client-type keys hold the pre-resolved override. get() returns the most
    specific match: client_type override → base → None.
    """

    def __init__(self, entries: dict[str, dict[str, FieldAuthority]]) -> None:
        self._entries = entries

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get(self, field_name: str, client_type: str = "") -> FieldAuthority | None:
        """
        Return compiled authority for a field, with client_type override applied.

        Returns None if the field has no SFC entry (no explicit authority).
        A None return means deficiency_allowed=False by the strict authority rule.
        """
        field_map = self._entries.get(field_name)
        if field_map is None:
            return None
        return field_map.get(client_type) or field_map.get("")

    def deficiency_allowed(self, field_name: str, client_type: str = "") -> bool:
        """True iff this field may be used as a synthetic deficiency for client_type."""
        auth = self.get(field_name, client_type)
        return auth is not None and auth.deficiency_allowed

    def all_deficiency_eligible(self, client_type: str = "") -> list[str]:
        """All canonical field names with deficiency_allowed=True for client_type."""
        return [f for f in self._entries if self.deficiency_allowed(f, client_type)]

    def all_locked(self, client_type: str = "") -> list[str]:
        """All canonical field names that are fixed_administrative for client_type."""
        return [
            f for f in self._entries
            if (a := self.get(f, client_type)) and a.is_locked
        ]

    def all_fields(self) -> list[str]:
        """All canonical field names with any SFC entry."""
        return list(self._entries.keys())

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        eligible = len(self.all_deficiency_eligible())
        locked   = len(self.all_locked())
        return (
            f"FieldAuthorityTable("
            f"fields={len(self)}, deficiency_eligible={eligible}, locked={locked})"
        )


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def _resolve_effective_class(entry: dict, client_type: str) -> str:
    """
    Resolve effective class for a slug entry, applying client_type_override.

    Override values may be:
      - a plain string: "fixed_administrative"  (current YAML format)
      - a dict:         {"class": "fixed_administrative"}  (future extended format)
    """
    base      = entry.get("class", "deficiency_eligible")
    overrides = entry.get("client_type_overrides") or {}
    if client_type and client_type in overrides:
        val = overrides[client_type]
        if isinstance(val, dict):
            return val.get("class", base)
        if isinstance(val, str):
            return val
    return base


def _compile_authority(
    field_name:   str,
    slug:         str,
    entry:        dict,
    tier_key:     str,
    client_type:  str = "",
) -> FieldAuthority:
    """Compile one SFC entry into a FieldAuthority for a given client_type."""
    eff_class = _resolve_effective_class(entry, client_type)

    # explicit deficiency_allowed: false always overrides class
    if entry.get("deficiency_allowed") is False:
        eff_class = "fixed_administrative"

    deficiency_allowed = eff_class == "deficiency_eligible"
    is_locked          = eff_class == "fixed_administrative"
    is_informational   = eff_class == "informational_only"

    sop_section = entry.get("sop_section_override") or _slug_to_section(slug)
    severity    = _coerce_severity(entry.get("severity_default"), tier_key)
    sw_raw      = entry.get("sampling_weight")
    sampling_w  = float(sw_raw) if sw_raw is not None else None

    return FieldAuthority(
        field_name         = field_name,
        deficiency_allowed = deficiency_allowed,
        is_locked          = is_locked,
        is_informational   = is_informational,
        sampling_weight    = sampling_w,
        sop_section        = sop_section,
        severity           = severity,
        client_type        = client_type,
    )


def _build_authority_table(
    sfc_path:   Path | None = None,
    tiers_path: Path | None = None,
) -> FieldAuthorityTable:
    """
    Compile sop_field_classes.yaml + field_tiers.yaml into a FieldAuthorityTable.

    Compiler-internal. Called only from sop_compiler.compile_sop().
    Use compiled_sop(sop_id) (SOPGraph) for all production authority queries.

    Conflict resolution (multiple slugs → same canonical_field):
        The more restrictive class wins (informational > fixed_admin > deficiency_eligible).
        A warning is logged when a conflict is detected.
    """
    _sfc   = sfc_path   or _SFC_PATH
    _tiers = tiers_path or _TIERS_PATH

    if not _sfc.exists():
        logger.warning("field_authority: sop_field_classes.yaml not found at %s", _sfc)
        return FieldAuthorityTable({})

    with open(_sfc, encoding="utf-8") as f:
        sfc_raw = yaml.safe_load(f) or {}

    # Tier lookup for severity defaults
    tier_meta: dict[str, str] = {}
    if _tiers.exists():
        with open(_tiers, encoding="utf-8") as f:
            tiers = yaml.safe_load(f) or {}
        for tier_key in ("tier1", "tier2"):
            for e in (tiers.get(tier_key) or []):
                if isinstance(e, dict) and "field" in e:
                    tier_meta[e["field"]] = tier_key

    # Collect all client_type_override keys across all entries (to pre-compile)
    all_client_types: set[str] = set()
    for slug, entry in (sfc_raw.get("fields") or {}).items():
        for ct in (entry.get("client_type_overrides") or {}):
            all_client_types.add(ct)

    # Compile: {canonical_field: {client_type: FieldAuthority}}
    # accumulator tracks best (most restrictive) authority seen per (field, client_type)
    _best_rank:   dict[tuple[str, str], int]             = {}
    _best_auth:   dict[tuple[str, str], FieldAuthority]  = {}

    for slug, entry in (sfc_raw.get("fields") or {}).items():
        canonical = entry.get("canonical_field")
        if not canonical:
            continue
        tier_key = tier_meta.get(canonical, "tier2")

        for client_type in ({"", *all_client_types}):
            auth = _compile_authority(canonical, slug, entry, tier_key, client_type)
            key  = (canonical, client_type)

            new_rank = _CLASS_RANK.get(
                "informational_only" if auth.is_informational else
                "fixed_administrative" if auth.is_locked else
                "deficiency_eligible",
                0,
            )
            existing_rank = _best_rank.get(key, -1)

            if new_rank > existing_rank:
                if existing_rank >= 0:
                    logger.warning(
                        "field_authority: conflict for field '%s' client_type=%r — "
                        "slug '%s' class=%s overrides previous (more restrictive wins)",
                        canonical, client_type or "(base)", slug,
                        "informational_only" if auth.is_informational else
                        "fixed_administrative" if auth.is_locked else "deficiency_eligible",
                    )
                _best_rank[key] = new_rank
                _best_auth[key] = auth

    # Restructure into {canonical: {client_type: FieldAuthority}}
    entries: dict[str, dict[str, FieldAuthority]] = {}
    for (canonical, client_type), auth in _best_auth.items():
        if canonical not in entries:
            entries[canonical] = {}
        entries[canonical][client_type] = auth

    table = FieldAuthorityTable(entries)
    logger.debug(
        "field_authority: compiled — %d fields, %d deficiency_eligible, %d locked",
        len(table), len(table.all_deficiency_eligible()), len(table.all_locked()),
    )
    return table
