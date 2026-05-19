"""
raw_to_training_pair/citation_resolver.py
==========================================
Citation normalization and hierarchy resolution for audit finding risk text.

Three rules applied in order
------------------------------
1. Hierarchy rule
   Within the same document, keep only the most specific reference.
   "GAS Chapter 3" + "GAS §3.36" → "GAS §3.36"  (§3.36 is inside Chapter 3)
   "AU-C 220" + "AU-C 220.26"   → "AU-C 220.26" (paragraph is inside section)

2. Dedup rule
   Normalize all citations to canonical display strings, then eliminate
   exact duplicates (case-insensitive after normalization).

3. Layer rule
   Re-expand results in canonical order:
       AU-C (AICPA auditing) → ET (AICPA ethics) → GAS → 2 CFR → other
   Within each layer, sort by section number.

Canonical abbreviations used in output
----------------------------------------
   "AU-C NNN"        — AICPA Auditing Standards Codification
   "ET N.NNN.NNN"    — AICPA Code of Professional Conduct
   "GAS §N.NN"       — Government Auditing Standards (Yellow Book)
   "2 CFR 200.NNN"   — Uniform Guidance

Public API
----------
    resolve(citations) -> list[str]
        Input:  raw strings from field_standards_context.yaml additive lists
        Output: normalized, hierarchy-collapsed list in canonical order
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum


# ---------------------------------------------------------------------------
# Layers — canonical display order
# ---------------------------------------------------------------------------

class _Layer(IntEnum):
    AICPA_UC = 1   # AU-C
    AICPA_ET = 2   # ET
    GAS      = 3   # Government Auditing Standards
    CFR      = 4   # 2 CFR (Uniform Guidance)
    OTHER    = 9


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_AUC_RE      = re.compile(r"^au-c\s+(\d+)(?:\.(\d+))?$", re.I)
_ET_RE       = re.compile(r"^et\s+([\d.]+)$", re.I)
_GAS_CHAP_RE = re.compile(r"government auditing standards[^§\n]*?chapter\s+(\d+)", re.I)
_GAS_SECT_RE = re.compile(r"government auditing standards\s*§\s*(\d+)[.\-](\d+)", re.I)
_CFR_RE      = re.compile(r"^2\s+cfr\s+200\.(\d+)([\w()\-]*)?$", re.I)


# ---------------------------------------------------------------------------
# Internal parsed citation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Cit:
    raw:        str
    layer:      _Layer
    sort_key:   tuple          # for ordering within layer
    hier_group: str            # citations sharing a group compete under hierarchy rule
    hier_depth: int            # higher = more specific; only max depth survives per group
    display:    str            # canonical output string


def _parse(raw: str) -> _Cit:
    """Parse one raw citation string into a _Cit."""
    s = raw.strip()

    # --- AU-C ---
    m = _AUC_RE.match(s)
    if m:
        sect = int(m.group(1))
        para = int(m.group(2)) if m.group(2) else None
        depth = 3 if para else 2
        display = f"AU-C {sect}" + (f".{para:02d}" if para else "")
        return _Cit(
            raw=raw, layer=_Layer.AICPA_UC,
            sort_key=(sect, para or 0),
            hier_group=f"auc_{sect}",
            hier_depth=depth,
            display=display,
        )

    # --- ET ---
    m = _ET_RE.match(s)
    if m:
        parts = [p for p in m.group(1).split(".") if p]
        depth = len(parts) + 1
        display = f"ET {m.group(1)}"
        sort_key = tuple(int(p) for p in parts if p.isdigit())
        hier_group = f"et_{'_'.join(parts[:2])}"
        return _Cit(
            raw=raw, layer=_Layer.AICPA_ET,
            sort_key=sort_key,
            hier_group=hier_group,
            hier_depth=depth,
            display=display,
        )

    # --- GAS section: §N.NN (check before chapter so §-containing strings go here) ---
    m = _GAS_SECT_RE.search(s)
    if m:
        chap = int(m.group(1))
        sect = int(m.group(2))
        return _Cit(
            raw=raw, layer=_Layer.GAS,
            sort_key=(chap, sect),
            hier_group=f"gas_{chap}",
            hier_depth=2,
            display=f"GAS §{chap}.{sect:02d}",
        )

    # --- GAS chapter: "(Chapter N)" with no § ---
    m = _GAS_CHAP_RE.search(s)
    if m and "§" not in s:
        chap = int(m.group(1))
        return _Cit(
            raw=raw, layer=_Layer.GAS,
            sort_key=(chap, 0),
            hier_group=f"gas_{chap}",
            hier_depth=1,
            display=f"GAS Chapter {chap}",
        )

    # --- 2 CFR 200.NNN ---
    m = _CFR_RE.match(s)
    if m:
        sect    = int(m.group(1))
        suffix  = (m.group(2) or "").strip()
        depth   = 3 if suffix else 2
        display = f"2 CFR 200.{sect}" + (suffix if suffix else "")
        return _Cit(
            raw=raw, layer=_Layer.CFR,
            sort_key=(sect,),
            hier_group=f"cfr_{sect}",
            hier_depth=depth,
            display=display,
        )

    # --- Unrecognised — pass through unchanged ---
    return _Cit(
        raw=raw, layer=_Layer.OTHER,
        sort_key=(0,),
        hier_group=raw.lower().strip(),
        hier_depth=1,
        display=s,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve(citations: list[str]) -> list[str]:
    """
    Normalise, deduplicate, and hierarchy-collapse a list of citation strings.

    Returns citations in canonical order: AU-C → ET → GAS → 2 CFR → other.
    Within the same document section, retains only the most specific reference.

    Examples
    --------
    >>> resolve(["Government Auditing Standards (Chapter 3)",
    ...          "Government Auditing Standards §3.36"])
    ['GAS §3.36']

    >>> resolve(["AU-C 220", "ET 1.295",
    ...          "Government Auditing Standards (Chapter 3)",
    ...          "Government Auditing Standards §3.36"])
    ['AU-C 220', 'ET 1.295', 'GAS §3.36']

    >>> resolve(["2 CFR 200.514", "2 CFR 200.516"])
    ['2 CFR 200.514', '2 CFR 200.516']
    """
    if not citations:
        return []

    parsed = [_parse(c) for c in citations]

    # Rule 1: Hierarchy — per hier_group, keep only max-depth citation(s)
    groups: dict[str, list[_Cit]] = defaultdict(list)
    for cit in parsed:
        groups[cit.hier_group].append(cit)

    after_hierarchy: list[_Cit] = []
    for group_cits in groups.values():
        max_depth = max(c.hier_depth for c in group_cits)
        for c in group_cits:
            if c.hier_depth == max_depth:
                after_hierarchy.append(c)

    # Rule 2: Dedup — by normalised display string
    seen:    set[str]   = set()
    deduped: list[_Cit] = []
    for c in after_hierarchy:
        key = c.display.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    # Rule 3: Layer order — AU-C → ET → GAS → 2 CFR → other; section-sorted within
    deduped.sort(key=lambda c: (int(c.layer), c.sort_key))

    return [c.display for c in deduped]
