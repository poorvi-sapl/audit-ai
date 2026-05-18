"""
auditai_data_normalization/extractors/structural_extractor.py
==============================================================
Phase 4 — Structural field extraction for financial statement PDFs.

A layered extraction system that uses semantic anchors to locate document
segments, then applies probabilistic structure inference within each
segment. Every extracted value carries explicit evidence metadata so
downstream systems can decide accept / review / reject without guessing.

Architecture
------------
Layer 1 — Segment detection (_find_segments)
    Locates the primary auditor report section and financial statement
    section by anchor scanning. Filters TOC and supplement occurrences.
    Returns page ranges — not fixed offsets.

Layer 2 — Role-based field inference (_infer_fields)
    Within each segment, extracts fields by role (salutation entity,
    standalone date, prose flag) rather than positional rules.
    Each value receives a FieldEvidence with confidence score + method.

Layer 3 — Evidence gating (extract)
    Only emits fields meeting the minimum confidence threshold.
    Returns dict[str, FieldEvidence] — caller merges into pipeline.

Design constraints
------------------
- Never encodes "line N = field X". Positions are probabilistic.
- engagement_partner is explicitly supported as null — registry-only.
- All-caps supplement blocks are filtered at segment detection time.
- City filter (City, State) confirms primary auditor report block vs
  supplement or TOC, validated across 6 diverse PDFs.
- Confidence scores are evidence-based, not tuned to pass a threshold.

Public API
----------
    extract(pages_text, pages_words) -> dict[str, FieldEvidence]
        pages_text  : list[str]  — extract_text() per page, 0-indexed
        pages_words : list[list] — extract_words() per page, 0-indexed
        Returns FieldEvidence per extracted field. Missing fields absent.

    FieldEvidence (dataclass)
        value        : str
        confidence   : float   [0.0, 1.0]
        source_page  : int     1-based
        method       : str     one of METHOD_* constants
        anchor       : str     the anchor that located this field

Caller integration (normalize.py)
----------------------------------
    Only called when document_category == 'financial_statement'.
    Results merged into fields_for_scoring slot B if slot B is empty.
    extraction_method gains 'structural_heuristic' tag.
    Never overwrites deterministic extractor values.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Evidence methods — what rule produced this value
# ---------------------------------------------------------------------------

METHOD_SALUTATION_BLOCK  = "salutation_block"   # entity after auditor salutation
METHOD_STANDALONE_DATE   = "standalone_date"    # isolated date line on cover page
METHOD_SIGNOFF_DATE      = "signoff_date"       # date in auditor report close block
METHOD_PROSE_FLAG        = "prose_flag"         # boolean field from prose keyword
METHOD_REGISTRY          = "registry"           # known-firm registry lookup

# ---------------------------------------------------------------------------
# FieldEvidence — carries value + how we got it
# ---------------------------------------------------------------------------

@dataclass
class FieldEvidence:
    """
    A single extracted field with full evidence chain.

    Attributes
    ----------
    value       : str   — the extracted value, normalized
    confidence  : float — [0.0, 1.0], evidence-based not threshold-tuned
    source_page : int   — 1-based page number where value was found
    method      : str   — which METHOD_* rule produced this
    anchor      : str   — the anchor text that located the segment
    """
    value:       str
    confidence:  float
    source_page: int
    method:      str
    anchor:      str = ""

    def __repr__(self) -> str:
        return (
            f"FieldEvidence({self.value!r} conf={self.confidence:.2f} "
            f"p{self.source_page} method={self.method})"
        )


# ---------------------------------------------------------------------------
# Minimum confidence to emit a field
# ---------------------------------------------------------------------------

_MIN_CONFIDENCE = 0.55   # below this → don't emit, let LLM fallback handle it


# ---------------------------------------------------------------------------
# Layer 1 — Segment detection
# ---------------------------------------------------------------------------

# Salutation patterns across audit report types
_SALUTATION_RE = re.compile(
    r"^(To the |The )(Board of Directors|Members|Governing Board|Trustees|Partners)",
    re.IGNORECASE,
)

# City, State — confirms primary report block vs supplement/TOC
# Matches "City Name, California" or "City Name, New York" etc.
_CITY_STATE_RE = re.compile(
    r"^[A-Za-z][A-Za-z\s\-]+,\s+[A-Za-z][A-Za-z\s]+$"
)

# Entity names that are actually supplement headers (all-caps, long)
_SUPPLEMENT_HEADER_RE = re.compile(r"^[A-Z\s,\.\(\)]{30,}$")

# Dot-leader pattern (TOC entries)
_DOT_LEADER_RE = re.compile(r"\.{5,}")

# Anchors for segment detection
_AUDITOR_REPORT_ANCHOR = "INDEPENDENT AUDITOR"
_FINANCIAL_STMT_ANCHOR = "FINANCIAL STATEMENTS"
_NOTES_ANCHOR          = "NOTES TO"


@dataclass
class _Segment:
    """One detected document segment."""
    name:       str   # "auditor_report" | "financial_statements" | "notes"
    start_page: int   # 1-based
    end_page:   int   # 1-based, inclusive — best-effort
    anchor:     str   # exact anchor text that fired


def _find_segments(pages_text: list[str]) -> list[_Segment]:
    """
    Locate primary document segments by anchor scanning.

    Filters:
    1. Dot-leader lines → TOC entry, skip.
    2. No salutation within 6 lines → not a primary auditor report, skip.
    3. No city line after entity → supplement block, skip.

    Returns list of _Segment in page order, earliest occurrence of each
    segment type only (primary report, not GAGAS supplements).
    """
    segments: list[_Segment] = []
    found_types: set[str] = set()

    for page_idx, text in enumerate(pages_text):
        page_num = page_idx + 1
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        for j, line in enumerate(lines):

            # ── Auditor report segment ─────────────────────────────────
            if _AUDITOR_REPORT_ANCHOR in line.upper() and "auditor_report" not in found_types:
                next6 = lines[j + 1: j + 7]

                # Filter 1: dot-leaders → TOC
                if any(_DOT_LEADER_RE.search(nl) for nl in next6):
                    continue

                # Filter 2: salutation must be present
                sal_idx = next(
                    (k for k, nl in enumerate(next6) if _SALUTATION_RE.match(nl)),
                    None,
                )
                if sal_idx is None:
                    continue

                # Filter 3: line after salutation must look like an entity name
                entity_line = next6[sal_idx + 1] if sal_idx + 1 < len(next6) else ""
                if _SUPPLEMENT_HEADER_RE.match(entity_line):
                    continue

                # Filter 4: city line must follow entity (confirms primary block)
                city_line = next6[sal_idx + 2] if sal_idx + 2 < len(next6) else ""
                if not _CITY_STATE_RE.match(city_line):
                    continue

                # Estimate end page: auditor report typically spans 2-3 pages
                end_page = min(page_num + 3, len(pages_text))
                segments.append(_Segment(
                    name="auditor_report",
                    start_page=page_num,
                    end_page=end_page,
                    anchor=line,
                ))
                found_types.add("auditor_report")

            # ── Financial statements segment ───────────────────────────
            elif _FINANCIAL_STMT_ANCHOR in line.upper() and "financial_statements" not in found_types:
                # Must be a standalone heading line (short) not embedded in prose
                if len(line) < 60:
                    segments.append(_Segment(
                        name="financial_statements",
                        start_page=page_num,
                        end_page=min(page_num + 5, len(pages_text)),
                        anchor=line,
                    ))
                    found_types.add("financial_statements")

            # ── Notes segment ──────────────────────────────────────────
            elif _NOTES_ANCHOR in line.upper() and "notes" not in found_types:
                segments.append(_Segment(
                    name="notes",
                    start_page=page_num,
                    end_page=len(pages_text),
                    anchor=line,
                ))
                found_types.add("notes")

    logger.debug(
        "structural_extractor: segments found: %s",
        [(s.name, s.start_page) for s in segments],
    )
    return segments


# ---------------------------------------------------------------------------
# Layer 2 — Role-based field inference
# ---------------------------------------------------------------------------

# Date pattern — matches full written dates and common audit date formats
_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September"
    r"|October|November|December)\s+\d{1,2},\s+\d{4}\b"
    r"|\b(?:June|December|September|March|June)\s+30,?\s*\d{4}\b"
    r"|\bDecember\s+31,?\s*\d{4}\b"
    r"|\bSeptember\s+30,?\s*\d{4}\b",
    re.IGNORECASE,
)

# All-caps standalone date (cover page fiscal year end)
_ALLCAPS_DATE_RE = re.compile(
    r"^(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER"
    r"|OCTOBER|NOVEMBER|DECEMBER)\s+\d{1,2},\s+\d{4}$"
)

# Prose flags — boolean Tier 1 fields detectable from narrative text
_PROSE_FLAGS: dict[str, list[str]] = {
    "includes_gagas": [
        "government auditing standards",
        "yellow book",
        "gagas",
        "gao standards",
    ],
    "includes_single_audit": [
        "uniform guidance",
        "single audit",
        "2 cfr",
        "a-133",
        "federal awards",
        "major federal program",
    ],
    "includes_gaas_audit": [
        "generally accepted auditing standards",
        "gaas",
        "aicpa",
        "au-c",
        "public company accounting",
    ],
    "reporting_framework": [
        "accounting principles generally accepted in the united states",
        "u.s. generally accepted accounting principles",
        "gaap",
        "gasb",
        "special purpose framework",
        "cash basis",
        "tax basis",
    ],
}

# Confidence assigned per prose flag hit count
def _prose_confidence(hit_count: int) -> float:
    if hit_count >= 3: return 0.90
    if hit_count == 2: return 0.80
    if hit_count == 1: return 0.70
    return 0.0


def _infer_from_auditor_report(
    segment: _Segment,
    pages_text: list[str],
) -> dict[str, FieldEvidence]:
    """
    Extract fields from the primary auditor report segment.

    Fields targeted:
        client_name      — entity name in salutation block
        client_address   — city/state in salutation block
        partner_sign_date — date in report close block (last 6 lines)
        includes_gagas, includes_single_audit, includes_gaas_audit,
        reporting_framework — prose flag detection
    """
    results: dict[str, FieldEvidence] = {}

    # Gather all text within segment
    segment_lines_by_page: list[tuple[int, list[str]]] = []
    for page_num in range(segment.start_page, segment.end_page + 1):
        idx = page_num - 1
        if idx >= len(pages_text):
            break
        lines = [l.strip() for l in pages_text[idx].splitlines() if l.strip()]
        segment_lines_by_page.append((page_num, lines))

    # ── client_name + client_address from salutation block ───────────────
    for page_num, lines in segment_lines_by_page:
        for j, line in enumerate(lines):
            if _AUDITOR_REPORT_ANCHOR not in line.upper():
                continue
            next6 = lines[j + 1: j + 7]
            if any(_DOT_LEADER_RE.search(nl) for nl in next6):
                continue
            sal_idx = next(
                (k for k, nl in enumerate(next6) if _SALUTATION_RE.match(nl)),
                None,
            )
            if sal_idx is None:
                continue
            entity = next6[sal_idx + 1] if sal_idx + 1 < len(next6) else ""
            city   = next6[sal_idx + 2] if sal_idx + 2 < len(next6) else ""

            if entity and not _SUPPLEMENT_HEADER_RE.match(entity):
                results["client_name"] = FieldEvidence(
                    value=entity,
                    confidence=0.85,
                    source_page=page_num,
                    method=METHOD_SALUTATION_BLOCK,
                    anchor=line,
                )
            if city and _CITY_STATE_RE.match(city):
                results["client_address"] = FieldEvidence(
                    value=city,
                    confidence=0.80,
                    source_page=page_num,
                    method=METHOD_SALUTATION_BLOCK,
                    anchor=line,
                )
            break
        if "client_name" in results:
            break

    # ── partner_sign_date from report close block ─────────────────────────
    # The sign date is in the last 6 lines of one of the auditor report pages.
    # It's a short line containing a written date, preceding the page number.
    for page_num, lines in segment_lines_by_page:
        for line in lines[-6:]:
            m = _DATE_RE.search(line)
            if m and len(line) < 40:
                # Confidence: shorter line = less likely to be prose sentence
                conf = 0.85 if len(line) < 25 else 0.72
                results["partner_sign_date"] = FieldEvidence(
                    value=m.group(0).strip(),
                    confidence=conf,
                    source_page=page_num,
                    method=METHOD_SIGNOFF_DATE,
                    anchor="auditor_report_close",
                )
                break
        if "partner_sign_date" in results:
            break

    # ── Prose flags ───────────────────────────────────────────────────────
    # Concatenate all segment text for keyword scanning
    segment_text = " ".join(
        " ".join(lines) for _, lines in segment_lines_by_page
    ).lower()

    for field, keywords in _PROSE_FLAGS.items():
        hits = sum(1 for kw in keywords if kw in segment_text)
        conf = _prose_confidence(hits)
        if conf >= _MIN_CONFIDENCE:
            # For boolean fields: value is "true"
            # For reporting_framework: infer the framework name
            if field == "reporting_framework":
                if "gasb" in segment_text:
                    value = "GASB"
                elif "special purpose" in segment_text or "cash basis" in segment_text:
                    value = "Special Purpose Framework"
                else:
                    value = "GAAP"
            else:
                value = "true"

            results[field] = FieldEvidence(
                value=value,
                confidence=conf,
                source_page=segment.start_page,
                method=METHOD_PROSE_FLAG,
                anchor=f"prose_flag:{field}",
            )

    return results


def _infer_fiscal_year_end(pages_text: list[str]) -> FieldEvidence | None:
    """
    Extract fiscal_year_end from cover page standalone date lines.

    Strategy: scan first 3 pages for standalone date lines. Score by:
    - All-caps date → highest confidence (cover page standard format)
    - Short line containing only a date → high confidence
    - Long prose line containing a date → lower confidence

    Returns the best-scoring candidate or None.
    """
    candidates: list[tuple[float, str, int]] = []  # (score, value, page)

    for page_idx in range(min(3, len(pages_text))):
        page_num = page_idx + 1
        lines = [l.strip() for l in pages_text[page_idx].splitlines() if l.strip()]

        for line in lines:
            m = _DATE_RE.search(line)
            if not m:
                continue

            date_val = m.group(0).strip()

            # Score this candidate
            if _ALLCAPS_DATE_RE.match(line):
                # All-caps standalone date — definitive cover page fiscal year
                score = 0.92
            elif len(line) < 30 and line.strip() == date_val:
                # Line is nothing but the date
                score = 0.85
            elif len(line) < 40:
                # Short line containing a date (e.g. "JUNE 30, 2024" with entity name)
                score = 0.75
            else:
                # Date embedded in prose — lower confidence
                score = 0.55

            candidates.append((score, date_val, page_num))

    if not candidates:
        return None

    # Pick highest-scoring candidate (stable sort: first occurrence wins ties)
    best_score, best_val, best_page = max(candidates, key=lambda c: c[0])

    if best_score < _MIN_CONFIDENCE:
        return None

    return FieldEvidence(
        value=best_val,
        confidence=best_score,
        source_page=best_page,
        method=METHOD_STANDALONE_DATE,
        anchor="cover_page_date",
    )


# ---------------------------------------------------------------------------
# Layer 3 — Evidence gating + public API
# ---------------------------------------------------------------------------

def extract(
    pages_text:  list[str],
    pages_words: list[list] | None = None,
) -> dict[str, FieldEvidence]:
    """
    Extract fields from a financial statement PDF using structural heuristics.

    Parameters
    ----------
    pages_text : list[str]
        Raw text per page from pdfplumber page.extract_text().
        Index 0 = page 1. Empty string for scanned/empty pages.
    pages_words : list[list] | None
        Word dicts per page from pdfplumber page.extract_words().
        Reserved for future bbox-based extraction (Phase 4 extension).
        Pass None to skip — all current extraction uses text layer only.

    Returns
    -------
    dict[str, FieldEvidence]
        Keys are canonical field names. Only fields meeting _MIN_CONFIDENCE
        are included. Missing fields are absent — not null, not 0.0.
        Caller is responsible for merging with other extractor results.

    Notes
    -----
    engagement_partner is intentionally not extracted here.
    The firm signature is an image in all observed PDFs.
    Use firm_registry.yaml lookup (separate concern) if needed.
    """
    if not pages_text:
        return {}

    results: dict[str, FieldEvidence] = {}

    # Layer 1 — segment detection
    segments = _find_segments(pages_text)

    if not segments:
        logger.debug("structural_extractor: no segments found — returning empty")
        return {}

    # Layer 2 — field inference per segment
    for segment in segments:
        if segment.name == "auditor_report":
            segment_results = _infer_from_auditor_report(segment, pages_text)
            # Only write fields not already found by an earlier segment
            for field, evidence in segment_results.items():
                if field not in results:
                    results[field] = evidence

    # fiscal_year_end is inferred from cover page, not a specific segment
    fy_evidence = _infer_fiscal_year_end(pages_text)
    if fy_evidence and "fiscal_year_end" not in results:
        results["fiscal_year_end"] = fy_evidence

    # Layer 3 — gate: drop anything below minimum confidence
    gated = {
        field: ev
        for field, ev in results.items()
        if ev.confidence >= _MIN_CONFIDENCE
    }

    logger.info(
        "structural_extractor: %d fields extracted (%d gated out) — %s",
        len(gated),
        len(results) - len(gated),
        {f: f"{ev.confidence:.2f}" for f, ev in gated.items()},
    )

    return gated