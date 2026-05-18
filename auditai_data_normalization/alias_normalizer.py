"""
auditai_data_normalization/alias_normalizer.py
===============================================
Normalization layer that runs AFTER deterministic alias lookup fails
and BEFORE fuzzy matching or LLM suggestion.

Pipeline (per unknown label)
-----------------------------
    raw label
        ↓
    1. compound split        — "Prepared By / Date" → ["Prepared By", "Date"]
        ↓
    2. OCR correction        — per unit: "cl1ent" → "client", "Dáte" → "Date"
        ↓
    3. text normalization    — lowercase, strip punctuation, expand abbreviations,
                               collapse whitespace
        ↓
    list of normalized units → each passed to alias_fuzzy.py independently

Why this ordering
-----------------
Compound split first: OCR correction on smaller semantic units is cleaner.
"Prepared By / Dáte" split first → ["Prepared By", "Dáte"] → correct each
unit independently. Correcting the combined string risks touching delimiters.

Relationship to text_normalizer.py
------------------------------------
text_normalizer.py runs on raw document text BEFORE field extraction.
alias_normalizer.py runs on extracted label keys AFTER extraction, when
a label failed deterministic alias lookup. Different stage, different purpose.
Do not merge these two modules.

Public API
----------
    normalize_label(raw_label: str) -> list[str]
        Full pipeline. Returns 1..N normalized units ready for fuzzy matching.
        Always returns at least one unit (may be empty string if input is blank).

    split_compound(label: str) -> list[str]
        Compound detection and split only. Exposed for unit testing.

    correct_ocr(label: str) -> str
        OCR character correction only. Exposed for unit testing.

    expand_abbreviations(label: str) -> str
        Abbreviation expansion only. Exposed for unit testing.

    normalize_unit(label: str) -> str
        Steps 2+3 on a single unit (no compound split). Exposed for testing.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Compound label delimiters
# Ordered by specificity — more specific patterns checked first.
# Sourced from real workpaper extraction:
#   "1st Year / New Client (Initial)"
#   "AU-C 220 | SAS No. 146 | Yellow Book"
#   "✅  ACCEPT / CONTINUE"
#   "Answer / Instruction"
# ---------------------------------------------------------------------------

_COMPOUND_DELIMITERS = re.compile(
    r"""
    \s*\|\s*        # pipe  — "AU-C 220 | SAS No. 146"
    | \s*/\s*       # slash — "Prepared By / Date", "ACCEPT / CONTINUE"
    | \s+and\s+     # "and" — "Name and Title"
    | \s+&\s+       # ampersand — "Name & Date"
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Minimum character length for a split unit to be kept.
# Units shorter than this after normalization are discarded as fragments.
_MIN_UNIT_LENGTH = 3

# ---------------------------------------------------------------------------
# OCR correction dictionary
# Maps common OCR substitution errors to correct characters.
# Sourced from audit workpaper extraction artifacts.
# Extend here — never in fuzzy or normalizer logic.
# ---------------------------------------------------------------------------

_OCR_CORRECTIONS: dict[str, str] = {
    # NOTE: digit-for-letter substitutions (0→o, 1→l) are intentionally
    # NOT in this translation table. They are context-sensitive and handled
    # exclusively by _OCR_DIGIT_PATTERNS below to avoid corrupting numeric
    # codes like "AU-C 220" or "SAS No. 146".
    # Accented / diacritic characters from bad PDF encoding
    "á": "a", "à": "a", "â": "a", "ä": "a", "ã": "a",
    "é": "e", "è": "e", "ê": "e", "ë": "e",
    "í": "i", "ì": "i", "î": "i", "ï": "i",
    "ó": "o", "ò": "o", "ô": "o", "ö": "o", "õ": "o",
    "ú": "u", "ù": "u", "û": "u", "ü": "u",
    "ý": "y", "ÿ": "y",
    "ñ": "n",
    "ç": "c",
    # Smart quotes and dashes that survive PDF extraction
    "\u2018": "'", "\u2019": "'",   # left/right single quotes
    "\u201c": '"', "\u201d": '"',   # left/right double quotes
    "\u2013": "-", "\u2014": "-",   # en-dash, em-dash
    "\u2012": "-",                  # figure dash
    # Non-breaking and zero-width spaces (seen in real workpapers)
    "\u00a0": " ",   # non-breaking space
    "\u200b": "",    # zero-width space
    "\u200c": "",    # zero-width non-joiner
    "\u200d": "",    # zero-width joiner
    "\u2002": " ",   # en space (seen in "Date:\u2002" patterns)
    "\u2003": " ",   # em space
    "\ufeff": "",    # BOM
}

# Regex patterns for digit-in-word OCR errors.
# Applied after character-level corrections.
# IMPORTANT: only fires inside words that are entirely alphabetic except
# for the digit — never inside numeric codes like "AU-C 220" or "SAS 146".
# Pattern requires: letter before digit, letter after digit (or word end),
# AND the token must not be preceded by a space+digit (numeric context).
_OCR_DIGIT_PATTERNS: list[tuple[re.Pattern, str]] = [
    # "c1ient" → "client": 1 letter before, 3+ letters after (avoids "SAS 146")
    (re.compile(r"(?<=[a-zA-Z])1(?=[a-zA-Z]{3,})"), "l"),
    # "c0mpany" → "company": same guard
    (re.compile(r"(?<=[a-zA-Z])0(?=[a-zA-Z]{3,})"), "o"),
    # "financia1" → "financial": 6+ letters before, word boundary after
    (re.compile(r"(?<=[a-zA-Z]{6})1(?=\s|$|[^a-zA-Z0-9])"), "l"),
]

# ---------------------------------------------------------------------------
# Abbreviation expansion
# Maps surface-form abbreviations to their expanded canonical form.
# Expansion happens AFTER OCR correction, BEFORE lowercasing.
# All keys must be lowercase for case-insensitive matching.
# Sourced from real workpaper label variants.
# ---------------------------------------------------------------------------

_ABBREVIATIONS: dict[str, str] = {
    # Organization / entity
    "org":          "organization",
    "org.":         "organization",
    "dept":         "department",
    "dept.":        "department",
    "div":          "division",
    "div.":         "division",
    "co":           "company",
    "co.":          "company",
    "corp":         "corporation",
    "corp.":        "corporation",
    "inc":          "incorporated",
    "inc.":         "incorporated",
    "llc":          "llc",            # keep — meaningful in audit context
    "llp":          "llp",            # keep
    "npo":          "nonprofit organization",
    "nfp":          "not-for-profit",

    # Fiscal / date
    "fy":           "fiscal year",
    "ye":           "year end",
    "y/e":          "year end",
    "ytd":          "year to date",
    "py":           "prior year",
    "cy":           "current year",
    "qtr":          "quarter",
    "q1":           "quarter 1",
    "q2":           "quarter 2",
    "q3":           "quarter 3",
    "q4":           "quarter 4",

    # Engagement / people
    "ep":           "engagement partner",
    "e.p.":         "engagement partner",
    "mgr":          "manager",
    "mgr.":         "manager",
    "sr":           "senior",
    "sr.":          "senior",
    "jr":           "junior",
    "jr.":          "junior",
    "prepd":        "prepared",
    "prep":         "prepared",
    "rev":          "reviewed",
    "appr":         "approved",

    # Audit / standards
    "gaas":         "generally accepted auditing standards",
    "gagas":        "government auditing standards",
    "gaap":         "generally accepted accounting principles",
    "gasb":         "governmental accounting standards board",
    "fasb":         "financial accounting standards board",
    "aicpa":        "aicpa",          # keep — proper acronym
    "sas":          "statement on auditing standards",
    "ssae":         "statement on standards for attestation engagements",
    "ssars":        "statements on standards for accounting and review services",

    # Document structure
    "ref":          "reference",
    "refs":         "references",
    "doc":          "document",
    "docs":         "documents",
    "wp":           "workpaper",
    "w/p":          "workpaper",
    "acct":         "account",
    "accts":        "accounts",
    "bal":          "balance",
    "stmt":         "statement",
    "stmts":        "statements",
    "fin":          "financial",
    "fin.":         "financial",
    "amt":          "amount",
    "amts":         "amounts",
    "no":           "number",         # "Form No" → "Form Number" (standalone only, not in "SAS No. 146")
    "no.":          "number",         # same — guarded by whole-word match but skipped post-standards tokens
    "num":          "number",
    "num.":         "number",
    "yr":           "year",
    "yr.":          "year",
    "dt":           "date",
    "dt.":          "date",
    "addr":         "address",
    "addr.":        "address",
    "ein":          "ein",            # keep — proper acronym in audit context
    "id":           "id",             # keep
}

# Pre-compiled pattern for whole-word abbreviation matching.
# Only expands when the abbreviation appears as a complete token.
_ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_ABBREVIATIONS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Emoji / symbol stripping
# Seen in real workpapers: "✅  ACCEPT / CONTINUE", "❌  DO NOT ACCEPT"
# ---------------------------------------------------------------------------

def _strip_emoji(text: str) -> str:
    """Remove emoji and dingbat characters that survive DOCX extraction."""
    return "".join(
        ch for ch in text
        if not unicodedata.category(ch).startswith("So")   # Symbol, other
        and ch not in ("\u2705", "\u274c", "\u2714", "\u2716")  # ✅ ❌ ✔ ✖
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_label(raw_label: str) -> list[str]:
    """
    Full normalization pipeline for a single raw label.

    Returns a list of normalized units (1..N). Each unit is a clean,
    lowercase, whitespace-collapsed string ready for fuzzy matching.
    Units that are too short or empty after normalization are dropped.

    Parameters
    ----------
    raw_label : str
        The label as it appeared in the document, post-extraction.

    Returns
    -------
    list[str]
        One or more normalized units. Empty list only if input is blank.
    """
    if not raw_label or not raw_label.strip():
        return []

    # Step 1 — compound split
    units = split_compound(raw_label)

    # Steps 2+3 — OCR correction + text normalization per unit
    normalized = []
    for unit in units:
        cleaned = normalize_unit(unit)
        if len(cleaned) >= _MIN_UNIT_LENGTH:
            normalized.append(cleaned)

    return normalized if normalized else []


def split_compound(label: str) -> list[str]:
    """
    Detect and split compound labels on known delimiters.

    Examples
    --------
    "Prepared By / Date"          → ["Prepared By", "Date"]
    "AU-C 220 | SAS No. 146"      → ["AU-C 220", "SAS No. 146"]
    "Name and Title"              → ["Name", "Title"]
    "✅  ACCEPT / CONTINUE"       → ["ACCEPT", "CONTINUE"]
    "Engagement Partner"          → ["Engagement Partner"]  (no split)

    Parameters
    ----------
    label : str
        Raw label string, may contain compound delimiters.

    Returns
    -------
    list[str]
        List of sub-labels. Always at least one element.
    """
    # Strip emoji before splitting — avoids "✅" becoming a fragment
    label = _strip_emoji(label).strip()

    parts = _COMPOUND_DELIMITERS.split(label)
    return [p.strip() for p in parts if p and p.strip()]


def correct_ocr(label: str) -> str:
    """
    Apply OCR character-level corrections to a single label unit.

    Handles: diacritic characters, smart quotes, non-breaking spaces,
    zero-width characters, and digit-in-word substitutions.

    Parameters
    ----------
    label : str
        Single label unit (post compound split).

    Returns
    -------
    str
        Label with OCR artifacts corrected.
    """
    # Character-level substitution via translation table
    table = str.maketrans(_OCR_CORRECTIONS)
    corrected = label.translate(table)

    # Digit-in-word patterns (context-sensitive, applied after char corrections)
    # Skip entirely if label contains standalone numbers (standards/code references)
    import re as _re
    if not _re.search(r"\b\d{2,}\b", corrected):
        for pattern, replacement in _OCR_DIGIT_PATTERNS:
            corrected = pattern.sub(replacement, corrected)

    return corrected


def expand_abbreviations(label: str) -> str:
    """
    Expand known abbreviations to their full canonical form.

    Expansion is whole-word only — "FY" expands but "FYI" does not.
    Applied after OCR correction, before lowercasing.

    Guards
    ------
    - Tokens following a digit (e.g. "No." in "SAS No. 146") are not expanded
      to avoid corrupting standards references like "AU-C 220" or "SAS No. 146".

    Parameters
    ----------
    label : str
        Label unit, post OCR correction.

    Returns
    -------
    str
        Label with abbreviations expanded.
    """
    # Guard: if label looks like a standards/code reference (contains digit chunks),
    # skip abbreviation expansion entirely to avoid corrupting "SAS No. 146" etc.
    if re.search(r"\b\d{2,}\b", label):
        return label

    def _replace(match: re.Match) -> str:
        token = match.group(0)
        return _ABBREVIATIONS.get(token.lower(), token)

    return _ABBREV_PATTERN.sub(_replace, label)


def normalize_unit(label: str) -> str:
    """
    Steps 2 and 3 on a single unit: OCR correction + text normalization.
    Does NOT perform compound splitting.

    Steps
    -----
    1. OCR character correction
    2. Emoji/symbol strip
    3. Abbreviation expansion
    4. Lowercase
    5. Strip leading/trailing punctuation and whitespace
    6. Collapse internal whitespace to single space
    7. Remove trailing punctuation artifacts (:, ., ?)

    Parameters
    ----------
    label : str
        Single label unit (post compound split).

    Returns
    -------
    str
        Normalized lowercase string.
    """
    # OCR correction
    result = correct_ocr(label)

    # Emoji strip (in case unit came from split without emoji strip)
    result = _strip_emoji(result)

    # Abbreviation expansion (before lowercasing for case-insensitive match)
    result = expand_abbreviations(result)

    # Lowercase
    result = result.lower()

    # Strip leading/trailing whitespace and punctuation
    result = result.strip()
    result = result.strip(":.?!,;\"'()")

    # Collapse internal whitespace
    result = re.sub(r"\s+", " ", result)

    # Remove trailing colon artifacts common in label cells ("Organization:")
    result = result.rstrip(":")

    return result.strip()