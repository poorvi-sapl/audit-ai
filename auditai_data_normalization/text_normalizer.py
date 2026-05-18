"""
auditai_data_normalization/text_normalizer.py
==============================================
Phase 1 — Pre-extraction normalization layer.

Runs BEFORE any field extraction logic. Cleans raw text coming out of
docx_extractor and pdf_text_extractor so that downstream regex and alias
resolution see consistent, parseable strings.

Pipeline
--------
    raw cell/line text
        → normalize_text()
            1. Wingding / Unicode checkbox  →  "true" / "false"
            2. Underscore OCR artifact      →  cleaned date/value
            3. Unicode whitespace           →  plain space
            4. Whitespace collapse          →  single spaces / clean newlines
            5. Inline engagement markers    →  structured value string
        → _extract_fields_from_record()  (normalize.py)

Why a separate module
---------------------
Keeping normalization isolated means:
- Every extractor (docx, pdf, ocr) can import one function
- Changes to normalization rules don't scatter across 4 files
- Unit-testable in isolation, independent of DocumentRecord

Public API
----------
    normalize_text(text: str) -> str
        Full normalization pipeline. Use this everywhere.

    normalize_checkbox(text: str) -> str
        Checkbox symbols only. Exposed for unit testing.

    normalize_underscores(text: str) -> str
        OCR underscore artifact cleanup only.

    normalize_inline_markers(text: str) -> str
        Inline (X) / (x) engagement decision markers only.

    normalize_unicode_whitespace(text: str) -> str
        Unicode space variants → ASCII space only.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# 1. Checkbox normalization
# ---------------------------------------------------------------------------
#
# Sources of checkbox symbols in audit workpapers:
#   - Wingdings font:  \uf0fe (checked box), \uf0a3 (bullet square), \uf061
#   - Unicode:         ☑ ☒ ✓ ✔  (checked)
#                      ☐ □ ○    (unchecked)
#   - PDF extraction:  ■ ▪ (filled square used as checked indicator)
#   - Word content controls: sometimes render as literal "☒" or "☑"
#
# Strategy: normalize ALL checked variants → "true", unchecked → "false".
# This makes column-position tracking in Phase 3 trivial (compare "true"/"false").

_CHECKED_CHARS: frozenset[str] = frozenset({
    # Unicode checkmarks and checked boxes
    "\u2713",   # ✓  CHECK MARK
    "\u2714",   # ✔  HEAVY CHECK MARK
    "\u2611",   # ☑  BALLOT BOX WITH CHECK
    "\u2612",   # ☒  BALLOT BOX WITH X
    # Filled squares (used as checked in many audit templates)
    "\u25a0",   # ■  BLACK SQUARE
    "\u25aa",   # ▪  BLACK SMALL SQUARE
    "\u25cf",   # ●  BLACK CIRCLE
    # Wingdings font codepoints (survive PDF/DOCX extraction as private-use chars)
    "\uf0fe",   # Wingdings checked box  (most common in PPC forms)
    "\uf061",   # Wingdings tick
    "\uf0a3",   # Wingdings filled square
    "\uf052",   # Wingdings checkmark variant
    # Letter X used as checkbox fill
    "\u2717",   # ✗  BALLOT X
    "\u2718",   # ✘  HEAVY BALLOT X
})

_UNCHECKED_CHARS: frozenset[str] = frozenset({
    "\u2610",   # ☐  BALLOT BOX (empty)
    "\u25a1",   # □  WHITE SQUARE
    "\u25cb",   # ○  WHITE CIRCLE
    "\u25ef",   # ◯  LARGE CIRCLE
    "\uf0a1",   # Wingdings empty box
})

# Pre-built translation table for O(1) per-character replacement
_CHECKBOX_TABLE = str.maketrans(
    {c: "true"  for c in _CHECKED_CHARS} |
    {c: "false" for c in _UNCHECKED_CHARS}
)


def normalize_checkbox(text: str) -> str:
    """
    Replace checkbox Unicode/Wingding symbols with 'true' or 'false'.

    Called first in the pipeline so subsequent steps see plain ASCII.

    Examples
    --------
    "☒ Accept  ☐ Decline"   →  "true Accept  false Decline"
    "\\uf0fe Single Audit"   →  "true Single Audit"
    "☑"                     →  "true"
    """
    if not text:
        return text
    return text.translate(_CHECKBOX_TABLE)


# ---------------------------------------------------------------------------
# 2. Underscore OCR artifact cleanup
# ---------------------------------------------------------------------------
#
# OCR frequently renders blank form fields (underline-filled) as runs of
# underscores interspersed with content:
#
#   "_08_/ _07_/ _2025__"   →  "08/ 07/ 2025"
#   "John___Smith"          →  "JohnSmith"   (then whitespace collapse handles space)
#   "___"                   →  ""            (pure underscore → empty)
#
# Strategy:
#   Step A — strip leading/trailing underscores around digits and slashes (dates)
#   Step B — remove remaining isolated underscore runs (≥2) that aren't
#             part of a legitimate identifier (snake_case field names are
#             already resolved before this runs — they live in the YAML keys,
#             not in the extracted values)

# Matches underscores wrapping digits/slashes — date artifacts like _08_/_07_/_2025_
_UNDERSCORE_DATE_RE = re.compile(r"_+([\d/ ]+)_*|_*([\d/ ]+)_+")

# Matches runs of 2+ underscores that are NOT between word characters
# (preserves snake_case in field names — but values shouldn't have those)
_UNDERSCORE_RUN_RE = re.compile(r"(?<!\w)_{2,}(?!\w)")


def normalize_underscores(text: str) -> str:
    """
    Remove OCR underscore artifacts from extracted text.

    Examples
    --------
    "_08_/ _07_/ _2025__"  →  "08/ 07/ 2025"
    "___"                  →  ""
    "John___Smith"         →  "John Smith"   (after whitespace collapse)
    """
    if not text or "_" not in text:
        return text

    # Step A — unwrap underscores around date-like content
    text = _UNDERSCORE_DATE_RE.sub(
        lambda m: (m.group(1) or m.group(2) or "").strip(),
        text,
    )

    # Step B — remove remaining isolated underscore runs
    text = _UNDERSCORE_RUN_RE.sub(" ", text)

    return text


# ---------------------------------------------------------------------------
# 3. Unicode whitespace normalization
# ---------------------------------------------------------------------------
#
# Beyond the 3 chars already handled in normalize.py (\u2002, \u200b, \u00a0),
# audit PDFs and DOCX files contain a range of Unicode space variants:
#
#   \u2002  EN SPACE          (already handled — kept for completeness)
#   \u2003  EM SPACE
#   \u2004  THREE-PER-EM SPACE
#   \u2005  FOUR-PER-EM SPACE
#   \u2009  THIN SPACE
#   \u00a0  NO-BREAK SPACE    (already handled)
#   \u200b  ZERO WIDTH SPACE  (already handled)
#   \u200c  ZERO WIDTH NON-JOINER
#   \u200d  ZERO WIDTH JOINER
#   \u2060  WORD JOINER
#   \u00ad  SOFT HYPHEN
#   \ufeff  BOM / ZERO WIDTH NO-BREAK SPACE
#
# All → plain ASCII space (or empty for zero-width).

_UNICODE_SPACE_TABLE = str.maketrans({
    "\u2002": " ",   # EN SPACE
    "\u2003": " ",   # EM SPACE
    "\u2004": " ",   # THREE-PER-EM SPACE
    "\u2005": " ",   # FOUR-PER-EM SPACE
    "\u2009": " ",   # THIN SPACE
    "\u00a0": " ",   # NO-BREAK SPACE
    "\u200b": "",    # ZERO WIDTH SPACE     → remove
    "\u200c": "",    # ZERO WIDTH NON-JOINER → remove
    "\u200d": "",    # ZERO WIDTH JOINER    → remove
    "\u2060": "",    # WORD JOINER          → remove
    "\u00ad": "",    # SOFT HYPHEN          → remove
    "\ufeff": "",    # BOM                  → remove
})


def normalize_unicode_whitespace(text: str) -> str:
    """
    Replace Unicode space variants with plain ASCII space or empty string.

    Extends the 3-char swap already in normalize.py to cover the full
    range of Unicode whitespace seen in audit PDFs and DOCX files.
    """
    if not text:
        return text
    return text.translate(_UNICODE_SPACE_TABLE)


# ---------------------------------------------------------------------------
# 4. Whitespace collapse
# ---------------------------------------------------------------------------

_MULTI_SPACE_RE  = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _collapse_whitespace(text: str) -> str:
    """
    Collapse runs of spaces/tabs → single space.
    Collapse 3+ newlines → two newlines (preserve paragraph breaks).
    Strip leading/trailing whitespace.
    """
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 5. Inline engagement marker normalization
# ---------------------------------------------------------------------------
#
# Audit engagement forms frequently use inline prose markers instead of
# checkbox controls:
#
#   "Accept (X)  Decline ( )"
#   "(x) Continue engagement"
#   "We should (X) / should not ( ) accept this engagement"
#   "Accepted: (X)  Rejected: ( )"
#
# These need to be converted to structured label: value pairs that
# _extract_fields_from_record() can resolve via alias lookup.
#
# Output convention:
#   Detected accepted/continue marker → emit "engagement_decision: Accept"
#   Detected decline/discontinue      → emit "engagement_decision: Decline"
#   These are appended to the text as new lines so they're picked up by
#   the section scanner in normalize.py.

# Patterns where (X) or (x) directly precedes or follows a decision keyword
# Both orderings required (per the optimization spec, item 8)
_MARKER_CHECKED   = r"\(\s*[xX✓✔]\s*\)"
_MARKER_UNCHECKED = r"\(\s*\)"

# Decision vocabulary — maps regex group name → canonical value
_ACCEPT_WORDS    = r"(?:Accept|Continue|Accepted|Go|Proceed)"
_DECLINE_WORDS   = r"(?:Decline|Discontinue|Reject|No.Go|Do Not Accept|Should Not Accept)"

# Forward: (X) Accept  or  (X) Continue
_INLINE_ACCEPT_FWD  = re.compile(
    rf"{_MARKER_CHECKED}\s*{_ACCEPT_WORDS}", re.IGNORECASE
)
_INLINE_DECLINE_FWD = re.compile(
    rf"{_MARKER_CHECKED}\s*{_DECLINE_WORDS}", re.IGNORECASE
)

# Backward: Accept (X)  or  Continue (X)
_INLINE_ACCEPT_BWD  = re.compile(
    rf"{_ACCEPT_WORDS}\s*{_MARKER_CHECKED}", re.IGNORECASE
)
_INLINE_DECLINE_BWD = re.compile(
    rf"{_DECLINE_WORDS}\s*{_MARKER_CHECKED}", re.IGNORECASE
)

# GAGAS / Single Audit inline markers:
# "(X) Government Auditing Standards"  or  "Yellow Book (X)"
_GAGAS_WORDS  = r"(?:Government Auditing Standards|Yellow Book|GAGAS|GAS Audit)"
_SA_WORDS     = r"(?:Single Audit|Uniform Guidance|2 CFR 200|A.133|Federal Program Audit)"

_INLINE_GAGAS_FWD  = re.compile(rf"{_MARKER_CHECKED}\s*{_GAGAS_WORDS}",  re.IGNORECASE)
_INLINE_GAGAS_BWD  = re.compile(rf"{_GAGAS_WORDS}\s*{_MARKER_CHECKED}",  re.IGNORECASE)
_INLINE_SA_FWD     = re.compile(rf"{_MARKER_CHECKED}\s*{_SA_WORDS}",     re.IGNORECASE)
_INLINE_SA_BWD     = re.compile(rf"{_SA_WORDS}\s*{_MARKER_CHECKED}",     re.IGNORECASE)


def normalize_inline_markers(text: str) -> str:
    """
    Detect inline (X) engagement decision markers and append structured
    label: value lines that _extract_fields_from_record() can parse.

    Original text is preserved — structured lines are APPENDED so the
    raw content remains available for LLM fallback.

    Examples
    --------
    "Accept (X)  Decline ( )"
        → original + "\\nengagement_decision: Accept"

    "(x) Government Auditing Standards  ( ) Not Applicable"
        → original + "\\nincludes_gagas: true"
    """
    if not text:
        return text

    appended: list[str] = []

    # engagement_decision
    if _INLINE_ACCEPT_FWD.search(text) or _INLINE_ACCEPT_BWD.search(text):
        appended.append("engagement_decision: Accept")
    elif _INLINE_DECLINE_FWD.search(text) or _INLINE_DECLINE_BWD.search(text):
        appended.append("engagement_decision: Decline")

    # includes_gagas
    if _INLINE_GAGAS_FWD.search(text) or _INLINE_GAGAS_BWD.search(text):
        appended.append("includes_gagas: true")

    # includes_single_audit
    if _INLINE_SA_FWD.search(text) or _INLINE_SA_BWD.search(text):
        appended.append("includes_single_audit: true")

    if appended:
        return text + "\n" + "\n".join(appended)
    return text


# ---------------------------------------------------------------------------
# Public API — full pipeline
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Full pre-extraction normalization pipeline.

    Run this on every cell value and line of text BEFORE field extraction.
    Steps run in dependency order: checkbox first (produces 'true'/'false'
    ASCII), then underscore cleanup, then unicode whitespace, then collapse,
    then inline marker detection.

    Parameters
    ----------
    text : str
        Raw text from a table cell, paragraph, or PDF line.

    Returns
    -------
    str
        Normalized text, ready for regex field extraction.

    Examples
    --------
    "\\uf0fe  Accept  \\uf0a1  Decline"
        → "true  Accept  false  Decline"
        → + appended "engagement_decision: Accept"

    "_08_/ _07_/ _2025__"
        → "08/ 07/ 2025"

    "Engagement Partner:\\u2002John Smith"
        → "Engagement Partner: John Smith"

    "Accept (X)  Decline ( )"
        → "Accept (X)  Decline ( )\\nengagement_decision: Accept"
    """
    if not text:
        return text

    # Step 1 — checkbox symbols → "true" / "false"
    text = normalize_checkbox(text)

    # Step 2 — OCR underscore artifacts
    text = normalize_underscores(text)

    # Step 3 — Unicode whitespace variants → ASCII
    text = normalize_unicode_whitespace(text)

    # Step 4 — collapse runs of spaces/newlines
    text = _collapse_whitespace(text)

    # Step 5 — inline (X) engagement markers → structured label lines
    text = normalize_inline_markers(text)

    return text