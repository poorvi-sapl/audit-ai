"""
raw_to_training_pair/generation/task_balance_reporter.py
==========================================================
Read a training-corpus JSONL file and report the distribution of
pair_type and task across all pairs. Warns when any single
pair_type/task share exceeds a configurable threshold.

Why this exists
---------------
Project context rule: "no task_type > 40% of total pairs". A
multi-task fine-tune corpus (review + generation + Q&A + validation
pairs) needs enforced balance or the model overfits the dominant
task. This reporter is the dashboard / CI signal for that rule.

It does NOT modify the JSONL — it only inspects. Use this:
    - After every batch run, to monitor balance drift
    - As a pre-training gate (fail CI if balance violation)
    - As an ad-hoc audit ("what does the corpus look like today?")

Public API
----------
    TaskBalanceReport       — dataclass of counts, shares, warnings
    report_task_balance(jsonl_path, max_share=0.40, taxonomy="pair_type")
        → TaskBalanceReport
    pretty_print(report)    → str
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

Taxonomy = Literal["pair_type", "task"]


@dataclass
class TaskBalanceReport:
    """Distribution of pairs across a chosen taxonomy."""
    jsonl_path: str
    taxonomy: Taxonomy
    total_pairs: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    shares: dict[str, float] = field(default_factory=dict)  # 0.0–1.0
    warnings: list[str] = field(default_factory=list)
    max_share_threshold: float = 0.40

    @property
    def is_balanced(self) -> bool:
        return not self.warnings


def report_task_balance(
    jsonl_path: str | Path,
    max_share: float = 0.40,
    taxonomy: Taxonomy = "pair_type",
) -> TaskBalanceReport:
    """Scan a JSONL training-corpus file and report the distribution.

    Args:
        jsonl_path: path to the JSONL file (one pair per line)
        max_share: threshold above which a single category is flagged
            as imbalanced. Default 0.40 per project_context rule.
        taxonomy: which metadata key to bucket by — "pair_type"
            (review / generation / etc.) or "task" (GENERATE_WORKPAPER
            / REVIEW_WORKPAPER / etc.)

    Returns:
        TaskBalanceReport with counts, share fractions, and any
        threshold-violation warnings.

    Raises:
        FileNotFoundError if jsonl_path is missing.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(
            f"report_task_balance: {path} does not exist"
        )

    report = TaskBalanceReport(
        jsonl_path=str(path),
        taxonomy=taxonomy,
        max_share_threshold=max_share,
    )

    counts: dict[str, int] = {}
    malformed_lines = 0

    with open(path) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                pair = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                logger.warning(
                    "report_task_balance: line %d of %s is not valid JSON",
                    line_num, path.name,
                )
                continue
            meta = pair.get("metadata") or {}
            key = str(meta.get(taxonomy, "(missing)"))
            counts[key] = counts.get(key, 0) + 1
            report.total_pairs += 1

    report.counts = dict(sorted(counts.items()))

    if report.total_pairs == 0:
        if malformed_lines:
            report.warnings.append(
                f"{malformed_lines} malformed line(s) in {path.name}; "
                "no valid pairs found"
            )
        return report

    # Compute shares + flag violations
    for key, count in report.counts.items():
        share = count / report.total_pairs
        report.shares[key] = share
        if share > max_share:
            report.warnings.append(
                f"{taxonomy}={key!r} has {count} pairs "
                f"({share:.1%}) which exceeds the {max_share:.0%} "
                f"max_share threshold."
            )

    if malformed_lines:
        report.warnings.append(
            f"{malformed_lines} malformed line(s) skipped during scan"
        )

    return report


def pretty_print(report: TaskBalanceReport) -> str:
    """Human-readable summary of a TaskBalanceReport."""
    lines: list[str] = [
        f"Task balance — {report.jsonl_path}",
        f"  Total pairs: {report.total_pairs}",
        f"  Taxonomy:    {report.taxonomy}",
        f"  Max share:   {report.max_share_threshold:.0%}",
        f"  Distribution:",
    ]
    if not report.counts:
        lines.append("    (no pairs)")
    else:
        max_key_len = max(len(k) for k in report.counts.keys())
        for key in sorted(
            report.counts.keys(), key=lambda k: -report.counts[k],
        ):
            count = report.counts[key]
            share = report.shares.get(key, 0.0)
            flag = "  ⚠️" if share > report.max_share_threshold else ""
            lines.append(
                f"    {key.ljust(max_key_len)}  "
                f"{count:>6}  ({share:.1%}){flag}"
            )
    if report.warnings:
        lines.append("  Warnings:")
        for w in report.warnings:
            lines.append(f"    - {w}")
    elif report.total_pairs > 0:
        lines.append("  ✓ Balanced — no threshold violations.")
    return "\n".join(lines)
