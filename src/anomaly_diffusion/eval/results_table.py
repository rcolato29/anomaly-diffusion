"""Aggregate per-run metrics into a Markdown results table.

Assembles the per-category comparison across MVTec (this method vs. PatchCore, PaDiM, CAE).
"""

from __future__ import annotations

import json
from pathlib import Path


def _fmt(v) -> str:
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:.3f}"  # v != v is True only for NaN
    return str(v)


def to_markdown(rows: dict[str, dict], columns: list[str], index_name: str = "method") -> str:
    """Render rows (label -> metrics dict) as a Markdown table over columns."""
    header = f"| {index_name} | " + " | ".join(columns) + " |"
    sep = "|" + "---|" * (len(columns) + 1)
    lines = [header, sep]
    for label, metrics in rows.items():
        cells = " | ".join(_fmt(metrics.get(c)) for c in columns)
        lines.append(f"| {label} | {cells} |")
    return "\n".join(lines)


def collect_json_results(results_dir: str | Path) -> dict[str, dict]:
    """Load every *.json metrics file in a directory keyed by filename stem."""
    out = {}
    for p in sorted(Path(results_dir).glob("*.json")):
        out[p.stem] = json.loads(p.read_text())
    return out
