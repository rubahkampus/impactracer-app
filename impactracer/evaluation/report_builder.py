"""Aggregated summary artifact generation.

Outputs:
  - summary_table.csv       - macro-averaged metrics per variant
  - per_category_table.csv  - metrics per variant per CR category
  - statistical_test.csv     - Wilcoxon signed-rank test comparing V7 against V5
  - latency_distribution.png - box plot of per-CR latency per variant

Reference: 10_evaluation_protocol.md §6.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_summary_artifacts(
    df: pd.DataFrame,
    stat_rows: list[dict],
    output_dir: Path,
) -> None:
    """Emit all post-evaluation artifacts to ``output_dir``."""
    raise NotImplementedError("Sprint 11")
