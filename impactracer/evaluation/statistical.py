"""Single pre-registered Wilcoxon test: V7 vs V5 on F1@10.

    Per thesis Bab III.7.5, the formal statistical test is
    restricted to one paired comparison. V0..V4 and V6 are retained for
    descriptive ablation plotting but are not part of the hypothesis test.
    No multiple-comparison correction is applied because exactly one test
    is performed.

    Reference: 10_evaluation_protocol.md §4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


ALPHA: float = 0.05
PRIMARY_COMPARISON: tuple[str, str] = ("V7", "V5")


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Non-parametric effect size in [-1, 1]."""
    raise NotImplementedError("Sprint 11")


def pairwise_wilcoxon(
    df: pd.DataFrame,
    var_a: str,
    var_b: str,
) -> dict[str, float | int]:
    """One-sided paired Wilcoxon signed-rank test on F1@10.

    Returns a dict with p_value, statistic, n, cliffs_delta, median_diff.
    """
    raise NotImplementedError("Sprint 11")


def run_primary_test(df: pd.DataFrame) -> dict[str, float | int | str | bool]:
    """Execute the single pre-registered V7 vs V5 test.

    Returns a dict with keys:
        hypothesis, variant_a, variant_b, p_value, statistic,
        cliffs_delta, median_diff, n, accepted.
    """
    raise NotImplementedError("Sprint 11")