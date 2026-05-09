"""Single pre-registered Wilcoxon test: V7 vs V5 on Total F1 (set-level).

Crucible Fix 4 (AV-1): primary metric changed from F1@10 to ``f1_set``.
F1@10 is structurally unable to distinguish a graph flood (V6's 372-node
CIS) from a focused result (V4's 8-node CIS). Total F1 (set-level) is
unbounded and therefore correctly attributes precision collapse to V6
and precision recovery to V7.

PROTOCOL (pre-registered):
  - Test:    one-sided paired Wilcoxon signed-rank.
  - Pair:    one CR with valid f1_set under both V5 and V7.
  - Metric:  ``f1_set`` (set-level F1 against ground-truth entity set).
  - N_min:   15 complete pairs. Below this, decline to report p-value.
  - Alpha:   0.05.
  - Effect:  Cliff's δ as non-parametric effect-size companion.
  - No multiple-comparison correction (one and only one test is run).
  - Incomplete-pair protocol: rows missing either V5 or V7 f1_set are
    excluded from the test. Substituting 0 or filling-forward is
    forbidden.

Reference: 10_evaluation_protocol.md §4 (revised under Crucible Plan).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ALPHA: float = 0.05
PRIMARY_COMPARISON: tuple[str, str] = ("V7", "V5")
PRIMARY_METRIC: str = "f1_set"
MIN_PAIRED_N: int = 15


class InsufficientPairsError(RuntimeError):
    """Raised when fewer than MIN_PAIRED_N complete pairs are available."""


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's δ effect size: ``(#(a > b) - #(a < b)) / (n_a * n_b)``.

    Range: [-1, 1]. Positive δ means group ``a`` tends to score higher.
    Returns 0.0 when both arrays are empty (degenerate).
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.size == 0 or b_arr.size == 0:
        return 0.0
    greater = 0
    less = 0
    for x in a_arr:
        greater += int(np.sum(x > b_arr))
        less += int(np.sum(x < b_arr))
    return (greater - less) / (a_arr.size * b_arr.size)


def pairwise_wilcoxon(
    df: pd.DataFrame,
    var_a: str,
    var_b: str,
    metric: str = PRIMARY_METRIC,
) -> dict[str, float | int]:
    """One-sided paired Wilcoxon signed-rank test on ``metric``.

    Expected DataFrame layout: rows = CRs, columns include
    ``f"{variant_id}_{metric}"`` (e.g. ``"V5_f1_set"``, ``"V7_f1_set"``).

    Args:
        df: per-CR results table.
        var_a, var_b: variant IDs (e.g. "V7", "V5"). Test is "var_a > var_b".
        metric: metric name suffix (default 'f1_set').

    Returns dict with: p_value, statistic, n, cliffs_delta, median_diff.

    Raises InsufficientPairsError if fewer than MIN_PAIRED_N complete pairs.
    """
    col_a = f"{var_a}_{metric}"
    col_b = f"{var_b}_{metric}"
    if col_a not in df.columns or col_b not in df.columns:
        raise KeyError(f"Missing columns: need {col_a} and {col_b}")

    pairs = df[[col_a, col_b]].dropna()
    pairs = pairs[
        ~pairs[col_a].apply(lambda x: isinstance(x, float) and math.isnan(x))
        & ~pairs[col_b].apply(lambda x: isinstance(x, float) and math.isnan(x))
    ]
    n = len(pairs)
    if n < MIN_PAIRED_N:
        raise InsufficientPairsError(
            f"Only {n} complete pairs for {var_a} vs {var_b} on {metric}; "
            f"need >= {MIN_PAIRED_N}. Reporting suppressed per protocol."
        )

    a = pairs[col_a].to_numpy(dtype=float)
    b = pairs[col_b].to_numpy(dtype=float)
    diff = a - b

    # One-sided alternative: var_a is greater than var_b.
    res = wilcoxon(diff, alternative="greater", zero_method="wilcox", correction=False)
    statistic = float(res.statistic)
    p_value = float(res.pvalue)

    median_diff = float(np.median(diff))
    delta = cliffs_delta(a, b)

    return {
        "p_value": p_value,
        "statistic": statistic,
        "n": n,
        "cliffs_delta": delta,
        "median_diff": median_diff,
    }


def run_primary_test(df: pd.DataFrame) -> dict[str, float | int | str | bool]:
    """Execute the single pre-registered V7 vs V5 test on f1_set.

    Returns a dict with keys: hypothesis, variant_a, variant_b, metric,
    p_value, statistic, cliffs_delta, median_diff, n, accepted,
    achieved_power_note.

    Raises InsufficientPairsError if n < MIN_PAIRED_N.
    """
    var_a, var_b = PRIMARY_COMPARISON
    res = pairwise_wilcoxon(df, var_a, var_b, metric=PRIMARY_METRIC)
    accepted = res["p_value"] < ALPHA
    return {
        "hypothesis": (
            f"{var_a}.{PRIMARY_METRIC} > {var_b}.{PRIMARY_METRIC} "
            f"(one-sided paired Wilcoxon)"
        ),
        "variant_a": var_a,
        "variant_b": var_b,
        "metric": PRIMARY_METRIC,
        "p_value": res["p_value"],
        "statistic": res["statistic"],
        "cliffs_delta": res["cliffs_delta"],
        "median_diff": res["median_diff"],
        "n": res["n"],
        "accepted": accepted,
        "achieved_power_note": (
            f"N={res['n']} complete pairs at alpha={ALPHA}; "
            "pre-registered moderate-power design (~0.60 at delta=0.2)."
        ),
    }
