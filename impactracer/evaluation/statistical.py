"""Single pre-registered Wilcoxon test: V7 vs V5 on F1@10.

Per thesis Bab III.7.5, the formal statistical test is restricted to one
paired comparison. V0..V4, V6, and V6.5 are retained for descriptive
ablation plotting but are not part of the hypothesis test. No
multiple-comparison correction is applied because exactly one test is
performed.

Phase 3.4 Protocol (A-6/A-5):

PAIRED-TEST PROTOCOL
--------------------
1. The unit of pairing is a single CR. For each CR, both V5 and V7 must
   produce a valid F1@10 score. If EITHER run fails (infrastructure error,
   API timeout, empty CIS), that CR is EXCLUDED from the paired test.
   Substituting 0 or filling-forward is forbidden.

2. Minimum paired N: ``MIN_PAIRED_N = 15``. If fewer than 15 complete
   pairs are available after exclusion, ``run_primary_test`` raises
   ``InsufficientPairsError`` and declines to report a p-value. The
   thesis must report the achieved N and acknowledge the limitation.

3. EXPECTED POWER ANALYSIS (pre-registered):
   - N=20 pairs (target), one-sided α=0.05
   - Expected effect size δ ≈ 0.2 (small-to-medium, based on literature
     comparing retrieval-only vs retrieval+propagation CIA systems)
   - Expected power ≈ 0.60 under the Wilcoxon signed-rank test
   - This is acknowledged as moderate power. The thesis reports the
     power estimate and acknowledges it as a limitation of N=20.
   - If achieved N < 20 due to exclusions, power may drop to < 0.50.

4. METRICS TARGET: F1@K is computed against ``cis.all_node_ids()`` order
   (Phase 3.2 / A-NEW-2), NOT against ``report.impacted_nodes``.
   See metrics.py for the invariant.

5. INCOMPLETE-PAIR HANDLING:
   - Collect results for all 20 CRs × all variants in a DataFrame.
   - Filter to rows where BOTH V5 and V7 f1_at_10 are non-null and non-NaN.
   - Rows failing this filter are logged with reason (timeout/error/empty-CIS).
   - The paired test runs on the filtered set only.

Reference: 10_evaluation_protocol.md §4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ALPHA: float = 0.05
PRIMARY_COMPARISON: tuple[str, str] = ("V7", "V5")
MIN_PAIRED_N: int = 15  # below this, decline to report p-value (A-6)


class InsufficientPairsError(RuntimeError):
    """Raised when fewer than MIN_PAIRED_N complete pairs are available."""


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Non-parametric effect size in [-1, 1].

    Cliff's δ = (# pairs where a > b) - (# pairs where a < b) / n²
    Positive δ means a tends to be larger than b.
    """
    raise NotImplementedError("Sprint 11")


def pairwise_wilcoxon(
    df: pd.DataFrame,
    var_a: str,
    var_b: str,
) -> dict[str, float | int]:
    """One-sided paired Wilcoxon signed-rank test on F1@10.

    Phase 3.4: respects the incomplete-pair protocol — only complete pairs
    (both var_a and var_b non-null) are included.

    Returns a dict with p_value, statistic, n, cliffs_delta, median_diff.

    Raises InsufficientPairsError if complete n < MIN_PAIRED_N.
    """
    raise NotImplementedError("Sprint 11")


def run_primary_test(df: pd.DataFrame) -> dict[str, float | int | str | bool]:
    """Execute the single pre-registered V7 vs V5 test.

    Phase 3.4 protocol:
    - Filters to complete pairs (V5 and V7 both have valid f1_at_10).
    - Raises InsufficientPairsError if n < MIN_PAIRED_N.
    - Reports achieved N alongside p-value and effect size.

    Returns a dict with keys:
        hypothesis, variant_a, variant_b, p_value, statistic,
        cliffs_delta, median_diff, n, accepted, achieved_power_note.
    """
    raise NotImplementedError("Sprint 11")