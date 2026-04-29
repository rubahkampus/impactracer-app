"""Precision@K, Recall@K, F1@K computation.

MRR is NOT reported (thesis III.7.4 remediation).

Reference: 10_evaluation_protocol.md §3.
"""

from __future__ import annotations


def compute_metrics(
    ranked: list[str],
    gt: set[str],
    k_values: list[int],
) -> dict[str, float]:
    """Compute P@K, R@K, F1@K for each K in ``k_values``.

    Returns a flat dict with keys like ``precision_at_10``, ``recall_at_10``,
    ``f1_at_10``, ``precision_at_5`` etc.

    Edge cases:
      - Empty ``ranked`` -> all metrics 0.0
      - Empty ``gt``     -> recall undefined; caller should exclude the CR
    """
    raise NotImplementedError("Sprint 11")
