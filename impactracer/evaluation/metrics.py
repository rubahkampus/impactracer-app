"""Precision@K, Recall@K, F1@K computation.

MRR is NOT reported (thesis III.7.4 remediation).

Phase 3.2 (A-NEW-2/A-4): metrics MUST be computed against the full CIS
node set, NOT against report.impacted_nodes.

The ImpactReport.impacted_nodes list is produced by LLM #5, which makes
an uncontrolled selection from the context window. Measuring F1 against
LLM #5's selection conflates pipeline CIA capability with LLM #5 summarisation
quality — these are orthogonal. The canonical evaluation target is:

    ranked = list of node_ids from cis.all_node_ids(), ordered by:
      1. SIS seeds first (depth=0), sorted by retrieval score desc
      2. Propagated nodes after seeds, sorted by severity then depth

    gt = set of ground-truth impacted node_ids from the annotation file.

The ``compute_metrics`` caller (ablation.py) is responsible for passing
the correctly ordered CIS-derived ranked list.  Passing report.impacted_nodes
instead is a methodological error that will be caught by unit tests.

Reference: 10_evaluation_protocol.md §3.
"""

from __future__ import annotations


def compute_metrics(
    ranked: list[str],
    gt: set[str],
    k_values: list[int],
) -> dict[str, float]:
    """Compute P@K, R@K, F1@K for each K in ``k_values``.

    Args:
        ranked: Ordered list of node_ids from the full CIS (NOT from
            report.impacted_nodes). Order determines which nodes are
            "top-K" — higher-priority nodes first.
        gt: Ground-truth set of impacted node_ids for this CR.
        k_values: List of K thresholds (e.g. [5, 10]).

    Returns:
        Flat dict with keys like ``precision_at_10``, ``recall_at_10``,
        ``f1_at_10``, ``precision_at_5`` etc.

    Edge cases:
      - Empty ``ranked``  -> all metrics 0.0
      - Empty ``gt``      -> recall undefined; caller should exclude the CR
                            from the paired Wilcoxon test.
      - K > len(ranked)   -> top-K is capped at len(ranked).

    Phase 3.2 invariant: this function does NOT accept ImpactReport objects.
    The caller must extract node_ids from CISResult, not from ImpactReport.
    """
    raise NotImplementedError("Sprint 11")
