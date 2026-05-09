"""Set-Level Precision, Recall, F1, R-Precision (Crucible Fix 4 / AV-1).

The architectural-horizon mandate forbids bounded F1@K because it cannot
detect the V6 graph flood: a 372-node CIS scored only on its top-10 looks
identical to a focused 50-node CIS, masking the precision collapse the
ablation study is designed to measure.

Primary metric: ``f1_set``. Computed against the full unpruned validated
CIS (``cis.all_node_ids()``). No top-K truncation.

Supplementary rank-aware metric: ``r_precision``. Defined as the precision
in the top-|gt| of a ranked list — gives one rank-sensitive diagnostic for
descriptive table purposes without compromising the unbounded primary
metric.

Phase 3.2 invariant (preserved): metrics MUST be computed against the CIS
node set, NOT against ``report.impacted_nodes``. After Crucible Fix 3
(Full Demotion of LLM #5), ``ImpactReport.impacted_nodes`` IS the full
CIS, so the two are equivalent — but the contract continues to require
CIS-level evaluation for clarity.

Reference: 10_evaluation_protocol.md §3 (revised under Crucible Plan).
"""

from __future__ import annotations

from typing import Iterable


def compute_set_metrics(
    predicted: set[str] | Iterable[str],
    gt: set[str] | Iterable[str],
) -> dict[str, float]:
    """Compute Set-Level Precision, Recall, F1.

    Args:
        predicted: full unpruned set of predicted-impacted node IDs (from
            ``cis.all_node_ids()``). Order does not matter.
        gt: ground-truth set of impacted node IDs.

    Returns:
        ``{precision_set, recall_set, f1_set, n_predicted, n_gt, n_intersect}``

    Edge cases:
      - Empty predicted -> precision_set = 0.0, recall_set = 0.0, f1_set = 0.0
      - Empty gt        -> recall_set = float('nan') (caller MUST exclude
                           the CR from the paired Wilcoxon test).
      - Both empty      -> all metrics 0.0 (degenerate; CR should be filtered).
    """
    predicted = set(predicted)
    gt = set(gt)

    n_predicted = len(predicted)
    n_gt = len(gt)
    n_intersect = len(predicted & gt)

    if n_predicted == 0:
        precision_set = 0.0
    else:
        precision_set = n_intersect / n_predicted

    if n_gt == 0:
        recall_set = float("nan")
    else:
        recall_set = n_intersect / n_gt

    if precision_set + recall_set <= 0 or recall_set != recall_set:
        # NaN guard: recall_set != recall_set is true only for NaN.
        f1_set = float("nan") if recall_set != recall_set else 0.0
    else:
        f1_set = 2 * precision_set * recall_set / (precision_set + recall_set)

    return {
        "precision_set": precision_set,
        "recall_set": recall_set,
        "f1_set": f1_set,
        "n_predicted": float(n_predicted),
        "n_gt": float(n_gt),
        "n_intersect": float(n_intersect),
    }


def compute_r_precision(
    ranked: list[str],
    gt: set[str] | Iterable[str],
) -> float:
    """R-Precision: precision at rank K = |gt|.

    A single rank-aware diagnostic retained for descriptive comparison
    with prior CIA literature. NOT a hypothesis-test target.

    Args:
        ranked: ordered list of predicted node IDs (highest priority first).
        gt: ground-truth set.

    Returns:
        ``|top-|gt| of ranked ∩ gt| / |gt|``, or 0.0 if gt is empty.
    """
    gt = set(gt)
    if not gt:
        return 0.0
    k = len(gt)
    top_k = ranked[:k]
    return sum(1 for nid in top_k if nid in gt) / k


def compute_dual_granularity_metrics(
    predicted_nodes: Iterable[str],
    predicted_files: Iterable[str],
    gt_nodes: Iterable[str],
    gt_files: Iterable[str],
) -> dict[str, float]:
    """Compute set metrics at both entity (node) and file granularity.

    Used by the ablation harness which requires file-level (routing
    accuracy) and entity-level (AST precision) figures separately.

    Returns flat dict with prefixed keys: ``entity_*`` and ``file_*``.
    """
    entity = compute_set_metrics(predicted_nodes, gt_nodes)
    files = compute_set_metrics(predicted_files, gt_files)
    return {
        **{f"entity_{k}": v for k, v in entity.items()},
        **{f"file_{k}": v for k, v in files.items()},
    }
