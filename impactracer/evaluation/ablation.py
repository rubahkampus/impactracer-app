"""Ablation harness: run V0..V7 over a GT dataset.

Reference: 09_ablation_harness.md.
"""

from __future__ import annotations

from pathlib import Path

from impactracer.shared.config import Settings


def run_single_cr_all_variants(
    cr_id: str,
    cr_text: str,
    gt_node_ids: set[str],
    settings: Settings,
    output_dir: Path,
) -> dict[str, dict]:
    """Execute all 8 variants on one CR.

    Returns a dict mapping variant_id to per-variant metrics and output paths.
    """
    raise NotImplementedError("Sprint 11")


def run_full_evaluation(
    cr_dataset: list[dict],
    settings: Settings,
    output_dir: Path,
) -> Path:
    """Execute the full ablation + metric aggregation.

    Writes:
      - per_cr_per_variant_metrics.csv
      - per_cr_per_variant_metrics.jsonl
      - <cr_id>/<variant>.json          (per-variant ImpactReport)
      - llm_audit.jsonl                 (NFR-05 audit trail)

    Returns the path to the CSV.
    """
    raise NotImplementedError("Sprint 11")
