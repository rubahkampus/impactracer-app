"""Ablation harness: run V0..V7 over a GT dataset.

Scores ``cis.all_node_ids()`` (full unpruned validated CIS) against the
ground-truth entity and file sets using set-level metrics — no F1@K.

Each (CR, variant) row records both entity-level and file-level metrics.
The output CSV feeds ``statistical.run_primary_test`` for the V7 vs V5
Wilcoxon test on f1_set.

Amendment 2: the harness instantiates one ``VariantCache`` per
(cr_id, anchor_priming) tuple and passes it through to every variant.
Cumulative pipeline boundaries (LLM #1, retrieval, rerank+gates, LLM #2,
LLM #3, BFS+collapse, LLM #4) are then memoised so V_{n+1} resumes from
V_n's cached output instead of recomputing.

Reference: 09_ablation_harness.md; docs/evaluation_protocol.md Amendment 2.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from impactracer.evaluation.metrics import compute_dual_granularity_metrics
from impactracer.evaluation.schemas import GTEntry
from impactracer.evaluation.variant_flags import VariantFlags, with_anchor_priming
from impactracer.pipeline.variant_cache import VariantCache
from impactracer.shared.config import Settings


def _extract_predicted(report) -> tuple[set[str], set[str]]:
    """Pull (entity_node_ids, file_paths) from an ImpactReport.

    Uses the canonical GT-aligned schema:
      - entity ids come from ``impacted_entities[i].node``
      - file paths come from the deterministic ``impacted_files`` array
        (the brief: "Extract file_path strings from the predicted
        impacted_files and compare against GT impacted_files").

    Apex Crucible Proposal A.4: defensive filter against File-type leakage.
    GT entities always carry a `::` qualifier; bare file paths in the entity
    set are guaranteed FPs. The synthesizer should already filter these
    (Proposal A.1) but we double-up here so replays of older runs benefit
    too.
    """
    nodes = {
        e.node
        for e in report.impacted_entities
        if e.node and "::" in e.node and getattr(e, "node_type", "") != "File"
    }
    files = {f.file_path for f in report.impacted_files if f.file_path}
    return nodes, files


def run_single_cr_all_variants(
    cr_id: str,
    cr_text: str,
    gt_entry: GTEntry,
    settings: Settings,
    output_dir: Path,
    shared_embedder=None,
    shared_reranker=None,
    shared_llm_client=None,
    anchor_priming: bool = True,
    cache_root: Path | None = None,
    run_tag: str | None = None,
) -> dict[str, dict]:
    """Execute every variant in VariantFlags.ALL_VARIANTS on one CR.

    Writes ``<output_dir>/<cr_id>/<variant_id>/impact_report.json`` and
    ``impact_report_full.json`` (step-by-step trace) per variant. Returns
    a dict mapping variant_id -> per-variant metrics + timing + report path.

    Variants that crash (exception during run_analysis) are recorded with
    status='error' and metric values set to None — these CRs are excluded
    from the paired Wilcoxon test by ``statistical.pairwise_wilcoxon``.
    """
    from impactracer.pipeline.runner import run_analysis

    cr_root = output_dir / cr_id
    cr_root.mkdir(parents=True, exist_ok=True)

    gt_nodes = gt_entry.entity_node_ids()
    gt_files = gt_entry.file_paths()

    # Amendment 2: per-CR cache scoped by (run_tag, cr_id, anchor_priming).
    # All eight variants on this CR share one VariantCache; intermediate
    # outputs computed by V_n are reused by V_{n+1}.
    variant_cache: VariantCache | None = None
    if cache_root is not None and run_tag is not None:
        variant_cache = VariantCache(
            root_dir=cache_root,
            run_tag=run_tag,
            cr_id=cr_id,
            anchor_priming=anchor_priming,
        )

    results: dict[str, dict] = {}

    for variant_id in VariantFlags.ALL_VARIANTS:
        flags = with_anchor_priming(VariantFlags.for_id(variant_id), anchor_priming)
        variant_dir = cr_root / variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)
        trace_sink: dict = {"cr_text": cr_text, "variant": variant_id, "cr_id": cr_id}
        t0 = time.perf_counter()
        try:
            report = run_analysis(
                cr_text,
                settings,
                variant_flags=flags,
                shared_embedder=shared_embedder,
                shared_reranker=shared_reranker,
                shared_llm_client=shared_llm_client,
                trace_sink=trace_sink,
                variant_cache=variant_cache,
            )
        except Exception as exc:
            logger.error(
                "[ablation] {} on {} crashed: {}", variant_id, cr_id, exc
            )
            # Persist the trace_sink even on crash for forensic inspection.
            try:
                (variant_dir / "impact_report_full.json").write_text(
                    json.dumps(trace_sink, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
            except Exception:
                pass
            results[variant_id] = {
                "status": "error",
                "error": str(exc),
                "elapsed_s": time.perf_counter() - t0,
                "anchor_priming": flags.anchor_priming,
                "f1_set": None,
                "precision_set": None,
                "recall_set": None,
            }
            continue

        elapsed = time.perf_counter() - t0
        predicted_nodes, predicted_files = _extract_predicted(report)
        metrics = compute_dual_granularity_metrics(
            predicted_nodes=predicted_nodes,
            predicted_files=predicted_files,
            gt_nodes=gt_nodes,
            gt_files=gt_files,
        )

        # Persist the canonical report and the full step-by-step trace.
        report_path = variant_dir / "impact_report.json"
        report_path.write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        full_path = variant_dir / "impact_report_full.json"
        full_path.write_text(
            json.dumps(trace_sink, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        results[variant_id] = {
            "status": "ok",
            "elapsed_s": elapsed,
            "report_path": str(report_path),
            "n_impacted_nodes": len(predicted_nodes),
            "n_impacted_files": len(predicted_files),
            "degraded_run": report.degraded_run,
            "analysis_mode": report.analysis_mode,
            "estimated_scope": report.estimated_scope,
            "anchor_priming": flags.anchor_priming,
            **metrics,
            # Promote the primary metrics to top-level for stat-test ease.
            "f1_set": metrics["entity_f1_set"],
            "precision_set": metrics["entity_precision_set"],
            "recall_set": metrics["entity_recall_set"],
        }

        logger.info(
            "[ablation] {}/{}: n={}, P={:.3f}, R={:.3f}, F1={:.3f}, "
            "elapsed={:.1f}s, degraded={}",
            cr_id, variant_id,
            len(predicted_nodes),
            metrics["entity_precision_set"],
            metrics["entity_recall_set"],
            metrics["entity_f1_set"],
            elapsed,
            report.degraded_run,
        )

    return results


def run_full_evaluation(
    cr_dataset: list[GTEntry],
    settings: Settings,
    output_dir: Path,
    anchor_priming: bool = True,
    cache_root: Path | None = None,
    run_tag: str | None = None,
) -> Path:
    """Execute the full ablation × CR matrix.

    Writes:
      - ``<output_dir>/per_cr_per_variant_metrics.csv``
      - ``<output_dir>/per_cr_per_variant_metrics.jsonl``
      - ``<output_dir>/<cr_id>/<variant_id>/impact_report.json``
      - ``<output_dir>/<cr_id>/<variant_id>/impact_report_full.json``

    Returns the CSV path so callers can pivot for the Wilcoxon test.

    Performance: shares embedder, reranker, and llm_client across runs so
    BFS/LLM context loading happens once per process invocation, not per
    CR-variant cell.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Amendment 2: variant-chain cache root + run_tag. Defaults to the same
    # output directory under a `cache` subfolder, with a fresh UTC timestamp
    # tag so cache trees never collide across runs.
    if cache_root is None:
        cache_root = output_dir / "cache"
    if run_tag is None:
        run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cache_root = Path(cache_root)
    logger.info(
        "[ablation] Variant cache: root={} run_tag={} anchor_priming={}",
        cache_root, run_tag, anchor_priming,
    )

    # Load shared infrastructure ONCE.
    from impactracer.pipeline.runner import load_pipeline_context

    seed_ctx = load_pipeline_context(settings, variant_flags=None)
    shared_embedder = seed_ctx.embedder
    shared_reranker = seed_ctx.reranker
    shared_llm_client = seed_ctx.llm_client
    # Note: SQLite/Chroma connections are NOT shared across runs because
    # load_pipeline_context creates a fresh PipelineContext per call (the
    # graph is loaded once per context, but ablation reuses the *seeded*
    # embedder/reranker/llm_client only).

    csv_path = output_dir / "per_cr_per_variant_metrics.csv"
    jsonl_path = output_dir / "per_cr_per_variant_metrics.jsonl"

    fieldnames = [
        "cr_id", "variant",
        "status", "elapsed_s",
        "n_impacted_nodes", "n_impacted_files",
        "degraded_run", "analysis_mode", "estimated_scope",
        # Amendment 1: anchor-priming methodology flag per cell.
        "anchor_priming",
        # Entity-level
        "entity_precision_set", "entity_recall_set", "entity_f1_set",
        "entity_n_predicted", "entity_n_gt", "entity_n_intersect",
        # File-level
        "file_precision_set", "file_recall_set", "file_f1_set",
        "file_n_predicted", "file_n_gt", "file_n_intersect",
        # Top-level (= entity_*)
        "f1_set", "precision_set", "recall_set",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as csv_f, \
            jsonl_path.open("w", encoding="utf-8") as jsonl_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()

        for gt_entry in cr_dataset:
            cr_id = gt_entry.cr_id
            cr_text = gt_entry.cr_description

            results = run_single_cr_all_variants(
                cr_id=cr_id,
                cr_text=cr_text,
                gt_entry=gt_entry,
                settings=settings,
                output_dir=output_dir,
                shared_embedder=shared_embedder,
                shared_reranker=shared_reranker,
                shared_llm_client=shared_llm_client,
                anchor_priming=anchor_priming,
                cache_root=cache_root,
                run_tag=run_tag,
            )

            for variant_id, payload in results.items():
                row = {"cr_id": cr_id, "variant": variant_id}
                for f in fieldnames:
                    if f in row:
                        continue
                    row[f] = payload.get(f, None)
                writer.writerow(row)
                jsonl_f.write(
                    json.dumps({"cr_id": cr_id, "variant": variant_id, **payload})
                    + "\n"
                )
                csv_f.flush()
                jsonl_f.flush()

    logger.info("[ablation] Wrote {}", csv_path)
    return csv_path
