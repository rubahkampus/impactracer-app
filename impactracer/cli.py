"""Typer CLI: the three user-facing commands.

Commands:
    index     - Build the knowledge representation for a target repository.
    analyze   - Analyze a Change Request and emit an ImpactReport.
    evaluate  - Run the ablation harness over a Ground Truth dataset.

See 11_configuration_and_cli.md for the full command contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from loguru import logger

app = typer.Typer(
    name="impactracer",
    help="Change Impact Analysis via RAG with multilingual embeddings and structural graph.",
    no_args_is_help=True,
)


def _configure_logging(verbose: bool = False) -> None:
    """Configure loguru sinks. Called once per CLI invocation."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level, format="{time:HH:mm:ss} | {level} | {message}")
    log_path = Path("./data/impactracer.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), level="DEBUG", rotation="10 MB")


@app.callback()
def _root(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
) -> None:
    _configure_logging(verbose=verbose)


@app.command()
def index(
    repo_path: Path = typer.Argument(..., help="Path to repository root."),
    force: bool = typer.Option(False, "--force", help="Reindex all files regardless of hash."),
) -> None:
    """Build or update the knowledge representation for a repository.

    Orchestrates the offline indexing pipeline:
      1. Scan repository for .md and .ts/.tsx files.
      2. Check file_hashes for changed/new files (unless --force).
      3. Chunk Markdown (FR-A1, FR-A2).
      4. Parse TypeScript AST (Pass 1: nodes; Pass 2: edges).
      5. Embed all pending chunks and nodes via BGE-M3.
      6. Compute layer-weighted traceability pairs (FR-A7).
      7. Write index_metadata and print statistics.

    Implementation entry point:
        impactracer.indexer.runner.run_indexing(repo_path, settings, force)
    """
    from impactracer.indexer.runner import run_indexing
    from impactracer.shared.config import get_settings

    settings = get_settings()
    stats = run_indexing(repo_path=repo_path, settings=settings, force=force)

    typer.echo(
        f"\n{'='*50}\n"
        f"  ImpacTracer Index Complete\n"
        f"{'='*50}\n"
        f"  Files scanned    : {stats['files_scanned']}\n"
        f"  Files reindexed  : {stats['files_reindexed']}\n"
        f"  Code nodes       : {stats['code_nodes']}\n"
        f"  Doc chunks       : {stats['doc_chunks']}\n"
        f"  Structural edges : {stats['edges']}\n"
        f"  Elapsed          : {stats['elapsed_seconds']}s\n"
        f"{'='*50}"
    )


@app.command()
def report(
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Write Markdown report to this file (optional, default: stdout).",
    ),
) -> None:
    """Generate a comprehensive indexing quality report.

    Queries SQLite + ChromaDB to report:
      - Global node / edge counts and type breakdown
      - Graph topology health (density, orphans, top-degree nodes)
      - Traceability score distribution and stranded doc chunks
      - Semantic benchmark top-1 resolution
      - FK integrity checks
      - ChromaDB ↔ SQLite alignment

    Implementation entry point:
        impactracer.indexer.auditor.generate_report(settings)
    """
    import sys
    import io

    from impactracer.indexer.auditor import generate_report
    from impactracer.shared.config import get_settings

    settings = get_settings()
    md = generate_report(settings)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(md, encoding="utf-8")
        typer.echo(f"Report written to {output}")
    else:
        # Use UTF-8 wrapper so Unicode glyphs don't crash on Windows cp1252
        out = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        out.write(md)
        out.write("\n")
        out.flush()


@app.command()
def analyze(
    cr_text: str = typer.Argument(..., help="Change Request text (Indonesian or English)."),
    output: Path = typer.Option(Path("./impact_report.json"), "--output", "-o"),
    variant: str = typer.Option("V7", "--variant", help="Ablation variant V0-V7 (default V7)."),
) -> None:
    """Analyze a CR against the indexed repository and emit an ImpactReport.

    Invokes the nine-step online pipeline. V7 runs all five LLM invocations;
    lower variants disable subsets as specified in 09_ablation_harness.md.

    Implementation entry point:
        impactracer.pipeline.runner.run_analysis(cr_text, settings, flags)
    """
    import json as _json

    from impactracer.evaluation.variant_flags import VariantFlags
    from impactracer.pipeline.runner import run_analysis
    from impactracer.shared.config import get_settings

    variant_upper = variant.upper()
    if variant_upper not in VariantFlags.ALL_VARIANTS:
        typer.echo(f"Unknown variant '{variant}'. Choose from {VariantFlags.ALL_VARIANTS}.", err=True)
        raise typer.Exit(1)

    settings = get_settings()
    flags = VariantFlags.for_id(variant_upper)

    typer.echo(f"Analyzing CR with variant {variant_upper}...", err=True)

    # Crucible E2E Task 2b: full-traceability dump.
    trace_sink: dict = {"cr_text": cr_text, "variant": variant_upper}
    report = run_analysis(
        cr_text=cr_text,
        settings=settings,
        variant_flags=flags,
        trace_sink=trace_sink,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )

    # Crucible E2E Task 2b: write the full step-by-step trace alongside.
    full_path = output.with_name(output.stem + "_full.json")
    full_path.write_text(
        _json.dumps(trace_sink, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    typer.echo(f"\nReport written to {output}", err=True)
    typer.echo(f"Full trace written to {full_path}", err=True)
    typer.echo(f"Impacted entities : {len(report.impacted_entities)}", err=True)
    typer.echo(f"Impacted files    : {len(report.impacted_files)}", err=True)
    typer.echo(f"Estimated scope   : {report.estimated_scope}", err=True)

    # Pipeable summary on stdout
    print(_json.dumps({
        "impacted_entities": len(report.impacted_entities),
        "impacted_files": len(report.impacted_files),
        "scope": report.estimated_scope,
        "degraded": report.degraded_run,
    }))


@app.command()
def evaluate(
    dataset: Path = typer.Option(..., "--dataset", help="Directory containing one GT JSON file per CR."),
    output_dir: Path = typer.Option(Path("./eval/results/"), "--output"),
    run_full_ablation: bool = typer.Option(True, "--run-full-ablation"),
    verify_nfr: bool = typer.Option(False, "--verify-nfr"),
) -> None:
    """Run the full evaluation protocol on a GT dataset directory.

    Executes:
      1. Canonical V0..V7 ablation across every ``*.json`` GT file in ``--dataset``.
      2. Dual-granularity set-level metrics (entity + file).
      3. One pre-registered Wilcoxon signed-rank test (V7 vs V5, entity ``f1_set``).
         Falls back to ``status='insufficient_pairs'`` when n < 15 (calibration).
      4. Optional NFR-01..NFR-05 verification (``--verify-nfr``).
      5. Calibration-set analytical commentary (Markdown) printed to stdout.

    Implementation entry point:
        impactracer.evaluation.ablation.run_full_evaluation(...)
    """
    import json as _json
    from datetime import datetime, timezone

    import pandas as pd

    from impactracer.evaluation.ablation import run_full_evaluation
    from impactracer.evaluation.nfr_verify import verify_all_nfrs
    from impactracer.evaluation.report_builder import build_summary_artifacts
    from impactracer.evaluation.schemas import GTEntry
    from impactracer.evaluation.statistical import (
        InsufficientPairsError,
        MIN_PAIRED_N,
        PRIMARY_COMPARISON,
        PRIMARY_METRIC,
        run_primary_test,
    )
    from impactracer.evaluation.variant_flags import VariantFlags as _VF
    from impactracer.shared.config import get_settings

    settings = get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load GT dataset directory.
    # ------------------------------------------------------------------
    if not dataset.exists() or not dataset.is_dir():
        typer.echo(f"--dataset must be an existing directory; got {dataset}", err=True)
        raise typer.Exit(2)

    gt_files = sorted(dataset.glob("*.json"))
    if not gt_files:
        typer.echo(
            f"No JSON files in {dataset}. Generating two mock GT files for harness sanity.",
            err=True,
        )
        _write_mock_gt(dataset)
        gt_files = sorted(dataset.glob("*.json"))

    cr_dataset: list[GTEntry] = []
    for p in gt_files:
        try:
            cr_dataset.append(GTEntry.model_validate_json(p.read_text(encoding="utf-8")))
        except Exception as exc:
            typer.echo(f"Skipping {p.name}: {exc}", err=True)
    if not cr_dataset:
        typer.echo("No valid GT entries loaded. Aborting.", err=True)
        raise typer.Exit(2)

    typer.echo(
        f"\nLoaded {len(cr_dataset)} GT entries from {dataset}\n"
        f"Variants: {_VF.ALL_VARIANTS}\n"
        f"Output  : {output_dir}\n",
        err=True,
    )

    run_start_iso = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Ablation matrix.
    # ------------------------------------------------------------------
    if not run_full_ablation:
        typer.echo("--run-full-ablation=False is currently a no-op (no partial mode wired).", err=True)

    csv_path = run_full_evaluation(cr_dataset, settings, output_dir)
    long_df = pd.read_csv(csv_path)

    # ------------------------------------------------------------------
    # Wilcoxon test: pivot long -> wide on f1_set, run V7 vs V5.
    # ------------------------------------------------------------------
    stat_path = output_dir / "statistical_tests.json"
    try:
        wide = long_df.pivot(index="cr_id", columns="variant", values="f1_set")
        wide.columns = [f"{c}_{PRIMARY_METRIC}" for c in wide.columns]
        stat_result = run_primary_test(wide)
        stat_result["status"] = "ok"
    except InsufficientPairsError as exc:
        var_a, var_b = PRIMARY_COMPARISON
        col_a, col_b = f"{var_a}_{PRIMARY_METRIC}", f"{var_b}_{PRIMARY_METRIC}"
        try:
            wide = long_df.pivot(index="cr_id", columns="variant", values="f1_set")
            wide.columns = [f"{c}_{PRIMARY_METRIC}" for c in wide.columns]
            pairs = wide[[col_a, col_b]].dropna() if (col_a in wide.columns and col_b in wide.columns) else None
        except Exception:
            pairs = None
        stat_result = {
            "status": "insufficient_pairs",
            "hypothesis": f"{var_a}.{PRIMARY_METRIC} > {var_b}.{PRIMARY_METRIC} (one-sided paired Wilcoxon)",
            "variant_a": var_a,
            "variant_b": var_b,
            "metric": PRIMARY_METRIC,
            "n": (int(len(pairs)) if pairs is not None else 0),
            "min_required": MIN_PAIRED_N,
            "note": str(exc),
        }
        if pairs is not None and not pairs.empty:
            import numpy as _np

            a = pairs[col_a].to_numpy(dtype=float)
            b = pairs[col_b].to_numpy(dtype=float)
            stat_result["median_diff_descriptive"] = float(_np.median(a - b))
            from impactracer.evaluation.statistical import cliffs_delta
            stat_result["cliffs_delta_descriptive"] = float(cliffs_delta(a, b))
    except Exception as exc:
        stat_result = {"status": "error", "error": repr(exc)}

    stat_path.write_text(
        _json.dumps(stat_result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Summary artifacts.
    # ------------------------------------------------------------------
    summary_csv = build_summary_artifacts(long_df, [stat_result], output_dir)
    summary_df = pd.read_csv(summary_csv)

    # ------------------------------------------------------------------
    # NFR sweep.
    # ------------------------------------------------------------------
    nfr_result: dict | None = None
    if verify_nfr:
        nfr01_cr = cr_dataset[0].cr_description  # first CR (typically Indonesian)
        nfr04_cr = next(
            (g.cr_description for g in cr_dataset if _looks_indonesian(g.cr_description)),
            cr_dataset[0].cr_description,
        )
        try:
            nfr_result = verify_all_nfrs(
                eval_csv_path=csv_path,
                settings=settings,
                output_dir=output_dir,
                audit_path=Path(settings.llm_audit_log_path),
                run_start_iso=run_start_iso,
                nfr01_cr_text=nfr01_cr,
                nfr04_cr_text=nfr04_cr,
            )
        except Exception as exc:
            typer.echo(f"NFR sweep crashed: {exc}", err=True)
            nfr_result = {"error": repr(exc)}

    # ------------------------------------------------------------------
    # Data-Scientist analysis (Task 6).
    # ------------------------------------------------------------------
    analysis_md = _calibration_analysis(summary_df, long_df, stat_result)
    (output_dir / "calibration_analysis.md").write_text(analysis_md, encoding="utf-8")

    # ------------------------------------------------------------------
    # Print all three required outputs to stdout (UTF-8-safe on Windows).
    # ------------------------------------------------------------------
    import io as _io

    out = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", newline="\n")

    def _emit(s: str) -> None:
        out.write(s)
        out.write("\n")

    _emit("\n" + "=" * 72)
    _emit("SUMMARY TABLE (summary_table.csv)")
    _emit("=" * 72)
    _emit(summary_csv.read_text(encoding="utf-8"))

    _emit("=" * 72)
    _emit("STATISTICAL TESTS (statistical_tests.json)")
    _emit("=" * 72)
    _emit(_json.dumps(stat_result, indent=2, ensure_ascii=False))
    _emit("")

    _emit("=" * 72)
    _emit("DATA-SCIENTIST ANALYSIS (calibration_analysis.md)")
    _emit("=" * 72)
    _emit(analysis_md)

    if nfr_result is not None:
        _emit("=" * 72)
        _emit("NFR VERIFICATION (nfr_verification.json)")
        _emit("=" * 72)
        _emit(_json.dumps(nfr_result, indent=2, ensure_ascii=False, default=str))

    out.flush()


def _looks_indonesian(text: str) -> bool:
    """Heuristic: Indonesian CR contains common ID stopwords."""
    t = text.lower()
    return any(w in t for w in (" yang ", " untuk ", " dan ", " dengan ", "tambahkan", "menambahkan"))


def _write_mock_gt(target_dir: Path) -> None:
    """Emit two mock GT JSON files into ``target_dir`` for harness sanity.

    Only invoked when the directory exists but contains zero JSON files.
    """
    import json as _json

    target_dir.mkdir(parents=True, exist_ok=True)
    mock1 = {
        "cr_id": "cr01_mock",
        "cr_description": "Tambahkan fitur untuk menampilkan badge verifikasi pada profil pengguna.",
        "impacted_files": [
            {"file_path": "src/lib/db/models/user.model.ts", "justification": "Mock"},
        ],
        "impacted_entities": [
            {"node": "src/lib/db/models/user.model.ts::IUser", "justification": "Mock"},
        ],
    }
    mock2 = {
        "cr_id": "cr02_mock",
        "cr_description": "Implement search history retention for the past 30 days.",
        "impacted_files": [
            {"file_path": "src/lib/services/search.service.ts", "justification": "Mock"},
        ],
        "impacted_entities": [
            {"node": "src/lib/services/search.service.ts::searchHistory", "justification": "Mock"},
        ],
    }
    (target_dir / "cr01_mock.json").write_text(_json.dumps(mock1, indent=2), encoding="utf-8")
    (target_dir / "cr02_mock.json").write_text(_json.dumps(mock2, indent=2), encoding="utf-8")


def _calibration_analysis(summary_df, long_df, stat_result: dict) -> str:
    """Generate the written Data-Scientist analysis (Task 6 of Sprint 11+12).

    Reads the actual numbers from summary_df / long_df / stat_result and
    answers the three pre-registered questions from the brief.
    """
    import numpy as np

    def _row(variant: str) -> dict | None:
        sub = summary_df[summary_df["variant"] == variant]
        if sub.empty:
            return None
        return sub.iloc[0].to_dict()

    rows = {v: _row(v) for v in ["V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7"]}
    v5, v6, v7 = rows.get("V5"), rows.get("V6"), rows.get("V7")

    def _g(d, k):
        if d is None or k not in d:
            return float("nan")
        try:
            return float(d[k])
        except Exception:
            return float("nan")

    lines: list[str] = []
    lines.append("# Calibration Analysis (5-CR set)")
    lines.append("")
    lines.append(
        "Entity-level set metrics, macro-averaged across calibration CRs. "
        "All numbers below are macro means of per-CR set-level Precision, "
        "Recall, and F1 (no F1@K masking)."
    )
    lines.append("")

    # ----- Q1: V5 -> V6 graph flood -----
    p5, r5, f5 = _g(v5, "entity_precision_set"), _g(v5, "entity_recall_set"), _g(v5, "entity_f1_set")
    p6, r6, f6 = _g(v6, "entity_precision_set"), _g(v6, "entity_recall_set"), _g(v6, "entity_f1_set")
    n5 = _g(v5, "median_n_impacted_nodes")
    n6 = _g(v6, "median_n_impacted_nodes")
    dp_v6 = p6 - p5
    dr_v6 = r6 - r5

    lines.append("## Q1 — Did V6 trigger the Graph Flood?")
    lines.append(
        f"- V5 (validated SIS, no BFS) — P={p5:.3f}, R={r5:.3f}, F1={f5:.3f}, median entities={n5:.1f}"
    )
    lines.append(
        f"- V6 (V5 + BFS, no LLM #4)   — P={p6:.3f}, R={r6:.3f}, F1={f6:.3f}, median entities={n6:.1f}"
    )
    lines.append(f"- Δ V5→V6: ΔP={dp_v6:+.3f}, ΔR={dr_v6:+.3f}, Δmedian-entities={n6 - n5:+.1f}")
    if not np.isnan(dp_v6) and not np.isnan(dr_v6):
        if dp_v6 < -0.05 and dr_v6 > 0.0:
            lines.append(
                "  - **Verdict:** classic Graph Flood signature observed — "
                "BFS expansion lifts recall but precision collapses because "
                "unvalidated topological neighbours are admitted indiscriminately."
            )
        elif dp_v6 < -0.05:
            lines.append(
                "  - **Verdict:** precision drop observed, recall did not lift "
                "by a corresponding amount — BFS admitted graph neighbours that "
                "are neither GT-correct nor blocked by a precision gate."
            )
        elif dr_v6 > 0.05:
            lines.append(
                "  - **Verdict:** recall lifted without a precision collapse — "
                "the BFS depth caps and node-type fan-in limits are doing their "
                "job. The mechanistic 'flood' is being absorbed by deterministic "
                "guard-rails rather than appearing as a precision regression."
            )
        else:
            lines.append(
                "  - **Verdict:** V5 and V6 are nearly indistinguishable at this "
                "calibration scale — BFS either contributed almost no new nodes "
                "or those nodes were chance-aligned with GT."
            )
    lines.append("")

    # ----- Q2: V6 -> V7 precision recovery via LLM #4 -----
    p7, r7, f7 = _g(v7, "entity_precision_set"), _g(v7, "entity_recall_set"), _g(v7, "entity_f1_set")
    n7 = _g(v7, "median_n_impacted_nodes")
    dp_v7 = p7 - p6
    dr_v7 = r7 - r6

    lines.append("## Q2 — Did V7 recover precision via LLM #4?")
    lines.append(
        f"- V6 (BFS, no LLM #4) — P={p6:.3f}, R={r6:.3f}, F1={f6:.3f}, median entities={n6:.1f}"
    )
    lines.append(
        f"- V7 (BFS + LLM #4)   — P={p7:.3f}, R={r7:.3f}, F1={f7:.3f}, median entities={n7:.1f}"
    )
    lines.append(f"- Δ V6→V7: ΔP={dp_v7:+.3f}, ΔR={dr_v7:+.3f}, Δmedian-entities={n7 - n6:+.1f}")
    if not np.isnan(dp_v7):
        if dp_v7 > 0.03:
            lines.append(
                "  - **Verdict:** LLM #4 is acting as a precision-recovery valve — "
                "validated-propagation entities pass, hallucinated graph traversals "
                "are rejected. The Distributed Justification Principle is empirically supported."
            )
        elif dp_v7 < -0.03:
            lines.append(
                "  - **Verdict:** unexpected — LLM #4 lowered precision. "
                "Likely cause: LLM #4 admitted spurious chains that V6 had not propagated, "
                "or scoring noise dominates at n=5. Re-examine LLM #4 verdicts in the per-CR logs."
            )
        else:
            lines.append(
                "  - **Verdict:** V6 and V7 precision are within sampling noise at n=5. "
                "Larger evaluation set (n=20) will be more discriminative."
            )
    lines.append("")

    # ----- Q3: highest entity F1 variant -----
    candidates: list[tuple[str, float]] = []
    for vid in ["V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7"]:
        v = _g(rows.get(vid), "entity_f1_set")
        if not np.isnan(v):
            candidates.append((vid, v))
    candidates.sort(key=lambda kv: kv[1], reverse=True)
    lines.append("## Q3 — Which variant has the highest entity-level F1?")
    if candidates:
        lines.append("Ranking by macro-averaged entity F1:")
        for vid, f in candidates:
            lines.append(f"- {vid}: F1={f:.3f}")
        top_vid, top_f = candidates[0]
        lines.append("")
        lines.append(
            f"- **Winner at calibration (n=5):** {top_vid} with entity F1 = {top_f:.3f}."
        )
        if top_vid == "V7":
            lines.append(
                "  V7 leading is the expected outcome: the full pipeline pairs "
                "wide retrieval with three validation gates and a propagation "
                "valve. The pre-registered V7-vs-V5 Wilcoxon test is the "
                "appropriate confirmation on the n=20 evaluation set."
            )
        else:
            lines.append(
                f"  {top_vid} leads V7 on this small calibration set. Possible "
                "causes: (a) LLM #4's recall-cost outweighs its precision-gain "
                "at this scale, (b) graph propagation is unnecessary for this "
                "CR mix, (c) sampling noise at n=5. Worth a deeper look in the "
                "per-CR breakdown if the same picture survives n=20."
            )
    else:
        lines.append("- No variants produced finite F1 values — pipeline likely crashed for every CR.")
    lines.append("")

    # ----- Statistical aside -----
    lines.append("## Pre-registered Wilcoxon test (V7 vs V5, entity f1_set)")
    if stat_result.get("status") == "ok":
        lines.append(
            f"- n={stat_result.get('n')}, p={stat_result.get('p_value'):.4f}, "
            f"statistic={stat_result.get('statistic'):.2f}, "
            f"Cliff's δ={stat_result.get('cliffs_delta'):.3f}, "
            f"median Δ={stat_result.get('median_diff'):.4f}, "
            f"H₀ rejected at α=0.05? {bool(stat_result.get('accepted'))}"
        )
    elif stat_result.get("status") == "insufficient_pairs":
        lines.append(
            f"- Test deferred: only n={stat_result.get('n')} complete pairs "
            f"(min required {stat_result.get('min_required')}). Calibration size is "
            "by design too small for a defensible p-value; the 20-CR evaluation set "
            "is where the pre-registered test runs."
        )
        if "cliffs_delta_descriptive" in stat_result:
            lines.append(
                f"- Descriptive: Cliff's δ = {stat_result['cliffs_delta_descriptive']:.3f}, "
                f"median Δ = {stat_result.get('median_diff_descriptive', float('nan')):.4f}."
            )
    else:
        lines.append(f"- Test status: {stat_result.get('status')}.")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    app()
