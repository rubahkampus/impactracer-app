"""Re-emit summary / statistical / NFR artefacts from an existing eval run.

Used after the first live evaluate completed all 40 ablation cells but the
final stdout print crashed on Windows cp1252 (Δ char) and the NFR sweep
needed an NFR-01 fix (validated-SIS set, not impacted_entities set).

Reads ``per_cr_per_variant_metrics.csv`` from --output, regenerates the
statistical_tests.json / summary_table.csv / nfr_verification.json /
calibration_analysis.md, and prints all four to UTF-8-safe stdout. Does
NOT re-run the 40-cell ablation.
"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from impactracer.evaluation.nfr_verify import verify_all_nfrs
from impactracer.evaluation.report_builder import build_summary_artifacts
from impactracer.evaluation.schemas import GTEntry
from impactracer.evaluation.statistical import (
    InsufficientPairsError,
    MIN_PAIRED_N,
    PRIMARY_COMPARISON,
    PRIMARY_METRIC,
    cliffs_delta,
    run_primary_test,
)
from impactracer.shared.config import get_settings


def _calibration_analysis(summary_df, long_df, stat_result):
    from impactracer.cli import _calibration_analysis as _impl

    return _impl(summary_df, long_df, stat_result)


def _looks_indonesian(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in (" yang ", " untuk ", " dan ", " dengan ", "tambahkan", "menambahkan"))


def main(output_dir: Path, dataset_dir: Path, run_start_iso: str) -> None:
    settings = get_settings()
    csv_path = output_dir / "per_cr_per_variant_metrics.csv"
    long_df = pd.read_csv(csv_path)

    # Wilcoxon / insufficient-pairs fallback.
    try:
        wide = long_df.pivot(index="cr_id", columns="variant", values="f1_set")
        wide.columns = [f"{c}_{PRIMARY_METRIC}" for c in wide.columns]
        stat_result = run_primary_test(wide)
        stat_result["status"] = "ok"
    except InsufficientPairsError as exc:
        var_a, var_b = PRIMARY_COMPARISON
        col_a, col_b = f"{var_a}_{PRIMARY_METRIC}", f"{var_b}_{PRIMARY_METRIC}"
        wide = long_df.pivot(index="cr_id", columns="variant", values="f1_set")
        wide.columns = [f"{c}_{PRIMARY_METRIC}" for c in wide.columns]
        pairs = wide[[col_a, col_b]].dropna() if (col_a in wide.columns and col_b in wide.columns) else None
        stat_result = {
            "status": "insufficient_pairs",
            "hypothesis": f"{var_a}.{PRIMARY_METRIC} > {var_b}.{PRIMARY_METRIC} (one-sided paired Wilcoxon)",
            "variant_a": var_a,
            "variant_b": var_b,
            "metric": PRIMARY_METRIC,
            "n": int(len(pairs)) if pairs is not None else 0,
            "min_required": MIN_PAIRED_N,
            "note": str(exc),
        }
        if pairs is not None and not pairs.empty:
            import numpy as _np

            a = pairs[col_a].to_numpy(dtype=float)
            b = pairs[col_b].to_numpy(dtype=float)
            stat_result["median_diff_descriptive"] = float(_np.median(a - b))
            stat_result["cliffs_delta_descriptive"] = float(cliffs_delta(a, b))

    (output_dir / "statistical_tests.json").write_text(
        json.dumps(stat_result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    summary_csv = build_summary_artifacts(long_df, [stat_result], output_dir)
    summary_df = pd.read_csv(summary_csv)

    # NFR sweep — only NFR-01 + NFR-04 re-run V7; NFR-03/NFR-05 are read from disk.
    gt_files = sorted(dataset_dir.glob("*.json"))
    cr_dataset = [GTEntry.model_validate_json(p.read_text(encoding="utf-8")) for p in gt_files]
    nfr01_cr = cr_dataset[0].cr_description
    nfr04_cr = next(
        (g.cr_description for g in cr_dataset if _looks_indonesian(g.cr_description)),
        cr_dataset[0].cr_description,
    )
    nfr_result = verify_all_nfrs(
        eval_csv_path=csv_path,
        settings=settings,
        output_dir=output_dir,
        audit_path=Path(settings.llm_audit_log_path),
        run_start_iso=run_start_iso,
        nfr01_cr_text=nfr01_cr,
        nfr04_cr_text=nfr04_cr,
    )

    analysis_md = _calibration_analysis(summary_df, long_df, stat_result)
    (output_dir / "calibration_analysis.md").write_text(analysis_md, encoding="utf-8")

    out = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", newline="\n")

    def _emit(s):
        out.write(s)
        out.write("\n")

    _emit("\n" + "=" * 72)
    _emit("SUMMARY TABLE (summary_table.csv)")
    _emit("=" * 72)
    _emit(summary_csv.read_text(encoding="utf-8"))
    _emit("=" * 72)
    _emit("STATISTICAL TESTS (statistical_tests.json)")
    _emit("=" * 72)
    _emit(json.dumps(stat_result, indent=2, ensure_ascii=False))
    _emit("")
    _emit("=" * 72)
    _emit("DATA-SCIENTIST ANALYSIS (calibration_analysis.md)")
    _emit("=" * 72)
    _emit(analysis_md)
    _emit("=" * 72)
    _emit("NFR VERIFICATION (nfr_verification.json)")
    _emit("=" * 72)
    _emit(json.dumps(nfr_result, indent=2, ensure_ascii=False, default=str))
    out.flush()


if __name__ == "__main__":
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "eval/results")
    ds_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "ground_truth/calibration")
    iso = sys.argv[3] if len(sys.argv) > 3 else datetime.now(timezone.utc).isoformat()
    main(out_dir, ds_dir, iso)
