"""Aggregated summary artifact generation.

Outputs:
  - summary_table.csv       - macro-averaged metrics per variant
  - summary_table.md        - same content, rendered as Markdown for the
                              thesis appendix.

The Wilcoxon test artifact (``statistical_tests.json``) is owned by the
CLI orchestrator, not this builder, so that descriptive statistics and
hypothesis testing remain separable concerns.

Reference: 10_evaluation_protocol.md §6.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from impactracer.evaluation.variant_flags import VariantFlags


_METRIC_COLS = [
    "entity_precision_set",
    "entity_recall_set",
    "entity_f1_set",
    "file_precision_set",
    "file_recall_set",
    "file_f1_set",
]

# ---------------------------------------------------------------------------
# Per-stratum change-type basis (the `summary_table_by_change_type.*` column).
# True  (default): stratify by the CR-CODE prefix (ADD/MOD/DEL of cr_id) — the
#                  designed GT strata, independent of how LLM #1 classified the
#                  CR. Stable buckets (e.g. always 8/8/8 on citrakara), so the
#                  table reads as "system performance per designed stratum".
# False: stratify by the LLM-ASSIGNED change_type column (what the pipeline
#        actually used). MOD CRs the LLM reclassified as ADDITION then count
#        under ADDITION. Hardcoded toggle — flip here, not via CLI.
# The emitted column is named `cr_change_type` in both modes to avoid implying
# the bucket equals the LLM-assigned label.
STRATIFY_BY_CR_CODE = True

_CR_CODE_TO_CHANGE_TYPE = {
    "ADD": "ADDITION",
    "MOD": "MODIFICATION",
    "DEL": "DELETION",
}


def _cr_code_change_type(cr_id: str, assigned_type: str) -> str:
    """Map a ``cr_id`` prefix (ADD-/MOD-/DEL-) to its change_type.

    Falls back to the LLM-``assigned_type`` when the prefix is unrecognised
    (the "if applicable" rule), so datasets whose ids don't encode a stratum
    still bucket by whatever the pipeline assigned.
    """
    prefix = str(cr_id).split("-")[0].strip().upper()
    return _CR_CODE_TO_CHANGE_TYPE.get(prefix, str(assigned_type).upper())


def _macro_average(group: pd.DataFrame, col: str) -> float:
    """Macro-average ``col`` across rows where ``status == 'ok'`` only.

    NaN-tolerant via ``np.nanmean``. Returns NaN if no usable values.
    """
    ok = group[group["status"] == "ok"]
    if col not in ok.columns or ok[col].empty:
        return float("nan")
    vals = pd.to_numeric(ok[col], errors="coerce").to_numpy(dtype=float)
    if vals.size == 0 or np.all(np.isnan(vals)):
        return float("nan")
    return float(np.nanmean(vals))


def build_summary_artifacts(
    df: pd.DataFrame,
    stat_rows: list[dict],
    output_dir: Path,
) -> Path:
    """Emit summary_table.csv + summary_table.md to ``output_dir``.

    Args:
        df: long-form DataFrame loaded from per_cr_per_variant_metrics.csv.
            Must have columns: cr_id, variant, status, elapsed_s,
            n_impacted_nodes, and the six metric columns.
        stat_rows: reserved for future descriptive stat-test rows. Unused
            here; the orchestrator writes statistical_tests.json directly.
        output_dir: target directory (must already exist).

    Returns:
        Path to summary_table.csv.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    for variant in VariantFlags.ALL_VARIANTS:
        sub = df[df["variant"] == variant]
        if sub.empty:
            continue
        row: dict = {"variant": variant}
        for col in _METRIC_COLS:
            row[col] = _macro_average(sub, col)
        ok = sub[sub["status"] == "ok"]
        row["n_ok"] = int(len(ok))
        row["n_error"] = int(len(sub) - len(ok))
        if not ok.empty:
            elapsed = pd.to_numeric(ok["elapsed_s"], errors="coerce").dropna()
            row["median_elapsed_s"] = (
                float(elapsed.median()) if not elapsed.empty else float("nan")
            )
            n_nodes = pd.to_numeric(ok["n_impacted_nodes"], errors="coerce").dropna()
            row["median_n_impacted_nodes"] = (
                float(n_nodes.median()) if not n_nodes.empty else float("nan")
            )
        else:
            row["median_elapsed_s"] = float("nan")
            row["median_n_impacted_nodes"] = float("nan")
        summary_rows.append(row)

    summary_df = pd.DataFrame(
        summary_rows,
        columns=[
            "variant",
            *_METRIC_COLS,
            "n_ok",
            "n_error",
            "median_elapsed_s",
            "median_n_impacted_nodes",
        ],
    )

    csv_path = output_dir / "summary_table.csv"
    summary_df.to_csv(csv_path, index=False, float_format="%.4f")

    md_path = output_dir / "summary_table.md"
    md_path.write_text(_render_markdown(summary_df), encoding="utf-8")

    # ------------------------------------------------------------------
    # Sprint 25: per-change-type stratified summary.
    # Macro F1 hides that the eval set mixes MODIFICATION (easy), ADDITION
    # (hard — anchor inference required), and DELETION (graph-flood prone).
    # Stratification gives the committee a per-change-type read on where
    # the pipeline actually works vs where it struggles.
    # ------------------------------------------------------------------
    # Derive the stratification key `cr_change_type` (see STRATIFY_BY_CR_CODE).
    # Default basis = CR-code prefix; toggle to LLM-assigned `change_type`.
    df = df.copy()
    if STRATIFY_BY_CR_CODE and "cr_id" in df.columns:
        df["cr_change_type"] = [
            _cr_code_change_type(cid, atype)
            for cid, atype in zip(
                df["cr_id"], df.get("change_type", pd.Series([""] * len(df)))
            )
        ]
    elif "change_type" in df.columns:
        df["cr_change_type"] = df["change_type"].astype(str).str.upper()

    if "cr_change_type" in df.columns:
        per_ct_rows: list[dict] = []
        for ct_value in ("ADDITION", "MODIFICATION", "DELETION"):
            ct_sub = df[df["cr_change_type"] == ct_value]
            if ct_sub.empty:
                continue
            for variant in VariantFlags.ALL_VARIANTS:
                vsub = ct_sub[ct_sub["variant"] == variant]
                if vsub.empty:
                    continue
                row: dict = {"cr_change_type": ct_value, "variant": variant}
                for col in _METRIC_COLS:
                    row[col] = _macro_average(vsub, col)
                ok = vsub[vsub["status"] == "ok"]
                row["n_ok"] = int(len(ok))
                row["n_error"] = int(len(vsub) - len(ok))
                per_ct_rows.append(row)

        if per_ct_rows:
            ct_df = pd.DataFrame(
                per_ct_rows,
                columns=["cr_change_type", "variant", *_METRIC_COLS, "n_ok", "n_error"],
            )
            ct_csv = output_dir / "summary_table_by_change_type.csv"
            ct_df.to_csv(ct_csv, index=False, float_format="%.4f")
            ct_md = output_dir / "summary_table_by_change_type.md"
            ct_md.write_text(_render_change_type_markdown(ct_df), encoding="utf-8")

    return csv_path


def _render_change_type_markdown(ct_df: pd.DataFrame) -> str:
    """Render the per-stratum summary as Markdown grouped by cr_change_type."""
    if ct_df.empty:
        return "# Summary by Change Type\n\n_(no rows)_\n"
    lines = ["# Summary Table — Per-Change-Type × Per-Variant Set-Level Metrics", ""]
    for ct in ("ADDITION", "MODIFICATION", "DELETION"):
        sub = ct_df[ct_df["cr_change_type"] == ct]
        if sub.empty:
            continue
        cols = [c for c in sub.columns if c != "cr_change_type"]
        lines.append(f"## cr_change_type = {ct}")
        lines.append("")
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, r in sub.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    cells.append("nan" if np.isnan(v) else f"{v:.4f}")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def _render_markdown(summary_df: pd.DataFrame) -> str:
    """Render the summary table as GitHub-flavored Markdown."""
    if summary_df.empty:
        return "# Summary Table\n\n_(no rows)_\n"
    cols = list(summary_df.columns)
    lines = ["# Summary Table — Macro-Averaged Set-Level Metrics per Variant", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in summary_df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                if np.isnan(v):
                    cells.append("nan")
                else:
                    cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
