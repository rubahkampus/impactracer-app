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

import json
from pathlib import Path

import numpy as np
import pandas as pd

from impactracer.evaluation.variant_flags import VariantFlags

# Final-report justification_source -> propagation path (arm).
# Path A = OUTWARD (BFS over dependency edges); Path B = IN-FILE (CONTAINS
# siblings). Both arms run together by default; this attribution makes each
# arm's contribution visible within the shared, co-run result.
_PATH_A_SOURCES = {"bfs_only", "llm4_propagation", "auto_exempt"}
_PATH_B_SOURCES = {"sibling_candidate", "llm4_sibling"}


def _trace_find(obj: object, key: str):
    """Depth-first search for the first value under ``key`` in a nested dict/list."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _trace_find(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _trace_find(v, key)
            if r is not None:
                return r
    return None


def _load_full_report(output_dir: Path, cr_id: str, variant: str) -> dict | None:
    p = output_dir / cr_id / variant / "impact_report_full.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def path_contribution_section(
    output_dir: Path,
    gt_by_cr: dict[str, set[str]],
) -> str:
    """Per-PATH propagation contribution: a funnel (expand -> deterministic
    filter -> LLM validate, per arm) plus final per-path attribution (TP / FP /
    unique-GT / precision) against ground truth.

    Reads the per-CR V6/V7 traces written by the runner. Pooled over all CRs
    that have a V6 (and, where present, V7) trace. Returns a Markdown section;
    empty string if no propagation traces are found (e.g. a retrieval-only run).
    """
    # Prefer V7 traces (full path incl. LLM validation); fall back to V6.
    variant = None
    for cand in ("V7", "V6"):
        if any((output_dir / cr / cand / "impact_report_full.json").exists()
               for cr in gt_by_cr):
            variant = cand
            break
    if variant is None:
        return ""

    # ---- Funnel accumulators (node counts + GT retained), per arm ----
    # A = outward BFS, B = in-file sibling.
    fa = {k: 0 for k in ("raw", "raw_gt", "post_filter", "post_filter_gt",
                          "final", "final_gt")}
    fb = {k: 0 for k in ("raw", "raw_gt", "post_filter", "post_filter_gt",
                          "final", "final_gt")}
    # Final attribution (against GT), per arm.
    aa = {"tp": 0, "fp": 0}
    ab = {"tp": 0, "fp": 0}
    seeds_tp = seeds_fp = 0  # SIS seeds, for context (not a propagation arm)
    n_cr = 0
    has_v7 = variant == "V7"

    for cr, gt in gt_by_cr.items():
        rep = _load_full_report(output_dir, cr, variant)
        if rep is None:
            continue
        n_cr += 1

        ents = rep.get("final_report", {}).get("impacted_entities", [])
        final_a = {e["node"] for e in ents if e.get("justification_source") in _PATH_A_SOURCES}
        final_b = {e["node"] for e in ents if e.get("justification_source") in _PATH_B_SOURCES}

        # Filter drops (always traced by 6.8/6.9, even on a cache-resume where
        # the expansion traces are absent — so we reconstruct the funnel from
        # drops + survivors rather than the expansion trace alone).
        p8 = _trace_find(rep, "step_6p8_weight_decay_prune") or {}
        p9 = _trace_find(rep, "step_6p9_sibling_precision_prune") or {}
        dropped_a = set(p8.get("dropped", []))
        dropped_b = set(p9.get("dropped", []))

        # --- Path A (outward BFS) funnel ---
        # post-6.8 survivors == the Path A entities that reached the scored set;
        # raw ~= survivors + 6.8 drops (collapse-folded leaves are not recoverable
        # from traces, so raw is a lower bound; final/post are exact).
        raw_a = final_a | dropped_a
        fa["raw"] += len(raw_a)
        fa["raw_gt"] += len(raw_a & gt)
        fa["post_filter"] += len(final_a)
        fa["post_filter_gt"] += len(final_a & gt)

        # --- Path B (in-file sibling) funnel (exact — siblings don't collapse) ---
        raw_b = final_b | dropped_b
        fb["raw"] += len(raw_b)
        fb["raw_gt"] += len(raw_b & gt)
        fb["post_filter"] += len(final_b)
        fb["post_filter_gt"] += len(final_b & gt)

        # --- Final attribution by justification_source ---
        for e in ents:
            nid = e.get("node")
            src = e.get("justification_source", "")
            is_gt = nid in gt
            if src in _PATH_A_SOURCES:
                fa["final"] += 1
                fa["final_gt"] += int(is_gt)
                aa["tp" if is_gt else "fp"] += 1
            elif src in _PATH_B_SOURCES:
                fb["final"] += 1
                fb["final_gt"] += int(is_gt)
                ab["tp" if is_gt else "fp"] += 1
            else:  # llm2_sis / llm3_trace -> SIS seeds (direct, not propagated)
                seeds_tp += int(is_gt)
                seeds_fp += int(not is_gt)

    if n_cr == 0:
        return ""

    def _prec(tp: int, fp: int) -> str:
        return f"{tp / (tp + fp):.3f}" if (tp + fp) else "—"

    lines: list[str] = []
    lines.append(f"## Per-Path Propagation Contribution ({variant}, N={n_cr} CRs)")
    lines.append("")
    lines.append("Two arms run together by default. **Path A = OUTWARD** (BFS over "
                 "dependency edges); **Path B = IN-FILE** (CONTAINS siblings). "
                 "Counts pooled over all scored CRs.")
    lines.append("")
    lines.append("### Funnel — nodes (and GT retained) through each stage")
    lines.append("")
    lines.append("| stage | Path A nodes | A: GT | Path B nodes | B: GT |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| 1. expand (raw, V6) | {fa['raw']} | {fa['raw_gt']} | "
                 f"{fb['raw']} | {fb['raw_gt']} |")
    lines.append(f"| 2. deterministic filter (6.8 / 6.9) | {fa['post_filter']} | "
                 f"{fa['post_filter_gt']} | {fb['post_filter']} | {fb['post_filter_gt']} |")
    lbl3 = "3. final (post-V7 LLM validate)" if has_v7 else "3. final (V6 — no LLM validate)"
    lines.append(f"| {lbl3} | {fa['final']} | {fa['final_gt']} | "
                 f"{fb['final']} | {fb['final_gt']} |")
    lines.append("")
    lines.append("_Filter row uses 6.8 weight-decay-prune drops for Path A and "
                 "6.9 anchor-rrf-prune drops for Path B._")
    lines.append("")
    lines.append("### Final attribution — each path's contribution to the scored set")
    lines.append("")
    lines.append("| path | TP | FP | precision | GT found |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| A — outward (BFS) | {aa['tp']} | {aa['fp']} | "
                 f"{_prec(aa['tp'], aa['fp'])} | {fa['final_gt']} |")
    lines.append(f"| B — in-file (sibling) | {ab['tp']} | {ab['fp']} | "
                 f"{_prec(ab['tp'], ab['fp'])} | {fb['final_gt']} |")
    lines.append(f"| (SIS seeds, for context) | {seeds_tp} | {seeds_fp} | "
                 f"{_prec(seeds_tp, seeds_fp)} | {seeds_tp} |")
    lines.append("")
    return "\n".join(lines) + "\n"


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
    # per-change-type stratified summary.
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
