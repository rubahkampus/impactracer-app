"""Sprint 17 — K-widening diagnostic.

Forensic; safe to run anytime. Counts ground-truth entities at cross-encoder
ranks {1-15, 16-25, 26-30, >30, not in pool} across the 5 calibration CRs.

Reads existing `impact_report_full.json` files that contain the
`step_3_reranked_full` trace (added in Sprint 17 to runner.py); writes ONE
CSV and prints a summary table. Does not modify any pipeline file.

Usage:
    python tools/diagnose_k_widening.py \\
        --results-dir eval/results_apex_v6_cot \\
        --gt-dir ground_truth/calibration \\
        --out-csv eval/results_apex_v6_cot/k_widening_diagnostic.csv

Apex Triage gate (threshold for proceeding to K=25):
    >= 3 GT entities at ranks 16-25 across the 5 calibration CRs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_gt(gt_dir: Path, cr_id: str) -> set[str]:
    """Load GT entity node-ids for a CR. cr_id is upper-case CR-XX form."""
    # CR-01 -> cr01.json (the calibration GT uses lowercase, no dash).
    suffix = cr_id.replace("CR-", "cr")
    gt_path = gt_dir / f"{suffix}.json"
    if not gt_path.is_file():
        return set()
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    entities = data.get("impacted_entities", [])
    return {e["node"] for e in entities if isinstance(e, dict) and "node" in e}


def _load_pool(report_path: Path) -> list[dict]:
    """Return step_3_reranked_full entries (or empty list if absent)."""
    if not report_path.is_file():
        return []
    data = json.loads(report_path.read_text(encoding="utf-8"))
    pool = data.get("step_3_reranked_full")
    if not isinstance(pool, list):
        return []
    return pool


def _rank_of(node_id: str, pool: list[dict]) -> int | None:
    """1-indexed rank of node_id in pool; None if not present."""
    for i, entry in enumerate(pool, start=1):
        if entry.get("node_id") == node_id:
            return i
    return None


def _bucket(rank: int | None) -> str:
    if rank is None:
        return "not_in_pool"
    if rank <= 15:
        return "1_to_15"
    if rank <= 25:
        return "16_to_25"
    if rank <= 30:
        return "26_to_30"
    return "gt_30"


def diagnose(
    results_dir: Path,
    gt_dir: Path,
    out_csv: Path,
) -> dict[str, int]:
    """Walk CR-01..CR-05/V7 reports, write CSV, return summary counts."""
    rows: list[dict] = []
    bucket_counts: dict[str, int] = {
        "1_to_15": 0,
        "16_to_25": 0,
        "26_to_30": 0,
        "gt_30": 0,
        "not_in_pool": 0,
    }

    # Discover CR-XX directories (sorted for deterministic output).
    cr_dirs = sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith("CR-")
    )

    for cr_dir in cr_dirs:
        cr_id = cr_dir.name
        gt_entities = _load_gt(gt_dir, cr_id)
        if not gt_entities:
            print(f"[warn] {cr_id}: no GT entities found at {gt_dir}/{cr_id.replace('CR-', 'cr')}.json")
            continue

        report_path = cr_dir / "V7" / "impact_report_full.json"
        pool = _load_pool(report_path)
        if not pool:
            print(f"[warn] {cr_id}: step_3_reranked_full missing or empty at {report_path}")

        for gt in sorted(gt_entities):
            rank = _rank_of(gt, pool)
            score = None
            if rank is not None:
                score = pool[rank - 1].get("raw_reranker_score")
            bucket = _bucket(rank)
            bucket_counts[bucket] += 1
            rows.append({
                "cr_id": cr_id,
                "gt_entity": gt,
                "rank": rank if rank is not None else "",
                "raw_reranker_score": f"{score:.4f}" if isinstance(score, (int, float)) else "",
                "in_top_15": "1" if rank is not None and rank <= 15 else "0",
                "in_top_25": "1" if rank is not None and rank <= 25 else "0",
                "in_top_30": "1" if rank is not None and rank <= 30 else "0",
                "bucket": bucket,
            })

    # Write CSV.
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cr_id", "gt_entity", "rank", "raw_reranker_score",
                "in_top_15", "in_top_25", "in_top_30", "bucket",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return bucket_counts


def print_summary(bucket_counts: dict[str, int], out_csv: Path) -> None:
    total = sum(bucket_counts.values())
    print()
    print("=" * 64)
    print("K-WIDENING DIAGNOSTIC — Sprint 17")
    print("=" * 64)
    print(f"Total GT entities scanned: {total}")
    print()
    print(f"{'Bucket':<14} {'Count':>8} {'%':>8}")
    print("-" * 32)
    for bucket in ("1_to_15", "16_to_25", "26_to_30", "gt_30", "not_in_pool"):
        n = bucket_counts[bucket]
        pct = (n / total * 100.0) if total else 0.0
        print(f"{bucket:<14} {n:>8d} {pct:>7.1f}%")
    print()
    gate_n = bucket_counts["16_to_25"]
    print(f"APEX TRIAGE GATE — GT entities at ranks 16-25: {gate_n}")
    if gate_n >= 3:
        print("  GATE: PROCEED to Step 5 (lift max_admitted_seeds to 25).")
    else:
        print("  GATE: DROP K-widening. Keep K=15. (Threshold = 3.)")
    print()
    print(f"CSV written to: {out_csv}")
    print("=" * 64)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing CR-XX/V7/impact_report_full.json files.",
    )
    p.add_argument(
        "--gt-dir",
        type=Path,
        default=_REPO_ROOT / "ground_truth" / "calibration",
        help="Directory containing cr0X.json GT files.",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <results-dir>/k_widening_diagnostic.csv",
    )
    args = p.parse_args()

    results_dir: Path = args.results_dir
    if not results_dir.is_absolute():
        results_dir = (_REPO_ROOT / results_dir).resolve()
    gt_dir: Path = args.gt_dir
    if not gt_dir.is_absolute():
        gt_dir = (_REPO_ROOT / gt_dir).resolve()
    out_csv: Path = args.out_csv or (results_dir / "k_widening_diagnostic.csv")

    if not results_dir.is_dir():
        print(f"ERROR: results dir not found: {results_dir}", file=sys.stderr)
        return 1
    if not gt_dir.is_dir():
        print(f"ERROR: GT dir not found: {gt_dir}", file=sys.stderr)
        return 1

    bucket_counts = diagnose(results_dir, gt_dir, out_csv)
    print_summary(bucket_counts, out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
