"""Crucible Fix 8 — pipeline-attrition diagnostic.

Runs one CR through V0..V7 and prints a per-stage attrition table so
the operator can verify that the V0..V7 metric movement matches the
architectural-horizon expectations BEFORE the full ablation study.

Usage:
    python tools/diagnose_pipeline.py --cr-text "<CR>"
    python tools/diagnose_pipeline.py --gt-file gt.json --cr-id CR-001

Output:
    Variant   RRF   rerank  gates  L2   L3   BFS   L4   final  degraded
    V0        50    15      15     -    -    -     -    15     False
    V4        50    15      12     8    -    -     -    8      False
    V6        50    15      12     8    6    372   -    378    False
    V7        50    15      12     8    6    372   41   47     False

Read-only: does NOT modify any files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the impactracer package importable when running as a script.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from loguru import logger

from impactracer.evaluation.variant_flags import VariantFlags
from impactracer.shared.config import get_settings


def _instrument_runner():
    """Monkey-patch runner.run_analysis to capture per-stage counts.

    We hook the loguru output stream and parse the structured info lines
    that the runner already emits ("Post-RRF pool: N", etc.). This avoids
    invasive changes to the production code.
    """
    captured: dict[str, dict[str, int | bool | str]] = {}
    current_variant = {"id": ""}

    def sink(message):
        text = message.record["message"]
        v = current_variant["id"]
        if not v:
            return
        cell = captured.setdefault(v, {
            "rrf_pool": None, "rerank": None, "gates": None,
            "l2": None, "l3": None, "bfs": None, "l4_kept": None,
            "final": None, "degraded": False,
        })
        if "Post-RRF pool:" in text:
            cell["rrf_pool"] = int(text.split(":")[-1].strip())
        elif "Post-rerank candidates:" in text:
            cell["rerank"] = int(text.split(":")[-1].strip())
        elif "post-gates=" in text:
            for token in text.split():
                if token.startswith("post-gates="):
                    cell["gates"] = int(token.split("=")[1])
        elif "LLM #2 complete:" in text:
            # "[validator] LLM #2 complete: 8/15 candidates confirmed (degraded=False)"
            try:
                cell["l2"] = int(text.split("complete:")[1].split("/")[0].strip())
            except Exception:
                pass
        elif "validated seeds (" in text:
            # "[traceability_validator] Result: 6 validated seeds (...)"
            try:
                cell["l3"] = int(text.split("Result:")[1].split("validated")[0].strip())
            except Exception:
                pass
        elif "BFS:" in text and "propagated" in text:
            # "[graph_bfs] BFS complete: 6 SIS seeds, 18 propagated nodes"
            try:
                pieces = text.split(",")
                sis = int(pieces[0].split(":")[-1].strip().split()[0])
                prop = int(pieces[1].strip().split()[0])
                cell["bfs"] = sis + prop
            except Exception:
                pass
        elif "After LLM #4:" in text:
            try:
                cell["l4_kept"] = int(text.split("After LLM #4:")[1].split("propagated")[0].strip())
            except Exception:
                pass
        elif "Analysis complete:" in text:
            # "[runner] Analysis complete: 12 impacted nodes, ..."
            try:
                cell["final"] = int(text.split("Analysis complete:")[1].split("impacted")[0].strip())
            except Exception:
                pass
            if "degraded=True" in text:
                cell["degraded"] = True

    handler_id = logger.add(sink, level="INFO")
    return captured, current_variant, handler_id


def _print_table(captured: dict) -> None:
    cols = ["RRF", "rerank", "gates", "L2", "L3", "BFS", "L4", "final", "degraded"]
    keys = ["rrf_pool", "rerank", "gates", "l2", "l3", "bfs", "l4_kept", "final", "degraded"]
    fmt = "{:<8}" + "  ".join("{:<8}" for _ in cols)
    print(fmt.format("Variant", *cols))
    for variant in VariantFlags.ALL_VARIANTS:
        cell = captured.get(variant, {})
        row = []
        for k in keys:
            v = cell.get(k)
            row.append(str(v) if v is not None else "-")
        print(fmt.format(variant, *row))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cr-text", help="Run this exact CR text")
    parser.add_argument("--gt-file", help="Path to a ground-truth JSON file")
    parser.add_argument("--cr-id", help="CR id within the GT file")
    args = parser.parse_args()

    if args.cr_text:
        cr_text = args.cr_text
        cr_id = "ad-hoc"
    elif args.gt_file and args.cr_id:
        gt_data = json.loads(Path(args.gt_file).read_text(encoding="utf-8"))
        # GT format may be a list of GTEntry dicts.
        entries = gt_data if isinstance(gt_data, list) else gt_data.get("entries", [])
        match = next((e for e in entries if e.get("cr_id") == args.cr_id), None)
        if match is None:
            print(f"CR {args.cr_id} not found in {args.gt_file}", file=sys.stderr)
            return 2
        cr_text = match["cr_description"]
        cr_id = match["cr_id"]
    else:
        parser.print_help()
        return 1

    settings = get_settings()
    from impactracer.pipeline.runner import load_pipeline_context, run_analysis

    seed_ctx = load_pipeline_context(settings, variant_flags=None)
    shared = (seed_ctx.embedder, seed_ctx.reranker, seed_ctx.llm_client)

    captured, current_variant, handler_id = _instrument_runner()
    try:
        for variant_id in VariantFlags.ALL_VARIANTS:
            current_variant["id"] = variant_id
            print(f"\n=== {variant_id} on {cr_id} ===", file=sys.stderr)
            try:
                run_analysis(
                    cr_text,
                    settings,
                    variant_flags=VariantFlags.for_id(variant_id),
                    shared_embedder=shared[0],
                    shared_reranker=shared[1],
                    shared_llm_client=shared[2],
                )
            except Exception as exc:
                print(f"  [ERROR] {exc}", file=sys.stderr)
    finally:
        logger.remove(handler_id)

    print()
    _print_table(captured)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
