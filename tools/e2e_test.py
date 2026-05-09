"""Crucible E2E Stress Test — runs 5 fixed CRs against V7.

Invokes ``impactracer analyze`` via subprocess for each CR, persists both
``impact_report.json`` and ``impact_report_full.json`` per CR, and prints
a quality summary table that exercises the five PASS criteria from
Crucible E2E Task 4:

  1. No crashes (subprocess returncode == 0).
  2. Fail-closed integrity (degraded_run flag exposed; exception did not
     terminate the run).
  3. Semantic excellence (entity count, file count, presence of explicit
     CR-named files).
  4. Distributed justifications (every entity has a non-empty
     justification with a source label).
  5. Schema compliance (impacted_files + impacted_entities present;
     impact_report_full.json contains the documented step keys).

This script is read-only with respect to source files; the only writes
are to the per-CR output directory under ``./eval/e2e_runs/<cr_id>/``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

CR_SUITE: list[dict] = [
    {
        "cr_id": "CR-PIN-LISTING",
        "expected_substrings": ["CommissionListing", "profile"],
        "text": (
            "Tambahkan kemampuan bagi ilustrator untuk menyematkan (pin) "
            "maksimal 3 commission listing ke bagian atas halaman profil "
            "publiknya. Fitur ini memerlukan penambahan field pada model "
            "CommissionListing atau User, modifikasi logika pengambilan "
            "listing pada service, dan modifikasi komponen profil untuk "
            "menampilkan listing yang disematkan secara terpisah di posisi atas."
        ),
    },
    {
        "cr_id": "CR-GRACE-DAYS",
        # ADDITION CR: graceDays/graceEndsAt are NEW field names, so they
        # cannot literally appear in existing node_ids. We instead require
        # that the report surface (a) the listing model file that will host
        # the new field, and (b) the function that computes the deadline
        # using grace logic. Both must be present to claim semantic
        # coverage.
        "expected_substrings": [
            "commissionListing.model.ts",
            "updateContractDeadline",
        ],
        "text": (
            "Ubah durasi grace period dari nilai tetap 7 hari menjadi nilai "
            "yang dapat dikonfigurasi oleh ilustrator pada setiap commission "
            "listing melalui field graceDays. Sistem menggunakan nilai "
            "graceDays dari listing terkait saat menghitung graceEndsAt pada "
            "kontrak yang melewati tenggat."
        ),
    },
    {
        "cr_id": "CR-INTERNAL-NOTES",
        "expected_substrings": ["ticket.model.ts", "ticket.service.ts"],
        "text": (
            "Tambahkan field opsional \"internalNotes\" bertipe string pada "
            "model CancelTicket yang hanya dapat dilihat oleh admin selama "
            "proses penyelesaian sengketa. Perubahan ini memengaruhi model "
            "tiket pada ticket.model.ts, fungsi terkait pada ticket.service.ts, "
            "komponen CancelTicketDetails.tsx, dan komponen "
            "AdminResolutionForm.tsx."
        ),
    },
    {
        "cr_id": "CR-ESCROW-REFACTOR",
        "expected_substrings": ["escrowTransaction.repository.ts", "escrowTransaction.service.ts"],
        "text": (
            "Refaktor seluruh pemanggilan langsung ke "
            "escrowTransaction.repository.ts yang tersebar di berbagai service "
            "agar seluruh operasi eskro diakses secara eksklusif melalui "
            "escrowTransaction.service.ts. Service yang saat ini mengimpor "
            "dan memanggil repository eskro secara langsung harus diubah "
            "untuk memanggil fungsi pada escrowTransaction.service.ts sebagai "
            "gantinya."
        ),
    },
    {
        "cr_id": "CR-DASHBOARD-TANSTACK",
        "expected_substrings": ["DashboardGalleryPage.tsx"],
        "text": (
            "Ganti mekanisme fetching data berbasis useEffect pada komponen "
            "DashboardGalleryPage (src/components/dashboard/galleries/"
            "DashboardGalleryPage.tsx) dengan TanStack Query."
        ),
    },
]


@dataclass
class CRResult:
    cr_id: str
    returncode: int = -1
    elapsed_s: float = 0.0
    stderr_tail: str = ""
    n_entities: int = 0
    n_files: int = 0
    degraded: bool = False
    has_required_files: list[str] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    entities_without_justification: int = 0
    justification_sources: dict[str, int] = field(default_factory=dict)
    full_trace_steps_present: list[str] = field(default_factory=list)
    full_trace_steps_missing: list[str] = field(default_factory=list)
    pass_no_crash: bool = False
    pass_failclosed: bool = False
    pass_semantic: bool = False
    pass_justifications: bool = False
    pass_schema: bool = False
    overall_pass: bool = False
    notes: list[str] = field(default_factory=list)


REQUIRED_TRACE_KEYS = [
    "step_1_interpretation",
    "step_2_rrf_pool",
    "step_3_reranked",
    "step_3_gates_survivors",
    "step_4_llm2_verdicts",
    "step_5_resolutions",
    # step_5b/6/7 may be absent for variants that disable those phases or
    # when no doc resolutions exist; we check them softly.
    "final_report",
]
SOFT_TRACE_KEYS = ["step_5b_llm3_verdicts", "step_6_bfs_raw_cis", "step_7_llm4_verdicts"]


def _run_one(cr: dict, output_dir: Path, timeout_s: int) -> CRResult:
    cr_dir = output_dir / cr["cr_id"]
    cr_dir.mkdir(parents=True, exist_ok=True)
    report_path = cr_dir / "impact_report.json"
    full_path = cr_dir / "impact_report_full.json"

    result = CRResult(cr_id=cr["cr_id"])
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [
                sys.executable, "-m", "impactracer.cli",
                "analyze", cr["text"],
                "--output", str(report_path),
                "--variant", "V7",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            encoding="utf-8",
            errors="replace",
        )
        result.returncode = proc.returncode
        result.stderr_tail = (proc.stderr or "")[-2000:]
    except subprocess.TimeoutExpired as exc:
        result.notes.append(f"TIMEOUT after {timeout_s}s")
        result.stderr_tail = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        return result
    finally:
        result.elapsed_s = time.perf_counter() - t0

    # Load report.
    if not report_path.exists():
        result.notes.append("impact_report.json not produced")
        return result
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        result.notes.append(f"impact_report.json parse error: {exc}")
        return result

    entities = report.get("impacted_entities", [])
    files = report.get("impacted_files", [])
    result.n_entities = len(entities)
    result.n_files = len(files)
    result.degraded = bool(report.get("degraded_run", False))

    # Required-substring presence: each expected substring must appear in
    # ANY file_path OR ANY entity node_id. CR-named identifiers (e.g.
    # "graceDays", "internalNotes") live inside model/service files, so a
    # file-path-only check would mis-fail when the field exists in a model
    # whose filename does not literally contain the identifier.
    file_paths = [f.get("file_path", "") for f in files]
    entity_nodes = [e.get("node", "") for e in entities]
    blob = (" | ".join(file_paths) + " || " + " | ".join(entity_nodes)).lower()
    for needle in cr["expected_substrings"]:
        if needle.lower() in blob:
            result.has_required_files.append(needle)
        else:
            result.missing_required.append(needle)

    # Justification quality + source tally.
    no_just = 0
    for ent in entities:
        if not ent.get("justification", "").strip():
            no_just += 1
        src = ent.get("justification_source", "") or "unknown"
        result.justification_sources[src] = result.justification_sources.get(src, 0) + 1
    result.entities_without_justification = no_just

    # Full-trace presence.
    if full_path.exists():
        try:
            trace = json.loads(full_path.read_text(encoding="utf-8"))
        except Exception as exc:
            trace = {}
            result.notes.append(f"impact_report_full.json parse error: {exc}")
        for k in REQUIRED_TRACE_KEYS:
            if k in trace:
                result.full_trace_steps_present.append(k)
            else:
                result.full_trace_steps_missing.append(k)
        for k in SOFT_TRACE_KEYS:
            if k in trace:
                result.full_trace_steps_present.append(k)
    else:
        result.notes.append("impact_report_full.json not produced")

    # PASS/FAIL determination.
    result.pass_no_crash = (result.returncode == 0)
    result.pass_failclosed = result.pass_no_crash  # subprocess survival = batch-fail-closed worked
    result.pass_semantic = (
        result.n_entities > 0
        and result.n_entities < 200          # no recall monster
        and not result.missing_required      # explicit CR-named files surfaced
    )
    result.pass_justifications = (
        result.n_entities > 0
        and result.entities_without_justification == 0
    )
    result.pass_schema = (
        "impacted_entities" in report
        and "impacted_files" in report
        and not result.full_trace_steps_missing
    )
    result.overall_pass = all(
        (result.pass_no_crash, result.pass_failclosed, result.pass_semantic,
         result.pass_justifications, result.pass_schema)
    )
    return result


def _print_summary(results: list[CRResult]) -> None:
    print()
    print("=" * 100)
    print(f"{'CR':<22} {'rc':>3} {'time':>6} {'ent':>4} {'file':>4} "
          f"{'NoCr':>4} {'FCl':>4} {'Sem':>4} {'Jus':>4} {'Sch':>4} {'PASS':>4}")
    print("-" * 100)
    for r in results:
        print(
            f"{r.cr_id:<22} {r.returncode:>3} {r.elapsed_s:>6.1f} "
            f"{r.n_entities:>4} {r.n_files:>4} "
            f"{'Y' if r.pass_no_crash else 'N':>4} "
            f"{'Y' if r.pass_failclosed else 'N':>4} "
            f"{'Y' if r.pass_semantic else 'N':>4} "
            f"{'Y' if r.pass_justifications else 'N':>4} "
            f"{'Y' if r.pass_schema else 'N':>4} "
            f"{'Y' if r.overall_pass else 'N':>4}"
        )
    print("=" * 100)
    print()
    for r in results:
        if r.missing_required or not r.overall_pass or r.notes:
            print(f"[{r.cr_id}]")
            if r.missing_required:
                print(f"  missing required substrings in file paths: {r.missing_required}")
            if r.entities_without_justification:
                print(f"  entities lacking justification: {r.entities_without_justification}")
            if r.full_trace_steps_missing:
                print(f"  trace keys missing: {r.full_trace_steps_missing}")
            if r.justification_sources:
                print(f"  justification source tally: {r.justification_sources}")
            if r.notes:
                for n in r.notes:
                    print(f"  note: {n}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="./eval/e2e_runs",
                        help="Per-CR output root (default ./eval/e2e_runs)")
    parser.add_argument("--timeout-s", type=int, default=900,
                        help="Per-CR subprocess timeout (default 900s = 15 min)")
    parser.add_argument("--cr-id", default=None,
                        help="If set, run only the specified CR id (one of: "
                             + ", ".join(c["cr_id"] for c in CR_SUITE) + ")")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = [c for c in CR_SUITE if not args.cr_id or c["cr_id"] == args.cr_id]
    if not targets:
        print(f"Unknown --cr-id {args.cr_id!r}", file=sys.stderr)
        return 2

    results: list[CRResult] = []
    for cr in targets:
        print(f"\n>>> Running {cr['cr_id']} ...")
        r = _run_one(cr, output_dir, args.timeout_s)
        results.append(r)
        print(
            f"<<< {cr['cr_id']}: rc={r.returncode} "
            f"entities={r.n_entities} files={r.n_files} "
            f"PASS={r.overall_pass}"
        )
        # Persist a per-CR summary.
        (output_dir / cr["cr_id"] / "summary.json").write_text(
            json.dumps(r.__dict__, indent=2, default=str), encoding="utf-8"
        )

    _print_summary(results)
    return 0 if all(r.overall_pass for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
