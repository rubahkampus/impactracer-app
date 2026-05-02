"""
Sprint 7 Incremental Ghost Test.
Run AFTER audit_e2e.py passes. Tests:
  1. No-change re-run (0 files reindexed)
  2. 1-file-changed re-run (correct detection + FK integrity)
"""
import sqlite3
import sys
import time
from pathlib import Path

from impactracer.indexer.runner import run_indexing
from impactracer.shared.config import get_settings

CITRAKARA = Path(r"C:\Users\Haidar\Documents\thesis\citrakara")
TARGET_FILE = CITRAKARA / "src" / "lib" / "services" / "wallet.service.ts"
DB_PATH = "./data/impactracer.db"


def run_with_timing(settings, force=False, label=""):
    t = time.perf_counter()
    stats = run_indexing(CITRAKARA, settings, force=force)
    elapsed = time.perf_counter() - t
    print(f"\n[{label}]")
    print(f"  files_scanned   : {stats['files_scanned']}")
    print(f"  files_reindexed : {stats['files_reindexed']}")
    print(f"  code_nodes      : {stats['code_nodes']}")
    print(f"  edges           : {stats['edges']}")
    print(f"  elapsed         : {elapsed:.1f}s")
    return stats


def check_fk(conn, label):
    bad = conn.execute(
        "SELECT COUNT(*) FROM structural_edges se "
        "WHERE NOT EXISTS (SELECT 1 FROM code_nodes cn WHERE cn.node_id = se.source_id)"
    ).fetchone()[0]
    bad2 = conn.execute(
        "SELECT COUNT(*) FROM doc_code_candidates dcc "
        "WHERE NOT EXISTS (SELECT 1 FROM code_nodes cn WHERE cn.node_id = dcc.code_id)"
    ).fetchone()[0]
    ok = bad == 0 and bad2 == 0
    print(f"  FK integrity [{label}]: {'✓ CLEAN' if ok else f'✗ BROKEN (edges={bad}, dcc={bad2})'}")
    return ok


def main():
    settings = get_settings()

    print("=" * 60)
    print("  INCREMENTAL GHOST TEST")
    print("=" * 60)

    # --- Run 2: No-change ---
    print("\n--- Run 2: No changes to repo ---")
    stats2 = run_with_timing(settings, force=False, label="no-change")
    assert stats2["files_reindexed"] == 0, (
        f"FAIL: expected 0 reindexed, got {stats2['files_reindexed']}"
    )
    print("  [CHECK] 0 files reindexed: ✓ PASS")

    conn = sqlite3.connect(DB_PATH)
    cands_before = conn.execute("SELECT COUNT(*) FROM doc_code_candidates").fetchone()[0]
    print(f"  [CHECK] doc_code_candidates: {cands_before}")
    check_fk(conn, "after no-change")
    conn.close()

    # --- Run 3: 1-file change ---
    print("\n--- Run 3: Modify wallet.service.ts ---")
    if not TARGET_FILE.exists():
        print(f"  WARNING: {TARGET_FILE} not found, skipping modification test")
        return

    original = TARGET_FILE.read_bytes()
    TARGET_FILE.write_bytes(original + b"\n// audit-marker\n")
    print(f"  Modified: {TARGET_FILE.name}")

    try:
        stats3 = run_with_timing(settings, force=False, label="1-file-changed")

        conn = sqlite3.connect(DB_PATH)

        # Check the modification was detected
        assert stats3["files_reindexed"] >= 1, (
            f"FAIL: expected >=1 reindexed, got {stats3['files_reindexed']}"
        )
        print(f"  [CHECK] files_reindexed >= 1: ✓ PASS ({stats3['files_reindexed']} files)")

        # Check wallet.service.ts nodes are present (not purged)
        wallet_nodes = conn.execute(
            "SELECT COUNT(*) FROM code_nodes WHERE file_path LIKE '%wallet.service%'"
        ).fetchone()[0]
        print(f"  [CHECK] wallet.service.ts nodes after reindex: {wallet_nodes}")

        # Check edges for wallet.service are present
        wallet_edges = conn.execute(
            "SELECT COUNT(*) FROM structural_edges se "
            "JOIN code_nodes cn ON se.source_id = cn.node_id "
            "WHERE cn.file_path LIKE '%wallet.service%'"
        ).fetchone()[0]
        print(f"  [CHECK] wallet.service.ts outgoing edges: {wallet_edges}")

        # FK check after modification
        check_fk(conn, "after 1-file-change")

        # Check reverse deps were re-extracted
        rev_deps = conn.execute(
            "SELECT DISTINCT dependent_file FROM file_dependencies "
            "WHERE target_file LIKE '%wallet.service%'"
        ).fetchall()
        print(f"  [CHECK] files that import wallet.service (reverse deps): {len(rev_deps)}")
        for r in rev_deps[:5]:
            print(f"    {r[0]}")

        conn.close()

    finally:
        # Restore the file
        TARGET_FILE.write_bytes(original)
        print(f"\n  Restored: {TARGET_FILE.name}")


if __name__ == "__main__":
    main()
