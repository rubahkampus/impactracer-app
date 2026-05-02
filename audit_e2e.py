"""
E2E Orchestration Audit Script — Sprint 7 Crucible.
Run AFTER the cold start completes.
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np

DB = Path("./data/impactracer.db")
CHROMA = Path("./data/chroma_store")
CITRAKARA = Path(r"C:\Users\Haidar\Documents\thesis\citrakara")

BENCHMARKS = [
    {
        "label": "Auth (sdd__v_1)",
        "chunk_id_pattern": "sdd__v_1_%",
        "expected_top1_contains": "auth.service",
    },
    {
        "label": "Wallet (sdd__v_17)",
        "chunk_id_pattern": "sdd__v_17_%",
        "expected_top1_contains": "wallet",
    },
    {
        "label": "Dispute/Sengketa (sdd__iii_12)",
        "chunk_id_pattern": "sdd__iii_12_%",
        "expected_top1_contains": "ticket",
    },
    {
        "label": "DB Design/Wallet entity (sdd__iv_2)",
        "chunk_id_pattern": "sdd__iv_2_%",
        "expected_top1_contains": "wallet",
    },
]


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_paths(conn):
    section("PATH ALIGNMENT AUDIT")

    # 1. SQLite file_hashes — should be absolute posix
    sample_hashes = conn.execute(
        "SELECT file_path FROM file_hashes LIMIT 5"
    ).fetchall()
    print("\n[file_hashes] Sample paths (expect absolute posix):")
    for r in sample_hashes:
        print(f"  {r[0]}")

    # 2. code_nodes.file_path — should be src/... relative
    sample_nodes = conn.execute(
        "SELECT file_path FROM code_nodes WHERE node_type='File' LIMIT 5"
    ).fetchall()
    print("\n[code_nodes.file_path] Sample paths (expect src/... relative):")
    for r in sample_nodes:
        print(f"  {r[0]}")

    # 3. file_dependencies — should be src/... relative on both columns
    sample_deps = conn.execute(
        "SELECT dependent_file, target_file FROM file_dependencies LIMIT 5"
    ).fetchall()
    print("\n[file_dependencies] Sample rows (expect src/... on both columns):")
    for r in sample_deps:
        print(f"  dependent={r[0]}  target={r[1]}")

    # 4. Check for any absolute paths in code_nodes (leakage)
    abs_leak = conn.execute(
        "SELECT COUNT(*) FROM code_nodes WHERE file_path LIKE 'C:%' OR file_path LIKE '/Users/%'"
    ).fetchone()[0]
    print(f"\n[CRITICAL] Absolute path leakage in code_nodes.file_path: {abs_leak} rows", end=" ")
    print("✓ CLEAN" if abs_leak == 0 else "✗ LEAKED — BUG")

    # 5. Check ExternalPackage node_ids (should be package names, not paths)
    ext_sample = conn.execute(
        "SELECT node_id FROM code_nodes WHERE node_type='ExternalPackage' LIMIT 5"
    ).fetchall()
    print("\n[code_nodes] ExternalPackage sample node_ids:")
    for r in ext_sample:
        print(f"  {r[0]}")

    # 6. Cross-check: ts file_hashes count vs code_nodes File count
    ts_hash_count = conn.execute(
        "SELECT COUNT(*) FROM file_hashes WHERE file_path LIKE '%.ts' OR file_path LIKE '%.tsx'"
    ).fetchone()[0]
    md_hash_count = conn.execute(
        "SELECT COUNT(*) FROM file_hashes WHERE file_path LIKE '%.md'"
    ).fetchone()[0]
    file_node_count = conn.execute(
        "SELECT COUNT(*) FROM code_nodes WHERE node_type='File'"
    ).fetchone()[0]
    print(f"\n[COUNTS] file_hashes: ts={ts_hash_count}, md={md_hash_count}")
    print(f"[COUNTS] code_nodes[File]: {file_node_count}")
    match = ts_hash_count == file_node_count
    print(f"[CHECK] ts hashes == File nodes: {'✓ MATCH' if match else '✗ MISMATCH'}")


def check_traceability(conn):
    section("SEMANTIC REGRESSION CHECK — 4 BENCHMARK CHUNKS")

    for bm in BENCHMARKS:
        # Find the chunk_id matching the pattern
        row = conn.execute(
            "SELECT doc_id FROM doc_code_candidates WHERE doc_id LIKE ? "
            "ORDER BY weighted_similarity_score DESC LIMIT 1",
            (bm["chunk_id_pattern"],)
        ).fetchone()

        if row is None:
            print(f"\n[{bm['label']}] ✗ NO CANDIDATES FOUND")
            continue

        chunk_id = row[0]

        # Top-3 for this chunk
        rows = conn.execute(
            "SELECT dcc.code_id, dcc.weighted_similarity_score, "
            "cn.node_type, cn.file_path "
            "FROM doc_code_candidates dcc "
            "JOIN code_nodes cn ON dcc.code_id = cn.node_id "
            "WHERE dcc.doc_id = ? "
            "ORDER BY dcc.weighted_similarity_score DESC LIMIT 3",
            (chunk_id,)
        ).fetchall()

        print(f"\n[{bm['label']}] chunk_id={chunk_id}")
        if not rows:
            print("  ✗ NO JOINED ROWS — FK broken or code_id missing")
            continue

        for i, (code_id, score, ntype, fpath) in enumerate(rows):
            marker = "→ TOP-1" if i == 0 else f"    #{i+1}"
            print(f"  {marker}  [{score:.4f}]  {code_id}  ({ntype})")

        top1_id = rows[0][0]
        expected = bm["expected_top1_contains"]
        ok = expected.lower() in top1_id.lower()
        print(f"  [CHECK] top-1 contains '{expected}': {'✓ PASS' if ok else '✗ FAIL'}")


def check_chroma_alignment(conn):
    section("CHROMADB ↔ SQLITE ALIGNMENT")

    from impactracer.persistence.chroma_client import get_client, init_collections
    client = get_client(str(CHROMA))
    doc_col, code_col = init_collections(client)

    chroma_code_count = code_col.count()
    chroma_doc_count = doc_col.count()

    sqlite_non_degen = conn.execute(
        "SELECT COUNT(*) FROM code_nodes "
        "WHERE embed_text IS NOT NULL AND length(embed_text) >= 50"
    ).fetchone()[0]

    print(f"\n[ChromaDB] code_units count      : {chroma_code_count}")
    print(f"[SQLite]   non-degen code_nodes  : {sqlite_non_degen}")
    ok = chroma_code_count == sqlite_non_degen
    print(f"[CHECK]    alignment              : {'✓ MATCH' if ok else '✗ MISMATCH'}")

    print(f"\n[ChromaDB] doc_chunks count      : {chroma_doc_count}")

    # Sample ChromaDB doc metadata source_file format
    docs = doc_col.get(limit=3, include=["metadatas"])
    print("\n[ChromaDB doc_chunks] Sample source_file metadata:")
    for meta in docs.get("metadatas", []):
        print(f"  {meta.get('source_file', 'MISSING')}")

    # Check doc source_file is absolute posix (should match file_hashes format)
    docs_all = doc_col.get(include=["metadatas"])
    bad_doc_paths = [
        m.get("source_file", "")
        for m in docs_all.get("metadatas", [])
        if not Path(m.get("source_file", "")).is_absolute()
    ]
    print(f"\n[CHECK] doc source_file absolute: {'✓ ALL ABSOLUTE' if not bad_doc_paths else f'✗ {len(bad_doc_paths)} RELATIVE PATHS'}")


def check_fk_integrity(conn):
    section("FOREIGN KEY INTEGRITY")

    bad_src = conn.execute(
        "SELECT COUNT(*) FROM structural_edges se "
        "WHERE NOT EXISTS (SELECT 1 FROM code_nodes cn WHERE cn.node_id = se.source_id)"
    ).fetchone()[0]

    bad_tgt = conn.execute(
        "SELECT COUNT(*) FROM structural_edges se "
        "WHERE NOT EXISTS (SELECT 1 FROM code_nodes cn WHERE cn.node_id = se.target_id)"
    ).fetchone()[0]

    bad_dcc = conn.execute(
        "SELECT COUNT(*) FROM doc_code_candidates dcc "
        "WHERE NOT EXISTS (SELECT 1 FROM code_nodes cn WHERE cn.node_id = dcc.code_id)"
    ).fetchone()[0]

    print(f"\n[structural_edges] orphan source_ids : {bad_src} {'✓' if bad_src == 0 else '✗ BUG'}")
    print(f"[structural_edges] orphan target_ids : {bad_tgt} {'✓' if bad_tgt == 0 else '✗ BUG'}")
    print(f"[doc_code_candidates] orphan code_ids: {bad_dcc} {'✓' if bad_dcc == 0 else '✗ BUG'}")


def main():
    if not DB.exists():
        print("ERROR: DB not found. Run the indexer first.")
        sys.exit(1)

    conn = sqlite3.connect(str(DB))

    check_paths(conn)
    check_traceability(conn)
    check_chroma_alignment(conn)
    check_fk_integrity(conn)

    section("SUMMARY COUNTS")
    for table in ["code_nodes", "structural_edges", "doc_code_candidates", "file_hashes"]:
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {n}")

    conn.close()


if __name__ == "__main__":
    main()
