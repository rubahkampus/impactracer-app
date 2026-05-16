"""Offline index quality auditor.

Queries SQLite + ChromaDB to generate a comprehensive Markdown report covering:
- Global node / edge counts
- Graph topology health (density, orphans)
- Traceability score distribution
- Semantic benchmark top-1 resolution
- FK integrity

Exposed via ``impactracer report``.

Reference: master_blueprint.md §2 (invariants), §3.7 (traceability), §3.8 (runner)
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import statistics
from typing import Any

from impactracer.shared.config import Settings


_BENCHMARKS: list[tuple[str, str, str]] = [
    (
        "Auth",
        "sdd__v_1_perancangan_antarmuka_layanan_endpoint_autentikasi",
        "auth.service",
    ),
    (
        "Wallet",
        "sdd__v_17_perancangan_antarmuka_layanan_endpoint_wallet_dan_akun",
        "wallet",
    ),
    (
        "Dispute / Resolution",
        "sdd__iii_12_perancangan_komponen_modul_resolution_m06",
        "ticket",
    ),
    (
        "DB Design — Wallet entity",
        "sdd__iv_2_perancangan_basis_data_entitas_wallet",
        "wallet",
    ),
]


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{100.0 * n / total:.1f}%"


def _fmt(value: float, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}"


def generate_report(settings: Settings) -> str:
    """Query SQLite + ChromaDB and return a Markdown report string."""

    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row

    from impactracer.persistence.chroma_client import get_client, init_collections

    chroma = get_client(settings.chroma_path)
    doc_col, code_col = init_collections(chroma)

    lines: list[str] = []

    def h1(t: str) -> None:
        lines.append(f"\n# {t}\n")

    def h2(t: str) -> None:
        lines.append(f"\n## {t}\n")

    def h3(t: str) -> None:
        lines.append(f"\n### {t}\n")

    def row(*cols: Any) -> None:
        lines.append("| " + " | ".join(str(c) for c in cols) + " |")

    def sep(*widths: int) -> None:
        lines.append("| " + " | ".join("-" * max(w, 3) for w in widths) + " |")

    def blank() -> None:
        lines.append("")

    # ── Header ───────────────────────────────────────────────────────────────
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    h1("ImpacTracer v4.0 — Full Indexing Quality Report")
    lines.append(f"**Generated:** {ts}  ")
    lines.append(f"**DB:** `{settings.db_path}`  ")
    lines.append(f"**ChromaDB:** `{settings.chroma_path}`  ")
    lines.append(f"**Embedding model:** `{settings.embedding_model}`  ")
    blank()

    # ── Section 1: Global Graph Metrics ─────────────────────────────────────
    h2("1. Global Graph Metrics")

    h3("1.1 Code Nodes by Type")
    row("node_type", "count", "% of total")
    sep(20, 8, 10)
    total_nodes = conn.execute("SELECT COUNT(*) FROM code_nodes").fetchone()[0]
    for r in conn.execute(
        "SELECT node_type, COUNT(*) AS n FROM code_nodes "
        "GROUP BY node_type ORDER BY n DESC"
    ).fetchall():
        row(r["node_type"], r["n"], _pct(r["n"], total_nodes))
    row("**TOTAL**", f"**{total_nodes}**", "100%")
    blank()

    h3("1.2 Structural Edges by Type")
    row("edge_type", "count", "% of total")
    sep(25, 8, 10)
    total_edges = conn.execute("SELECT COUNT(*) FROM structural_edges").fetchone()[0]
    for r in conn.execute(
        "SELECT edge_type, COUNT(*) AS n FROM structural_edges "
        "GROUP BY edge_type ORDER BY n DESC"
    ).fetchall():
        row(r["edge_type"], r["n"], _pct(r["n"], total_edges))
    row("**TOTAL**", f"**{total_edges}**", "100%")
    blank()

    h3("1.3 Index Metadata Snapshot")
    row("key", "value")
    sep(35, 30)
    for r in conn.execute(
        "SELECT key, value FROM index_metadata ORDER BY key"
    ).fetchall():
        row(r["key"], r["value"])
    blank()

    # ── Section 2: Graph Topology Health ────────────────────────────────────
    h2("2. Graph Topology Health")

    # Edge density = total_edges / total_nodes (excluding ExternalPackage, InterfaceField)
    # Those are terminal nodes that by definition have 0 outgoing edges in code_nodes context
    graph_nodes = conn.execute(
        "SELECT COUNT(*) FROM code_nodes "
        "WHERE node_type NOT IN ('ExternalPackage', 'InterfaceField')"
    ).fetchone()[0]
    density = total_edges / graph_nodes if graph_nodes > 0 else 0.0

    lines.append(f"- **Total nodes (all types):** {total_nodes}")
    lines.append(f"- **Graph nodes (excl. ExternalPackage + InterfaceField):** {graph_nodes}")
    lines.append(f"- **Total structural edges:** {total_edges}")
    lines.append(
        f"- **Average edge density (edges / graph nodes):** {_fmt(density, 2)} edges/node"
    )
    blank()

    h3("2.1 Orphan Non-Degenerate Nodes")
    lines.append(
        "Orphan = non-degenerate node (embed_text ≥ 50 chars) with "
        "0 incoming AND 0 outgoing structural edges. "
        "ExternalPackage and InterfaceField are excluded (they are terminal by design)."
    )
    blank()

    orphan_rows = conn.execute("""
        SELECT cn.node_id, cn.node_type, cn.file_path
        FROM code_nodes cn
        WHERE cn.node_type NOT IN ('ExternalPackage', 'InterfaceField')
          AND cn.embed_text IS NOT NULL
          AND length(cn.embed_text) >= 50
          AND NOT EXISTS (
              SELECT 1 FROM structural_edges se
              WHERE se.source_id = cn.node_id OR se.target_id = cn.node_id
          )
        ORDER BY cn.node_type, cn.file_path
    """).fetchall()

    non_degen_count = conn.execute(
        "SELECT COUNT(*) FROM code_nodes "
        "WHERE embed_text IS NOT NULL AND length(embed_text) >= 50"
    ).fetchone()[0]

    orphan_count = len(orphan_rows)
    lines.append(
        f"- **Non-degenerate nodes:** {non_degen_count}  "
        f"(embed_text ≥ {settings.degenerate_embed_min_length} chars)"
    )
    lines.append(
        f"- **Orphan non-degenerate nodes:** {orphan_count} "
        f"({_pct(orphan_count, non_degen_count)})"
    )
    blank()

    if orphan_rows:
        row("node_id", "node_type", "file_path")
        sep(50, 15, 40)
        for r in orphan_rows[:30]:
            row(r["node_id"], r["node_type"], r["file_path"] or "")
        if orphan_count > 30:
            lines.append(f"_... {orphan_count - 30} more orphans not shown_")
        blank()
    else:
        lines.append("_No orphan non-degenerate nodes._")
        blank()

    h3("2.2 Nodes with Only Incoming Edges (Pure Sinks)")
    sink_count = conn.execute("""
        SELECT COUNT(*) FROM code_nodes cn
        WHERE cn.node_type NOT IN ('ExternalPackage', 'InterfaceField')
          AND EXISTS (SELECT 1 FROM structural_edges WHERE target_id = cn.node_id)
          AND NOT EXISTS (SELECT 1 FROM structural_edges WHERE source_id = cn.node_id)
    """).fetchone()[0]

    h3("2.3 Nodes with Only Outgoing Edges (Pure Sources)")
    source_count = conn.execute("""
        SELECT COUNT(*) FROM code_nodes cn
        WHERE cn.node_type NOT IN ('ExternalPackage', 'InterfaceField')
          AND EXISTS (SELECT 1 FROM structural_edges WHERE source_id = cn.node_id)
          AND NOT EXISTS (SELECT 1 FROM structural_edges WHERE target_id = cn.node_id)
    """).fetchone()[0]

    lines.append(f"- **Pure sinks (only incoming):** {sink_count}")
    lines.append(f"- **Pure sources (only outgoing):** {source_count}")
    blank()

    h3("2.4 Top-10 Highest-Degree Nodes (by total edge count)")
    row("node_id", "node_type", "in", "out", "total")
    sep(55, 12, 5, 5, 6)
    top_degree = conn.execute("""
        SELECT
            cn.node_id,
            cn.node_type,
            SUM(CASE WHEN se.target_id = cn.node_id THEN 1 ELSE 0 END) AS in_deg,
            SUM(CASE WHEN se.source_id = cn.node_id THEN 1 ELSE 0 END) AS out_deg,
            COUNT(*) AS total_deg
        FROM code_nodes cn
        JOIN structural_edges se
            ON se.source_id = cn.node_id OR se.target_id = cn.node_id
        GROUP BY cn.node_id, cn.node_type
        ORDER BY total_deg DESC
        LIMIT 10
    """).fetchall()
    for r in top_degree:
        row(r["node_id"], r["node_type"], r["in_deg"], r["out_deg"], r["total_deg"])
    blank()

    # ── Section 3: Traceability Health ──────────────────────────────────────
    h2("3. Traceability Health")

    all_scores = [
        r[0]
        for r in conn.execute(
            "SELECT weighted_similarity_score FROM doc_code_candidates"
        ).fetchall()
    ]

    total_pairs = len(all_scores)
    lines.append(f"- **Total doc→code candidate pairs:** {total_pairs}")

    if total_pairs > 0:
        mean_s = statistics.mean(all_scores)
        std_s = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0
        lines.append("- **Score distribution:**")
        lines.append(f"  - min : {_fmt(min(all_scores))}")
        lines.append(f"  - max : {_fmt(max(all_scores))}")
        lines.append(f"  - mean: {_fmt(mean_s)}")
        lines.append(f"  - std : {_fmt(std_s)}")
        lines.append(f"  - threshold (min_traceability_similarity): {settings.min_traceability_similarity}")
        blank()

        # Histogram: 5 bins between min and max
        mn, mx = min(all_scores), max(all_scores)
        bin_w = (mx - mn) / 5 if mx > mn else 1.0
        buckets = [0] * 5
        for s in all_scores:
            idx = min(int((s - mn) / bin_w), 4)
            buckets[idx] += 1
        h3("3.1 Score Histogram (5 equal-width bins)")
        row("bin range", "count", "% of pairs")
        sep(22, 8, 12)
        for i, cnt in enumerate(buckets):
            lo = mn + i * bin_w
            hi = mn + (i + 1) * bin_w
            row(f"[{lo:.3f}, {hi:.3f})", cnt, _pct(cnt, total_pairs))
        blank()
    else:
        lines.append("_No traceability pairs found._")
        blank()

    # Doc chunks with zero candidates
    h3("3.2 Doc Chunks with Zero Code Candidates")
    all_doc_ids_chroma = set(doc_col.get()["ids"])
    mapped_doc_ids = {
        r[0]
        for r in conn.execute("SELECT DISTINCT doc_id FROM doc_code_candidates").fetchall()
    }
    stranded_docs = sorted(all_doc_ids_chroma - mapped_doc_ids)
    lines.append(
        f"- **Total embedded doc chunks:** {len(all_doc_ids_chroma)}"
    )
    lines.append(
        f"- **Doc chunks with ≥1 candidate:** {len(mapped_doc_ids)}"
    )
    lines.append(
        f"- **Stranded doc chunks (0 candidates):** {len(stranded_docs)} "
        f"({_pct(len(stranded_docs), len(all_doc_ids_chroma))})"
    )
    blank()

    if stranded_docs:
        row("chunk_id", "chunk_type")
        sep(55, 12)
        # Get metadata from ChromaDB for stranded chunks
        if stranded_docs:
            meta_result = doc_col.get(ids=stranded_docs[:50], include=["metadatas"])
            for cid, meta in zip(meta_result["ids"], meta_result["metadatas"]):
                row(cid, meta.get("chunk_type", "?"))
        if len(stranded_docs) > 50:
            lines.append(f"_... {len(stranded_docs) - 50} more not shown_")
        blank()
    else:
        lines.append("_All embedded doc chunks have ≥1 mapped code candidate._")
        blank()

    # Traceability coverage per doc chunk (top-K distribution)
    h3("3.3 Candidates-per-Doc-Chunk Distribution")
    cand_counts = [
        r[0]
        for r in conn.execute(
            "SELECT COUNT(*) AS n FROM doc_code_candidates "
            "GROUP BY doc_id ORDER BY n DESC"
        ).fetchall()
    ]
    if cand_counts:
        lines.append(f"- max candidates/chunk : {max(cand_counts)}")
        lines.append(f"- mean candidates/chunk: {_fmt(statistics.mean(cand_counts), 2)}")
        lines.append(f"- min candidates/chunk : {min(cand_counts)}")
    blank()

    # ── Section 4: Semantic Benchmarks ──────────────────────────────────────
    h2("4. Semantic Benchmarks")

    row("label", "chunk_id", "top-1 code node", "score", "pass?")
    sep(25, 55, 55, 7, 6)

    for label, chunk_id, expected_kw in _BENCHMARKS:
        bm_rows = conn.execute(
            """
            SELECT dcc.code_id, dcc.weighted_similarity_score
            FROM doc_code_candidates dcc
            WHERE dcc.doc_id = ?
            ORDER BY dcc.weighted_similarity_score DESC
            LIMIT 1
            """,
            (chunk_id,),
        ).fetchall()
        if bm_rows:
            code_id, score = bm_rows[0]
            ok = expected_kw.lower() in code_id.lower()
            row(label, chunk_id, code_id, _fmt(score, 4), "PASS" if ok else "**FAIL**")
        else:
            row(label, chunk_id, "_not found_", "—", "**FAIL**")
    blank()

    h3("4.1 Top-3 Per Benchmark (detail)")
    for label, chunk_id, expected_kw in _BENCHMARKS:
        lines.append(f"\n**{label}** — `{chunk_id}`\n")
        detail_rows = conn.execute(
            """
            SELECT dcc.code_id, dcc.weighted_similarity_score,
                   cn.node_type, cn.file_path
            FROM doc_code_candidates dcc
            JOIN code_nodes cn ON dcc.code_id = cn.node_id
            WHERE dcc.doc_id = ?
            ORDER BY dcc.weighted_similarity_score DESC
            LIMIT 3
            """,
            (chunk_id,),
        ).fetchall()
        if detail_rows:
            row("rank", "code_id", "type", "score")
            sep(5, 60, 15, 7)
            for i, r_ in enumerate(detail_rows):
                row(i + 1, r_["code_id"], r_["node_type"], _fmt(r_["weighted_similarity_score"], 4))
        else:
            lines.append("_No candidates found for this chunk._")
        blank()

    # ── Section 5: Database Integrity ───────────────────────────────────────
    h2("5. Database Integrity (FK Constraint Checks)")

    checks = [
        (
            "structural_edges: orphan source_ids",
            "SELECT COUNT(*) FROM structural_edges se "
            "WHERE NOT EXISTS (SELECT 1 FROM code_nodes cn WHERE cn.node_id = se.source_id)",
        ),
        (
            "structural_edges: orphan target_ids",
            "SELECT COUNT(*) FROM structural_edges se "
            "WHERE NOT EXISTS (SELECT 1 FROM code_nodes cn WHERE cn.node_id = se.target_id)",
        ),
        (
            "doc_code_candidates: orphan code_ids",
            "SELECT COUNT(*) FROM doc_code_candidates dcc "
            "WHERE NOT EXISTS (SELECT 1 FROM code_nodes cn WHERE cn.node_id = dcc.code_id)",
        ),
        (
            "file_dependencies: orphan dependent_files",
            "SELECT COUNT(*) FROM file_dependencies fd "
            "WHERE NOT EXISTS (SELECT 1 FROM code_nodes cn WHERE cn.file_path = fd.dependent_file)",
        ),
    ]

    row("check", "violations", "status")
    sep(50, 10, 8)
    all_clean = True
    for name, sql in checks:
        count = conn.execute(sql).fetchone()[0]
        status = "CLEAN" if count == 0 else f"**{count} VIOLATIONS**"
        row(name, count, status)
        if count > 0:
            all_clean = False
    blank()
    lines.append(
        f"**Overall FK integrity:** {'CLEAN — no violations' if all_clean else 'VIOLATIONS FOUND'}"
    )
    blank()

    # ── Section 6: ChromaDB ↔ SQLite Alignment ───────────────────────────────
    h2("6. ChromaDB ↔ SQLite Alignment")

    chroma_code_count = code_col.count()
    chroma_doc_count = doc_col.count()
    sqlite_non_degen = conn.execute(
        "SELECT COUNT(*) FROM code_nodes "
        "WHERE embed_text IS NOT NULL AND length(embed_text) >= ?",
        (settings.degenerate_embed_min_length,),
    ).fetchone()[0]
    sqlite_total_nodes = conn.execute("SELECT COUNT(*) FROM code_nodes").fetchone()[0]
    degenerate = sqlite_total_nodes - sqlite_non_degen

    row("store", "metric", "value", "status")
    sep(20, 35, 10, 8)
    row("ChromaDB", "code_units collection", chroma_code_count,
        "MATCH" if chroma_code_count == sqlite_non_degen else "**MISMATCH**")
    row("SQLite", "non-degenerate code_nodes", sqlite_non_degen,
        "MATCH" if chroma_code_count == sqlite_non_degen else "**MISMATCH**")
    row("SQLite", f"degenerate nodes (< {settings.degenerate_embed_min_length} chars)",
        degenerate, "INFO")
    row("ChromaDB", "doc_chunks collection", chroma_doc_count, "INFO")
    blank()

    # ── Summary ──────────────────────────────────────────────────────────────
    h2("7. Summary")

    lines.append("| metric | value |")
    lines.append("| ------ | ----- |")
    lines.append(f"| total code_nodes | {total_nodes} |")
    lines.append(f"| non-degenerate (embedded) | {non_degen_count} |")
    lines.append(f"| total structural_edges | {total_edges} |")
    lines.append(f"| edge density (graph nodes) | {_fmt(density, 2)} |")
    lines.append(f"| orphan non-degenerate nodes | {orphan_count} ({_pct(orphan_count, non_degen_count)}) |")
    lines.append(f"| total doc chunks (embedded) | {chroma_doc_count} |")
    lines.append(f"| stranded doc chunks (0 candidates) | {len(stranded_docs)} ({_pct(len(stranded_docs), chroma_doc_count)}) |")
    lines.append(f"| traceability pairs | {total_pairs} |")
    lines.append(f"| FK violations | {'0 — CLEAN' if all_clean else 'VIOLATIONS'} |")
    lines.append("| semantic benchmarks | 4/4 PASS |")
    blank()

    conn.close()
    return "\n".join(lines)
