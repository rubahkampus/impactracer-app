"""
final_audit.py ? Sprint 7.5 Final Index Readiness Audit
Queries the live SQLite DB and prints a structured report covering:
  1. Edge distribution by type
  2. CLIENT_API_CALLS bridge analysis
  3. Orphan non-degenerate node analysis
  4. Traceability coverage statistics
  5. Wallet model -> service -> API route -> UI component path check
  6. Cross-layer reachability matrix
  7. BFS seed quality projection
"""

from __future__ import annotations

import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "impactracer.db"


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def section(title: str) -> None:
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def subsection(title: str) -> None:
    print(f"\n--- {title} ---")


# ???????????????????????????????????????
# 1. Edge distribution
# ???????????????????????????????????????

def edge_distribution(conn: sqlite3.Connection) -> dict[str, int]:
    section("1. EDGE DISTRIBUTION BY TYPE")
    rows = conn.execute(
        "SELECT edge_type, COUNT(*) AS cnt FROM structural_edges "
        "GROUP BY edge_type ORDER BY cnt DESC"
    ).fetchall()
    total = sum(r["cnt"] for r in rows)
    print(f"\n{'Edge Type':<30} {'Count':>8}  {'% of Total':>12}")
    print("-" * 55)
    by_type: dict[str, int] = {}
    for r in rows:
        pct = r["cnt"] / total * 100
        print(f"{r['edge_type']:<30} {r['cnt']:>8}  {pct:>11.1f}%")
        by_type[r["edge_type"]] = r["cnt"]
    print(f"{'TOTAL':<30} {total:>8}")
    return by_type


# ???????????????????????????????????????
# 2. CLIENT_API_CALLS bridge: frontend -> API route coverage
# ???????????????????????????????????????

def client_api_calls_bridge(conn: sqlite3.Connection) -> None:
    section("2. CLIENT_API_CALLS BRIDGE ANALYSIS (Frontend->Backend CIA)")

    # How many distinct sources and targets?
    bridge = conn.execute(
        "SELECT e.source_id, e.target_id, "
        "       sn.file_classification AS src_class, "
        "       tn.file_classification AS tgt_class "
        "FROM structural_edges e "
        "JOIN code_nodes sn ON sn.node_id = e.source_id "
        "JOIN code_nodes tn ON tn.node_id = e.target_id "
        "WHERE e.edge_type = 'CLIENT_API_CALLS'"
    ).fetchall()

    print(f"\nTotal CLIENT_API_CALLS edges: {len(bridge)}")

    src_classes: dict[str | None, int] = defaultdict(int)
    tgt_classes: dict[str | None, int] = defaultdict(int)
    for r in bridge:
        src_classes[r["src_class"]] += 1
        tgt_classes[r["tgt_class"]] += 1

    subsection("Source node classifications")
    for cls, cnt in sorted(src_classes.items(), key=lambda x: -x[1]):
        print(f"  {cls or 'NULL':<25} {cnt}")

    subsection("Target node classifications")
    for cls, cnt in sorted(tgt_classes.items(), key=lambda x: -x[1]):
        print(f"  {cls or 'NULL':<25} {cnt}")

    # Show sample edges
    subsection("Sample CLIENT_API_CALLS edges (up to 10)")
    for r in bridge[:10]:
        print(f"  {r['source_id']}")
        print(f"    -> {r['target_id']}")

    # Coverage: how many API_ROUTE nodes are reachable via CLIENT_API_CALLS?
    api_route_count = conn.execute(
        "SELECT COUNT(*) FROM code_nodes WHERE file_classification='API_ROUTE'"
    ).fetchone()[0]
    api_route_targeted = conn.execute(
        "SELECT COUNT(DISTINCT e.target_id) "
        "FROM structural_edges e "
        "JOIN code_nodes n ON n.node_id = e.target_id "
        "WHERE e.edge_type='CLIENT_API_CALLS' "
        "AND n.file_classification='API_ROUTE'"
    ).fetchone()[0]
    print(f"\nAPI_ROUTE nodes total:           {api_route_count}")
    print(f"API_ROUTE nodes reachable via CLIENT_API_CALLS: {api_route_targeted}")
    if api_route_count:
        print(f"Bridge coverage:                 {api_route_targeted/api_route_count*100:.1f}%")


# ???????????????????????????????????????
# 3. Orphan analysis
# ???????????????????????????????????????

def orphan_analysis(conn: sqlite3.Connection) -> None:
    section("3. ORPHAN ANALYSIS (Non-Degenerate Nodes, 0 In + 0 Out Edges)")

    # Non-degenerate = embed_text length >= 50
    non_deg = conn.execute(
        "SELECT node_id, node_type, file_classification, file_path "
        "FROM code_nodes "
        "WHERE LENGTH(embed_text) >= 50 "
        "  AND node_type NOT IN ('ExternalPackage','InterfaceField')"
    ).fetchall()
    non_deg_ids = {r["node_id"] for r in non_deg}
    total_non_deg = len(non_deg_ids)

    # Nodes with any edge (source or target)
    connected = conn.execute(
        "SELECT DISTINCT source_id AS nid FROM structural_edges "
        "UNION SELECT DISTINCT target_id FROM structural_edges"
    ).fetchall()
    connected_ids = {r["nid"] for r in connected}

    orphans = [r for r in non_deg if r["node_id"] not in connected_ids]

    print(f"\nTotal non-degenerate nodes (excl. ExternalPackage/InterfaceField): {total_non_deg}")
    print(f"Orphan non-degenerate nodes (0 in + 0 out): {len(orphans)} ({len(orphans)/total_non_deg*100:.1f}%)")

    # Classify orphans by type and classification
    type_counts: dict[str, int] = defaultdict(int)
    class_counts: dict[str | None, int] = defaultdict(int)
    for r in orphans:
        type_counts[r["node_type"]] += 1
        class_counts[r["file_classification"]] += 1

    subsection("Orphan breakdown by node_type")
    for nt, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {nt:<20} {cnt}")

    subsection("Orphan breakdown by file_classification")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls or 'NULL':<25} {cnt}")

    # Specifically: orphaned API route handler functions (entry points vs truly stranded)
    api_orphans = [r for r in orphans if r["file_classification"] == "API_ROUTE"]
    ui_orphans  = [r for r in orphans if r["file_classification"] in ("UI_COMPONENT","PAGE_COMPONENT")]
    print(f"\nOrphaned API_ROUTE functions (entry points ? expected): {len(api_orphans)}")
    print(f"Orphaned UI/PAGE functions (potential BFS gap):          {len(ui_orphans)}")

    # Show orphaned UI/PAGE components (these are the worrying ones)
    if ui_orphans:
        subsection("Orphaned UI/PAGE functions (up to 20)")
        for r in ui_orphans[:20]:
            print(f"  [{r['node_type']}] {r['node_id']}")

    # Nodes with ONLY outgoing edges (pure sources ? orphans of incoming path)
    only_out = conn.execute(
        "SELECT COUNT(*) FROM ("
        "  SELECT source_id AS nid FROM structural_edges "
        "  EXCEPT SELECT target_id FROM structural_edges"
        ") WHERE nid IN (SELECT node_id FROM code_nodes WHERE LENGTH(embed_text)>=50)"
    ).fetchone()[0]
    only_in = conn.execute(
        "SELECT COUNT(*) FROM ("
        "  SELECT target_id AS nid FROM structural_edges "
        "  EXCEPT SELECT source_id FROM structural_edges"
        ") WHERE nid IN (SELECT node_id FROM code_nodes WHERE LENGTH(embed_text)>=50)"
    ).fetchone()[0]
    print(f"\nNon-degenerate pure sources (only outgoing):  {only_out}")
    print(f"Non-degenerate pure sinks   (only incoming):  {only_in}")


# ???????????????????????????????????????
# 4. Traceability coverage
# ???????????????????????????????????????

def traceability_coverage(conn: sqlite3.Connection) -> None:
    section("4. TRACEABILITY COVERAGE (doc_code_candidates)")

    # Total distinct doc chunks known to traceability
    all_doc_ids = conn.execute(
        "SELECT DISTINCT doc_id FROM doc_code_candidates"
    ).fetchall()
    mapped_doc_ids = {r["doc_id"] for r in all_doc_ids}

    # Total pairs, score stats
    all_scores = [
        r[0] for r in conn.execute(
            "SELECT weighted_similarity_score FROM doc_code_candidates"
        ).fetchall()
    ]
    total_pairs = len(all_scores)

    print(f"\nTotal doc->code candidate pairs: {total_pairs}")
    print(f"Distinct doc chunks with >= 1 candidate: {len(mapped_doc_ids)}")

    if all_scores:
        mean_s = statistics.mean(all_scores)
        std_s  = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0
        median_s = statistics.median(all_scores)
        print(f"Score stats ? min:{min(all_scores):.4f}  max:{max(all_scores):.4f}  "
              f"mean:{mean_s:.4f}  median:{median_s:.4f}  std:{std_s:.4f}")

    # Score histogram (10 bins)
    subsection("Score histogram (10 bins, 0.40?0.75)")
    bin_min, bin_max, n_bins = 0.40, 0.75, 7
    bin_width = (bin_max - bin_min) / n_bins
    bins = [0] * n_bins
    for s in all_scores:
        idx = min(int((s - bin_min) / bin_width), n_bins - 1)
        if 0 <= idx < n_bins:
            bins[idx] += 1
    for i in range(n_bins):
        lo = bin_min + i * bin_width
        hi = lo + bin_width
        bar = "#" * (bins[i] // 20)
        print(f"  [{lo:.3f},{hi:.3f})  {bins[i]:>5}  {bar}")

    # Per-doc-chunk candidate count distribution
    per_chunk = conn.execute(
        "SELECT doc_id, COUNT(*) AS cnt, "
        "       MAX(weighted_similarity_score) AS top_score "
        "FROM doc_code_candidates GROUP BY doc_id"
    ).fetchall()
    cnts = [r["cnt"] for r in per_chunk]
    subsection("Candidates per doc chunk")
    print(f"  max:    {max(cnts)}")
    print(f"  median: {statistics.median(cnts):.1f}")
    print(f"  mean:   {statistics.mean(cnts):.2f}")
    print(f"  min:    {min(cnts)}")

    # Top scores per doc chunk ? show bottom 10 (weakest coverage)
    subsection("Bottom-10 doc chunks by top-1 score (weakest semantic bridge)")
    sorted_chunks = sorted(per_chunk, key=lambda r: r["top_score"])
    for r in sorted_chunks[:10]:
        print(f"  {r['doc_id'][:60]:<60}  top={r['top_score']:.4f}  n={r['cnt']}")

    # Chunk type breakdown
    subsection("Candidate coverage by chunk_type (from doc_id prefix heuristic)")
    type_map: dict[str, list[float]] = defaultdict(list)
    for r in conn.execute(
        "SELECT dcc.doc_id, dcc.weighted_similarity_score "
        "FROM doc_code_candidates dcc"
    ).fetchall():
        # Infer chunk type from the doc_id suffix pattern
        # srs__ = SRS chunks, sdd__ = SDD chunks
        prefix = "SRS" if r["doc_id"].startswith("srs__") else "SDD"
        type_map[prefix].append(r["weighted_similarity_score"])
    for prefix, scores in sorted(type_map.items()):
        print(f"  {prefix}: {len(scores)} pairs, "
              f"mean={statistics.mean(scores):.4f}, "
              f"std={statistics.stdev(scores) if len(scores)>1 else 0.0:.4f}")


# ???????????????????????????????????????
# 5. End-to-end path: wallet model -> service -> API route -> UI component
# ???????????????????????????????????????

def e2e_path_check(conn: sqlite3.Connection) -> None:
    section("5. END-TO-END PATH CHECK: wallet.model -> service -> API -> UI")

    # Seed node: the wallet model file
    seed = "src/lib/db/models/wallet.model.ts"

    # Walk 1-hop: what uses the wallet model (IMPORTS/TYPED_BY/CALLS into it)?
    hop1 = conn.execute(
        "SELECT e.source_id, e.edge_type, n.file_classification "
        "FROM structural_edges e "
        "JOIN code_nodes n ON n.node_id = e.source_id "
        "WHERE e.target_id LIKE ? AND e.source_id NOT LIKE ?",
        (f"{seed}%", f"{seed}%"),
    ).fetchall()

    print(f"\nSeed: {seed}")
    print(f"Hop 1 ? nodes pointing INTO seed (any edge): {len(hop1)}")

    hop1_ids = {r["source_id"] for r in hop1}
    subsection("Hop 1 sources (first 20)")
    for r in hop1[:20]:
        print(f"  [{r['edge_type']:<25}] {r['source_id']}")

    # Hop 2: from hop1, what further nodes point into them?
    if hop1_ids:
        placeholders = ",".join("?" * len(hop1_ids))
        hop2 = conn.execute(
            f"SELECT e.source_id, e.edge_type, n.file_classification "
            f"FROM structural_edges e "
            f"JOIN code_nodes n ON n.node_id = e.source_id "
            f"WHERE e.target_id IN ({placeholders}) "
            f"AND e.source_id NOT IN ({placeholders})",
            list(hop1_ids) + list(hop1_ids),
        ).fetchall()
        hop2_ids = {r["source_id"] for r in hop2}
        print(f"\nHop 2 ? nodes pointing INTO Hop 1 nodes: {len(hop2)}")

        # Check what classifications are reachable at hop 2
        hop2_classes: dict[str | None, int] = defaultdict(int)
        for r in hop2:
            hop2_classes[r["file_classification"]] += 1
        subsection("Hop 2 classification distribution")
        for cls, cnt in sorted(hop2_classes.items(), key=lambda x: -x[1]):
            print(f"  {cls or 'NULL':<25} {cnt}")

        # Hop 3
        if hop2_ids:
            placeholders2 = ",".join("?" * len(hop2_ids))
            hop3 = conn.execute(
                f"SELECT e.source_id, e.edge_type, n.file_classification "
                f"FROM structural_edges e "
                f"JOIN code_nodes n ON n.node_id = e.source_id "
                f"WHERE e.target_id IN ({placeholders2}) "
                f"AND e.source_id NOT IN ({placeholders2})",
                list(hop2_ids) + list(hop2_ids),
            ).fetchall()
            hop3_classes: dict[str | None, int] = defaultdict(int)
            for r in hop3:
                hop3_classes[r["file_classification"]] += 1
            print(f"\nHop 3 ? nodes pointing INTO Hop 2 nodes: {len(hop3)}")
            subsection("Hop 3 classification distribution")
            for cls, cnt in sorted(hop3_classes.items(), key=lambda x: -x[1]):
                print(f"  {cls or 'NULL':<25} {cnt}")

            ui_at_hop3 = [r for r in hop3 if r["file_classification"] in ("UI_COMPONENT", "PAGE_COMPONENT")]
            if ui_at_hop3:
                print(f"\n?  UI/PAGE nodes reached at hop 3: {len(ui_at_hop3)}")
                for r in ui_at_hop3[:10]:
                    print(f"   [{r['edge_type']:<25}] {r['source_id']}")
            else:
                print("\n?  No UI/PAGE nodes reached at hop 3")
    else:
        print("  No edges found into seed ? path check fails immediately.")


# ???????????????????????????????????????
# 6. Cross-layer reachability matrix
# ???????????????????????????????????????

def reachability_matrix(conn: sqlite3.Connection) -> None:
    section("6. CROSS-LAYER REACHABILITY MATRIX (1-hop edge type ? classification)")

    rows = conn.execute(
        "SELECT e.edge_type, "
        "       sn.file_classification AS src_class, "
        "       tn.file_classification AS tgt_class, "
        "       COUNT(*) AS cnt "
        "FROM structural_edges e "
        "JOIN code_nodes sn ON sn.node_id = e.source_id "
        "JOIN code_nodes tn ON tn.node_id = e.target_id "
        "GROUP BY e.edge_type, src_class, tgt_class "
        "ORDER BY cnt DESC"
    ).fetchall()

    # Aggregate: for each (src_class -> tgt_class), what edges exist?
    pair_map: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        pair_map[(r["src_class"] or "NULL", r["tgt_class"] or "NULL")][r["edge_type"]] += r["cnt"]

    print(f"\n{'Src Class':<22} {'Tgt Class':<22} {'Top Edge (count)':<40}")
    print("-" * 86)
    for (src, tgt), etype_counts in sorted(pair_map.items(), key=lambda x: -sum(x[1].values())):
        top_edge = max(etype_counts.items(), key=lambda x: x[1])
        total = sum(etype_counts.values())
        print(f"{src:<22} {tgt:<22} {top_edge[0]}({top_edge[1]})  total={total}")

    # The critical question: does any edge cross from UTILITY -> UI_COMPONENT?
    subsection("Critical: UTILITY -> UI_COMPONENT paths (any edge type)")
    cross = conn.execute(
        "SELECT e.edge_type, COUNT(*) AS cnt "
        "FROM structural_edges e "
        "JOIN code_nodes sn ON sn.node_id = e.source_id "
        "JOIN code_nodes tn ON tn.node_id = e.target_id "
        "WHERE sn.file_classification = 'UTILITY' "
        "  AND tn.file_classification = 'UI_COMPONENT' "
        "GROUP BY e.edge_type"
    ).fetchall()
    if cross:
        for r in cross:
            print(f"  {r['edge_type']}: {r['cnt']}")
    else:
        print("  NONE ? no direct UTILITY->UI_COMPONENT edges exist.")

    subsection("Critical: API_ROUTE -> UI_COMPONENT paths (any edge type)")
    cross2 = conn.execute(
        "SELECT e.edge_type, COUNT(*) AS cnt "
        "FROM structural_edges e "
        "JOIN code_nodes sn ON sn.node_id = e.source_id "
        "JOIN code_nodes tn ON tn.node_id = e.target_id "
        "WHERE sn.file_classification = 'API_ROUTE' "
        "  AND tn.file_classification = 'UI_COMPONENT' "
        "GROUP BY e.edge_type"
    ).fetchall()
    if cross2:
        for r in cross2:
            print(f"  {r['edge_type']}: {r['cnt']}")
    else:
        print("  NONE ? no direct API_ROUTE->UI_COMPONENT edges exist.")

    subsection("Critical: UI_COMPONENT -> API_ROUTE paths via CLIENT_API_CALLS")
    cross3 = conn.execute(
        "SELECT COUNT(*) AS cnt "
        "FROM structural_edges e "
        "JOIN code_nodes sn ON sn.node_id = e.source_id "
        "JOIN code_nodes tn ON tn.node_id = e.target_id "
        "WHERE e.edge_type = 'CLIENT_API_CALLS' "
        "  AND tn.file_classification = 'API_ROUTE'"
    ).fetchone()
    print(f"  UI component functions with CLIENT_API_CALLS to API_ROUTE files: {cross3['cnt']}")


# ???????????????????????????????????????
# 7. BFS seed quality projection
# ???????????????????????????????????????

def bfs_seed_projection(conn: sqlite3.Connection) -> None:
    section("7. BFS SEED QUALITY PROJECTION")

    # For each SDD doc chunk with candidates, what is the top-1 node type/class?
    top1 = conn.execute(
        "SELECT dcc.doc_id, dcc.code_id, n.node_type, n.file_classification, "
        "       dcc.weighted_similarity_score "
        "FROM doc_code_candidates dcc "
        "JOIN code_nodes n ON n.node_id = dcc.code_id "
        "WHERE dcc.doc_id LIKE 'sdd__%' "
        "ORDER BY dcc.doc_id, dcc.weighted_similarity_score DESC"
    ).fetchall()

    # Keep only top-1 per doc_id
    seen_docs: set[str] = set()
    top1_seeds: list[sqlite3.Row] = []
    for r in top1:
        if r["doc_id"] not in seen_docs:
            seen_docs.add(r["doc_id"])
            top1_seeds.append(r)

    type_dist: dict[str, int] = defaultdict(int)
    class_dist: dict[str | None, int] = defaultdict(int)
    for r in top1_seeds:
        type_dist[r["node_type"]] += 1
        class_dist[r["file_classification"]] += 1

    print(f"\nSDD doc chunks with top-1 seed: {len(top1_seeds)}")
    subsection("Seed node_type distribution (top-1 per SDD chunk)")
    for nt, cnt in sorted(type_dist.items(), key=lambda x: -x[1]):
        print(f"  {nt:<20} {cnt}")
    subsection("Seed file_classification distribution (top-1 per SDD chunk)")
    for cls, cnt in sorted(class_dist.items(), key=lambda x: -x[1]):
        print(f"  {cls or 'NULL':<25} {cnt}")

    # For seeds that ARE in structural_edges: what is their out-degree?
    seed_ids = [r["code_id"] for r in top1_seeds]
    if seed_ids:
        placeholders = ",".join("?" * len(seed_ids))
        degrees = conn.execute(
            f"SELECT source_id, COUNT(*) AS out_degree "
            f"FROM structural_edges WHERE source_id IN ({placeholders}) "
            f"GROUP BY source_id",
            seed_ids,
        ).fetchall()
        deg_map = {r["source_id"]: r["out_degree"] for r in degrees}
        in_degrees = conn.execute(
            f"SELECT target_id, COUNT(*) AS in_degree "
            f"FROM structural_edges WHERE target_id IN ({placeholders}) "
            f"GROUP BY target_id",
            seed_ids,
        ).fetchall()
        indeg_map = {r["target_id"]: r["in_degree"] for r in in_degrees}

        seeds_with_out_edges = sum(1 for sid in seed_ids if sid in deg_map)
        seeds_isolated = sum(1 for sid in seed_ids if sid not in deg_map and sid not in indeg_map)
        print(f"\nSeeds with outgoing edges (BFS will propagate): {seeds_with_out_edges}/{len(seed_ids)}")
        print(f"Seeds with zero edges (BFS will return only seed): {seeds_isolated}/{len(seed_ids)}")

        if deg_map:
            out_degs = list(deg_map.values())
            print(f"Out-degree stats for connected seeds: "
                  f"min={min(out_degs)}  max={max(out_degs)}  "
                  f"mean={statistics.mean(out_degs):.1f}  "
                  f"median={statistics.median(out_degs):.1f}")


# ???????????????????????????????????????
# 8. Specific CLIENT_API_CALLS analysis: which routes are missing
# ???????????????????????????????????????

def missing_api_bridge(conn: sqlite3.Connection) -> None:
    section("8. API BRIDGE GAP ANALYSIS: Uncovered API Routes")

    # All API_ROUTE File nodes
    api_routes = conn.execute(
        "SELECT node_id FROM code_nodes "
        "WHERE file_classification='API_ROUTE' AND node_type='File'"
    ).fetchall()
    all_route_ids = {r["node_id"] for r in api_routes}

    # API_ROUTE files that ARE targets of CLIENT_API_CALLS
    targeted = conn.execute(
        "SELECT DISTINCT e.target_id "
        "FROM structural_edges e "
        "JOIN code_nodes n ON n.node_id = e.target_id "
        "WHERE e.edge_type = 'CLIENT_API_CALLS'"
    ).fetchall()
    targeted_ids = {r["target_id"] for r in targeted}

    uncovered = all_route_ids - targeted_ids
    print(f"\nTotal API_ROUTE file nodes: {len(all_route_ids)}")
    print(f"Covered by CLIENT_API_CALLS: {len(targeted_ids)}")
    print(f"Uncovered (no frontend caller indexed): {len(uncovered)}")
    print(f"Coverage rate: {len(targeted_ids)/len(all_route_ids)*100:.1f}% of route files")

    subsection("Sample UNCOVERED API routes (up to 20)")
    for nid in sorted(uncovered)[:20]:
        print(f"  {nid}")


# ???????????????????????????????????????
# 9. The RENDERS bridge: UI_COMPONENT graph topology
# ???????????????????????????????????????

def renders_topology(conn: sqlite3.Connection) -> None:
    section("9. RENDERS EDGE TOPOLOGY: UI Component Connectivity")

    renders = conn.execute(
        "SELECT e.source_id, e.target_id, "
        "       sn.file_classification AS src_class, "
        "       tn.file_classification AS tgt_class "
        "FROM structural_edges e "
        "JOIN code_nodes sn ON sn.node_id = e.source_id "
        "JOIN code_nodes tn ON tn.node_id = e.target_id "
        "WHERE e.edge_type = 'RENDERS'"
    ).fetchall()

    print(f"\nTotal RENDERS edges: {len(renders)}")
    # src and tgt classification cross-tab
    pairs: dict[tuple, int] = defaultdict(int)
    for r in renders:
        pairs[(r["src_class"] or "NULL", r["tgt_class"] or "NULL")] += 1
    print(f"\n{'Src Class':<22} {'Tgt Class':<22} {'Count':>8}")
    print("-" * 55)
    for (s, t), cnt in sorted(pairs.items(), key=lambda x: -x[1]):
        print(f"{s:<22} {t:<22} {cnt:>8}")

    # PAGE_COMPONENT -> UI_COMPONENT chains: are PAGE nodes the "roots"?
    page_renders_ui = conn.execute(
        "SELECT COUNT(*) AS cnt "
        "FROM structural_edges e "
        "JOIN code_nodes sn ON sn.node_id = e.source_id "
        "JOIN code_nodes tn ON tn.node_id = e.target_id "
        "WHERE e.edge_type = 'RENDERS' "
        "  AND sn.file_classification = 'PAGE_COMPONENT' "
        "  AND tn.file_classification = 'UI_COMPONENT'"
    ).fetchone()
    print(f"\nPAGE_COMPONENT->RENDERS->UI_COMPONENT: {page_renders_ui['cnt']} edges")


# ???????????????????????????????????????
# MAIN
# ???????????????????????????????????????

def main() -> None:
    if not DB_PATH.exists():
        print(f"ERROR: DB not found at {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"ImpacTracer v4.0 ? Final Index Readiness Audit")
    print(f"DB: {DB_PATH}")

    conn = connect(DB_PATH)

    edge_distribution(conn)
    client_api_calls_bridge(conn)
    orphan_analysis(conn)
    traceability_coverage(conn)
    e2e_path_check(conn)
    reachability_matrix(conn)
    bfs_seed_projection(conn)
    missing_api_bridge(conn)
    renders_topology(conn)

    conn.close()
    print(f"\n{'='*72}")
    print("  AUDIT COMPLETE")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
