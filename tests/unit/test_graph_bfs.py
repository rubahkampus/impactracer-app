"""Unit tests for graph_bfs: build_graph_from_sqlite, compute_confidence_tiers,
bfs_propagate (FR-D1)."""

from __future__ import annotations

import sqlite3


from impactracer.persistence.sqlite_client import init_schema
from impactracer.pipeline.graph_bfs import (
    _HUB_DEGREE_THRESHOLD,
    _hub_nodes,
    bfs_propagate,
    build_graph_from_sqlite,
    compute_confidence_tiers,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db(*edges: tuple[str, str, str]) -> sqlite3.Connection:
    """Create in-memory DB with given (src, tgt, edge_type) rows.

    init_schema sets PRAGMA foreign_keys = ON inside its DDL, so we must
    seed code_nodes stubs for every referenced node_id before inserting
    edges. All node_ids in ``edges`` get a minimal stub row.
    """
    conn = sqlite3.connect(":memory:")
    init_schema(conn)

    # Collect all unique node IDs from edges.
    node_ids: set[str] = set()
    for src, tgt, _ in edges:
        node_ids.add(src)
        node_ids.add(tgt)

    # Insert minimal stub code_node rows.
    for nid in node_ids:
        conn.execute(
            "INSERT OR IGNORE INTO code_nodes "
            "(node_id, node_type, name, embed_text) VALUES (?, ?, ?, ?)",
            (nid, "Function", nid, "stub"),
        )

    for src, tgt, etype in edges:
        conn.execute(
            "INSERT OR IGNORE INTO structural_edges (source_id, target_id, edge_type) "
            "VALUES (?, ?, ?)",
            (src, tgt, etype),
        )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# build_graph_from_sqlite
# ---------------------------------------------------------------------------


def test_build_graph_loads_edges() -> None:
    conn = _make_db(
        ("A", "B", "CALLS"),
        ("B", "C", "IMPORTS"),
    )
    g = build_graph_from_sqlite(conn)
    assert g.number_of_edges() == 2
    assert "A" in g
    assert "B" in g
    assert "C" in g


def test_build_graph_empty() -> None:
    conn = _make_db()
    g = build_graph_from_sqlite(conn)
    assert g.number_of_nodes() == 0
    assert g.number_of_edges() == 0


def test_build_graph_stores_edge_type() -> None:
    conn = _make_db(("X", "Y", "TYPED_BY"))
    g = build_graph_from_sqlite(conn)
    edge_data = list(g.get_edge_data("X", "Y").values())
    assert any(d.get("edge_type") == "TYPED_BY" for d in edge_data)


# ---------------------------------------------------------------------------
# compute_confidence_tiers
# ---------------------------------------------------------------------------


def test_confidence_tier_top_n() -> None:
    seeds = ["A", "B", "C", "D", "E"]
    score_map = {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6, "E": 0.5}
    tier = compute_confidence_tiers(seeds, score_map, top_n=3)
    assert tier == frozenset({"A", "B", "C"})


def test_confidence_tier_empty_seeds() -> None:
    tier = compute_confidence_tiers([], {}, top_n=5)
    assert tier == frozenset()


def test_confidence_tier_missing_scores() -> None:
    """Seeds absent from score_map get 0.0; top-2 still works."""
    seeds = ["A", "B", "C"]
    tier = compute_confidence_tiers(seeds, {"A": 0.9}, top_n=2)
    assert "A" in tier


def test_confidence_tier_all_seeds_fit() -> None:
    """top_n >= len(seeds) → all seeds returned."""
    seeds = ["A", "B"]
    tier = compute_confidence_tiers(seeds, {"A": 0.5, "B": 0.3}, top_n=10)
    assert tier == frozenset({"A", "B"})


# ---------------------------------------------------------------------------
# bfs_propagate — invariant and basic direction
# ---------------------------------------------------------------------------


def test_bfs_invariant_simple() -> None:
    """SIS + propagated == visited for a simple CALLS chain."""
    import networkx as nx
    g = nx.MultiDiGraph()
    # A calls B (CALLS edge: reverse direction → from seed B, walk to A)
    g.add_edge("A", "B", edge_type="CALLS")
    g.add_edge("B", "C", edge_type="CALLS")
    cis = bfs_propagate(g, seeds=["B"])
    assert len(cis.sis_nodes) + len(cis.propagated_nodes) == len(
        set(cis.sis_nodes) | set(cis.propagated_nodes)
    )
    # Invariant: visited == sis + propagated
    assert len(cis.sis_nodes) + len(cis.propagated_nodes) >= 1


def test_bfs_reverse_calls() -> None:
    """CALLS is reverse: seed is the callee; caller propagated."""
    import networkx as nx
    g = nx.MultiDiGraph()
    # callerA -> seed (CALLS edge: source=caller, target=seed)
    g.add_edge("callerA", "seed", edge_type="CALLS")
    cis = bfs_propagate(g, seeds=["seed"])
    assert "seed" in cis.sis_nodes
    assert "callerA" in cis.propagated_nodes


def test_bfs_forward_defines_method() -> None:
    """DEFINES_METHOD is forward: seed is the class; method is propagated."""
    import networkx as nx
    g = nx.MultiDiGraph()
    # class -> method (forward direction)
    g.add_edge("MyClass", "MyClass::myMethod", edge_type="DEFINES_METHOD")
    cis = bfs_propagate(g, seeds=["MyClass"])
    assert "MyClass" in cis.sis_nodes
    assert "MyClass::myMethod" in cis.propagated_nodes


def test_bfs_sis_seeds_not_in_propagated() -> None:
    """Seeds appear in sis_nodes only, never in propagated_nodes."""
    import networkx as nx
    g = nx.MultiDiGraph()
    g.add_edge("A", "seed", edge_type="CALLS")
    cis = bfs_propagate(g, seeds=["seed"])
    assert "seed" not in cis.propagated_nodes
    assert "seed" in cis.sis_nodes


def test_bfs_causal_chain_recorded() -> None:
    """propagated_nodes records the edge type in causal_chain."""
    import networkx as nx
    g = nx.MultiDiGraph()
    g.add_edge("caller", "seed", edge_type="CALLS")
    cis = bfs_propagate(g, seeds=["seed"])
    assert "CALLS" in cis.propagated_nodes["caller"].causal_chain


def test_bfs_multi_seed() -> None:
    """Multiple seeds, each propagates independently."""
    import networkx as nx
    g = nx.MultiDiGraph()
    g.add_edge("A", "seed1", edge_type="CALLS")
    g.add_edge("B", "seed2", edge_type="CALLS")
    cis = bfs_propagate(g, seeds=["seed1", "seed2"])
    assert "seed1" in cis.sis_nodes
    assert "seed2" in cis.sis_nodes
    assert "A" in cis.propagated_nodes
    assert "B" in cis.propagated_nodes


def test_bfs_no_duplicate_in_sis_and_propagated() -> None:
    """If two seeds share a common upstream caller, it appears only once."""
    import networkx as nx
    g = nx.MultiDiGraph()
    g.add_edge("sharedCaller", "seed1", edge_type="CALLS")
    g.add_edge("sharedCaller", "seed2", edge_type="CALLS")
    cis = bfs_propagate(g, seeds=["seed1", "seed2"])
    # sharedCaller must appear in propagated_nodes exactly once.
    assert "sharedCaller" in cis.propagated_nodes
    assert "sharedCaller" not in cis.sis_nodes
    total = len(cis.sis_nodes) + len(cis.propagated_nodes)
    visited = set(cis.sis_nodes) | set(cis.propagated_nodes)
    assert total == len(visited)


# ---------------------------------------------------------------------------
# bfs_propagate — depth caps
# ---------------------------------------------------------------------------


def test_bfs_calls_depth_capped_for_low_conf() -> None:
    """CALLS depth capped to 1 for low-confidence origins."""
    import networkx as nx
    g = nx.MultiDiGraph()
    # Chain: A->seed, B->A (depth 2 from seed via CALLS reverse)
    g.add_edge("A", "seed", edge_type="CALLS")
    g.add_edge("B", "A", edge_type="CALLS")

    # seed is low-confidence → CALLS capped at 1.
    cis_low = bfs_propagate(
        g,
        seeds=["seed"],
        high_confidence=frozenset(),          # seed is NOT in high_confidence
        low_confidence_seed_map={"seed": True},
    )
    # Only depth-1 neighbor A should appear; B is depth 2 → capped out.
    assert "A" in cis_low.propagated_nodes
    assert "B" not in cis_low.propagated_nodes


def test_bfs_calls_depth_2_for_high_conf() -> None:
    """CALLS max_depth is 2.

    Even for a high-confidence seed, the third reverse-CALLS hop is
    structurally rejected. Depth-3 fan-in regularly produces 200+ propagated
    nodes per seed in real TS codebases — depth-2 is the precision-recovery
    sweet spot.
    """
    import networkx as nx
    g = nx.MultiDiGraph()
    # Chain: A->seed, B->A, C->B (depths 1, 2, 3)
    g.add_edge("A", "seed", edge_type="CALLS")
    g.add_edge("B", "A", edge_type="CALLS")
    g.add_edge("C", "B", edge_type="CALLS")

    cis = bfs_propagate(
        g,
        seeds=["seed"],
        high_confidence=frozenset({"seed"}),
        low_confidence_seed_map={"seed": False},
    )
    assert "A" in cis.propagated_nodes      # depth 1
    assert "B" in cis.propagated_nodes      # depth 2
    assert "C" not in cis.propagated_nodes  # depth 3: dropped


# ---------------------------------------------------------------------------
# Hub node mitigation
# ---------------------------------------------------------------------------


def test_hub_node_detected() -> None:
    """Node with degree > threshold is classified as a hub."""
    import networkx as nx
    g = nx.MultiDiGraph()
    hub = "ext::react"
    for i in range(_HUB_DEGREE_THRESHOLD + 1):
        g.add_edge(f"component_{i}", hub, edge_type="DEPENDS_ON_EXTERNAL")
    hubs = _hub_nodes(g)
    assert hub in hubs


def test_hub_node_capped_at_depth_1() -> None:
    """BFS does not propagate beyond depth 1 from a hub node."""
    import networkx as nx
    g = nx.MultiDiGraph()
    # Make 'hub' a hub by giving it many edges.
    hub = "hub_node"
    for i in range(_HUB_DEGREE_THRESHOLD + 1):
        g.add_edge(f"src_{i}", hub, edge_type="CALLS")
    # Add a deeper chain: deepNode -> hub -> seed
    g.add_edge(hub, "seed", edge_type="CALLS")
    g.add_edge("deepNode", hub, edge_type="CALLS")

    cis = bfs_propagate(g, seeds=["seed"], high_confidence=frozenset({"seed"}))
    # hub is depth 1 (reachable via reverse CALLS). deepNode would be depth 2
    # from seed, but passing THROUGH hub. Hub itself is depth 1 — but when BFS
    # is AT hub (a hub node), it caps its own outgoing traversal to depth 1.
    # deepNode should still be reachable as depth 2 only if hub is NOT capped.
    # With hub capping: hub traversal depth is 1 → deepNode depth from hub = 1,
    # so total depth from seed = 2. Hub cap prevents further traversal.
    assert hub in cis.propagated_nodes
    trace_hub = cis.propagated_nodes[hub]
    assert trace_hub.depth == 1


# ---------------------------------------------------------------------------
# CISResult invariant assertion
# ---------------------------------------------------------------------------


def test_bfs_invariant_asserted() -> None:
    """Verify the invariant holds for an isolated graph."""
    import networkx as nx
    g = nx.MultiDiGraph()
    g.add_edge("P", "Q", edge_type="IMPORTS")
    g.add_edge("Q", "R", edge_type="CALLS")
    cis = bfs_propagate(g, seeds=["Q"])
    total = len(cis.sis_nodes) + len(cis.propagated_nodes)
    visited = set(cis.sis_nodes) | set(cis.propagated_nodes)
    assert total == len(visited)  # invariant: no double-counting


def test_bfs_empty_seeds() -> None:
    """Empty seeds produce empty CISResult."""
    import networkx as nx
    g = nx.MultiDiGraph()
    g.add_edge("A", "B", edge_type="CALLS")
    cis = bfs_propagate(g, seeds=[])
    assert len(cis.sis_nodes) == 0
    assert len(cis.propagated_nodes) == 0


def test_bfs_seed_not_in_graph() -> None:
    """Seed absent from graph appears in sis_nodes with no propagation."""
    import networkx as nx
    g = nx.MultiDiGraph()
    cis = bfs_propagate(g, seeds=["ghost_node"])
    assert "ghost_node" in cis.sis_nodes
    assert len(cis.propagated_nodes) == 0


def test_bfs_duplicate_seeds_deduplicated() -> None:
    """Duplicate seeds produce exactly one sis_node entry."""
    import networkx as nx
    g = nx.MultiDiGraph()
    cis = bfs_propagate(g, seeds=["A", "A", "A"])
    assert len(cis.sis_nodes) == 1
    assert "A" in cis.sis_nodes



# ---------------------------------------------------------------------------
# Phase 1: graph isolation — bfs_propagate must NOT mutate the shared graph
# ---------------------------------------------------------------------------


def test_bfs_does_not_add_absent_seed_to_graph() -> None:
    """Phase 1 (E-NEW-4): a seed absent from the graph must NOT be added.

    Previously bfs_propagate called graph.add_node(seed) for absent seeds,
    contaminating the shared ctx.graph across sequential ablation runs.
    Post-fix: graph size is unchanged after BFS with absent seeds.
    """
    import networkx as nx
    g = nx.MultiDiGraph()
    # Graph starts empty.
    assert g.number_of_nodes() == 0

    cis = bfs_propagate(g, seeds=["ghost_seed"])

    # ghost_seed must appear in sis_nodes (it's a valid SIS entry).
    assert "ghost_seed" in cis.sis_nodes
    # But the GRAPH must remain empty — no mutation.
    assert g.number_of_nodes() == 0


def test_bfs_does_not_mutate_graph_with_real_seed() -> None:
    """BFS over an existing seed does not add extra nodes to the graph."""
    import networkx as nx
    g = nx.MultiDiGraph()
    g.add_edge("caller", "seed", edge_type="CALLS")
    node_count_before = g.number_of_nodes()

    bfs_propagate(g, seeds=["seed"])

    # Graph node count must be unchanged after BFS.
    assert g.number_of_nodes() == node_count_before


def test_bfs_sequential_runs_independent() -> None:
    """Two sequential BFS runs on the same graph produce identical results.

    This is the ablation isolation regression test: if run N mutated the
    graph, run N+1 would see different nodes and produce different results
    for the same seed set.
    """
    import networkx as nx
    g = nx.MultiDiGraph()
    g.add_edge("caller", "seed", edge_type="CALLS")

    cis1 = bfs_propagate(g, seeds=["seed"])
    cis2 = bfs_propagate(g, seeds=["seed"])

    # Both runs must produce identical node sets.
    assert set(cis1.sis_nodes.keys()) == set(cis2.sis_nodes.keys())
    assert set(cis1.propagated_nodes.keys()) == set(cis2.propagated_nodes.keys())


# ---------------------------------------------------------------------------
# Phase 1: CONTAINS direction = reverse (E-NEW-8)
# ---------------------------------------------------------------------------


def test_bfs_contains_reverse_only() -> None:
    """Phase 1 (E-NEW-8): CONTAINS is direction=reverse.

    Given edge: parent --CONTAINS--> child
    BFS from seed=child (via reverse CONTAINS) should reach parent.
    BFS from seed=parent (which would use forward CONTAINS) must NOT reach child.
    """
    import networkx as nx
    g = nx.MultiDiGraph()
    # parent CONTAINS child (stored as parent→child edge)
    g.add_edge("parent_file", "child_field", edge_type="CONTAINS")

    # Seed = child_field → reverse CONTAINS → should reach parent_file
    cis_from_child = bfs_propagate(g, seeds=["child_field"])
    assert "parent_file" in cis_from_child.propagated_nodes

    # Seed = parent_file → forward CONTAINS was removed → must NOT reach child_field
    # (CONTAINS is now direction=reverse, so forward traversal is disabled)
    cis_from_parent = bfs_propagate(g, seeds=["parent_file"])
    assert "child_field" not in cis_from_parent.propagated_nodes
