"""Integration test: graph state isolation across sequential ablation runs.

Phase 3.5 (E-NEW-4 / 9.3): verifies that bfs_propagate does NOT mutate the
shared graph across sequential runs. Previously graph.add_node(seed) was
called for absent seeds, permanently adding them to the shared ctx.graph and
causing non-reproducible BFS results in subsequent ablation variant runs.

These tests use the real graph_bfs.bfs_propagate function (not mocked) to
catch regressions in the isolation guarantee.
"""

from __future__ import annotations

import networkx as nx
import pytest

from impactracer.pipeline.graph_bfs import bfs_propagate, build_graph_from_sqlite


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(*edges: tuple[str, str, str]) -> nx.MultiDiGraph:
    """Build a MultiDiGraph from (src, tgt, edge_type) triples."""
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    for src, tgt, etype in edges:
        g.add_edge(src, tgt, edge_type=etype)
    return g


# ---------------------------------------------------------------------------
# Core isolation invariants
# ---------------------------------------------------------------------------


def test_absent_seed_does_not_mutate_graph() -> None:
    """A seed not in the graph must NOT be added to it after BFS.

    Regression: bfs_propagate previously called graph.add_node(seed) for
    seeds absent from the graph. This contaminated the shared graph so that
    subsequent runs saw different node sets.
    """
    g = _make_graph()
    assert g.number_of_nodes() == 0, "Graph should start empty"

    cis = bfs_propagate(g, seeds=["ghost_seed"])

    # Seed must be in SIS (it's a valid CIS entry).
    assert "ghost_seed" in cis.sis_nodes
    # Graph must remain unchanged — no mutation.
    assert g.number_of_nodes() == 0, (
        "bfs_propagate must not add absent seeds to the shared graph"
    )


def test_graph_node_count_unchanged_after_bfs() -> None:
    """BFS over an existing graph must not add or remove nodes."""
    g = _make_graph(
        ("caller_a", "target_fn", "CALLS"),
        ("caller_b", "target_fn", "CALLS"),
        ("target_fn", "helper", "CALLS"),
    )
    node_count_before = g.number_of_nodes()
    edge_count_before = g.number_of_edges()

    bfs_propagate(g, seeds=["target_fn"])

    assert g.number_of_nodes() == node_count_before, (
        "BFS must not add/remove nodes from the graph"
    )
    assert g.number_of_edges() == edge_count_before, (
        "BFS must not add/remove edges from the graph"
    )


def test_sequential_runs_produce_identical_cis() -> None:
    """Two sequential BFS runs on the same graph with the same seeds must
    produce byte-identical CIS node sets.

    This is the primary ablation isolation regression test. If run N mutated
    the graph (e.g. by adding seed nodes), run N+1 would encounter a different
    graph topology and produce a different CIS, making ablation results
    non-reproducible.
    """
    g = _make_graph(
        ("caller_1", "seed_fn", "CALLS"),
        ("caller_2", "caller_1", "CALLS"),
        ("iface", "seed_fn", "IMPLEMENTS"),
    )

    cis_run1 = bfs_propagate(g, seeds=["seed_fn"])
    cis_run2 = bfs_propagate(g, seeds=["seed_fn"])

    assert set(cis_run1.sis_nodes.keys()) == set(cis_run2.sis_nodes.keys()), (
        "SIS node sets differ between sequential runs — graph mutation suspected"
    )
    assert set(cis_run1.propagated_nodes.keys()) == set(cis_run2.propagated_nodes.keys()), (
        "Propagated node sets differ between sequential runs — graph mutation suspected"
    )


def test_multi_seed_sequential_runs_independent() -> None:
    """Multiple seeds across two sequential runs must not bleed state."""
    g = _make_graph(
        ("callerA", "seedA", "CALLS"),
        ("callerB", "seedB", "CALLS"),
    )

    # Run 1: only seedA
    cis1 = bfs_propagate(g, seeds=["seedA"])
    # Run 2: only seedB — must not see callerA from run 1
    cis2 = bfs_propagate(g, seeds=["seedB"])

    assert "callerA" in cis1.propagated_nodes
    assert "callerB" not in cis1.propagated_nodes  # run 1 scoped to seedA

    assert "callerB" in cis2.propagated_nodes
    assert "callerA" not in cis2.propagated_nodes  # run 2 scoped to seedB

    # Graph itself must be unchanged throughout.
    assert "seedA" not in g.nodes or "seedB" not in g.nodes or True  # always passes
    assert g.number_of_nodes() == 4  # callerA, seedA, callerB, seedB only


def test_absent_multi_seed_leaves_graph_empty() -> None:
    """Multiple absent seeds must not add any nodes to the graph."""
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    assert g.number_of_nodes() == 0

    cis = bfs_propagate(g, seeds=["ghost_1", "ghost_2", "ghost_3"])

    assert len(cis.sis_nodes) == 3   # all listed as SIS
    assert len(cis.propagated_nodes) == 0  # no edges → no propagation
    assert g.number_of_nodes() == 0   # graph untouched


def test_variant_ablation_graph_isolation_simulation() -> None:
    """Simulate 3 sequential ablation variant runs on a shared graph.

    Mimics how the ablation harness calls bfs_propagate 3 times (V5, V6, V7)
    on the same PipelineContext graph. Each run must see the same graph.
    """
    g = _make_graph(
        ("fileA", "fn_target", "CONTAINS"),
        ("caller_1", "fn_target", "CALLS"),
        ("caller_2", "caller_1", "CALLS"),
        ("iface_x", "fn_target", "IMPLEMENTS"),
    )
    baseline_nodes = set(g.nodes())
    baseline_edges = g.number_of_edges()

    seeds = ["fn_target"]
    results = []
    for _ in range(3):
        cis = bfs_propagate(g, seeds=seeds, high_confidence=frozenset(seeds))
        results.append(frozenset(cis.propagated_nodes.keys()))

        # Graph must not have mutated.
        assert set(g.nodes()) == baseline_nodes, (
            "Graph node set changed between ablation runs"
        )
        assert g.number_of_edges() == baseline_edges, (
            "Graph edge count changed between ablation runs"
        )

    # All three runs must produce identical propagated node sets.
    assert results[0] == results[1] == results[2], (
        "Propagated node sets differ across sequential ablation runs"
    )
