"""Multi-seed BFS with confidence-tiered CALLS depth cap (FR-D1).

Reference: 07_online_pipeline.md §10.
"""

from __future__ import annotations

import sqlite3

import networkx as nx

from impactracer.shared.models import CISResult


def build_graph_from_sqlite(conn: sqlite3.Connection) -> nx.MultiDiGraph:
    """Load ``structural_edges`` into a NetworkX MultiDiGraph."""
    raise NotImplementedError("Sprint 10")


def compute_confidence_tiers(
    code_seeds: list[str],
    sis_reranker_map: dict[str, float],
    top_n: int,
) -> frozenset[str]:
    """Return the top-N seeds by reranker score as the high-confidence set."""
    raise NotImplementedError("Sprint 10")


def bfs_propagate(
    graph: nx.MultiDiGraph,
    seeds: list[str],
    high_confidence: frozenset[str] | None = None,
    low_confidence_seed_map: dict[str, bool] | None = None,
) -> CISResult:
    """Execute multi-seed BFS with per-edge-type direction and depth limits.

    Invariant: ``len(sis_nodes) + len(propagated_nodes) == len(visited_set)``.
    Asserted in unit tests.
    """
    raise NotImplementedError("Sprint 10")
