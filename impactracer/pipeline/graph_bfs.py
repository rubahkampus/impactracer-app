"""Multi-seed BFS with confidence-tiered CALLS depth cap (FR-D1).

Hub Node Mitigation (mandate #1): any node with in-degree + out-degree > 20
(typical for generic interfaces / framework primitives) has its traversal
depth capped to 1 for ALL edge types.

Reference: master_blueprint.md §4 Step 6.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict, deque

import networkx as nx
from loguru import logger

from impactracer.shared.constants import (
    EDGE_CONFIG,
    EXCLUDED_PROPAGATION_NODE_TYPES,
    LOW_CONF_CAPPED_EDGES,
    NODE_TYPE_MAX_FAN_IN,
    UTILITY_FILE_CALLS_DEPTH_CAP,
)
from impactracer.shared.models import CISResult, NodeTrace

# Hub node threshold: nodes with total degree above this are capped at depth 1.
_HUB_DEGREE_THRESHOLD = 20

_UTILITY_DEPTH_CAPPED_EDGES: frozenset[str] = frozenset({"CALLS"})


def build_graph_from_sqlite(conn: sqlite3.Connection) -> nx.MultiDiGraph:
    """Load ``structural_edges`` into a NetworkX MultiDiGraph.

    Blueprint §4 Step 0: called once at pipeline startup.
    """
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    rows = conn.execute(
        "SELECT source_id, target_id, edge_type FROM structural_edges"
    ).fetchall()
    for src, tgt, etype in rows:
        g.add_edge(src, tgt, edge_type=etype)
    logger.info(
        "[graph_bfs] Graph loaded: {} nodes, {} edges",
        g.number_of_nodes(),
        g.number_of_edges(),
    )
    return g


def _hub_nodes(graph: nx.MultiDiGraph) -> frozenset[str]:
    """Return the set of nodes whose total degree exceeds _HUB_DEGREE_THRESHOLD.

    Hub nodes (generic interfaces, framework primitives like ext::react) cause
    combinatorial BFS explosions. Capping their traversal depth to 1 prevents
    this without requiring a special allowlist.
    """
    hubs: set[str] = set()
    for node in graph.nodes():
        total = graph.in_degree(node) + graph.out_degree(node)
        if total > _HUB_DEGREE_THRESHOLD:
            hubs.add(node)
    if hubs:
        logger.debug("[graph_bfs] Hub nodes detected ({}): capped at depth 1", len(hubs))
    return frozenset(hubs)


def compute_confidence_tiers(
    code_seeds: list[str],
    sis_reranker_map: dict[str, float],
    top_n: int,
) -> frozenset[str]:
    """Return the top-N seeds by reranker score as the high-confidence set.

    Blueprint §4 Step 6: doc-chunk reranker scores propagate to their
    resolved code seeds via dict.setdefault in runner.py.
    """
    if not code_seeds:
        return frozenset()
    scored = sorted(
        code_seeds,
        key=lambda sid: sis_reranker_map.get(sid, 0.0),
        reverse=True,
    )
    return frozenset(scored[:top_n])


def bfs_propagate(
    graph: nx.MultiDiGraph,
    seeds: list[str],
    high_confidence: frozenset[str] | None = None,
    low_confidence_seed_map: dict[str, bool] | None = None,
    seed_file_classification: dict[str, str] | None = None,
    node_type_by_id: dict[str, str] | None = None,
) -> CISResult:
    """Execute multi-seed BFS with per-edge-type direction and depth limits.

    Algorithm:
    1. Initialise sis_nodes from seeds (depth=0).
    2. BFS queue: (node_id, depth, causal_chain, path, source_seed).
    3. For each node, iterate over EDGE_CONFIG entries:
       - Determine neighbor direction (forward=successors, reverse=predecessors).
       - Apply max_depth from EDGE_CONFIG.
       - For CALLS edges, cap to 1 if origin seed is low-confidence.
       - For hub nodes (degree > 20), cap all edges to depth 1.
       - For CALLS chains originating at a UTILITY-file seed, cap depth to
         UTILITY_FILE_CALLS_DEPTH_CAP (prevents service-layer fan-out explosion).
       - Skip neighbours in EXCLUDED_PROPAGATION_NODE_TYPES (e.g. ExternalPackage).
       - Skip neighbours whose in-degree exceeds NODE_TYPE_MAX_FAN_IN[node_type].
       - Skip already-visited nodes.
    4. Record NodeTrace per propagated node.

    Invariant: ``len(sis_nodes) + len(propagated_nodes) == len(visited_set)``.

    Blueprint §4 Step 6.
    """
    if seed_file_classification is None:
        seed_file_classification = {}
    if node_type_by_id is None:
        node_type_by_id = {}
    if high_confidence is None:
        high_confidence = frozenset()
    if low_confidence_seed_map is None:
        low_confidence_seed_map = {}

    hubs = _hub_nodes(graph)

    sis_nodes: dict[str, NodeTrace] = {}
    propagated_nodes: dict[str, NodeTrace] = {}

    # Deduplicate seeds while preserving order.
    seen_seeds: set[str] = set()
    unique_seeds: list[str] = []
    for s in seeds:
        if s not in seen_seeds:
            seen_seeds.add(s)
            unique_seeds.append(s)

    # Initialise SIS nodes and the BFS frontier.
    # Queue entry: (node_id, depth, causal_chain, path, source_seed)
    queue: deque[tuple[str, int, list[str], list[str], str]] = deque()

    for seed in unique_seeds:
        is_low_conf = low_confidence_seed_map.get(seed, False) or (
            seed not in high_confidence and len(high_confidence) > 0
        )
        sis_nodes[seed] = NodeTrace(
            depth=0,
            causal_chain=[],
            path=[seed],
            source_seed=seed,
            low_confidence_seed=is_low_conf,
        )
        # Do NOT mutate the shared graph — adding absent seeds would contaminate
        # sequential ablation runs. Pure terminal nodes (no edges) simply produce
        # no BFS neighbors; they are recorded in sis_nodes only.
        if seed not in graph:
            # Absent from graph → no edges → no BFS expansion, only seed itself.
            logger.debug(
                "[graph_bfs] Seed '{}' not in graph (pure terminal node) — "
                "included as SIS seed, no BFS expansion",
                seed,
            )
        queue.append((seed, 0, [], [seed], seed))

    visited: set[str] = set(unique_seeds)

    while queue:
        node_id, depth, causal_chain, path, source_seed = queue.popleft()

        # Skip BFS expansion for nodes absent from the graph (pure terminal
        # nodes or seeds not yet indexed). They are already recorded in
        # sis_nodes / propagated_nodes; they simply produce no neighbors.
        if node_id not in graph:
            continue

        # Determine if this origin seed is low-confidence for CALLS cap.
        # The low_confidence_seed_map may have direct_code_seeds too.
        origin_is_low_conf = (
            low_confidence_seed_map.get(source_seed, False)
            or (source_seed not in high_confidence and len(high_confidence) > 0)
        )

        # Hub mitigation: if the CURRENT node is a hub, cap its traversal to 1.
        node_is_hub = node_id in hubs

        origin_is_utility = (
            seed_file_classification.get(source_seed, "") == "UTILITY"
        )

        for edge_type, cfg in EDGE_CONFIG.items():
            direction: str = cfg["direction"]
            max_depth: int = cfg["max_depth"]

            if edge_type in LOW_CONF_CAPPED_EDGES and origin_is_low_conf:
                max_depth = 1

            # Hub mitigation: cap ALL edges to 1 when traversing FROM a hub.
            if node_is_hub:
                max_depth = 1

            if (
                edge_type in _UTILITY_DEPTH_CAPPED_EDGES
                and origin_is_utility
            ):
                max_depth = min(max_depth, UTILITY_FILE_CALLS_DEPTH_CAP)

            if depth >= max_depth:
                continue

            # Collect neighbors according to edge direction.
            neighbors: set[str] = set()
            if direction in ("forward", "both"):
                for nbr in graph.successors(node_id):
                    for edge_data in graph.get_edge_data(node_id, nbr, default={}).values():
                        if edge_data.get("edge_type") == edge_type:
                            neighbors.add(nbr)
                            break
            if direction in ("reverse", "both"):
                for nbr in graph.predecessors(node_id):
                    for edge_data in graph.get_edge_data(nbr, node_id, default={}).values():
                        if edge_data.get("edge_type") == edge_type:
                            neighbors.add(nbr)
                            break

            for nbr in neighbors:
                nbr_type = node_type_by_id.get(nbr, "")
                if nbr_type in EXCLUDED_PROPAGATION_NODE_TYPES:
                    continue

                # Per-node-type fan-in cap. Seeds are never excluded.
                if nbr_type and nbr not in visited:
                    fan_in_cap = NODE_TYPE_MAX_FAN_IN.get(nbr_type)
                    if fan_in_cap is not None and fan_in_cap > 0:
                        if graph.in_degree(nbr) > fan_in_cap:
                            logger.debug(
                                "[graph_bfs] Skipped {} ({}) — fan-in {} > cap {}",
                                nbr, nbr_type, graph.in_degree(nbr), fan_in_cap,
                            )
                            continue

                new_chain = causal_chain + [edge_type]
                new_path = path + [nbr]
                new_depth = depth + 1

                if nbr in visited:
                    # Best-path semantics: if the new trace has higher severity
                    # than the existing one, replace and re-enqueue so BFS
                    # continues from the improved chain.
                    existing = propagated_nodes.get(nbr)
                    if existing is None:
                        # nbr is a SIS seed — seeds always keep depth=0, skip.
                        continue
                    from impactracer.shared.constants import severity_for_chain
                    _RANK = {"Tinggi": 0, "Menengah": 1, "Rendah": 2}
                    existing_rank = _RANK[severity_for_chain(existing.causal_chain)]
                    new_rank = _RANK[severity_for_chain(new_chain)]
                    if new_rank < existing_rank:
                        # New trace has higher severity — replace and re-enqueue.
                        propagated_nodes[nbr] = NodeTrace(
                            depth=new_depth,
                            causal_chain=new_chain,
                            path=new_path,
                            source_seed=source_seed,
                            low_confidence_seed=origin_is_low_conf,
                        )
                        # Re-enqueue so BFS propagates from this improved trace.
                        queue.append((nbr, new_depth, new_chain, new_path, source_seed))
                    continue

                visited.add(nbr)
                propagated_nodes[nbr] = NodeTrace(
                    depth=new_depth,
                    causal_chain=new_chain,
                    path=new_path,
                    source_seed=source_seed,
                    low_confidence_seed=origin_is_low_conf,
                )
                queue.append((nbr, new_depth, new_chain, new_path, source_seed))

    # Best-path re-enqueuing does not change the visited set size — a re-enqueued
    # node was already in visited; only its NodeTrace changes.
    assert len(sis_nodes) + len(propagated_nodes) == len(visited), (
        f"BFS invariant violated: "
        f"{len(sis_nodes)} sis + {len(propagated_nodes)} prop != {len(visited)} visited"
    )

    logger.info(
        "[graph_bfs] BFS complete: {} SIS seeds, {} propagated nodes",
        len(sis_nodes),
        len(propagated_nodes),
    )
    return CISResult(sis_nodes=sis_nodes, propagated_nodes=propagated_nodes)


#: Node types that can act as CONTAINS parents (aggregation candidates).
_CONTAINS_PARENT_TYPES: frozenset[str] = frozenset({
    "Interface", "Enum", "Class", "File",
})

#: Node types that are collapsed into their parent (leaf children).
_CONTAINS_CHILD_TYPES: frozenset[str] = frozenset({
    "InterfaceField",
})


def collapse_contains_subtrees(
    cis: CISResult,
    graph: nx.MultiDiGraph,
    node_meta_by_id: dict[str, dict],
) -> CISResult:
    """Collapse CONTAINS-only children into their parent's NodeTrace.

    Dense CONTAINS edges (File→InterfaceField, Interface→InterfaceField) cause
    BFS to visit hundreds of leaf nodes, generating thousands of LLM #4 prompt
    tokens. This function identifies propagated parent nodes whose CONTAINS
    children are pure leaves (no non-CONTAINS edges), removes those children
    from propagated_nodes, and records their IDs in the parent's
    ``collapsed_children`` list.

    Collapse conditions for a child node C relative to parent P:
    1. P is in ``propagated_nodes`` (or ``sis_nodes``) and is a
       CONTAINS-parent type.
    2. C is in ``propagated_nodes`` and is a CONTAINS-child type.
    3. There exists a CONTAINS edge between P and C (either direction).
    4. C has **no other** incoming or outgoing edges in the graph *except*
       CONTAINS edges (i.e. C is purely a leaf).  This prevents collapsing
       a node that is also reachable via CALLS/IMPORTS etc.

    The modified CISResult preserves the BFS invariant for downstream
    consumers: the combined() view now contains parent records instead of
    the removed child records.

    Blueprint reference: master_blueprint.md §4 Step 6.5.
    """
    if not cis.propagated_nodes:
        return cis

    # Build the set of all nodes currently in the CIS (SIS + propagated).
    all_cis_ids: set[str] = set(cis.sis_nodes) | set(cis.propagated_nodes)

    # Build the collapsed set incrementally.
    children_to_remove: set[str] = set()
    # Map parent_id → list[child_id]
    parent_to_children: dict[str, list[str]] = {}

    # Candidate parents = any CIS node (SIS or propagated) that is a parent type.
    candidate_parents: list[str] = [
        nid for nid in all_cis_ids
        if node_meta_by_id.get(nid, {}).get("node_type", "") in _CONTAINS_PARENT_TYPES
    ]

    for parent_id in candidate_parents:
        if parent_id not in graph:
            continue

        # Collect CONTAINS neighbors of this parent in the graph.
        contains_neighbors: set[str] = set()

        # Forward: parent → child
        for nbr in graph.successors(parent_id):
            for ed in graph.get_edge_data(parent_id, nbr, default={}).values():
                if ed.get("edge_type") == "CONTAINS":
                    contains_neighbors.add(nbr)
                    break

        # Reverse: child → parent (edge stored as child→parent or parent→child)
        for nbr in graph.predecessors(parent_id):
            for ed in graph.get_edge_data(nbr, parent_id, default={}).values():
                if ed.get("edge_type") == "CONTAINS":
                    contains_neighbors.add(nbr)
                    break

        # Filter: keep only those in CIS propagated_nodes of a child type.
        collapsible: list[str] = []
        for nbr_id in contains_neighbors:
            if nbr_id not in cis.propagated_nodes:
                continue
            if node_meta_by_id.get(nbr_id, {}).get("node_type", "") not in _CONTAINS_CHILD_TYPES:
                continue
            # Condition 4: the child must have ONLY CONTAINS edges in the full graph.
            nbr_edge_types: set[str] = set()
            if nbr_id in graph:
                for _, _, ed in graph.in_edges(nbr_id, data=True):
                    nbr_edge_types.add(ed.get("edge_type", ""))
                for _, _, ed in graph.out_edges(nbr_id, data=True):
                    nbr_edge_types.add(ed.get("edge_type", ""))
            # Allow collapse only if CONTAINS is the sole edge type.
            if nbr_edge_types and nbr_edge_types != {"CONTAINS"}:
                continue
            collapsible.append(nbr_id)

        if not collapsible:
            continue

        children_to_remove.update(collapsible)
        parent_to_children.setdefault(parent_id, []).extend(collapsible)

    if not children_to_remove:
        logger.info("[graph_bfs] collapse_contains_subtrees: nothing to collapse")
        return cis

    # Build new propagated_nodes: remove collapsed children, annotate parents.
    new_propagated: dict[str, NodeTrace] = {}
    for node_id, trace in cis.propagated_nodes.items():
        if node_id in children_to_remove:
            continue
        new_trace = trace
        if node_id in parent_to_children:
            # Attach collapsed children list (create a new dataclass instance).
            from dataclasses import replace as _dc_replace
            new_trace = _dc_replace(
                trace,
                collapsed_children=list(parent_to_children[node_id]),
            )
        new_propagated[node_id] = new_trace

    # Also annotate SIS parents if any (rare but possible).
    new_sis: dict[str, NodeTrace] = {}
    for node_id, trace in cis.sis_nodes.items():
        new_trace = trace
        if node_id in parent_to_children:
            from dataclasses import replace as _dc_replace
            new_trace = _dc_replace(
                trace,
                collapsed_children=list(parent_to_children[node_id]),
            )
        new_sis[node_id] = new_trace

    total_collapsed = len(children_to_remove)
    logger.info(
        "[graph_bfs] collapse_contains_subtrees: collapsed {} child nodes into {} parents",
        total_collapsed,
        len(parent_to_children),
    )
    return CISResult(sis_nodes=new_sis, propagated_nodes=new_propagated)


# =========================================================================
# Apex Crucible Proposal A.2 — Sibling promotion via CONTAINS
# =========================================================================

#: Node types that can appear as anchors for sibling promotion.
#: We anchor on qualified (file::symbol) entities; bare File nodes don't
#: anchor a "promote my siblings" pass — that would be every node in the file.
_SIBLING_ANCHOR_EXCLUDED_NODE_TYPES: frozenset[str] = frozenset({
    "File", "ExternalPackage",
})

#: Node types eligible as sibling candidates within an anchor's file.
#: InterfaceField is excluded — Step 6.5 already collapses these into their
#: parent Interface, so promoting them again would re-introduce token bloat
#: and InterfaceFields are never in GT.
_SIBLING_CANDIDATE_ALLOWED_NODE_TYPES: frozenset[str] = frozenset({
    "Function", "Method", "Interface", "TypeAlias",
    "Enum", "Class", "Variable",
})


def collect_file_local_siblings(
    anchor_ids: list[str],
    conn: sqlite3.Connection,
    already_in_cis: set[str],
    max_per_file: int = 12,
) -> dict[str, list[tuple[str, str, str]]]:
    """Return per-file sibling candidates for promotion via CONTAINS.

    For each anchor (a validated qualified node), look up the anchor's
    ``file_path`` and fetch every other qualified node in the same file
    that:
      - has a node_type in _SIBLING_CANDIDATE_ALLOWED_NODE_TYPES;
      - is not already in ``already_in_cis``;
      - is not the anchor itself.

    Multiple anchors in the same file collapse to a single sibling list
    (so LLM #4 only adjudicates each sibling once). Returns a dict keyed by
    file_path, value is a list of (sibling_id, node_type, anchor_id) tuples
    capped at ``max_per_file``. The anchor_id is the *first* anchor we
    encountered in that file — used only as a justification reference.

    Apex Crucible Proposal A.2: 7 of 8 missed GT entities on CR-01 live in
    files we already named correctly; 4 of 6 on CR-03. CONTAINS-based
    file-local sibling enumeration is the cheapest way to surface them.
    """
    if not anchor_ids:
        return {}

    # Fetch anchor file_paths.
    placeholders = ",".join("?" * len(anchor_ids))
    rows = conn.execute(
        f"SELECT node_id, file_path, node_type "
        f"FROM code_nodes WHERE node_id IN ({placeholders})",
        anchor_ids,
    ).fetchall()
    anchor_file_paths: dict[str, str] = {}
    file_first_anchor: dict[str, str] = {}
    for nid, fp, ntype in rows:
        if not fp or not nid:
            continue
        if ntype in _SIBLING_ANCHOR_EXCLUDED_NODE_TYPES:
            continue
        anchor_file_paths[nid] = fp
        file_first_anchor.setdefault(fp, nid)

    target_files = list(file_first_anchor.keys())
    if not target_files:
        return {}

    # One query for every qualified node living in any of those files.
    placeholders_f = ",".join("?" * len(target_files))
    rows = conn.execute(
        f"SELECT node_id, node_type, file_path "
        f"FROM code_nodes "
        f"WHERE file_path IN ({placeholders_f}) "
        f"  AND node_id LIKE '%::%'",
        target_files,
    ).fetchall()

    by_file: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    anchor_set = set(anchor_ids)
    for nid, ntype, fp in rows:
        if nid in anchor_set or nid in already_in_cis:
            continue
        if ntype not in _SIBLING_CANDIDATE_ALLOWED_NODE_TYPES:
            continue
        first_anchor = file_first_anchor.get(fp, "")
        by_file[fp].append((nid, ntype, first_anchor))

    # Apply per-file cap.
    capped: dict[str, list[tuple[str, str, str]]] = {}
    for fp, sibs in by_file.items():
        if not sibs:
            continue
        capped[fp] = sibs[:max_per_file]

    total = sum(len(v) for v in capped.values())
    if capped:
        logger.info(
            "[graph_bfs] sibling promotion: {} candidate siblings across {} files "
            "(anchors={}, max_per_file={})",
            total, len(capped), len(anchor_ids), max_per_file,
        )
    return capped
