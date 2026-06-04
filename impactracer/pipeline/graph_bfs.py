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





# =========================================================================
# Sibling promotion via CONTAINS (Step 7.5 support)
# =========================================================================

#: Node types that can appear as anchors for sibling promotion.
#: We anchor on qualified (file::symbol) entities; bare File nodes don't
#: anchor a "promote my siblings" pass — that would be every node in the file.
_SIBLING_ANCHOR_EXCLUDED_NODE_TYPES: frozenset[str] = frozenset({
    "File", "ExternalPackage",
})

#: Node types eligible as sibling candidates within an anchor's file.
#: InterfaceField is not listed — it is a retired node type (absent from the
#: index) and never appears in GT.
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

    Motivation: many missed GT entities live in files the pipeline has
    already named correctly. CONTAINS-based file-local sibling enumeration
    is the cheapest way to surface them.
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


# =========================================================================
# DELETION-only backward-compat filter (post-BFS, pre-LLM #4)
# =========================================================================

#: Edge types that, when they are the SOLE reason a node is in the CIS for
#: a DELETION CR, are likely "import-only" / "dead-reference" patterns that
#: a linter would clean up rather than evidence of true semantic impact.
#: When BFS reaches a node via ONLY these edges (no CALLS / TYPED_BY /
#: FIELDS_ACCESSED / IMPLEMENTS / INHERITS), the node is demoted out of
#: the CIS for the DELETION case.
_DELETION_IMPORT_ONLY_EDGES: frozenset[str] = frozenset({
    "IMPORTS",
    "DYNAMIC_IMPORT",
})

#: Edge types that ALWAYS represent a real consumption surface for a
#: deletion target. If ANY of these appears in the chain, the candidate
#: stays in the CIS.
_DELETION_SUBSTANTIVE_EDGES: frozenset[str] = frozenset({
    "CALLS",
    "TYPED_BY",
    "FIELDS_ACCESSED",
    "IMPLEMENTS",
    "INHERITS",
    "DEFINES_METHOD",
    "RENDERS",
    "CONTAINS",
    # Retired edges (PASSES_CALLBACK, HOOK_DEPENDS_ON, CLIENT_API_CALLS,
    # DEPENDS_ON_EXTERNAL) removed 2026-06 — propagation-inert, so they can
    # never appear in a causal_chain reaching the filter anyway.
})


def apply_deletion_import_only_filter(cis: CISResult) -> tuple[CISResult, int]:
    """Demote propagated nodes whose ONLY path to a SIS seed is via IMPORTS.

    Rationale (sprint 25):
      For DELETION CRs the BFS reaches many modules that merely import the
      deletion target without using it in any behaviour-affecting position.
      These are "dead references" that a linter (tsc / eslint
      no-unused-imports) would clean up after the deletion. They are not
      part of the cognitive impact set the GT annotator considered. Demoting
      them at this stage tightens precision on DELETION without changing
      ADDITION / MODIFICATION behaviour at all.

    Filter rule (applied per propagated node):
      Keep iff the node's ``causal_chain`` contains at least one edge in
      ``_DELETION_SUBSTANTIVE_EDGES``. Drop iff every edge in the chain is
      in ``_DELETION_IMPORT_ONLY_EDGES``.
      SIS seeds are never demoted (the deleted symbol itself + its directly
      retrieved + LLM-2-validated co-changes stay).

    Returns:
        (filtered_cis, n_demoted) where n_demoted is the count of nodes
        removed from propagated_nodes.

    NOTE: This is a deterministic post-BFS filter. It does NOT call any LLM.
    It is gated on change_type==DELETION by the caller.
    """
    if not cis.propagated_nodes:
        return cis, 0

    kept: dict[str, NodeTrace] = {}
    demoted: list[str] = []

    for nid, trace in cis.propagated_nodes.items():
        chain = trace.causal_chain or []
        if not chain:
            # Empty chain == this is effectively a direct seed; keep.
            kept[nid] = trace
            continue
        has_substantive = any(e in _DELETION_SUBSTANTIVE_EDGES for e in chain)
        if has_substantive:
            kept[nid] = trace
        else:
            demoted.append(nid)

    if demoted:
        logger.info(
            "[graph_bfs] deletion-import-only filter: demoted {} of {} propagated "
            "nodes (kept {})",
            len(demoted), len(cis.propagated_nodes), len(kept),
        )

    filtered = CISResult(sis_nodes=cis.sis_nodes, propagated_nodes=kept)
    return filtered, len(demoted)


# =========================================================================
# Propagator — the deterministic propagation component (steps 6 + 6.6-6.9).
# =========================================================================
# Composes the building blocks above into the full V6 propagation pass and
# owns its cache + trace surface, so the runner calls it once. Two arms:
#   OUTWARD : BFS over dependency edges + DELETION import-only filter +
#             weight-decay top-K prune.
#   IN-FILE : raw CONTAINS-sibling expansion + anchor-rrf top-K prune.
# Both arms inject unvalidated nodes (LLM #4 prunes them at V7). The raw pool
# is cached BEFORE the prunes so the top-K stays a re-tunable view-time knob.


def _bfs_cis_trace(cis: CISResult) -> dict:
    return {
        "sis_seeds": [
            {"node_id": k, "depth": v.depth, "source_seed": v.source_seed}
            for k, v in cis.sis_nodes.items()
        ],
        "propagated_nodes": [
            {"node_id": k, "depth": v.depth,
             "causal_chain": v.causal_chain, "source_seed": v.source_seed}
            for k, v in cis.propagated_nodes.items()
        ],
    }


def _expand_siblings(cis, conn, sis_justifications, trace_mechanisms,
                     trace_justifications, admitted_candidates, settings, variant_cache):
    """In-file arm: inject raw CONTAINS-sibling candidates of mechanism-carrying
    seeds, tagged ``sibling_candidate``. Caches the per-candidate context
    (file, anchor, anchor rrf) the V7 sibling validator and the 6.9 prune need.
    Returns the trace payload."""
    anchors: list[str] = []
    anchor_just: dict[str, str] = {}
    for nid in list(cis.sis_nodes) + list(cis.propagated_nodes):
        if "::" not in nid:
            continue
        v2 = sis_justifications.get(nid) or {}
        mechanism = (v2.get("mechanism_of_impact") or "").strip()
        fallback = v2.get("justification") or ""
        if not mechanism:
            mechanism = (trace_mechanisms.get(nid) or "").strip()
            if mechanism and not fallback:
                fallback = trace_justifications.get(nid) or ""
        if not mechanism:
            continue
        anchors.append(nid)
        anchor_just[nid] = mechanism or fallback

    already = set(cis.sis_nodes) | set(cis.propagated_nodes)
    per_file = collect_file_local_siblings(
        anchor_ids=anchors, conn=conn, already_in_cis=already,
        max_per_file=getattr(settings, "sibling_promotion_max_per_file", 12),
    )

    anchor_rrf = {c.node_id: float(c.rrf_score or 0.0) for c in admitted_candidates}
    meta: dict[str, dict[str, str]] = {}
    injected = 0
    for file_path, sibs in per_file.items():
        for sib_id, sib_type, first_anchor in sibs:
            if sib_id in cis.sis_nodes or sib_id in cis.propagated_nodes:
                continue
            cis.propagated_nodes[sib_id] = NodeTrace(
                depth=1, causal_chain=["CONTAINS"], path=[first_anchor, sib_id],
                source_seed=first_anchor, low_confidence_seed=False,
                justification="", justification_source="sibling_candidate",
            )
            meta[sib_id] = {
                "file_path": file_path, "node_type": sib_type, "anchor": first_anchor,
                "anchor_justification": anchor_just.get(first_anchor, ""),
                "anchor_rrf": anchor_rrf.get(first_anchor, 0.0),
            }
            injected += 1

    if variant_cache is not None:
        variant_cache.put_sibling_candidates(meta)
    logger.info(
        "[graph_bfs] sibling EXPANSION injected {} raw candidates across {} files "
        "(anchors={}, unvalidated)",
        injected, sum(1 for s in per_file.values() if s), len(anchors),
    )
    return {
        "anchors": list(anchors),
        "raw_candidates": list(meta),
        "candidates_per_file": {
            fp: [sid for sid, _t, _a in sibs] for fp, sibs in per_file.items()
        },
    }


def _prune_bfs_arm(cis, settings):
    """Outward arm precision: keep the top-K BFS nodes by edge-weight decay.
    Siblings are exempt (they tie under decay). Returns the trace payload or None."""
    if not getattr(settings, "enable_propagation_weight_prune", True):
        return None
    from impactracer.shared.constants import propagation_decay_score
    k = getattr(settings, "propagation_prune_top_k", 10)
    siblings = {n: t for n, t in cis.propagated_nodes.items()
                if t.justification_source == "sibling_candidate"}
    bfs = {n: t for n, t in cis.propagated_nodes.items()
           if t.justification_source != "sibling_candidate"}
    if not (k and k > 0 and len(bfs) > k):
        return None
    scored = sorted(
        bfs.items(),
        key=lambda kv: propagation_decay_score(kv[1].causal_chain or [], kv[1].depth),
        reverse=True,
    )
    cis.propagated_nodes = {**dict(scored[:k]), **siblings}
    logger.info(
        "[graph_bfs] weight-decay prune {} -> {} BFS nodes (top-{}); {} siblings exempt",
        len(bfs), k, k, len(siblings),
    )
    return {"pre_count": len(bfs), "kept_count": k, "top_k": k,
            "dropped": [n for n, _ in scored[k:]]}


def _prune_sibling_arm(cis, settings, variant_cache):
    """In-file arm precision: keep the top-K siblings by anchor rrf_score.
    Returns the trace payload or None."""
    if not (getattr(settings, "enable_sibling_precision_prune", True)
            and getattr(settings, "enable_sibling_promotion", True)):
        return None
    k = getattr(settings, "sibling_prune_top_k", 10)
    meta = (variant_cache.get_sibling_candidates() if variant_cache is not None else None) or {}
    sib_nodes = [n for n, t in cis.propagated_nodes.items()
                 if t.justification_source == "sibling_candidate"]
    if not (k and k > 0 and len(sib_nodes) > k):
        return None

    def anchor_rrf(sib_id: str) -> float:
        m = meta.get(sib_id)
        return float(m["anchor_rrf"]) if m and "anchor_rrf" in m else 0.0

    ranked = sorted(sib_nodes, key=anchor_rrf, reverse=True)
    dropped = ranked[k:]
    for nid in dropped:
        del cis.propagated_nodes[nid]
    kept = len(sib_nodes) - len(dropped)
    logger.info(
        "[graph_bfs] sibling-precision prune {} -> {} siblings (anchor-rrf top-{}, dropped {})",
        len(sib_nodes), kept, k, len(dropped),
    )
    return {"pre_count": len(sib_nodes), "kept_count": kept, "top_k": k, "dropped": dropped}


def propagate(
    *,
    cis_from_seeds,
    graph: nx.MultiDiGraph,
    conn: sqlite3.Connection,
    all_code_seeds: list[str],
    low_conf: dict[str, bool],
    validated_code_seeds: list[str],
    resolutions: list,
    admitted_candidates: list,
    sis_justifications: dict,
    trace_mechanisms: dict,
    trace_justifications: dict,
    change_type: str,
    settings,
    variant_cache,
    trace,
) -> CISResult:
    """Run the full deterministic propagation pass and return the CIS.

    ``cis_from_seeds`` is a zero-arg builder for the BFS-disabled / no-seed
    fallback (the runner supplies it so candidate hydration stays its concern).
    ``trace`` is the runner's trace callback. Cache-hit on ``bfs_cis`` replays
    the raw pool; the prunes re-apply every run so K stays a view-time knob.
    """
    cache_hit = False
    if variant_cache is not None and all_code_seeds:
        cached = variant_cache.get_bfs_cis()
        if cached is not None:
            cis = cached
            logger.info(
                "[graph_bfs] [cache HIT] bfs_cis ({} sis + {} propagated, BFS skipped)",
                len(cis.sis_nodes), len(cis.propagated_nodes),
            )
            trace("step_6_bfs_raw_cis", _bfs_cis_trace(cis))
            cache_hit = True

    if cache_hit:
        pass
    elif all_code_seeds:
        sis_reranker = {
            c.node_id: c.reranker_score
            for c in admitted_candidates if c.node_id in set(all_code_seeds)
        }
        for r in resolutions:
            for cid in r["code_ids"]:
                sis_reranker.setdefault(cid, 0.0)
        high_conf = compute_confidence_tiers(
            all_code_seeds, sis_reranker, settings.bfs_high_conf_top_n
        )
        logger.info("[graph_bfs] High-confidence seeds (top-{}): {}",
                    settings.bfs_high_conf_top_n, len(high_conf))

        node_type_by_id: dict[str, str] = {}
        seed_file_classification: dict[str, str] = {}
        ids_to_fetch = list(set(graph.nodes()) | set(all_code_seeds))
        for i in range(0, len(ids_to_fetch), 500):
            chunk = ids_to_fetch[i:i + 500]
            ph = ",".join("?" * len(chunk))
            for nid_, ntype_, fclass_ in conn.execute(
                f"SELECT node_id, node_type, file_classification "
                f"FROM code_nodes WHERE node_id IN ({ph})", chunk,
            ).fetchall():
                if ntype_:
                    node_type_by_id[nid_] = ntype_
                if fclass_:
                    seed_file_classification[nid_] = fclass_

        cis = bfs_propagate(
            graph, all_code_seeds, high_confidence=high_conf,
            low_confidence_seed_map=low_conf,
            seed_file_classification=seed_file_classification,
            node_type_by_id=node_type_by_id,
        )
        logger.info("[graph_bfs] BFS: {} SIS seeds, {} propagated nodes",
                    len(cis.sis_nodes), len(cis.propagated_nodes))
        trace("step_6_bfs_raw_cis", _bfs_cis_trace(cis))
    else:
        # No code seeds: CIS = admitted candidates only (runner-hydrated).
        cis = cis_from_seeds()
        logger.info("[graph_bfs] No code seeds — CIS from admitted candidates")

    # OUTWARD arm — DELETION import-only filter (deterministic; DEL CRs only).
    if change_type.upper() == "DELETION" and cis.propagated_nodes:
        cis, n_demoted = apply_deletion_import_only_filter(cis)
        trace("step_6p6_deletion_import_only_filter", {
            "n_demoted": n_demoted, "kept_propagated_count": len(cis.propagated_nodes),
        })

    # IN-FILE arm — raw sibling expansion (skipped on cache-hit; cached pool
    # already holds the siblings). Runs BEFORE put_bfs_cis so they persist.
    if (not cache_hit and getattr(settings, "enable_sibling_promotion", True)):
        payload = _expand_siblings(
            cis, conn, sis_justifications, trace_mechanisms, trace_justifications,
            admitted_candidates, settings, variant_cache,
        )
        trace("step_6p7_sibling_expansion", payload)

    # Cache the RAW pool (BFS + raw siblings) BEFORE the prunes, so K is a
    # re-tunable view-time knob and the cache is a faithful unpruned record.
    if not cache_hit and variant_cache is not None and all_code_seeds:
        variant_cache.put_bfs_cis(cis)

    # Deterministic precision prunes (re-applied every run, incl. cache-hit).
    p8 = _prune_bfs_arm(cis, settings)
    if p8 is not None:
        trace("step_6p8_weight_decay_prune", p8)
    p9 = _prune_sibling_arm(cis, settings, variant_cache)
    if p9 is not None:
        trace("step_6p9_sibling_precision_prune", p9)

    return cis
