"""System-wide constants: edge config, weight tables, builtin blacklists.

References:
    03_data_models.md §10-15
    05_ast_edge_catalog.md for edge rationale
"""

from __future__ import annotations

from impactracer.shared.models import ChangeType, Severity


# =========================================================================
# RRF path weights (indexed by change_type) — RETIRED, ARCHIVAL.
# =========================================================================
# Ablated over all 24 citrakara CRs (variance-free, V0-V3) and found inert:
# moves only V2 on 8/24 CRs, pooled delta -0.0019 (inside noise), washed out
# by V3's cross-encoder + every LLM stage. Production now fuses with uniform
# weight 1.0 (Settings.uniform_rrf_weights defaults True). This table is no
# longer consulted unless that flag is set False; kept for reversibility.
# Evidence: eval/rrf_weight_ablation/. NOTE change_type stays load-bearing
# elsewhere (LLM #2 ADDITION framing) — only retrieval fusion drops it.

RRF_PATH_WEIGHTS: dict[ChangeType, dict[str, float]] = {
    "ADDITION": {
        "dense_doc": 1.2,
        "bm25_doc": 1.0,
        "dense_code": 1.0,
        "bm25_code": 0.8,
    },
    "MODIFICATION": {
        "dense_doc": 1.0,
        "bm25_doc": 1.0,
        "dense_code": 1.2,
        "bm25_code": 1.0,
    },
    "DELETION": {
        "dense_doc": 1.0,
        "bm25_doc": 0.8,
        "dense_code": 1.2,
        "bm25_code": 1.0,
    },
}


# =========================================================================
# Layer compatibility matrix for traceability precomputation
# =========================================================================

LAYER_COMPAT: dict[str | None, dict[str, float]] = {
    # API_ROUTE: Next.js route handlers — entry points for FRs, described in
    # SDD endpoint-spec (Design) sections and process flows (General).
    "API_ROUTE":       {"FR": 1.0, "NFR": 0.6, "Design": 0.9, "General": 0.8},

    # PAGE_COMPONENT: Next.js pages — primary FR/Design targets for UI flows.
    "PAGE_COMPONENT":  {"FR": 1.0, "NFR": 0.5, "Design": 0.9, "General": 0.7},

    # UI_COMPONENT: React components, client hooks, Zustand stores.
    # Now also covers src/hooks/ and src/lib/stores/ (client-side state).
    # General raised to 0.7 — process docs reference UI interactions.
    "UI_COMPONENT":    {"FR": 0.9, "NFR": 0.5, "Design": 0.9, "General": 0.7},

    # UTILITY: services, repositories, api helpers, utils — the backend
    # business logic layer.  Directly implements FRs and is the primary
    # subject of SDD component (Design) and SRS process (General) sections.
    "UTILITY":         {"FR": 1.0, "NFR": 0.7, "Design": 1.0, "General": 0.8},

    # TYPE_DEFINITION: Mongoose models (src/lib/db/models/**) and TypeScript
    # type files (src/types/**).  These are the verbatim subject of SDD
    # "Perancangan Basis Data" (Database Design) sections → 1.0 × Design.
    # Also relevant to FRs (data structures used by FRs) and General
    # (process docs reference entity names).  Low NFR relevance.
    "TYPE_DEFINITION": {"FR": 0.8, "NFR": 0.3, "Design": 1.0, "General": 0.6},

    # None: unclassified files (config.ts, middleware.ts, theme/).
    # Treated conservatively — plausible match for any doc type.
    None:              {"FR": 0.8, "NFR": 0.5, "Design": 0.8, "General": 0.6},
}


def layer_compat(code_classification: str | None, doc_chunk_type: str) -> float:
    """Return the layer-compatibility multiplier for a (code, doc) pair."""
    row = LAYER_COMPAT.get(code_classification, LAYER_COMPAT[None])
    return row.get(doc_chunk_type, 0.5)


# =========================================================================
# Edge propagation configuration
# =========================================================================

EDGE_CONFIG: dict[str, dict] = {
    # Behavioral dependencies (reverse direction)
    # Reverse-CALLS: each hop multiplies fan-in by ~3-8 in TS codebases.
    # Depth-2 caps fan-out while still catching transitive same-feature callers.
    "CALLS":               {"direction": "reverse", "max_depth": 2},
    "INHERITS":            {"direction": "reverse", "max_depth": 3},
    "IMPLEMENTS":          {"direction": "reverse", "max_depth": 3},
    "TYPED_BY":            {"direction": "reverse", "max_depth": 3},
    # FIELDS_ACCESSED retired together with the InterfaceField node type
    # (its only possible target). See RETIRED_PROPAGATION_EDGES / RETIRED_NODE_TYPES.
    # Structural ownership (forward direction)
    # Containment relation, not a semantic propagation pathway. Depth-1 only.
    "DEFINES_METHOD":      {"direction": "forward", "max_depth": 1},
    # Module composition
    "IMPORTS":             {"direction": "reverse", "max_depth": 1},
    "RENDERS":             {"direction": "reverse", "max_depth": 1},
    "DYNAMIC_IMPORT":      {"direction": "reverse", "max_depth": 1},
    # --- RETIRED EDGES (propagation-inert; see RETIRED_PROPAGATION_EDGES) -----
    # Empirically retired as methodological deadweight: across the
    # citrakara+nova corpora these contributed ZERO true positives to BFS.
    # Two are empty in every index (PASSES_CALLBACK, HOOK_DEPENDS_ON); two are
    # populated but structurally propagation-dead (CLIENT_API_CALLS never sat on
    # a seed->node chain; DEPENDS_ON_EXTERNAL targets only ExternalPackage nodes,
    # which EXCLUDED_PROPAGATION_NODE_TYPES bars BFS from entering). They are
    # left OUT of EDGE_CONFIG so the BFS never walks them. The extractor still
    # emits them (schema/provenance intact) and re-enabling is a one-liner via
    # settings.enable_retired_edges -> see _RETIRED_EDGE_CONFIG below.
    #   "PASSES_CALLBACK":     {"direction": "forward", "max_depth": 1},
    #   "HOOK_DEPENDS_ON":     {"direction": "reverse", "max_depth": 1},
    #   "DEPENDS_ON_EXTERNAL": {"direction": "reverse", "max_depth": 1},
    #   "CLIENT_API_CALLS":    {"direction": "reverse", "max_depth": 1},
    # -------------------------------------------------------------------------
    # CONTAINS bridges the File↔symbol membrane. Reverse only: given a changed
    # symbol, find which files contain it — not enumerate all sibling symbols.
    "CONTAINS":            {"direction": "reverse", "max_depth": 1},
}

#: Edge types retired from BFS propagation. Inert by construction —
#: zero true-positive contribution on citrakara+nova. Kept here so the
#: extractor/schema stay honest and the decision is reversible: passing
#: ``settings.enable_retired_edges=True`` merges these back into the active
#: traversal config via :func:`active_edge_config`.
#:
#: FIELDS_ACCESSED is a SECOND-WAVE retirement: it is retired not because it is
#: merely empty (INHERITS/IMPLEMENTS are also empty but RETAINED as class/
#: interface-granularity contract edges), but because its ONLY possible target
#: is an ``InterfaceField`` node — and InterfaceField is itself retired (see
#: RETIRED_NODE_TYPES). With no InterfaceField nodes the edge cannot fire.
RETIRED_PROPAGATION_EDGES: dict[str, dict] = {
    "PASSES_CALLBACK":     {"direction": "forward", "max_depth": 1},
    "HOOK_DEPENDS_ON":     {"direction": "reverse", "max_depth": 1},
    "DEPENDS_ON_EXTERNAL": {"direction": "reverse", "max_depth": 1},
    "CLIENT_API_CALLS":    {"direction": "reverse", "max_depth": 1},
    "FIELDS_ACCESSED":     {"direction": "reverse", "max_depth": 2},
}

#: Node types retired from the index. Defined in the schema/enum for
#: reversibility but NOT emitted by default. ``ExternalPackage`` was orphaned
#: once DEPENDS_ON_EXTERNAL (its only inbound edge) was retired. ``InterfaceField``
#: is field-level granularity: 46% of indexed nodes on citrakara yet 0% of
#: ground truth and 0% of propagated nodes — below the file/named-declaration
#: granularity that the ground truth and the CIA literature operate at. Both are
#: re-emitted (with their edges) when ``settings.enable_retired_edges`` is True.
RETIRED_NODE_TYPES: frozenset[str] = frozenset({
    "ExternalPackage",
    "InterfaceField",
})


def active_edge_config(enable_retired_edges: bool = False) -> dict[str, dict]:
    """Return the BFS edge config; optionally re-admit the retired edges.

    Default (``False``) returns ``EDGE_CONFIG`` unchanged — the four retired
    edges stay propagation-inert. ``True`` overlays ``RETIRED_PROPAGATION_EDGES``
    for diagnostic/ablation use (e.g. confirming they still catch nothing).
    """
    if not enable_retired_edges:
        return EDGE_CONFIG
    return {**EDGE_CONFIG, **RETIRED_PROPAGATION_EDGES}

LOW_CONF_CAPPED_EDGES: frozenset[str] = frozenset({"CALLS"})
"""Edges whose depth is capped to 1 for low-confidence seeds."""

PROPAGATION_VALIDATION_EXEMPT_EDGES: frozenset[str] = frozenset({
    "IMPLEMENTS",
    "DEFINES_METHOD",
})
"""Single-hop edges that skip LLM #4 validation (direct contracts).

TYPED_BY is intentionally NOT in the exempt set. Auto-admitting every
TYPED_BY-linked file produced too many false positives (e.g. a model file
admitted because it references a typed payload, even though it never
changes). LLM #4 adjudicates these single-hop type contracts on the same
footing as deeper propagation chains.
"""


# =========================================================================
# Structural propagation limits (BFS graph-flood defence)
# =========================================================================
#
# Two orthogonal mechanisms beyond the hub-degree cap:
#
# 1. UTILITY_FILE_CALLS_DEPTH_CAP — when a propagation chain ORIGINATES at
#    a UTILITY-classified file (e.g. lib/format-date.ts, lib/error.ts),
#    its reverse-CALLS traversal is capped to depth 1. Utility functions
#    are called from everywhere; reverse-CALLS at depth ≥2 from a UTILITY
#    seed produces a near-guaranteed flood across unrelated features.
#
# 2. NODE_TYPE_MAX_FAN_IN — propagated nodes whose total in-degree exceeds
#    this cap are dropped from the CIS unless they are SIS seeds. This is
#    a sharper version of the hub-degree threshold (which uses degree=20
#    to cap depth=1, but does not exclude). Primitive React components,
#    base utility classes, and framework defaults all exceed 50 incoming
#    edges and are pure noise in the impact set.

UTILITY_FILE_CALLS_DEPTH_CAP: int = 1
"""Reverse-CALLS depth cap for chains originating at UTILITY files."""

NODE_TYPE_MAX_FAN_IN: dict[str, int] = {
    "Function": 50,
    "Method": 50,
    "Class": 50,
    "Interface": 100,        # Type definitions are referenced widely; allow more.
    "TypeAlias": 100,
    "Enum": 100,
    "InterfaceField": 200,   # RETIRED node type; retained for reversibility.
    "File": 200,             # File nodes act as containers; high fan-in is expected.
    "ExternalPackage": 0,    # RETIRED node type; never propagated regardless.
    "Variable": 80,          # Top-level const declarations (schemas, constant data).
}
"""Max in-degree before a node is excluded from BFS propagation (not seeds)."""

EXCLUDED_PROPAGATION_NODE_TYPES: frozenset[str] = frozenset({
    "ExternalPackage",
})
"""Node types that are NEVER added to the propagated set, regardless of edges.

Rationale: ExternalPackage represents a third-party dependency. We do not
analyse third-party code as 'impacted' — at most, we observe that our code
DEPENDS_ON_EXTERNAL it. The dependency edge is informative, but reaching
INTO the package node and treating it as a unit-of-impact is incorrect.
"""


# =========================================================================
# Weight-decay propagation prune (Step 6.8) — deterministic flood control.
# =========================================================================
#
# 3. PROPAGATION_DECAY_WEIGHTS — the third flood-defence mechanism. After BFS
#    + sibling expansion, every propagated node is scored by an edge-weight-
#    aware decay along its causal chain, then only the top-K are kept (SIS
#    seeds are never scored or cut). Score:
#
#        decay(node) = prod(PROPAGATION_DECAY_WEIGHTS[e] for e in chain)
#                      / (1 + depth)
#
#    Weights encode MEASURED edge productivity on the GT (BFS true-positive
#    audit): RENDERS is the workhorse (13/17 BFS TPs) -> 1.0; CALLS strong;
#    IMPORTS the dominant flood source (long dead-reference fan-out) -> low.
#    Selected over PPR and semantic-cosine by an offline sweep on the V6 pool
#    (eval results,): at equal K it removed ~68% of propagated false
#    positives while retaining ~70% of propagated true positives — strictly
#    more surgical than PPR (40% TP kept) or semantic (55%) at the same FP cut.
#    Distinct from graph_rerank._EDGE_WEIGHTS (the default-OFF PPR rerank);
#    these are the authoritative decay weights for the always-on Step 6.8.

PROPAGATION_DECAY_WEIGHTS: dict[str, float] = {
    "RENDERS": 1.0,          # workhorse: 13/17 BFS true positives
    "CALLS": 0.8,
    "TYPED_BY": 0.6,
    "CONTAINS": 0.5,         # in-file sibling arm
    "DEFINES_METHOD": 0.5,
    "IMPLEMENTS": 0.5,
    "INHERITS": 0.5,
    "DYNAMIC_IMPORT": 0.3,
    "IMPORTS": 0.3,          # dominant flood source: dead-reference fan-out
}
"""Per-edge weights for the Step 6.8 weight-decay propagation prune."""


def propagation_decay_score(causal_chain: list[str], depth: int | None = None) -> float:
    """Edge-weight-aware decay score for a propagated node's causal chain.

    score = prod(weight[edge]) / (1 + depth). Unknown edges default to the
    IMPORTS weight (0.3) — conservative (treated as weak). Empty chain (a
    direct/depth-0 admit) scores 1.0. Higher = more likely genuine impact.
    """
    if depth is None:
        depth = len(causal_chain)
    w = 1.0
    for et in causal_chain:
        w *= PROPAGATION_DECAY_WEIGHTS.get(et, 0.3)
    return w / (1.0 + depth)


# =========================================================================
# Severity mapping
# =========================================================================

SEVERITY_BY_EDGE_CHAIN_TYPE: dict[str, Severity] = {
    # Contract dependency chain -> HIGH
    "IMPLEMENTS": "Tinggi",
    "TYPED_BY": "Tinggi",
    # FIELDS_ACCESSED retired (InterfaceField target retired); if
    # re-enabled, severity_for_chain falls back to "Rendah".
    # Behavioral dependency chain -> MEDIUM
    "CALLS": "Menengah",
    "INHERITS": "Menengah",
    "DEFINES_METHOD": "Menengah",
    # Module composition chain -> LOW
    "IMPORTS": "Rendah",
    "RENDERS": "Rendah",
    "DYNAMIC_IMPORT": "Rendah",
    # Retired edges (HOOK_DEPENDS_ON, PASSES_CALLBACK, DEPENDS_ON_EXTERNAL,
    # CLIENT_API_CALLS) omitted — propagation-inert (see EDGE_CONFIG). If
    # re-enabled via enable_retired_edges, severity_for_chain falls back to
    # "Rendah" for them, which is the correct tier anyway.
    # File ownership -> LOW (structural containment, not semantic dependency)
    "CONTAINS": "Rendah",
}


def severity_for_chain(causal_chain: list[str]) -> Severity:
    """Severity = the edge type of the LAST hop in the causal chain.

    The last hop is the proximate structural dependency; its type indicates
    *why* the node is impacted rather than laundering through earlier hops.
    SIS seeds (empty chain) are Tinggi — they are direct retrieval targets.
    """
    if not causal_chain:
        return "Tinggi"
    return SEVERITY_BY_EDGE_CHAIN_TYPE.get(causal_chain[-1], "Rendah")


# =========================================================================
# Extraction blacklists and pattern sets
# =========================================================================

BUILTIN_PATTERNS: frozenset[str] = frozenset({
    "console", "Object", "Array", "Math", "JSON", "Promise",
    "setTimeout", "setInterval", "clearTimeout", "clearInterval",
    "parseInt", "parseFloat", "String", "Number", "Boolean",
    "Error", "Date", "RegExp", "Map", "Set", "WeakMap", "WeakSet",
    "Symbol", "Proxy", "Reflect", "Intl",
    "fetch", "URL", "URLSearchParams", "FormData",
    "Headers", "Request", "Response",
    "Buffer", "process", "require",
    "window", "document", "globalThis",
})
"""Identifiers never emitted as CALLS targets."""

PRIMITIVE_TYPES: frozenset[str] = frozenset({
    "string", "number", "boolean", "void", "any", "unknown",
    "null", "undefined", "never", "object", "symbol", "bigint",
})
"""Type identifiers never emitted as TYPED_BY targets."""

HOOK_NAMES: frozenset[str] = frozenset({
    "useEffect", "useCallback", "useMemo", "useLayoutEffect",
})
"""React hooks whose dep array drives HOOK_DEPENDS_ON extraction."""
