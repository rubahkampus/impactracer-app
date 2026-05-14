"""System-wide constants: edge config, weight tables, builtin blacklists.

References:
    03_data_models.md §10-15
    05_ast_edge_catalog.md for edge rationale
"""

from __future__ import annotations

from impactracer.shared.models import ChangeType, Severity


# =========================================================================
# Adaptive RRF weights (indexed by change_type)
# =========================================================================

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
    "FIELDS_ACCESSED":     {"direction": "reverse", "max_depth": 2},
    # Structural ownership (forward direction)
    # Containment relation, not a semantic propagation pathway. Depth-1 only.
    "DEFINES_METHOD":      {"direction": "forward", "max_depth": 1},
    "PASSES_CALLBACK":     {"direction": "forward", "max_depth": 1},
    # Reactive
    "HOOK_DEPENDS_ON":     {"direction": "reverse", "max_depth": 1},
    # Module composition
    "IMPORTS":             {"direction": "reverse", "max_depth": 1},
    "RENDERS":             {"direction": "reverse", "max_depth": 1},
    "DEPENDS_ON_EXTERNAL": {"direction": "reverse", "max_depth": 1},
    "CLIENT_API_CALLS":    {"direction": "reverse", "max_depth": 1},
    "DYNAMIC_IMPORT":      {"direction": "reverse", "max_depth": 1},
    # CONTAINS bridges the File↔symbol membrane. Reverse only: given a changed
    # symbol, find which files contain it — not enumerate all sibling symbols.
    "CONTAINS":            {"direction": "reverse", "max_depth": 1},
}

LOW_CONF_CAPPED_EDGES: frozenset[str] = frozenset({"CALLS"})
"""Edges whose depth is capped to 1 for low-confidence seeds."""

PROPAGATION_VALIDATION_EXEMPT_EDGES: frozenset[str] = frozenset({
    "IMPLEMENTS",
    "DEFINES_METHOD",
})
"""Single-hop edges that skip LLM #4 validation (direct contracts).

Apex Crucible Proposal A.1: TYPED_BY removed from the exempt set. Forensic
audit showed that auto-admitting every TYPED_BY-linked file produced 10 of
28 false positives on CR-01 (e.g. review.model.ts admitted because it
references CommissionListing, even though it never changes). LLM #4 now
adjudicates these single-hop type contracts on the same footing as deeper
propagation chains.
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
    "InterfaceField": 200,   # Field-level access has the highest fan-in by nature.
    "File": 200,             # File nodes act as containers; high fan-in is expected.
    "ExternalPackage": 0,    # Never propagate INTO an ExternalPackage from BFS.
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
# Severity mapping
# =========================================================================

SEVERITY_BY_EDGE_CHAIN_TYPE: dict[str, Severity] = {
    # Contract dependency chain -> HIGH
    "IMPLEMENTS": "Tinggi",
    "TYPED_BY": "Tinggi",
    "FIELDS_ACCESSED": "Tinggi",
    # Behavioral dependency chain -> MEDIUM
    "CALLS": "Menengah",
    "INHERITS": "Menengah",
    "DEFINES_METHOD": "Menengah",
    "HOOK_DEPENDS_ON": "Menengah",
    "PASSES_CALLBACK": "Menengah",
    # Module composition chain -> LOW
    "IMPORTS": "Rendah",
    "RENDERS": "Rendah",
    "DEPENDS_ON_EXTERNAL": "Rendah",
    "CLIENT_API_CALLS": "Rendah",
    "DYNAMIC_IMPORT": "Rendah",
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
