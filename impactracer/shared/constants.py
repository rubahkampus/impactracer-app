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
    "CALLS":               {"direction": "reverse", "max_depth": 3},
    "INHERITS":            {"direction": "reverse", "max_depth": 3},
    "IMPLEMENTS":          {"direction": "reverse", "max_depth": 3},
    "TYPED_BY":            {"direction": "reverse", "max_depth": 3},
    "FIELDS_ACCESSED":     {"direction": "reverse", "max_depth": 2},
    # Structural ownership (forward direction)
    # Phase 1 fix (F-1/F-NEW-2): max_depth reduced 3→1. DEFINES_METHOD is a
    # containment relation (Class owns its Methods); it is NOT a semantic
    # propagation pathway. Allowing depth-3 forward traversal caused Class →
    # Method → anything-3-hops-deep explosions that are architecturally
    # unjustifiable. Depth-1 means a class reaches only its own direct methods.
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
    # CONTAINS bridges the File↔symbol membrane.
    # Phase 1 fix (E-NEW-8/A-NEW-7): direction changed "both"→"reverse".
    # Forward traversal (File→all-children) caused a BFS explosion that reached
    # every InterfaceField and Method in any touched file, producing 372+ nodes
    # on live CRs and a 97K-token synthesis timeout. The academically correct
    # direction is reverse only: given a changed symbol, we want to know which
    # files CONTAIN it (i.e. which files are affected), NOT to enumerate all
    # other symbols that live in the same file. Forward CONTAINS is structural
    # noise, not semantic propagation.
    "CONTAINS":            {"direction": "reverse", "max_depth": 1},
}

LOW_CONF_CAPPED_EDGES: frozenset[str] = frozenset({"CALLS"})
"""Edges whose depth is capped to 1 for low-confidence seeds (Fix D)."""

PROPAGATION_VALIDATION_EXEMPT_EDGES: frozenset[str] = frozenset({
    "IMPLEMENTS",
    "DEFINES_METHOD",
    "TYPED_BY",
})
"""Single-hop edges that skip LLM #4 validation (direct contracts)."""


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

    Phase 1 fix (F-NEW-4): the previous implementation used min(rank) across
    the entire chain, which caused "severity laundering" — a node reached via
    CALLS → CALLS → IMPLEMENTS inherited Tinggi severity from the final
    IMPLEMENTS edge even though the intervening CALLS hops were speculative.
    The last hop is the proximate structural dependency; its edge type is the
    correct indicator of *why* the node is impacted.

    SIS seeds (empty chain) are ``Tinggi`` by convention: they are the direct
    retrieval targets of the CR and have the highest semantic certainty.
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
