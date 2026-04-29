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
    "API_ROUTE":       {"FR": 1.0, "NFR": 0.5, "Design": 0.8, "General": 0.5},
    "PAGE_COMPONENT":  {"FR": 1.0, "NFR": 0.5, "Design": 0.9, "General": 0.5},
    "UI_COMPONENT":    {"FR": 0.9, "NFR": 0.5, "Design": 0.9, "General": 0.5},
    "UTILITY":         {"FR": 0.7, "NFR": 0.7, "Design": 0.8, "General": 0.5},
    "TYPE_DEFINITION": {"FR": 0.6, "NFR": 0.3, "Design": 0.9, "General": 0.5},
    None:              {"FR": 0.8, "NFR": 0.5, "Design": 0.8, "General": 0.5},
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
    "DEFINES_METHOD":      {"direction": "forward", "max_depth": 3},
    "PASSES_CALLBACK":     {"direction": "forward", "max_depth": 1},
    # Reactive
    "HOOK_DEPENDS_ON":     {"direction": "reverse", "max_depth": 1},
    # Module composition
    "IMPORTS":             {"direction": "reverse", "max_depth": 1},
    "RENDERS":             {"direction": "reverse", "max_depth": 1},
    "DEPENDS_ON_EXTERNAL": {"direction": "reverse", "max_depth": 1},
    "CLIENT_API_CALLS":    {"direction": "reverse", "max_depth": 1},
    "DYNAMIC_IMPORT":      {"direction": "reverse", "max_depth": 1},
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
}


def severity_for_chain(causal_chain: list[str]) -> Severity:
    """Severity = highest (most-severe) category across the edge chain.

    SIS seeds (empty chain) are ``Tinggi`` by convention.
    """
    if not causal_chain:
        return "Tinggi"
    rank = {"Tinggi": 0, "Menengah": 1, "Rendah": 2}
    return min(
        (SEVERITY_BY_EDGE_CHAIN_TYPE.get(e, "Rendah") for e in causal_chain),
        key=lambda s: rank[s],
    )


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
