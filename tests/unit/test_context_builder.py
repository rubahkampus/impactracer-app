"""Unit tests for pipeline/context_builder.py (FR-E1, FR-E2) and runner._compute_scope (N8)."""

from __future__ import annotations

import sqlite3


from impactracer.pipeline.context_builder import fetch_snippets, _ILA_NODE_TYPES
from impactracer.pipeline.runner import _compute_scope
from impactracer.shared.models import CISResult, NodeTrace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db_with_nodes(rows: list[tuple]) -> sqlite3.Connection:
    """Create in-memory SQLite with code_nodes rows.

    rows: list of (node_id, node_type, ila, source_code)
    """
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE code_nodes (
            node_id TEXT PRIMARY KEY,
            node_type TEXT,
            internal_logic_abstraction TEXT,
            source_code TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO code_nodes VALUES (?, ?, ?, ?)", rows
    )
    conn.commit()
    return conn


def _make_cis(node_ids: list[str]) -> CISResult:
    sis = {
        nid: NodeTrace(depth=0, causal_chain=[], path=[nid], source_seed=nid)
        for nid in node_ids
    }
    return CISResult(sis_nodes=sis, propagated_nodes={})


# ---------------------------------------------------------------------------
# fetch_snippets — N6: ILA preference for Function/Method
# ---------------------------------------------------------------------------

def test_fetch_snippets_prefers_ila_for_function():
    """N6: Function nodes use ILA over source_code when ILA is present."""
    conn = _make_db_with_nodes([
        ("fn1", "Function", "const x = foo();", "function fn1() { /* raw */ }"),
    ])
    result = fetch_snippets(["fn1"], conn)
    assert result["fn1"] == "const x = foo();"


def test_fetch_snippets_prefers_ila_for_method():
    """N6: Method nodes also use ILA."""
    conn = _make_db_with_nodes([
        ("mth1", "Method", "return this.repo.find();", "method() { /* raw */ }"),
    ])
    result = fetch_snippets(["mth1"], conn)
    assert result["mth1"] == "return this.repo.find();"


def test_fetch_snippets_falls_back_to_source_code_when_no_ila():
    """N6: Function node without ILA falls back to source_code."""
    conn = _make_db_with_nodes([
        ("fn1", "Function", None, "function fn1() { /* raw */ }"),
    ])
    result = fetch_snippets(["fn1"], conn)
    assert result["fn1"] == "function fn1() { /* raw */ }"


def test_fetch_snippets_non_function_uses_source_code():
    """N6: Class/Interface/TypeAlias nodes do NOT use ILA even if present."""
    conn = _make_db_with_nodes([
        ("cls1", "Class", "class ILA text", "class Foo { ... }"),
    ])
    result = fetch_snippets(["cls1"], conn)
    # Class is not in _ILA_NODE_TYPES → source_code
    assert "cls1" in result
    assert result["cls1"] == "class Foo { ... }"


def test_fetch_snippets_empty_input():
    conn = _make_db_with_nodes([])
    result = fetch_snippets([], conn)
    assert result == {}


def test_fetch_snippets_missing_node_not_in_result():
    conn = _make_db_with_nodes([])
    # Node not in DB → not in result (will be fetched from ChromaDB if doc_col given)
    result = fetch_snippets(["nonexistent"], conn)
    assert "nonexistent" not in result


def test_fetch_snippets_multiple_types():
    """Mix of Function (ILA), Class (source_code), and Method (ILA)."""
    conn = _make_db_with_nodes([
        ("fn1", "Function", "ila for fn1", "source for fn1"),
        ("cls1", "Class", "ila for cls1", "source for cls1"),
        ("mth1", "Method", "ila for mth1", "source for mth1"),
    ])
    result = fetch_snippets(["fn1", "cls1", "mth1"], conn)
    assert result["fn1"] == "ila for fn1"
    assert result["cls1"] == "source for cls1"  # Class uses source_code
    assert result["mth1"] == "ila for mth1"


def test_ila_node_types_are_function_and_method():
    """_ILA_NODE_TYPES must contain exactly Function and Method."""
    assert "Function" in _ILA_NODE_TYPES
    assert "Method" in _ILA_NODE_TYPES
    # Sanity: Class and others are NOT in the set
    assert "Class" not in _ILA_NODE_TYPES
    assert "Interface" not in _ILA_NODE_TYPES


# ---------------------------------------------------------------------------
# _compute_scope — N8: deterministic scope computation
# ---------------------------------------------------------------------------

def test_compute_scope_terlokalisasi_small():
    """≤10 nodes → terlokalisasi (Phase 2.9: default scope_local_max=10)."""
    cis = _make_cis(["n1", "n2", "n3"])
    assert _compute_scope(cis) == "terlokalisasi"


def test_compute_scope_terlokalisasi_boundary():
    """Exactly 10 nodes → terlokalisasi (at the local_max boundary)."""
    cis = _make_cis([f"n{i}" for i in range(10)])
    assert _compute_scope(cis) == "terlokalisasi"


def test_compute_scope_menengah_lower_bound():
    """11 nodes → menengah (one above scope_local_max=10)."""
    cis = _make_cis([f"n{i}" for i in range(11)])
    assert _compute_scope(cis) == "menengah"


def test_compute_scope_menengah_upper_bound():
    """Exactly 30 nodes → menengah (at scope_medium_max=30)."""
    cis = _make_cis([f"n{i}" for i in range(30)])
    assert _compute_scope(cis) == "menengah"


def test_compute_scope_ekstensif():
    """>30 nodes → ekstensif."""
    cis = _make_cis([f"n{i}" for i in range(31)])
    assert _compute_scope(cis) == "ekstensif"


def test_compute_scope_includes_propagated_nodes():
    """Propagated nodes also count toward scope.

    Phase 2.9: with default thresholds (10/30), 1+30=31 → ekstensif.
    """
    sis = {"n1": NodeTrace(depth=0, causal_chain=[], path=["n1"], source_seed="n1")}
    propagated = {
        f"p{i}": NodeTrace(depth=1, causal_chain=["CALLS"], path=["n1", f"p{i}"], source_seed="n1")
        for i in range(30)
    }
    cis = CISResult(sis_nodes=sis, propagated_nodes=propagated)
    # 1 + 30 = 31 nodes → ekstensif
    assert _compute_scope(cis) == "ekstensif"


def test_compute_scope_empty_cis():
    """Empty CIS → terlokalisasi (no nodes = no scope)."""
    cis = CISResult(sis_nodes={}, propagated_nodes={})
    assert _compute_scope(cis) == "terlokalisasi"


def test_compute_scope_custom_thresholds():
    """Phase 2.9 (F-NEW-5): _compute_scope respects custom settings thresholds."""
    class _FakeSettings2:
        scope_local_max = 5
        scope_medium_max = 15

    # 5 nodes → terlokalisasi (≤5)
    assert _compute_scope(_make_cis([f"n{i}" for i in range(5)]), _FakeSettings2()) == "terlokalisasi"
    # 6 nodes → menengah (6-15)
    assert _compute_scope(_make_cis([f"n{i}" for i in range(6)]), _FakeSettings2()) == "menengah"
    # 16 nodes → ekstensif (>15)
    assert _compute_scope(_make_cis([f"n{i}" for i in range(16)]), _FakeSettings2()) == "ekstensif"


# ---------------------------------------------------------------------------
# Sprint 10.1 — Strategy 2: _apply_hard_limit tests
# ---------------------------------------------------------------------------


from impactracer.pipeline.context_builder import _apply_hard_limit, _HARD_CHAR_LIMIT


def _make_combined(nodes: list[tuple[str, int]]) -> dict:
    """Build a {node_id: NodeTrace} dict from (node_id, depth) pairs."""
    result = {}
    for nid, depth in nodes:
        chain = ["CALLS"] * depth if depth > 0 else []
        result[nid] = NodeTrace(
            depth=depth,
            causal_chain=chain,
            path=["seed"] + [f"n{i}" for i in range(depth)],
            source_seed="seed",
        )
    return result


def test_apply_hard_limit_no_op_when_under_limit() -> None:
    """Small node list well within limit — no nodes dropped, no warning."""
    nodes = [("n1", 0), ("n2", 1), ("n3", 2)]
    combined = _make_combined(nodes)
    sorted_ids = ["n1", "n2", "n3"]
    trimmed, warning = _apply_hard_limit(sorted_ids, combined, char_limit=_HARD_CHAR_LIMIT)
    assert trimmed == sorted_ids
    assert warning == ""


def test_apply_hard_limit_drops_deepest_first() -> None:
    """When limit is tight, deepest nodes are dropped first; shallowest are kept.

    Phase 1 (F-2/E-3): proxy raised to 2000 chars/node.
    With char_limit=6000:
      early-exit check: 4 * 2000 = 8000 > 6000 → proceeds to trim logic.
      immune (seed, depth=0): immune_budget = 1 * 2000 = 2000
      available_for_droppable = 6000 - 2000 = 4000
      max_keep = 4000 // 2000 = 2
      droppable sorted ascending by depth: [(1,n1), (2,n2), (3,n3)]
      kept = [(1,n1), (2,n2)], dropped = [(3,n3)]
    """
    nodes = [("seed", 0), ("n1", 1), ("n2", 2), ("n3", 3)]
    combined = _make_combined(nodes)
    sorted_ids = ["seed", "n1", "n2", "n3"]
    trimmed, warning = _apply_hard_limit(sorted_ids, combined, char_limit=6000)
    assert "seed" in trimmed    # depth-0 is immune
    assert "n1" in trimmed      # shallowest propagated — kept
    assert "n2" in trimmed      # second-shallowest — kept
    assert "n3" not in trimmed  # deepest — dropped
    assert warning != ""         # warning emitted


def test_apply_hard_limit_seed_always_survives() -> None:
    """SIS seeds (depth=0) are immune to hard-limit truncation.

    Phase 1 (F-2/E-3): with 2000 chars/node proxy and 10 seeds:
      immune_budget = 10 * 2000 = 20000 > any droppable budget → all survive.
    """
    nodes = [(f"seed{i}", 0) for i in range(10)]
    combined = _make_combined(nodes)
    sorted_ids = [f"seed{i}" for i in range(10)]
    # Even with a tiny char_limit, depth-0 nodes survive (immune).
    trimmed, warning = _apply_hard_limit(sorted_ids, combined, char_limit=100)
    for nid in sorted_ids:
        assert nid in trimmed


def test_apply_hard_limit_returns_warning_with_max_depth() -> None:
    """Warning message mentions the max depth of surviving nodes.

    Phase 1 (F-2/E-3): char_limit=6000 keeps n1 and n2, drops n3.
    Warning must include 'SYSTEM WARNING' and the truncation notice.
    """
    nodes = [("seed", 0), ("n1", 1), ("n2", 2), ("n3", 3)]
    combined = _make_combined(nodes)
    sorted_ids = ["seed", "n1", "n2", "n3"]
    trimmed, warning = _apply_hard_limit(sorted_ids, combined, char_limit=6000)
    assert "SYSTEM WARNING" in warning
    assert "truncated" in warning.lower() or "Truncated" in warning
    assert "n1" in trimmed   # n1 kept (depth 1)
    assert "n2" in trimmed   # n2 kept (depth 2)


def test_apply_hard_limit_no_propagated_nodes() -> None:
    """Only SIS seeds (all depth=0) — no drops possible regardless of limit."""
    nodes = [("s1", 0), ("s2", 0), ("s3", 0)]
    combined = _make_combined(nodes)
    sorted_ids = ["s1", "s2", "s3"]
    # 3 * 2000 = 6000 > 500 → would normally enter trim logic,
    # but all are depth-0 (immune) → available_for_droppable = 0 → no drops.
    trimmed, warning = _apply_hard_limit(sorted_ids, combined, char_limit=500)
    assert set(trimmed) == {"s1", "s2", "s3"}
    assert warning == ""


def test_apply_hard_limit_preserves_original_order() -> None:
    """Surviving nodes preserve their relative order from sorted_ids."""
    nodes = [("seed", 0), ("a", 1), ("b", 2), ("c", 3)]
    combined = _make_combined(nodes)
    sorted_ids = ["b", "a", "seed", "c"]  # arbitrary order
    # Use a limit large enough that everything fits (4 * 2000 = 8000 <= 240000)
    trimmed, _ = _apply_hard_limit(sorted_ids, combined, char_limit=_HARD_CHAR_LIMIT)
    assert trimmed == sorted_ids  # order preserved when no drops


# ---------------------------------------------------------------------------
# Phase 1: context sort order — SIS seeds must come before propagated nodes
# ---------------------------------------------------------------------------


from impactracer.pipeline.context_builder import build_context
from impactracer.shared.models import CRInterpretation


def _make_cr_interp() -> CRInterpretation:
    return CRInterpretation(
        is_actionable=True,
        primary_intent="Change X",
        change_type="MODIFICATION",
        affected_layers=["code"],
        domain_concepts=["auth"],
        search_queries=["auth change", "auth modification"],
    )


class _FakeSettings:
    llm_max_context_tokens = 100_000
    synthesis_system_prompt_tokens = 3_000
    output_reserve_tokens = 200
    top_k_backlinks_per_node = 3


def test_context_sort_seeds_before_propagated() -> None:
    """Phase 1 (E-NEW-6): SIS seeds (group 0) appear BEFORE propagated nodes (group 1).

    The previous code had this inverted — propagated nodes got sort key (0, ...)
    and seeds got (1, ...), meaning seeds were truncated first under budget pressure.
    The corrected code must place all SIS seeds before any propagated node.
    """
    # Build a CIS with 1 seed and 2 propagated nodes of varying severity.
    seed_trace = NodeTrace(depth=0, causal_chain=[], path=["seed1"], source_seed="seed1")
    prop_high = NodeTrace(
        depth=1, causal_chain=["IMPLEMENTS"], path=["seed1", "prop_high"],
        source_seed="seed1",
    )
    prop_low = NodeTrace(
        depth=1, causal_chain=["IMPORTS"], path=["seed1", "prop_low"],
        source_seed="seed1",
    )

    cis = CISResult(
        sis_nodes={"seed1": seed_trace},
        propagated_nodes={"prop_high": prop_high, "prop_low": prop_low},
    )

    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE code_nodes (node_id TEXT PRIMARY KEY, node_type TEXT, "
        "internal_logic_abstraction TEXT, source_code TEXT)"
    )
    for nid in ["seed1", "prop_high", "prop_low"]:
        conn.execute(
            "INSERT INTO code_nodes VALUES (?, ?, ?, ?)",
            (nid, "Function", None, f"// source of {nid}"),
        )
    conn.commit()

    context = build_context(
        cr_text="Change X",
        cr_interp=_make_cr_interp(),
        cis=cis,
        backlinks={},
        snippets={"seed1": "seed code", "prop_high": "prop high code", "prop_low": "prop low code"},
        settings=_FakeSettings(),
    )

    # Find positions of each node block in the assembled context string.
    pos_seed = context.find("--- NODE: seed1 ---")
    pos_prop_high = context.find("--- NODE: prop_high ---")
    pos_prop_low = context.find("--- NODE: prop_low ---")

    # The seed must appear BEFORE both propagated nodes.
    assert pos_seed != -1, "seed1 not in context"
    assert pos_prop_high != -1, "prop_high not in context"
    assert pos_prop_low != -1, "prop_low not in context"
    assert pos_seed < pos_prop_high, "seed1 must appear before prop_high"
    assert pos_seed < pos_prop_low, "seed1 must appear before prop_low"


def test_context_sort_high_severity_propagated_before_low() -> None:
    """Phase 1 (E-NEW-6): among propagated nodes, higher severity appears first."""
    seed_trace = NodeTrace(depth=0, causal_chain=[], path=["seed1"], source_seed="seed1")
    prop_tinggi = NodeTrace(
        depth=1, causal_chain=["IMPLEMENTS"], path=["seed1", "prop_tinggi"],
        source_seed="seed1",
    )
    prop_rendah = NodeTrace(
        depth=1, causal_chain=["IMPORTS"], path=["seed1", "prop_rendah"],
        source_seed="seed1",
    )

    cis = CISResult(
        sis_nodes={"seed1": seed_trace},
        propagated_nodes={"prop_tinggi": prop_tinggi, "prop_rendah": prop_rendah},
    )

    context = build_context(
        cr_text="Change X",
        cr_interp=_make_cr_interp(),
        cis=cis,
        backlinks={},
        snippets={},
        settings=_FakeSettings(),
    )

    pos_tinggi = context.find("--- NODE: prop_tinggi ---")
    pos_rendah = context.find("--- NODE: prop_rendah ---")

    assert pos_tinggi != -1 and pos_rendah != -1
    # Tinggi severity → sort key (1, 0, 1, "prop_tinggi") < (1, 2, 1, "prop_rendah")
    assert pos_tinggi < pos_rendah, "Tinggi severity node must appear before Rendah"
