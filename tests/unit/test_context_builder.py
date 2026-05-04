"""Unit tests for pipeline/context_builder.py (FR-E1, FR-E2) and runner._compute_scope (N8)."""

from __future__ import annotations

import sqlite3

import pytest

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
    """≤5 nodes → terlokalisasi."""
    cis = _make_cis(["n1", "n2", "n3"])
    assert _compute_scope(cis) == "terlokalisasi"


def test_compute_scope_terlokalisasi_boundary():
    """Exactly 5 nodes → terlokalisasi."""
    cis = _make_cis([f"n{i}" for i in range(5)])
    assert _compute_scope(cis) == "terlokalisasi"


def test_compute_scope_menengah_lower_bound():
    """6 nodes → menengah."""
    cis = _make_cis([f"n{i}" for i in range(6)])
    assert _compute_scope(cis) == "menengah"


def test_compute_scope_menengah_upper_bound():
    """Exactly 15 nodes → menengah."""
    cis = _make_cis([f"n{i}" for i in range(15)])
    assert _compute_scope(cis) == "menengah"


def test_compute_scope_ekstensif():
    """>15 nodes → ekstensif."""
    cis = _make_cis([f"n{i}" for i in range(16)])
    assert _compute_scope(cis) == "ekstensif"


def test_compute_scope_includes_propagated_nodes():
    """Propagated nodes also count toward scope."""
    sis = {"n1": NodeTrace(depth=0, causal_chain=[], path=["n1"], source_seed="n1")}
    propagated = {
        f"p{i}": NodeTrace(depth=1, causal_chain=["CALLS"], path=["n1", f"p{i}"], source_seed="n1")
        for i in range(15)
    }
    cis = CISResult(sis_nodes=sis, propagated_nodes=propagated)
    # 1 + 15 = 16 nodes → ekstensif
    assert _compute_scope(cis) == "ekstensif"


def test_compute_scope_empty_cis():
    """Empty CIS → terlokalisasi (no nodes = no scope)."""
    cis = CISResult(sis_nodes={}, propagated_nodes={})
    assert _compute_scope(cis) == "terlokalisasi"
