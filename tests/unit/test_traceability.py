"""Tests for indexer/traceability.py (FR-A7).

Uses crafted 2-D vectors so cosine scores and layer_compat weights are
easy to reason about exactly.
"""

from __future__ import annotations

import math
import sqlite3

import numpy as np
import pytest

from impactracer.indexer.traceability import compute_and_store
from impactracer.persistence.sqlite_client import init_schema
from impactracer.shared.constants import layer_compat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    init_schema(conn)
    return conn


def _seed_code_nodes(conn: sqlite3.Connection, code_ids: list[str]) -> None:
    """Insert minimal code_nodes rows so FK constraints are satisfied."""
    conn.executemany(
        "INSERT OR IGNORE INTO code_nodes (node_id, node_type, name) VALUES (?, 'File', ?)",
        [(cid, cid) for cid in code_ids],
    )
    conn.commit()


def _unit(angle_deg: float) -> np.ndarray:
    """2-D unit vector at the given angle."""
    a = math.radians(angle_deg)
    return np.array([math.cos(a), math.sin(a)], dtype=np.float32)


def _rows(conn: sqlite3.Connection) -> list[tuple]:
    return conn.execute(
        "SELECT code_id, doc_id, weighted_similarity_score "
        "FROM doc_code_candidates ORDER BY code_id, doc_id"
    ).fetchall()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_db():
    """
    2 code nodes × 3 doc nodes.

    code_0 (API_ROUTE, angle=0°):   [1, 0]
    code_1 (UTILITY, angle=90°):    [0, 1]

    doc_A (FR, angle=10°):  nearly aligned with code_0
    doc_B (FR, angle=80°):  nearly aligned with code_1
    doc_C (General, angle=45°): equal to both

    layer_compat:
        API_ROUTE / FR      = 1.0
        API_ROUTE / General = 0.5
        UTILITY   / FR      = 0.7
        UTILITY   / General = 0.5
    """
    conn = _make_db()

    code_vecs = {
        "code_0": _unit(0),
        "code_1": _unit(90),
    }
    doc_vecs = {
        "doc_A": _unit(10),
        "doc_B": _unit(80),
        "doc_C": _unit(45),
    }
    code_meta = {
        "code_0": {"file_classification": "API_ROUTE"},
        "code_1": {"file_classification": "UTILITY"},
    }
    doc_meta = {
        "doc_A": {"chunk_type": "FR"},
        "doc_B": {"chunk_type": "FR"},
        "doc_C": {"chunk_type": "General"},
    }
    _seed_code_nodes(conn, list(code_vecs.keys()))
    return conn, code_vecs, doc_vecs, code_meta, doc_meta


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_returns_row_count(simple_db):
    conn, cv, dv, cm, dm = simple_db
    n = compute_and_store(cv, dv, cm, dm, top_k=3, min_similarity=0.0, conn=conn)
    assert isinstance(n, int)
    assert n > 0


def test_rows_written_to_db(simple_db):
    conn, cv, dv, cm, dm = simple_db
    compute_and_store(cv, dv, cm, dm, top_k=3, min_similarity=0.0, conn=conn)
    rows = _rows(conn)
    assert len(rows) > 0


def test_weighted_score_formula(simple_db):
    """Spot-check: code_0 (API_ROUTE) vs doc_A (FR, angle=10°).

    cosine(0°, 10°) = cos(10°) ≈ 0.9848
    layer_compat(API_ROUTE, FR) = 1.0
    weighted = 0.9848 * 1.0 ≈ 0.9848
    """
    conn, cv, dv, cm, dm = simple_db
    compute_and_store(cv, dv, cm, dm, top_k=3, min_similarity=0.0, conn=conn)
    row = conn.execute(
        "SELECT weighted_similarity_score FROM doc_code_candidates "
        "WHERE code_id='code_0' AND doc_id='doc_A'"
    ).fetchone()
    assert row is not None
    expected = math.cos(math.radians(10)) * layer_compat("API_ROUTE", "FR")
    assert abs(row[0] - expected) < 1e-5


def test_layer_compat_applied_correctly(simple_db):
    """code_0 (API_ROUTE) vs doc_C (General, 45°).

    cosine(0°, 45°) = cos(45°) ≈ 0.7071
    layer_compat(API_ROUTE, General) = 0.5
    weighted ≈ 0.3536
    """
    conn, cv, dv, cm, dm = simple_db
    compute_and_store(cv, dv, cm, dm, top_k=3, min_similarity=0.0, conn=conn)
    row = conn.execute(
        "SELECT weighted_similarity_score FROM doc_code_candidates "
        "WHERE code_id='code_0' AND doc_id='doc_C'"
    ).fetchone()
    assert row is not None
    expected = math.cos(math.radians(45)) * layer_compat("API_ROUTE", "General")
    assert abs(row[0] - expected) < 1e-5


def test_min_similarity_filter(simple_db):
    """With a high min_similarity, only the best pairs should survive."""
    conn, cv, dv, cm, dm = simple_db
    # code_0/doc_A weighted ≈ 0.985, code_1/doc_B weighted ≈ 0.985*0.7 ≈ 0.689
    # All General pairs are ≈ 0.35  → filtered out at 0.60
    n = compute_and_store(cv, dv, cm, dm, top_k=5, min_similarity=0.60, conn=conn)
    rows = _rows(conn)
    for _, _, score in rows:
        assert score >= 0.60
    assert n == len(rows)


def test_top_k_limit_per_code_node():
    """top_k=1 per direction. With 1 code node and 6 doc nodes:
    forward pass gives 1 pair (code→top doc); reverse pass gives 6 pairs
    (each doc→its top code = c0). Union = 6 unique pairs.
    """
    conn = _make_db()
    code_vecs = {"c0": _unit(0)}
    doc_vecs = {f"d{i}": _unit(i * 5) for i in range(6)}
    code_meta = {"c0": {"file_classification": "API_ROUTE"}}
    doc_meta = {f"d{i}": {"chunk_type": "FR"} for i in range(6)}
    _seed_code_nodes(conn, list(code_vecs.keys()))

    compute_and_store(code_vecs, doc_vecs, code_meta, doc_meta,
                      top_k=1, min_similarity=0.0, conn=conn)
    rows = _rows(conn)
    # 6 unique (code, doc) pairs — forward contributes 1, reverse contributes 6,
    # union deduplicates to 6.
    assert len(rows) == 6


def test_idempotent(simple_db):
    """Running twice on identical input yields identical rows."""
    conn, cv, dv, cm, dm = simple_db
    compute_and_store(cv, dv, cm, dm, top_k=3, min_similarity=0.0, conn=conn)
    rows_first = _rows(conn)
    compute_and_store(cv, dv, cm, dm, top_k=3, min_similarity=0.0, conn=conn)
    rows_second = _rows(conn)
    assert rows_first == rows_second


def test_empty_code_vecs_returns_zero():
    conn = _make_db()
    n = compute_and_store({}, {"d0": _unit(0)},
                          {}, {"d0": {"chunk_type": "FR"}},
                          top_k=5, min_similarity=0.0, conn=conn)
    assert n == 0


def test_empty_doc_vecs_returns_zero():
    conn = _make_db()
    n = compute_and_store({"c0": _unit(0)}, {},
                          {"c0": {"file_classification": "API_ROUTE"}}, {},
                          top_k=5, min_similarity=0.0, conn=conn)
    assert n == 0


def test_none_classification(simple_db):
    """None file_classification falls back to LAYER_COMPAT[None] row."""
    conn = _make_db()
    code_vecs = {"c_none": _unit(0)}
    doc_vecs = {"d_fr": _unit(10)}
    code_meta = {"c_none": {"file_classification": None}}
    doc_meta = {"d_fr": {"chunk_type": "FR"}}
    _seed_code_nodes(conn, list(code_vecs.keys()))

    compute_and_store(code_vecs, doc_vecs, code_meta, doc_meta,
                      top_k=1, min_similarity=0.0, conn=conn)
    row = conn.execute(
        "SELECT weighted_similarity_score FROM doc_code_candidates "
        "WHERE code_id='c_none' AND doc_id='d_fr'"
    ).fetchone()
    assert row is not None
    expected = math.cos(math.radians(10)) * layer_compat(None, "FR")
    assert abs(row[0] - expected) < 1e-5


def test_stale_rows_replaced_on_rerun():
    """Old rows for a code_id are deleted before new ones are inserted."""
    conn = _make_db()
    cv = {"c0": _unit(0)}
    dv_first = {"d0": _unit(5), "d1": _unit(85)}
    cm = {"c0": {"file_classification": "API_ROUTE"}}
    dm_first = {"d0": {"chunk_type": "FR"}, "d1": {"chunk_type": "FR"}}
    _seed_code_nodes(conn, list(cv.keys()))

    compute_and_store(cv, dv_first, cm, dm_first, top_k=2, min_similarity=0.0, conn=conn)
    rows_first = {(r[0], r[1]) for r in _rows(conn)}

    # Second run with completely different doc set
    dv_second = {"d2": _unit(20)}
    dm_second = {"d2": {"chunk_type": "Design"}}
    compute_and_store(cv, dv_second, cm, dm_second, top_k=2, min_similarity=0.0, conn=conn)
    rows_second = {(r[0], r[1]) for r in _rows(conn)}

    # Old doc IDs must be gone
    assert ("c0", "d0") not in rows_second
    assert ("c0", "d1") not in rows_second
    assert ("c0", "d2") in rows_second
