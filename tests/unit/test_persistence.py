"""Acceptance tests for SQLite schema and ChromaDB collection initialization."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from impactracer.persistence.sqlite_client import connect, init_schema
from impactracer.persistence.chroma_client import get_client, init_collections
from impactracer.shared.constants import EDGE_CONFIG, RETIRED_PROPAGATION_EDGES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_NODE_TYPE = "Function"
_VALID_CLASSIFICATION = "UTILITY"

_ALL_NODE_TYPES = (
    "File", "Class", "Function", "Method",
    "Interface", "TypeAlias", "Enum",
    "ExternalPackage", "InterfaceField",
)

# All 14 extracted/stored edge types: 10 propagated (EDGE_CONFIG) + 4 retired.
# The DDL CHECK accepts all 14; only EDGE_CONFIG was trimmed to 10 in 2026-06.
_ALL_EDGE_TYPES = tuple(EDGE_CONFIG.keys()) + tuple(RETIRED_PROPAGATION_EDGES.keys())


def _fresh_conn() -> sqlite3.Connection:
    """Return an in-memory SQLite connection with the schema initialized."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema(conn)
    return conn


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r[0] for r in rows}


def _insert_code_node(conn: sqlite3.Connection, node_id: str = "src/lib/a.ts::fn") -> None:
    conn.execute(
        "INSERT INTO code_nodes (node_id, node_type, name) VALUES (?, ?, ?)",
        (node_id, _VALID_NODE_TYPE, "fn"),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# 1. Six tables exist after init_schema
# ---------------------------------------------------------------------------

EXPECTED_TABLES = {
    "code_nodes",
    "structural_edges",
    "doc_code_candidates",
    "file_hashes",
    "file_dependencies",
    "index_metadata",
}


def test_init_schema_creates_all_six_tables() -> None:
    """init_schema on a fresh in-memory DB creates all 6 required tables."""
    conn = _fresh_conn()
    assert _table_names(conn) >= EXPECTED_TABLES


# ---------------------------------------------------------------------------
# 2. init_schema is idempotent
# ---------------------------------------------------------------------------


def test_init_schema_idempotent() -> None:
    """Calling init_schema twice on the same connection does not raise."""
    conn = _fresh_conn()
    init_schema(conn)  # second call
    assert _table_names(conn) >= EXPECTED_TABLES


# ---------------------------------------------------------------------------
# 3. code_nodes.node_type CHECK constraint rejects invalid value
# ---------------------------------------------------------------------------


def test_node_type_check_rejects_invalid() -> None:
    """INSERT with an unknown node_type raises IntegrityError."""
    conn = _fresh_conn()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO code_nodes (node_id, node_type, name) VALUES (?, ?, ?)",
            ("x::y", "Widget", "y"),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# 4. structural_edges.edge_type CHECK rejects invalid value
# ---------------------------------------------------------------------------


def test_edge_type_check_rejects_invalid() -> None:
    """INSERT with an unknown edge_type raises IntegrityError."""
    conn = _fresh_conn()
    _insert_code_node(conn, "src/a.ts::fn1")
    _insert_code_node(conn, "src/a.ts::fn2")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO structural_edges (source_id, target_id, edge_type) VALUES (?, ?, ?)",
            ("src/a.ts::fn1", "src/a.ts::fn2", "REFERENCES"),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# 5. All 13 canonical edge types are accepted
# ---------------------------------------------------------------------------


def test_all_14_edge_types_accepted() -> None:
    """All 14 stored edge types (10 propagated + 4 retired) insert without error."""
    conn = _fresh_conn()
    src_id = "src/lib/a.ts::source"
    tgt_id = "src/lib/a.ts::target"
    _insert_code_node(conn, src_id)
    _insert_code_node(conn, tgt_id)

    for edge_type in _ALL_EDGE_TYPES:
        conn.execute(
            "INSERT OR IGNORE INTO structural_edges (source_id, target_id, edge_type)"
            " VALUES (?, ?, ?)",
            (src_id, tgt_id, edge_type),
        )
    conn.commit()

    rows = conn.execute("SELECT edge_type FROM structural_edges").fetchall()
    inserted_types = {r[0] for r in rows}
    assert inserted_types == set(_ALL_EDGE_TYPES)


def test_edge_config_has_exactly_9_propagated_entries() -> None:
    """EDGE_CONFIG defines the 9 propagated edge types (post 2026-06 retirements).

    14 edge types are extracted/stored. 5 are retired from propagation and live
    in RETIRED_PROPAGATION_EDGES: PASSES_CALLBACK, HOOK_DEPENDS_ON,
    DEPENDS_ON_EXTERNAL, CLIENT_API_CALLS (wave 1, inert), and FIELDS_ACCESSED
    (wave 2 — its only target, the InterfaceField node type, was retired). The
    two sets together still total 14.
    """
    assert len(EDGE_CONFIG) == 9
    assert len(RETIRED_PROPAGATION_EDGES) == 5
    assert len(set(EDGE_CONFIG) | set(RETIRED_PROPAGATION_EDGES)) == 14
    assert set(EDGE_CONFIG).isdisjoint(RETIRED_PROPAGATION_EDGES)
    # FIELDS_ACCESSED specifically is retired, not active.
    assert "FIELDS_ACCESSED" in RETIRED_PROPAGATION_EDGES
    assert "FIELDS_ACCESSED" not in EDGE_CONFIG


# ---------------------------------------------------------------------------
# 6. code_nodes.file_classification CHECK rejects invalid non-NULL value
# ---------------------------------------------------------------------------


def test_file_classification_check_rejects_invalid() -> None:
    """INSERT with an unknown non-NULL file_classification raises IntegrityError."""
    conn = _fresh_conn()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO code_nodes (node_id, node_type, name, file_classification)"
            " VALUES (?, ?, ?, ?)",
            ("src/x.ts::x", _VALID_NODE_TYPE, "x", "SERVICE"),
        )
        conn.commit()


def test_file_classification_accepts_null() -> None:
    """NULL file_classification is permitted (unclassified file)."""
    conn = _fresh_conn()
    conn.execute(
        "INSERT INTO code_nodes (node_id, node_type, name, file_classification)"
        " VALUES (?, ?, ?, ?)",
        ("src/x.ts::x", _VALID_NODE_TYPE, "x", None),
    )
    conn.commit()
    row = conn.execute(
        "SELECT file_classification FROM code_nodes WHERE node_id = ?",
        ("src/x.ts::x",),
    ).fetchone()
    assert row[0] is None


def test_file_classification_accepts_all_valid_values() -> None:
    """All 5 canonical file_classification values insert without error."""
    valid_values = ("API_ROUTE", "PAGE_COMPONENT", "UI_COMPONENT", "UTILITY", "TYPE_DEFINITION")
    conn = _fresh_conn()
    for i, cls in enumerate(valid_values):
        conn.execute(
            "INSERT INTO code_nodes (node_id, node_type, name, file_classification)"
            " VALUES (?, ?, ?, ?)",
            (f"src/x{i}.ts::fn", _VALID_NODE_TYPE, "fn", cls),
        )
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM code_nodes").fetchone()[0]
    assert count == 5


# ---------------------------------------------------------------------------
# 7. ChromaDB init_collections returns two cosine-space collections
# ---------------------------------------------------------------------------


def test_init_collections_returns_two_cosine_collections(tmp_path: pytest.TempPathFactory) -> None:
    """init_collections creates doc_chunks and code_units with hnsw:space='cosine'."""
    chroma_dir = str(tmp_path / "chroma")
    client = get_client(chroma_dir)
    doc_col, code_col = init_collections(client)

    assert doc_col.name == "doc_chunks"
    assert code_col.name == "code_units"
    assert doc_col.metadata.get("hnsw:space") == "cosine"
    assert code_col.metadata.get("hnsw:space") == "cosine"


def test_init_collections_idempotent(tmp_path: pytest.TempPathFactory) -> None:
    """Calling init_collections twice returns collections, not raises."""
    chroma_dir = str(tmp_path / "chroma2")
    client = get_client(chroma_dir)
    doc1, code1 = init_collections(client)
    doc2, code2 = init_collections(client)

    assert doc1.name == doc2.name
    assert code1.name == code2.name


# ---------------------------------------------------------------------------
# 8. connect() helper applies pragmas
# ---------------------------------------------------------------------------


def test_connect_creates_parent_directory(tmp_path: pytest.TempPathFactory) -> None:
    """connect() creates parent directories and returns a working connection."""
    db_path = str(tmp_path / "nested" / "dir" / "test.db")
    conn = connect(db_path)
    # Connection is usable
    conn.execute("SELECT 1")
    conn.close()
    assert Path(db_path).exists()
