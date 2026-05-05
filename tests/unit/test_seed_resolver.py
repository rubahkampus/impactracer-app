"""Unit tests for seed_resolver.resolve_doc_to_code (FR-C6)."""

from __future__ import annotations

import sqlite3

import pytest

from impactracer.pipeline.seed_resolver import resolve_doc_to_code
from impactracer.persistence.sqlite_client import init_schema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def conn() -> sqlite3.Connection:
    """In-memory SQLite DB with schema and minimal seed data."""
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys = ON")
    init_schema(c)
    # Insert a real code node so it's in code_node_set.
    c.execute(
        "INSERT INTO code_nodes (node_id, node_type, name, file_path, embed_text) "
        "VALUES (?, ?, ?, ?, ?)",
        ("src/lib/services/auth.service.ts::loginUser", "Function",
         "loginUser", "src/lib/services/auth.service.ts", "loginUser function"),
    )
    # Insert a second code node that appears in traceability.
    c.execute(
        "INSERT INTO code_nodes (node_id, node_type, name, file_path, embed_text) "
        "VALUES (?, ?, ?, ?, ?)",
        ("src/lib/services/wallet.service.ts", "File",
         "wallet.service.ts", "src/lib/services/wallet.service.ts", "wallet service file"),
    )
    # Insert doc_code_candidates for a doc chunk.
    c.execute(
        "INSERT INTO doc_code_candidates (code_id, doc_id, weighted_similarity_score) "
        "VALUES (?, ?, ?)",
        ("src/lib/services/auth.service.ts::loginUser", "sdd__v_1_auth", 0.6671),
    )
    c.execute(
        "INSERT INTO doc_code_candidates (code_id, doc_id, weighted_similarity_score) "
        "VALUES (?, ?, ?)",
        ("src/lib/services/wallet.service.ts", "sdd__v_1_auth", 0.5000),
    )
    c.commit()
    return c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_direct_code_seed_passthrough(conn: sqlite3.Connection) -> None:
    """A SIS entry that is already a code node bypasses doc resolution."""
    resolutions, doc_to_code_map, direct_code_seeds = resolve_doc_to_code(
        sis_ids=["src/lib/services/auth.service.ts::loginUser"],
        conn=conn,
        top_k=5,
    )
    assert direct_code_seeds == ["src/lib/services/auth.service.ts::loginUser"]
    assert resolutions == []
    assert doc_to_code_map == {}


def test_doc_chunk_resolves_to_code(conn: sqlite3.Connection) -> None:
    """A doc-chunk SIS entry resolves via doc_code_candidates."""
    resolutions, doc_to_code_map, direct_code_seeds = resolve_doc_to_code(
        sis_ids=["sdd__v_1_auth"],
        conn=conn,
        top_k=5,
    )
    assert direct_code_seeds == []
    assert len(resolutions) == 1
    assert resolutions[0]["doc_id"] == "sdd__v_1_auth"
    # Both code nodes should be present, ordered by score desc.
    assert resolutions[0]["code_ids"][0] == "src/lib/services/auth.service.ts::loginUser"
    assert "sdd__v_1_auth" in doc_to_code_map


def test_top_k_limits_resolution(conn: sqlite3.Connection) -> None:
    """top_k=1 returns only the highest-scored code node per doc chunk."""
    resolutions, doc_to_code_map, _ = resolve_doc_to_code(
        sis_ids=["sdd__v_1_auth"],
        conn=conn,
        top_k=1,
    )
    assert len(resolutions[0]["code_ids"]) == 1
    assert resolutions[0]["code_ids"][0] == "src/lib/services/auth.service.ts::loginUser"


def test_stranded_doc_chunk_skipped(conn: sqlite3.Connection) -> None:
    """A doc chunk with no candidates produces no resolution entry."""
    resolutions, doc_to_code_map, direct_code_seeds = resolve_doc_to_code(
        sis_ids=["srs__i_1_general_description"],
        conn=conn,
        top_k=5,
    )
    assert resolutions == []
    assert doc_to_code_map == {}
    assert direct_code_seeds == []


def test_mixed_sis_ids(conn: sqlite3.Connection) -> None:
    """Mix of direct code node and doc chunk in one call."""
    resolutions, doc_to_code_map, direct_code_seeds = resolve_doc_to_code(
        sis_ids=[
            "src/lib/services/auth.service.ts::loginUser",
            "sdd__v_1_auth",
        ],
        conn=conn,
        top_k=5,
    )
    assert "src/lib/services/auth.service.ts::loginUser" in direct_code_seeds
    assert len(resolutions) == 1
    assert resolutions[0]["doc_id"] == "sdd__v_1_auth"


def test_empty_sis_ids(conn: sqlite3.Connection) -> None:
    """Empty input returns empty outputs."""
    resolutions, doc_to_code_map, direct_code_seeds = resolve_doc_to_code(
        sis_ids=[],
        conn=conn,
        top_k=5,
    )
    assert resolutions == []
    assert doc_to_code_map == {}
    assert direct_code_seeds == []


def test_doc_to_code_map_matches_resolutions(conn: sqlite3.Connection) -> None:
    """doc_to_code_map has identical entries as resolutions."""
    resolutions, doc_to_code_map, _ = resolve_doc_to_code(
        sis_ids=["sdd__v_1_auth"],
        conn=conn,
        top_k=5,
    )
    for r in resolutions:
        assert r["doc_id"] in doc_to_code_map
        assert doc_to_code_map[r["doc_id"]] == r["code_ids"]
