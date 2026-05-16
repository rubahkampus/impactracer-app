"""Sprint 13-W2 acceptance: raw-CR multilingual bridge + traceability pool seeding.

These tests use small synthetic doubles for embedder / chroma collection /
SQLite so they exercise the new branches without standing up the full index.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from impactracer.evaluation.variant_flags import VariantFlags
from impactracer.pipeline.retriever import hybrid_search
from impactracer.shared.models import CRInterpretation


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------

class _StubEmbedder:
    """Returns a deterministic vector per text. Records all embed_single calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_single(self, text: str):
        self.calls.append(text)
        # Vector encodes a simple feature: keyword presence flags.
        v = [
            1.0 if "grace" in text.lower() else 0.0,
            1.0 if "template" in text.lower() else 0.0,
            1.0 if "schema" in text.lower() else 0.0,
            1.0 if "ilustrator" in text.lower() else 0.0,  # Indonesian for "illustrator"
        ]
        return v


class _StubCol:
    """ChromaDB collection double. ``query`` returns canned ids / distances by
    matching against query_embeddings[0] keyword flags."""

    def __init__(self, name: str, scenario: dict) -> None:
        self.name = name
        self._scenario = scenario  # {feature_idx: [(id, distance), ...]}

    def query(self, query_embeddings, n_results, where=None, include=None):
        vec = query_embeddings[0]
        # Pick the feature axis with the largest score.
        if not vec or max(vec) == 0:
            return {"ids": [[]], "distances": [[]]}
        idx = vec.index(max(vec))
        pairs = self._scenario.get(idx, [])
        ids = [p[0] for p in pairs[:n_results]]
        dists = [p[1] for p in pairs[:n_results]]
        return {"ids": [ids], "distances": [dists]}

    def get(self, ids=None, include=None):
        # For hydration step; return minimal stubs.
        if ids is None:
            return {"ids": [], "documents": [], "metadatas": []}
        return {
            "ids": list(ids),
            "documents": [f"doc for {cid}" for cid in ids],
            "metadatas": [{"file_path": cid.split("::")[0] if "::" in cid else cid,
                           "chunk_type": "Design",
                           "source_file": cid.split("::")[0] if "::" in cid else cid,
                           "section_title": cid} for cid in ids],
        }


def _make_settings(**overrides):
    """Build a minimal settings object covering all flags hybrid_search reads."""
    base = {
        "top_k_per_query": 30,
        "top_k_rrf_pool": 200,
        "rrf_k": 60,
        "max_admitted_seeds": 15,
        "enable_raw_cr_dense_pass": True,
        "raw_cr_dense_top_k": 60,
        "enable_traceability_pool_seeding": True,
        "traceability_seed_top_k_per_doc": 5,
        "traceability_seed_min_score": 0.40,
        "traceability_seed_synthetic_rank": 5,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_ctx(*, code_col, doc_col, conn=None, code_node_ids=()):
    """Build a stub PipelineContext.

    ``code_node_ids`` is an optional list of node_ids to pre-insert into
    code_nodes so the hydration query inside hybrid_search succeeds. If
    ``conn`` is None and ``code_node_ids`` is non-empty, a fresh in-memory
    SQLite is created with the canonical schema.
    """
    if conn is None and code_node_ids:
        from impactracer.persistence.sqlite_client import init_schema
        conn = sqlite3.connect(":memory:")
        init_schema(conn)
    if conn is not None and code_node_ids:
        for nid in code_node_ids:
            fp = nid.split("::", 1)[0] if "::" in nid else nid
            nm = nid.split("::", 1)[1] if "::" in nid else nid
            conn.execute(
                "INSERT OR REPLACE INTO code_nodes (node_id, node_type, name, file_path) "
                "VALUES (?, ?, ?, ?)",
                (nid, "Function", nm, fp),
            )
        conn.commit()
    return SimpleNamespace(
        variant_flags=VariantFlags.v7_full(),
        embedder=_StubEmbedder(),
        code_col=code_col,
        doc_col=doc_col,
        doc_bm25=None,
        doc_bm25_ids=[],
        code_bm25=None,
        code_bm25_ids=[],
        doc_meta_cache={},
        conn=conn,
    )


def _make_cr(*, change_type="MODIFICATION", layers=("requirement", "design", "code")):
    return CRInterpretation(
        is_actionable=True,
        actionability_reason=None,
        change_type=change_type,
        affected_layers=list(layers),
        primary_intent="x",
        domain_concepts=["grace", "schema"],
        search_queries=["graceDays", "schemaConfig"],
        named_entry_points=["graceDays"],
        out_of_scope_operations=[],
    )


# ---------------------------------------------------------------------------
# W2B — raw-CR multilingual bridge
# ---------------------------------------------------------------------------

def test_w2b_raw_cr_dense_pass_adds_code_candidates():
    """When cr_text is provided and the dense path is on, the retriever
    should call embed_single on the raw CR and add its nearest-neighbour
    code candidates to the dense_code pool."""
    code_scenario = {
        # idx 0 = "grace" feature: returned by the LLM #1 queries
        0: [("file_q.ts::Query", 0.10)],
        # idx 3 = "ilustrator" feature: only the raw Indonesian CR has this
        3: [("file_raw.ts::RawHit", 0.05)],
    }
    code_col = _StubCol("code", code_scenario)
    doc_col = _StubCol("doc", {0: [("doc_q.md", 0.10)]})
    ctx = _make_ctx(
        code_col=code_col, doc_col=doc_col,
        code_node_ids=["file_q.ts::Query", "file_raw.ts::RawHit"],
    )
    settings = _make_settings(enable_traceability_pool_seeding=False)
    cr = _make_cr()
    cr_text = "Tambahkan fitur untuk ilustrator menyematkan listing."

    candidates = hybrid_search(cr, ctx, settings, cr_text=cr_text)
    ids = {c.node_id for c in candidates}
    assert "file_raw.ts::RawHit" in ids, (
        "Raw-CR dense pass should pull in the Indonesian-only nearest neighbour"
    )
    # The embedder was called for each LLM #1 query plus once for cr_text.
    assert cr_text in ctx.embedder.calls


def test_w2b_disabled_when_setting_off():
    code_col = _StubCol("code", {0: [("a.ts::A", 0.10)], 3: [("raw.ts::Raw", 0.05)]})
    doc_col = _StubCol("doc", {0: [("d.md", 0.10)]})
    ctx = _make_ctx(
        code_col=code_col, doc_col=doc_col,
        code_node_ids=["a.ts::A", "raw.ts::Raw"],
    )
    settings = _make_settings(enable_raw_cr_dense_pass=False, enable_traceability_pool_seeding=False)
    candidates = hybrid_search(_make_cr(), ctx, settings, cr_text="ilustrator")
    ids = {c.node_id for c in candidates}
    assert "raw.ts::Raw" not in ids


def test_w2b_no_crash_without_cr_text():
    """Backward-compat: calling without cr_text (omitted) must work."""
    code_col = _StubCol("code", {0: [("a.ts::A", 0.10)]})
    doc_col = _StubCol("doc", {0: [("d.md", 0.10)]})
    ctx = _make_ctx(
        code_col=code_col, doc_col=doc_col,
        code_node_ids=["a.ts::A"],
    )
    settings = _make_settings(enable_traceability_pool_seeding=False)
    # cr_text omitted — must not raise.
    candidates = hybrid_search(_make_cr(), ctx, settings)
    assert any(c.node_id == "a.ts::A" for c in candidates)


# ---------------------------------------------------------------------------
# W2C — traceability pool seeding
# ---------------------------------------------------------------------------

@pytest.fixture
def _conn_with_doc_code_links(tmp_path):
    from impactracer.persistence.sqlite_client import init_schema

    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    init_schema(conn)
    # Pre-insert the code nodes the seeded rows point at so hydration works.
    for nid, fp, nm in [
        ("src/x.ts::SeededHigh", "src/x.ts", "SeededHigh"),
        ("src/x.ts::SeededMid", "src/x.ts", "SeededMid"),
        ("src/x.ts::SeededLow", "src/x.ts", "SeededLow"),
    ]:
        conn.execute(
            "INSERT OR REPLACE INTO code_nodes (node_id, node_type, name, file_path) "
            "VALUES (?, 'Function', ?, ?)",
            (nid, nm, fp),
        )
    conn.executemany(
        "INSERT INTO doc_code_candidates (doc_id, code_id, weighted_similarity_score) "
        "VALUES (?,?,?)",
        [
            # doc_q.md links to three code nodes; only those past min_score cut off.
            ("doc_q.md", "src/x.ts::SeededHigh", 0.90),
            ("doc_q.md", "src/x.ts::SeededMid", 0.65),
            ("doc_q.md", "src/x.ts::SeededLow", 0.30),
        ],
    )
    conn.commit()
    yield conn
    conn.close()


def test_w2c_traceability_seeding_injects_code_neighbours(_conn_with_doc_code_links):
    """Code nodes traceability-linked to a retrieved doc chunk are injected
    into the RRF pool even when no LLM #1 query embedding reaches them."""
    # doc collection returns doc_q.md, code collection returns NOTHING for the
    # LLM #1 search-query features; the only way SeededHigh appears is via
    # traceability seeding.
    doc_col = _StubCol("doc", {0: [("doc_q.md", 0.05)]})
    code_col = _StubCol("code", {})  # No direct dense hits.
    ctx = _make_ctx(code_col=code_col, doc_col=doc_col, conn=_conn_with_doc_code_links)
    settings = _make_settings(
        enable_raw_cr_dense_pass=False,  # isolate W2C
        traceability_seed_top_k_per_doc=2,
        traceability_seed_min_score=0.40,
    )
    candidates = hybrid_search(_make_cr(), ctx, settings)
    ids = {c.node_id for c in candidates}
    assert "src/x.ts::SeededHigh" in ids
    assert "src/x.ts::SeededMid" in ids
    # SeededLow is below the min_score floor — must NOT be seeded.
    assert "src/x.ts::SeededLow" not in ids


def test_w2c_respects_min_score_floor(_conn_with_doc_code_links):
    doc_col = _StubCol("doc", {0: [("doc_q.md", 0.05)]})
    code_col = _StubCol("code", {})
    ctx = _make_ctx(code_col=code_col, doc_col=doc_col, conn=_conn_with_doc_code_links)
    settings = _make_settings(
        enable_raw_cr_dense_pass=False,
        traceability_seed_min_score=0.70,  # only SeededHigh (0.90) qualifies
    )
    candidates = hybrid_search(_make_cr(), ctx, settings)
    ids = {c.node_id for c in candidates}
    assert "src/x.ts::SeededHigh" in ids
    assert "src/x.ts::SeededMid" not in ids


def test_w2c_disabled_when_setting_off(_conn_with_doc_code_links):
    doc_col = _StubCol("doc", {0: [("doc_q.md", 0.05)]})
    code_col = _StubCol("code", {})
    ctx = _make_ctx(code_col=code_col, doc_col=doc_col, conn=_conn_with_doc_code_links)
    settings = _make_settings(
        enable_raw_cr_dense_pass=False,
        enable_traceability_pool_seeding=False,
    )
    candidates = hybrid_search(_make_cr(), ctx, settings)
    ids = {c.node_id for c in candidates}
    assert "src/x.ts::SeededHigh" not in ids
