"""Unit tests for the per-CR variant cache.

Verifies the four contract invariants:

  1. Cache miss returns None for every getter.
  2. Cache put -> cache get round-trips identical data.
  3. Getters return deep copies — mutating the returned object does NOT
     corrupt the cache.
  4. anchor_priming=True and =False are isolated keys; cross-contamination
     is impossible.

The cache must also tolerate the full Candidate field surface, including
``merged_doc_contexts`` (tuples). CISResult round-trip preserves
NodeTrace metadata.

Reference: ``docs/evaluation_protocol.md``.
"""

from __future__ import annotations

from pathlib import Path

from impactracer.pipeline.variant_cache import VariantCache
from impactracer.shared.models import (
    CISResult,
    Candidate,
    CRInterpretation,
    NodeTrace,
)


# ----------------------------------------------------------------------
# Fixture helpers
# ----------------------------------------------------------------------


def _make_interp(anchor_candidates: list[str] | None = None) -> CRInterpretation:
    return CRInterpretation(
        is_actionable=True,
        primary_intent="Test refactor of wallet service to use escrow service.",
        change_type="MODIFICATION",
        affected_layers=["code"],
        domain_concepts=["wallet", "escrow"],
        search_queries=["wallet service escrow import", "refactor wallet to call escrow"],
        named_entry_points=[],
        anchor_candidates=anchor_candidates if anchor_candidates is not None else [],
        out_of_scope_operations=[],
    )


def _make_candidate(
    node_id: str = "src/lib/services/wallet.service.ts::getTransactions",
) -> Candidate:
    return Candidate(
        node_id=node_id,
        node_type="Function",
        collection="code_units",
        rrf_score=0.5,
        reranker_score=0.7,
        raw_reranker_score=2.5,
        file_path="src/lib/services/wallet.service.ts",
        file_classification="UTILITY",
        chunk_type=None,
        name="getTransactions",
        text_snippet="async function getTransactions(...) { ... }",
        internal_logic_abstraction="seq: db.find, return list",
        merged_doc_ids=["doc1", "doc2"],
        merged_doc_contexts=[("Section Title", "Some doc text here")],
        bm25_score=1.2,
        cosine_score=0.8,
    )


def _make_cis() -> CISResult:
    return CISResult(
        sis_nodes={
            "node_a": NodeTrace(
                depth=0,
                causal_chain=[],
                path=["node_a"],
                source_seed="node_a",
                low_confidence_seed=False,
                justification="seed admitted via LLM #2",
                justification_source="llm2_sis",
                function_purpose="reads wallet",
                mechanism_of_impact="signature stays identical",
            ),
        },
        propagated_nodes={
            "node_b": NodeTrace(
                depth=1,
                causal_chain=["CALLS"],
                path=["node_a", "node_b"],
                source_seed="node_a",
                low_confidence_seed=True,
                collapsed_children=["node_c"],
                justification="propagated via LLM #4",
                justification_source="llm4_propagation",
            ),
        },
    )


# ----------------------------------------------------------------------
# Contract 1: cache miss returns None
# ----------------------------------------------------------------------


def test_cache_miss_returns_none(tmp_path: Path) -> None:
    cache = VariantCache(
        root_dir=tmp_path,
        run_tag="2026-05-20T13-00-00Z",
        cr_id="C1",
        anchor_priming=True,
    )

    assert cache.get_interp() is None
    assert cache.get_retrieval_v0() is None
    assert cache.get_retrieval_v1() is None
    assert cache.get_retrieval_v2plus() is None
    assert cache.get_rerank_gated() is None
    assert cache.get_sis_verdicts() is None
    assert cache.get_trace_verdicts() is None
    assert cache.get_bfs_cis() is None
    assert cache.get_llm4_verdicts() is None


# ----------------------------------------------------------------------
# Contract 2: put -> get round-trips data
# ----------------------------------------------------------------------


def test_interp_roundtrip(tmp_path: Path) -> None:
    cache = VariantCache(tmp_path, "run1", "C1", anchor_priming=True)
    original = _make_interp(anchor_candidates=["WalletService", "EscrowService"])

    cache.put_interp(original)
    loaded = cache.get_interp()

    assert loaded is not None
    assert loaded.is_actionable == original.is_actionable
    assert loaded.primary_intent == original.primary_intent
    assert loaded.change_type == original.change_type
    assert loaded.affected_layers == original.affected_layers
    assert loaded.domain_concepts == original.domain_concepts
    assert loaded.search_queries == original.search_queries
    assert loaded.anchor_candidates == ["WalletService", "EscrowService"]


def test_candidates_roundtrip_preserves_all_fields(tmp_path: Path) -> None:
    cache = VariantCache(tmp_path, "run1", "C1", anchor_priming=True)
    original = [
        _make_candidate(),
        _make_candidate("other::func"),
    ]

    cache.put_rerank_gated(original)
    loaded = cache.get_rerank_gated()

    assert loaded is not None
    assert len(loaded) == 2
    assert loaded[0].node_id == original[0].node_id
    assert loaded[1].node_id == original[1].node_id
    # merged_doc_contexts must round-trip as tuples (not lists).
    assert loaded[0].merged_doc_contexts == [("Section Title", "Some doc text here")]
    assert isinstance(loaded[0].merged_doc_contexts[0], tuple)
    # Numeric fields preserved exactly.
    assert loaded[0].raw_reranker_score == 2.5
    assert loaded[0].rrf_score == 0.5


def test_sis_verdicts_roundtrip(tmp_path: Path) -> None:
    cache = VariantCache(tmp_path, "run1", "C1", anchor_priming=True)
    ids = ["node_a", "node_b"]
    justifs = {
        "node_a": {"function_purpose": "p1", "mechanism": "m1", "summary": "s1"},
    }

    cache.put_sis_verdicts(ids, justifs, degraded=False)
    loaded = cache.get_sis_verdicts()

    assert loaded is not None
    confirmed, j, deg = loaded
    assert confirmed == ids
    assert j == justifs
    assert deg is False


def test_trace_verdicts_roundtrip(tmp_path: Path) -> None:
    cache = VariantCache(tmp_path, "run1", "C1", anchor_priming=True)
    seeds = ["code_a", "code_b"]
    low = {"code_a": True, "code_b": False}
    just = {"code_a": "low conf seed", "code_b": "validated"}
    # code_b is CONFIRMED with a mechanism (anchor-eligible); code_a is
    # PARTIAL/low-confidence with no mechanism.
    mech = {"code_a": "", "code_b": "add discount field to schema"}

    cache.put_trace_verdicts(seeds, low, just, mech, degraded=True)
    loaded = cache.get_trace_verdicts()

    assert loaded is not None
    s, lc, j, m, deg = loaded
    assert s == seeds
    assert lc == low
    assert j == just
    assert m == mech
    assert deg is True


def test_trace_verdicts_roundtrip_legacy_cache_without_mechanisms(tmp_path: Path) -> None:
    """A pre-fix cache file lacking the 'mechanisms' key must load as {}."""
    cache = VariantCache(tmp_path, "run1", "C1", anchor_priming=True)
    # Simulate a stale cache written before the mechanisms field existed.
    cache._write_json(  # type: ignore[attr-defined]
        "trace_verdicts",
        {
            "validated_code_seeds": ["code_a"],
            "low_confidence": {"code_a": True},
            "justifications": {"code_a": "legacy"},
            "degraded": False,
        },
    )
    loaded = cache.get_trace_verdicts()
    assert loaded is not None
    s, lc, j, m, deg = loaded
    assert s == ["code_a"]
    assert m == {}  # graceful default, no crash


def test_bfs_cis_roundtrip(tmp_path: Path) -> None:
    cache = VariantCache(tmp_path, "run1", "C1", anchor_priming=True)
    original = _make_cis()

    cache.put_bfs_cis(original)
    loaded = cache.get_bfs_cis()

    assert loaded is not None
    assert set(loaded.sis_nodes.keys()) == {"node_a"}
    assert set(loaded.propagated_nodes.keys()) == {"node_b"}
    nt_a = loaded.sis_nodes["node_a"]
    assert nt_a.depth == 0
    assert nt_a.justification_source == "llm2_sis"
    assert nt_a.function_purpose == "reads wallet"
    nt_b = loaded.propagated_nodes["node_b"]
    assert nt_b.depth == 1
    assert nt_b.causal_chain == ["CALLS"]
    assert nt_b.collapsed_children == ["node_c"]


def test_llm4_verdicts_roundtrip(tmp_path: Path) -> None:
    cache = VariantCache(tmp_path, "run1", "C1", anchor_priming=True)
    cis = _make_cis()
    just = {"node_b": "validated by LLM #4"}

    cache.put_llm4_verdicts(cis, just, degraded=False)
    loaded = cache.get_llm4_verdicts()

    assert loaded is not None
    loaded_cis, loaded_just, deg = loaded
    assert set(loaded_cis.sis_nodes.keys()) == {"node_a"}
    assert loaded_just == just
    assert deg is False


# ----------------------------------------------------------------------
# Contract 3: getters return deep copies
# ----------------------------------------------------------------------


def test_candidate_get_returns_deep_copy(tmp_path: Path) -> None:
    cache = VariantCache(tmp_path, "run1", "C1", anchor_priming=True)
    original = [_make_candidate()]

    cache.put_rerank_gated(original)

    # First read, mutate aggressively.
    loaded1 = cache.get_rerank_gated()
    assert loaded1 is not None
    loaded1[0].reranker_score = 99.0
    loaded1[0].raw_reranker_score = 99.0
    loaded1[0].merged_doc_ids.append("injected")
    loaded1[0].merged_doc_contexts.append(("injected", "junk"))

    # Second read must be unaffected.
    loaded2 = cache.get_rerank_gated()
    assert loaded2 is not None
    assert loaded2[0].reranker_score == 0.7
    assert loaded2[0].raw_reranker_score == 2.5
    assert loaded2[0].merged_doc_ids == ["doc1", "doc2"]
    assert loaded2[0].merged_doc_contexts == [("Section Title", "Some doc text here")]


def test_cis_get_returns_deep_copy(tmp_path: Path) -> None:
    cache = VariantCache(tmp_path, "run1", "C1", anchor_priming=True)
    cache.put_bfs_cis(_make_cis())

    loaded1 = cache.get_bfs_cis()
    assert loaded1 is not None
    # Mutate the loaded CIS.
    loaded1.sis_nodes["node_a"].depth = 99
    loaded1.sis_nodes["node_a"].causal_chain.append("INJECTED")
    loaded1.propagated_nodes["node_b"].collapsed_children.append("injected")
    loaded1.sis_nodes["mutant"] = NodeTrace(
        depth=42, causal_chain=[], path=[], source_seed="mutant",
    )

    loaded2 = cache.get_bfs_cis()
    assert loaded2 is not None
    assert loaded2.sis_nodes["node_a"].depth == 0
    assert loaded2.sis_nodes["node_a"].causal_chain == []
    assert loaded2.propagated_nodes["node_b"].collapsed_children == ["node_c"]
    assert "mutant" not in loaded2.sis_nodes


def test_interp_get_returns_deep_copy(tmp_path: Path) -> None:
    cache = VariantCache(tmp_path, "run1", "C1", anchor_priming=True)
    cache.put_interp(_make_interp(anchor_candidates=["A", "B"]))

    loaded1 = cache.get_interp()
    assert loaded1 is not None
    loaded1.anchor_candidates.append("INJECTED")
    loaded1.search_queries.append("INJECTED")

    loaded2 = cache.get_interp()
    assert loaded2 is not None
    assert loaded2.anchor_candidates == ["A", "B"]
    assert "INJECTED" not in loaded2.search_queries


# ----------------------------------------------------------------------
# Contract 4: anchor_priming True / False are isolated
# ----------------------------------------------------------------------


def test_anchor_priming_key_isolation(tmp_path: Path) -> None:
    cache_on = VariantCache(tmp_path, "run1", "C1", anchor_priming=True)
    cache_off = VariantCache(tmp_path, "run1", "C1", anchor_priming=False)

    cache_on.put_interp(_make_interp(anchor_candidates=["WalletService"]))
    cache_off.put_interp(_make_interp(anchor_candidates=[]))

    on_loaded = cache_on.get_interp()
    off_loaded = cache_off.get_interp()

    assert on_loaded is not None and off_loaded is not None
    assert on_loaded.anchor_candidates == ["WalletService"]
    assert off_loaded.anchor_candidates == []


def test_cr_id_key_isolation(tmp_path: Path) -> None:
    cache_c1 = VariantCache(tmp_path, "run1", "C1", anchor_priming=True)
    cache_c2 = VariantCache(tmp_path, "run1", "C2", anchor_priming=True)

    cache_c1.put_interp(_make_interp(anchor_candidates=["alpha"]))
    cache_c2.put_interp(_make_interp(anchor_candidates=["beta"]))

    assert cache_c1.get_interp().anchor_candidates == ["alpha"]
    assert cache_c2.get_interp().anchor_candidates == ["beta"]


def test_run_tag_key_isolation(tmp_path: Path) -> None:
    cache_a = VariantCache(tmp_path, "runA", "C1", anchor_priming=True)
    cache_b = VariantCache(tmp_path, "runB", "C1", anchor_priming=True)

    cache_a.put_interp(_make_interp(anchor_candidates=["a-only"]))
    # cache_b never written.

    assert cache_a.get_interp().anchor_candidates == ["a-only"]
    assert cache_b.get_interp() is None
