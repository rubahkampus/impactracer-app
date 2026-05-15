"""Unit tests for Apex Crucible Proposal C — graph_rerank.

Verifies the 2-iteration label propagation lifts structurally-connected
candidates and admits new graph-discovered nodes in mode B.
"""

from __future__ import annotations

import networkx as nx
import pytest

from impactracer.pipeline.graph_rerank import graph_rerank
from impactracer.shared.models import Candidate


def _cand(nid: str, score: float, name: str = "") -> Candidate:
    return Candidate(
        node_id=nid,
        node_type="Function",
        collection="code_units",
        rrf_score=0.5,
        reranker_score=score,
        raw_reranker_score=score,
        name=name or nid.split("::")[-1],
        text_snippet="",
    )


def test_graph_rerank_lifts_adjacent_candidate():
    """A candidate with low CE score but a TYPED_BY edge to a high-CE seed
    should be lifted after graph propagation (alpha=0.5 to make the effect
    visible in the blended score).
    """
    g = nx.MultiDiGraph()
    g.add_edge("schema", "consumer", edge_type="TYPED_BY")

    seed = _cand("schema", score=1.0)         # top cross-encoder
    adjacent = _cand("consumer", score=0.1)   # bottom of pool
    distant = _cand("isolated", score=0.5)    # mid pool, no graph link

    cands = [seed, adjacent, distant]
    out = graph_rerank(
        cands,
        graph=g,
        alpha=0.5,
        iterations=2,
        personalization_top_n=1,
        add_top_n=0,  # mode A only
    )

    by_id = {c.node_id: c for c in out}

    # The adjacent candidate should now outrank the distant one because
    # graph propagation gave it a TYPED_BY lift from the seed.
    assert by_id["consumer"].raw_reranker_score > by_id["isolated"].raw_reranker_score, (
        f"consumer should be lifted via TYPED_BY: "
        f"consumer={by_id['consumer'].raw_reranker_score:.3f} "
        f"isolated={by_id['isolated'].raw_reranker_score:.3f}"
    )


def test_graph_rerank_mode_b_adds_graph_discovered_node():
    """Mode B should discover a TYPED_BY neighbour that is NOT in the pool
    and add it as a new Candidate."""
    g = nx.MultiDiGraph()
    g.add_edge("schema", "ui_component", edge_type="TYPED_BY")

    seed = _cand("schema", score=1.0)
    other = _cand("other", score=0.3)

    code_meta = {
        "ui_component": {
            "node_type": "Function",
            "file_path": "src/ui.tsx",
            "file_classification": "UI_COMPONENT",
            "internal_logic_abstraction": "function UIComponent() {}",
            "source_code": "function UIComponent() {}",
        },
    }

    out = graph_rerank(
        [seed, other],
        graph=g,
        alpha=0.7,
        iterations=2,
        personalization_top_n=1,
        add_top_n=5,
        add_min_score=0.0,  # admit anything above zero
        code_meta_by_id=code_meta,
    )

    ids = {c.node_id for c in out}
    assert "ui_component" in ids, "Mode B should have added ui_component"
    added = next(c for c in out if c.node_id == "ui_component")
    assert added.file_path == "src/ui.tsx"
    assert added.node_type == "Function"


def test_graph_rerank_respects_external_package_exclusion():
    """ExternalPackage neighbours must not be added by mode B."""
    g = nx.MultiDiGraph()
    g.add_edge("seed", "ext::lodash", edge_type="DEPENDS_ON_EXTERNAL")

    seed = _cand("seed", score=1.0)
    code_meta = {
        "ext::lodash": {
            "node_type": "ExternalPackage",
            "file_path": "",
            "file_classification": None,
            "internal_logic_abstraction": "",
            "source_code": "",
        },
    }
    out = graph_rerank(
        [seed],
        graph=g,
        alpha=0.7,
        iterations=2,
        personalization_top_n=1,
        add_top_n=5,
        add_min_score=0.0,
        code_meta_by_id=code_meta,
    )
    ids = {c.node_id for c in out}
    assert "ext::lodash" not in ids


def test_graph_rerank_no_op_when_empty():
    """Empty candidate list is a no-op."""
    g = nx.MultiDiGraph()
    out = graph_rerank([], graph=g)
    assert out == []


def test_graph_rerank_blends_with_alpha():
    """alpha=1.0 should preserve cross-encoder order; alpha=0.0 should make
    the order entirely structural.
    """
    g = nx.MultiDiGraph()
    g.add_edge("schema", "consumer", edge_type="TYPED_BY")

    seed = _cand("schema", score=1.0)
    adjacent = _cand("consumer", score=0.0)
    other = _cand("other", score=0.5)

    # alpha=1 (cross-encoder only): seed > other > adjacent
    out_alpha1 = graph_rerank(
        [_cand("schema", 1.0), _cand("consumer", 0.0), _cand("other", 0.5)],
        graph=g, alpha=1.0, iterations=2, personalization_top_n=1, add_top_n=0,
    )
    ranked = sorted(out_alpha1, key=lambda c: c.raw_reranker_score, reverse=True)
    assert ranked[0].node_id == "schema"
    assert ranked[1].node_id == "other"
    assert ranked[2].node_id == "consumer"

    # alpha=0 (graph only): consumer should rank above other (consumer is
    # TYPED_BY-linked to schema, other is isolated).
    out_alpha0 = graph_rerank(
        [_cand("schema", 1.0), _cand("consumer", 0.0), _cand("other", 0.5)],
        graph=g, alpha=0.0, iterations=2, personalization_top_n=1, add_top_n=0,
    )
    by_id = {c.node_id: c for c in out_alpha0}
    assert by_id["consumer"].raw_reranker_score > by_id["other"].raw_reranker_score


def test_graph_rerank_two_hop_propagation():
    """A candidate two hops from a seed (via two TYPED_BY edges) should
    receive a smaller but non-zero structural boost relative to a one-hop
    neighbour."""
    g = nx.MultiDiGraph()
    g.add_edge("seed", "mid", edge_type="TYPED_BY")
    g.add_edge("mid", "far", edge_type="TYPED_BY")

    seed = _cand("seed", 1.0)
    mid = _cand("mid", 0.0)
    far = _cand("far", 0.0)
    isolated = _cand("isolated", 0.0)

    out = graph_rerank(
        [seed, mid, far, isolated],
        graph=g, alpha=0.0, iterations=2, personalization_top_n=1, add_top_n=0,
    )
    by_id = {c.node_id: c for c in out}
    # mid should outrank far (1 hop > 2 hop). far should outrank isolated.
    assert by_id["mid"].raw_reranker_score >= by_id["far"].raw_reranker_score
    assert by_id["far"].raw_reranker_score > by_id["isolated"].raw_reranker_score
