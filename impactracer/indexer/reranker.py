"""BGE-Reranker-v2-M3 cross-encoder wrapper (FR-C3).

Returns sigmoid-normalized scores in [0, 1]. Used by the online pipeline
at Step 3 (after RRF, before pre-validation gates).

Reference: master_blueprint.md §3.6.
"""

from __future__ import annotations

from impactracer.shared.models import Candidate


class Reranker:
    """Wraps ``FlagEmbedding.FlagReranker``."""

    def __init__(self, model_name: str) -> None:
        import torch
        from FlagEmbedding import FlagReranker
        # Use CUDA if available; fp16 is only beneficial on GPU (CPU fp16 has
        # no native hardware support and adds conversion overhead).
        # FlagReranker uses ``devices`` (plural) not ``device``.
        use_cuda = torch.cuda.is_available()
        self.model = FlagReranker(
            model_name,
            use_fp16=use_cuda,
            devices=["cuda:0"] if use_cuda else ["cpu"],
        )

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        top_k: int,
    ) -> list[Candidate]:
        """Score each ``(query, candidate.text_snippet)`` pair.

        Sets ``reranker_score`` in-place on each candidate and returns
        them sorted descending, truncated to ``top_k``.
        """
        if not candidates:
            return []
        pairs = [(query, c.text_snippet) for c in candidates]
        scores = self.model.compute_score(pairs, normalize=True)
        for c, s in zip(candidates, scores):
            c.reranker_score = float(s)
        return sorted(candidates, key=lambda c: c.reranker_score, reverse=True)[:top_k]

    def rerank_multi_query(
        self,
        queries: list[str],
        fallback_query: str,
        candidates: list[Candidate],
        top_k: int,
    ) -> list[Candidate]:
        """Score each candidate against ALL queries and take the max score.

        N4 fix: using only primary_intent (single query) systematically
        penalises BM25-code candidates whose embed_text is identifier-level
        (low semantic overlap with natural-language intent).  Scoring against
        all search_queries and keeping the maximum ensures a candidate that
        precisely matches any query phrase is not under-scored.

        Falls back to ``fallback_query`` when ``queries`` is empty.

        Sets ``reranker_score`` on each candidate to the max cross-encoder
        score across all queries, then returns top_k sorted descending.
        The caller snapshots ``raw_reranker_score`` from this value BEFORE
        min-max normalisation (B4).
        """
        if not candidates:
            return []

        effective_queries = queries if queries else [fallback_query]

        # Build (query, text) pairs for ALL queries in one batch to avoid
        # N separate model.compute_score calls (each call has per-call overhead).
        # Pairs layout: [q0c0, q0c1, ..., q0cN, q1c0, ..., qMcN]
        all_pairs: list[tuple[str, str]] = []
        for q in effective_queries:
            for c in candidates:
                # Prefer ILA for scoring: it is a compact skeleton that
                # preserves call sites and signatures — exactly what the
                # cross-encoder needs to judge structural relevance.
                text = c.internal_logic_abstraction or c.text_snippet or ""
                all_pairs.append((q, text))

        all_scores = self.model.compute_score(all_pairs, normalize=True)

        n_cands = len(candidates)
        n_queries = len(effective_queries)

        # Reshape and take max across queries for each candidate
        for cand_idx, c in enumerate(candidates):
            max_score = max(
                float(all_scores[q_idx * n_cands + cand_idx])
                for q_idx in range(n_queries)
            )
            c.reranker_score = max_score

        return sorted(candidates, key=lambda c: c.reranker_score, reverse=True)[:top_k]
