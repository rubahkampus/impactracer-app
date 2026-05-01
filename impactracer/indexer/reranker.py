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
        from FlagEmbedding import FlagReranker
        self.model = FlagReranker(model_name, use_fp16=True)

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
