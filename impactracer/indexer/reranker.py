"""BGE-Reranker-v2-M3 cross-encoder wrapper (FR-C3).

Returns sigmoid-normalized scores in [0, 1]. Used both by the online
pipeline (Step 3) and optionally during indexing for calibration runs.

Reference: 07_online_pipeline.md §5.
"""

from __future__ import annotations

from impactracer.shared.models import Candidate


class Reranker:
    """Wraps ``FlagEmbedding.FlagReranker``."""

    def __init__(self, model_name: str) -> None:
        """Load the model, warm the HF cache if needed."""
        raise NotImplementedError("Sprint 6")

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
        raise NotImplementedError("Sprint 6")
