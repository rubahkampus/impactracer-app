"""BGE-M3 dense embedding wrapper (FR-A5).

Output dimension: 1024. Multilingual: Indonesian + English in a shared
space. Uses FP16 on GPU when available.

Reference: 06_offline_indexer.md §6.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class Embedder:
    """Wraps ``FlagEmbedding.BGEM3FlagModel``."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        """Load the model, warm the HF cache if needed."""
        raise NotImplementedError("Sprint 6")

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts. Returns ``(N, 1024)`` float32 array."""
        raise NotImplementedError("Sprint 6")

    def embed_single(self, text: str) -> list[float]:
        """Encode one text; return a plain Python list (ChromaDB compatible)."""
        raise NotImplementedError("Sprint 6")


def ensure_model_cached(model_name: str) -> None:
    """Pre-warm the Hugging Face cache for a given model name."""
    raise NotImplementedError("Sprint 6")
