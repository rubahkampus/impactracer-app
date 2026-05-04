"""BGE-M3 dense embedding wrapper (FR-A5).

Output dimension: 1024. Multilingual: Indonesian + English in a shared
space. Uses FP16 on GPU when available.

Reference: master_blueprint.md §3.5.
"""

from __future__ import annotations

import numpy as np
from huggingface_hub import snapshot_download


class Embedder:
    """Wraps ``FlagEmbedding.BGEM3FlagModel``."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        import torch
        from FlagEmbedding import BGEM3FlagModel
        # Use CUDA if available; fp16 is only beneficial on GPU.
        # BGEM3FlagModel uses ``devices`` (plural) not ``device``.
        use_cuda = torch.cuda.is_available()
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=use_cuda,
            devices=["cuda:0"] if use_cuda else ["cpu"],
        )
        self.batch_size = batch_size
        self.max_length = max_length

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts. Returns ``(N, 1024)`` float32 array."""
        out = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return out["dense_vecs"].astype(np.float32)

    def embed_single(self, text: str) -> list[float]:
        """Encode one text; return a plain Python list (ChromaDB compatible)."""
        return self.embed_batch([text])[0].tolist()


def ensure_model_cached(model_name: str) -> None:
    """Pre-warm the Hugging Face cache for a given model name."""
    snapshot_download(repo_id=model_name)
