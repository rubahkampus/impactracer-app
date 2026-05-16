"""Tests for indexer/embedder.py (FR-A5). All model calls are monkeypatched."""

from __future__ import annotations

import numpy as np
import pytest

from impactracer.indexer.embedder import Embedder, ensure_model_cached


class _FakeBGEM3:
    """Fake BGEM3FlagModel: returns deterministic unit vectors."""

    def __init__(self, model_name, use_fp16=True, devices=None, **kwargs):
        self._dim = 1024

    def encode(self, texts, batch_size, max_length, return_dense, return_sparse, return_colbert_vecs):
        assert return_dense is True
        assert return_sparse is False
        assert return_colbert_vecs is False
        # Return same deterministic vector for identical text by keying on content
        result = []
        for t in texts:
            seed = sum(ord(c) for c in t) % (2**31)
            r = np.random.default_rng(seed=seed)
            result.append(r.random(self._dim, dtype=np.float32).astype(np.float32))
        return {"dense_vecs": np.stack(result)}


@pytest.fixture()
def embedder(monkeypatch):
    import impactracer.indexer.embedder as mod
    monkeypatch.setattr(mod, "BGEM3FlagModel", _FakeBGEM3, raising=False)

    import sys
    import types
    # Ensure FlagEmbedding module is patchable at import time inside __init__
    flag_mod = types.ModuleType("FlagEmbedding")
    flag_mod.BGEM3FlagModel = _FakeBGEM3
    sys.modules["FlagEmbedding"] = flag_mod

    return Embedder("BAAI/bge-m3", batch_size=4, max_length=128)


def test_embed_batch_shape(embedder):
    texts = ["hello world", "foo bar baz", "another text"]
    result = embedder.embed_batch(texts)
    assert result.shape == (3, 1024)


def test_embed_batch_dtype_float32(embedder):
    result = embedder.embed_batch(["test"])
    assert result.dtype == np.float32


def test_embed_single_returns_list(embedder):
    result = embedder.embed_single("createCommissionListing function")
    assert isinstance(result, list)
    assert len(result) == 1024


def test_embed_single_elements_are_floats(embedder):
    result = embedder.embed_single("some text")
    assert all(isinstance(v, float) for v in result)


def test_embed_batch_single_text(embedder):
    result = embedder.embed_batch(["only one"])
    assert result.shape == (1, 1024)


def test_embed_batch_many_texts(embedder):
    texts = [f"text number {i}" for i in range(10)]
    result = embedder.embed_batch(texts)
    assert result.shape == (10, 1024)


def test_determinism(embedder):
    text = "createCommissionListing function"
    v1 = embedder.embed_single(text)
    v2 = embedder.embed_single(text)
    diff = max(abs(a - b) for a, b in zip(v1, v2))
    assert diff < 1e-6


def test_embed_batch_determinism(embedder):
    texts = ["alpha", "beta", "gamma"]
    r1 = embedder.embed_batch(texts)
    r2 = embedder.embed_batch(texts)
    assert np.max(np.abs(r1 - r2)) < 1e-6


def test_ensure_model_cached_calls_snapshot_download(monkeypatch):
    calls = []

    def fake_snapshot_download(repo_id):
        calls.append(repo_id)

    import impactracer.indexer.embedder as mod
    monkeypatch.setattr(mod, "snapshot_download", fake_snapshot_download)
    ensure_model_cached("BAAI/bge-m3")
    assert calls == ["BAAI/bge-m3"]
