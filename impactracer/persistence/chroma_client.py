"""ChromaDB persistent client wrapper.

Schema reference: 04_database_schema.md §2. Both collections MUST use
cosine space; the default L2 metric produces incorrect rankings on
un-normalized BGE-M3 vectors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb


COLLECTION_CONFIG: dict[str, Any] = {"hnsw:space": "cosine"}


def get_client(chroma_path: str) -> chromadb.PersistentClient:
    """Return (or create) a PersistentClient rooted at ``chroma_path``."""
    Path(chroma_path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=chroma_path)


def _assert_cosine_space(col: chromadb.Collection) -> None:
    """Fail fast if a collection was not created with hnsw:space=cosine.

    Silent score corruption occurs when 1.0-dist is applied to L2 distances;
    an assertion here surfaces the root cause immediately instead of
    producing numerically wrong RRF scores for an entire evaluation run.
    """
    space = (col.metadata or {}).get("hnsw:space", "l2")
    if space != "cosine":
        raise RuntimeError(
            f"ChromaDB collection '{col.name}' uses hnsw:space='{space}', "
            f"expected 'cosine'. Re-index with `impactracer index --force` to rebuild."
        )


def init_collections(
    client: chromadb.PersistentClient,
) -> tuple[chromadb.Collection, chromadb.Collection]:
    """Return ``(doc_chunks, code_units)``, creating if absent.

    Asserts both collections use cosine distance (thesis methodology
    requires cosine similarity for reproducibility).
    """
    doc = client.get_or_create_collection(name="doc_chunks", metadata=COLLECTION_CONFIG)
    code = client.get_or_create_collection(name="code_units", metadata=COLLECTION_CONFIG)
    _assert_cosine_space(doc)
    _assert_cosine_space(code)
    return doc, code
