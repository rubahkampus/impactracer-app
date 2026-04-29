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


def init_collections(
    client: chromadb.PersistentClient,
) -> tuple[chromadb.Collection, chromadb.Collection]:
    """Return ``(doc_chunks, code_units)``, creating if absent."""
    doc = client.get_or_create_collection(name="doc_chunks", metadata=COLLECTION_CONFIG)
    code = client.get_or_create_collection(name="code_units", metadata=COLLECTION_CONFIG)
    return doc, code
