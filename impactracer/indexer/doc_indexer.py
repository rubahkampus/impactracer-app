"""Markdown chunking and chunk-type classification.

Implements FR-A1 (boundary-based chunking at H2/H3) and FR-A2
(deterministic substring classification into FR | NFR | Design | General).

Reference: 06_offline_indexer.md §2.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict


CHUNK_TYPE_RULES: dict[str, list[str]] = {
    "FR":     ["kebutuhan fungsional", "functional requirement", "use case"],
    "NFR":    ["non-fungsional", "non-functional", "kebutuhan non"],
    "Design": ["perancangan", "desain", "arsitektur", "design", "architecture"],
}


class DocChunk(TypedDict):
    chunk_id: str
    source_file: str
    section_title: str
    chunk_type: str
    text: str


def classify_chunk(section_title: str) -> str:
    """Return one of FR, NFR, Design, or General."""
    t = section_title.lower()
    for ctype, keywords in CHUNK_TYPE_RULES.items():
        if any(kw in t for kw in keywords):
            return ctype
    return "General"


def chunk_markdown(filepath: Path) -> list[DocChunk]:
    """Split a Markdown file at H2/H3 boundaries.

    Chunks are deterministic: identical input produces identical output
    across runs. Chunk IDs follow ``{file_stem}__{slugified_title}``.
    """
    raise NotImplementedError("Sprint 3")
