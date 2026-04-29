"""TypeScript/TSX AST parser and edge extractor (FR-A3, FR-A4).

Two-pass design:

- Pass 1 (:func:`extract_nodes`): walks the AST to produce :class:`File`,
  :class:`Class`, :class:`Function`, :class:`Method`, :class:`Interface`,
  :class:`TypeAlias`, :class:`Enum`, and :class:`InterfaceField` nodes.
  Populates ``internal_logic_abstraction`` via :func:`skeletonize_node`.

- Pass 2 (:func:`extract_edges`): walks the AST again, with the full
  set of node IDs from Pass 1 available, and emits all 13 edge types.
  Populates ``file_dependencies`` for incremental reindex.

Reference: 05_ast_edge_catalog.md, 06_offline_indexer.md §3-4.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from tree_sitter import Parser


def get_ts_parser(file_path: Path) -> Parser:
    """Return the correct parser for a .ts or .tsx file."""
    raise NotImplementedError("Sprint 4")


def classify_file(rel_path: Path) -> str | None:
    """Return the ``file_classification`` for a given relative path.

    Uses the NEXTJS_ROUTE_PATTERNS path glob rules.
    """
    raise NotImplementedError("Sprint 4")


def derive_route_path(rel_path: Path) -> str | None:
    """Derive the ``route_path`` for API_ROUTE and PAGE_COMPONENT files."""
    raise NotImplementedError("Sprint 4")


def extract_nodes(
    file_path: Path,
    source_bytes: bytes,
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Pass 1: extract all nodes from a source file.

    Populates the ``code_nodes`` table. Returns the list of nodes created
    so the caller can collect exported names for File-node enrichment.
    """
    raise NotImplementedError("Sprint 4")


def extract_edges(
    file_path: Path,
    source_bytes: bytes,
    known_node_ids: set[str],
    conn: sqlite3.Connection,
) -> int:
    """Pass 2: extract all 13 edge types.

    Returns the number of edges emitted. Also populates
    ``file_dependencies`` for incremental reindex support.
    """
    raise NotImplementedError("Sprint 5")


def compose_embed_text(node: dict[str, Any]) -> str:
    """Compose the BGE-M3 input text for a non-File node.

    Layout: ``docstring\nsignature`` (empty lines omitted). For nodes
    without a docstring or signature, falls back to ``name``.
    """
    raise NotImplementedError("Sprint 4")


def compose_file_embed_text(
    file_node: dict[str, Any],
    exported_names: list[str],
    rel_dir: str,
) -> str:
    """Compose the enriched BGE-M3 input text for a File node."""
    raise NotImplementedError("Sprint 4")


def synthesize_ui_docstring(name: str, signature: str) -> str:
    """Synthesize a docstring for an exported UI component without JSDoc."""
    raise NotImplementedError("Sprint 4")
