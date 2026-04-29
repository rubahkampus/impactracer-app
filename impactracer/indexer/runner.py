"""Offline indexer orchestrator.

Invoked by :func:`impactracer.cli.index`. Executes the full offline
pipeline:

1. Scan repository for Markdown and TypeScript/TSX source files.
2. Diff against file_hashes to determine work set (unless ``force``).
3. Chunk Markdown -> :mod:`doc_indexer`.
4. AST Pass 1 -> :mod:`code_indexer` -> skeletonize each node body.
5. AST Pass 2 -> :mod:`code_indexer` -> 13 edge types.
6. Embed all pending texts -> :mod:`embedder`.
7. Traceability precompute -> :mod:`traceability`.
8. Write index_metadata, print summary statistics.

Reference: 06_offline_indexer.md
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from impactracer.shared.config import Settings


class IndexingStats(TypedDict):
    code_nodes: int
    doc_chunks: int
    edges: int
    files_scanned: int
    files_reindexed: int
    elapsed_seconds: float


def run_indexing(
    repo_path: Path,
    settings: Settings,
    force: bool = False,
) -> IndexingStats:
    """Execute the offline indexing pipeline end-to-end.

    Args:
        repo_path: Root of the target repository.
        settings: Runtime configuration.
        force: If True, invalidate all file_hashes and reindex everything.

    Returns:
        Statistics dictionary for CLI reporting.

    Raises:
        FileNotFoundError: If ``repo_path`` does not exist.
    """
    raise NotImplementedError("Sprint 7")
