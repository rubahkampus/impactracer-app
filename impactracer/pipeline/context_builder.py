"""Backlink retrieval and synthesis context assembly (FR-E1, FR-E2).

Reference: 07_online_pipeline.md §12.
"""

from __future__ import annotations

import sqlite3

from impactracer.shared.models import CISResult, CRInterpretation


def fetch_backlinks(
    code_ids: list[str],
    conn: sqlite3.Connection,
    top_k: int,
) -> dict[str, list[tuple[str, float]]]:
    """Fetch top-K traceability backlinks per code node (FR-E1)."""
    raise NotImplementedError("Sprint 8")


def fetch_snippets(
    code_ids: list[str],
    conn: sqlite3.Connection,
) -> dict[str, str]:
    """Fetch ``source_code`` for each node id."""
    raise NotImplementedError("Sprint 8")


def build_context(
    cr_text: str,
    cr_interp: CRInterpretation,
    cis: CISResult,
    backlinks: dict[str, list[tuple[str, float]]],
    snippets: dict[str, str],
    settings: object,
) -> str:
    """Assemble the final LLM Call #5 user message (FR-E2).

    Enforces the token budget; truncates lowest-severity nodes first and
    emits a truncation note inside the context itself.
    """
    raise NotImplementedError("Sprint 8")
