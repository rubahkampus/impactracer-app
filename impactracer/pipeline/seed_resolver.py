"""Trace resolution: doc-chunk SIS -> code seeds (FR-C6).

Uses the precomputed ``doc_code_candidates`` table. Direct code-node SIS
entries bypass resolution. The output feeds LLM #3 when enabled.

Reference: 07_online_pipeline.md §8.
"""

from __future__ import annotations

import sqlite3


def resolve_doc_to_code(
    sis_ids: list[str],
    conn: sqlite3.Connection,
    top_k: int,
) -> tuple[list[dict], dict[str, list[str]], list[str]]:
    """Return (resolutions, doc_to_code_map, direct_code_seeds).

    - ``resolutions``: list of ``{"doc_id": ..., "code_ids": [...]}``
      for doc-chunk SIS entries that resolved to at least one code node.
    - ``doc_to_code_map``: ``{doc_id: [code_ids]}`` for synthesis backlinks.
    - ``direct_code_seeds``: SIS entries that were already code nodes.
    """
    raise NotImplementedError("Sprint 10")
