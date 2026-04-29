"""CLI helper for Ground Truth construction.

Lets an annotator browse code_nodes by file, mark nodes with a confidence
label (core / supporting / peripheral), and emit a GT JSON entry.

Reference: 10_evaluation_protocol.md §2.
"""

from __future__ import annotations

from pathlib import Path


def annotate_cr(cr_text: str, db_path: Path) -> dict:
    """Interactive annotation session; returns a GT entry dict."""
    raise NotImplementedError("Sprint 12")
