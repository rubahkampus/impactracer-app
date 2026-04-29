"""Pre-validation deterministic gates (FR-C4).

Three sub-steps executed in order:

- Step 3.5 - Reranker score floor (default 0.01).
- Step 3.6 - Cross-collection semantic deduplication.
- Step 3.7 - Layer-aware affinity rescoring + file-density plausibility
             gate with named_entry_points exemption.

No LLM calls. Purely deterministic.

Reference: 08_gating_and_filters.md.
"""

from __future__ import annotations

import sqlite3

from impactracer.shared.models import Candidate, CRInterpretation


def apply_prevalidation_gates(
    candidates: list[Candidate],
    cr_interp: CRInterpretation,
    settings: object,
    conn: sqlite3.Connection,
    enable_dedup: bool = True,
    enable_plausibility: bool = True,
) -> list[Candidate]:
    """Apply Steps 3.5, 3.6, 3.7 in order.

    ``enable_dedup`` and ``enable_plausibility`` are variant flags; Step
    3.5 is always active because it is a precondition for LLM #2 sanity.
    """
    raise NotImplementedError("Sprint 9")


def step_3_5_score_filter(
    candidates: list[Candidate],
    threshold: float,
) -> list[Candidate]:
    """Drop candidates with ``reranker_score < threshold``."""
    raise NotImplementedError("Sprint 9")


def step_3_6_semantic_dedup(
    candidates: list[Candidate],
    conn: sqlite3.Connection,
) -> list[Candidate]:
    """Merge doc chunks whose top-1 resolution is already a code candidate."""
    raise NotImplementedError("Sprint 9")


def step_3_7_plausibility_and_affinity(
    candidates: list[Candidate],
    cr_interp: CRInterpretation,
    settings: object,
) -> list[Candidate]:
    """Rescore by layer affinity, then enforce file-density gate."""
    raise NotImplementedError("Sprint 9")
