"""Online pipeline orchestrator (nine steps, five LLM invocations).

Invoked by :func:`impactracer.cli.analyze`. Consumes a :class:`VariantFlags`
instance so the same code powers both full V7 analysis and the ablation
harness variants V0 through V6.

Reference: 07_online_pipeline.md, 09_ablation_harness.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from impactracer.shared.config import Settings
from impactracer.shared.models import ImpactReport

if TYPE_CHECKING:
    from impactracer.evaluation.variant_flags import VariantFlags


@dataclass
class PipelineContext:
    """Loaded persistent stores, shared across all pipeline steps."""

    conn: Any
    doc_col: Any
    code_col: Any
    graph: Any
    doc_bm25: Any
    doc_bm25_ids: list[str]
    code_bm25: Any
    code_bm25_ids: list[str]
    embedder: Any
    reranker: Any
    llm_client: Any


def load_pipeline_context(settings: Settings) -> PipelineContext:
    """Step 0 of the online pipeline: load every persistent dependency."""
    raise NotImplementedError("Sprint 8")


def run_analysis(
    cr_text: str,
    settings: Settings,
    variant_flags: "VariantFlags | None" = None,
) -> ImpactReport:
    """End-to-end online analysis for one CR.

    Defaults to variant V7 (full system) when ``variant_flags`` is None.
    """
    raise NotImplementedError("Sprint 8-10")
