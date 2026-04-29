"""Sprint 1 acceptance tests: schema round-trips and truncation."""

from __future__ import annotations

import pytest

from impactracer.shared.models import (
    CRInterpretation,
    CandidateVerdict,
    ImpactReport,
    ImpactedNode,
    TruncatingModel,
)


def test_truncating_model_silently_truncates() -> None:
    """Overlong strings are truncated rather than raising ValidationError."""
    raise NotImplementedError


def test_cr_interpretation_change_type_enum() -> None:
    """change_type accepts only ADDITION / MODIFICATION / DELETION."""
    raise NotImplementedError


def test_impact_report_round_trip() -> None:
    """ImpactReport -> JSON -> ImpactReport is identity."""
    raise NotImplementedError
