"""Sprint 1 acceptance tests: schema round-trips, truncation, constants, settings."""

from __future__ import annotations

import os

import pytest
from pydantic import Field, ValidationError

from impactracer.shared.models import (
    CRInterpretation,
    CandidateVerdict,
    ImpactReport,
    ImpactedNode,
    PropagationVerdict,
    PropagationValidationResult,
    SISValidationResult,
    TraceVerdict,
    TraceValidationResult,
    TruncatingModel,
)
from impactracer.shared.constants import layer_compat, severity_for_chain
from impactracer.shared.config import Settings


# ---------------------------------------------------------------------------
# TruncatingModel
# ---------------------------------------------------------------------------


class _Short(TruncatingModel):
    label: str = Field(max_length=10)


def test_truncating_model_silently_truncates() -> None:
    """Overlong strings are truncated to max_length rather than raising."""
    obj = _Short(label="A" * 20)
    assert obj.label == "A" * 10


def test_truncating_model_exact_length_passes() -> None:
    """Strings at exactly max_length are not modified."""
    obj = _Short(label="B" * 10)
    assert obj.label == "B" * 10


def test_truncating_model_short_string_passes() -> None:
    """Strings shorter than max_length pass through unchanged."""
    obj = _Short(label="hello")
    assert obj.label == "hello"


# ---------------------------------------------------------------------------
# CRInterpretation — change_type enum + round-trip
# ---------------------------------------------------------------------------


def _make_cr_interpretation(change_type: str = "MODIFICATION") -> CRInterpretation:
    return CRInterpretation(
        is_actionable=True,
        actionability_reason=None,
        primary_intent="Add OAuth login to the platform.",
        change_type=change_type,  # type: ignore[arg-type]
        affected_layers=["requirement", "design", "code"],
        domain_concepts=["authentication", "OAuth"],
        search_queries=["OAuth login implementation", "auth middleware"],
        named_entry_points=["loginHandler"],
        out_of_scope_operations=["logout flow"],
    )


def test_cr_interpretation_change_type_enum() -> None:
    """change_type accepts only ADDITION / MODIFICATION / DELETION."""
    for valid in ("ADDITION", "MODIFICATION", "DELETION"):
        cr = _make_cr_interpretation(valid)
        assert cr.change_type == valid

    with pytest.raises(ValidationError):
        _make_cr_interpretation("UPDATE")  # invalid literal


def test_cr_interpretation_round_trip() -> None:
    """CRInterpretation -> JSON -> CRInterpretation is identity."""
    original = _make_cr_interpretation()
    restored = CRInterpretation.model_validate_json(original.model_dump_json())
    assert restored == original


# ---------------------------------------------------------------------------
# CandidateVerdict + SISValidationResult round-trips
# ---------------------------------------------------------------------------


def _make_candidate_verdict() -> CandidateVerdict:
    return CandidateVerdict(
        node_id="src/lib/auth.ts::loginHandler",
        function_purpose="Handles OAuth token exchange.",
        mechanism_of_impact="Must update token validation logic.",
        justification="Directly processes login tokens.",
        confirmed=True,
    )


def test_candidate_verdict_round_trip() -> None:
    v = _make_candidate_verdict()
    assert CandidateVerdict.model_validate_json(v.model_dump_json()) == v


def test_candidate_verdict_truncation() -> None:
    """function_purpose > 150 chars is silently truncated."""
    v = CandidateVerdict(
        node_id="x",
        function_purpose="X" * 200,
        mechanism_of_impact="",
        justification="Rejected.",
        confirmed=False,
    )
    assert len(v.function_purpose) == 150


def test_sis_validation_result_round_trip() -> None:
    result = SISValidationResult(verdicts=[_make_candidate_verdict()])
    assert SISValidationResult.model_validate_json(result.model_dump_json()) == result


# ---------------------------------------------------------------------------
# TraceVerdict + TraceValidationResult round-trips
# ---------------------------------------------------------------------------


def _make_trace_verdict(decision: str = "CONFIRMED") -> TraceVerdict:
    return TraceVerdict(
        doc_chunk_id="srs__functional_requirements",
        code_node_id="src/lib/auth.ts::loginHandler",
        decision=decision,  # type: ignore[arg-type]
        justification="Code implements the login requirement.",
    )


def test_trace_verdict_round_trip() -> None:
    v = _make_trace_verdict()
    assert TraceVerdict.model_validate_json(v.model_dump_json()) == v


def test_trace_verdict_decisions() -> None:
    """TraceDecision accepts CONFIRMED / PARTIAL / REJECTED only."""
    for valid in ("CONFIRMED", "PARTIAL", "REJECTED"):
        tv = _make_trace_verdict(valid)
        assert tv.decision == valid

    with pytest.raises(ValidationError):
        _make_trace_verdict("MAYBE")


def test_trace_validation_result_round_trip() -> None:
    result = TraceValidationResult(verdicts=[_make_trace_verdict()])
    assert TraceValidationResult.model_validate_json(result.model_dump_json()) == result


# ---------------------------------------------------------------------------
# PropagationVerdict + PropagationValidationResult round-trips
# ---------------------------------------------------------------------------


def _make_propagation_verdict() -> PropagationVerdict:
    return PropagationVerdict(
        node_id="src/components/LoginForm.tsx::LoginForm",
        semantically_impacted=True,
        justification="Calls loginHandler which changes its token contract.",
    )


def test_propagation_verdict_round_trip() -> None:
    v = _make_propagation_verdict()
    assert PropagationVerdict.model_validate_json(v.model_dump_json()) == v


def test_propagation_validation_result_round_trip() -> None:
    result = PropagationValidationResult(verdicts=[_make_propagation_verdict()])
    assert PropagationValidationResult.model_validate_json(result.model_dump_json()) == result


# ---------------------------------------------------------------------------
# ImpactedNode + ImpactReport round-trips
# ---------------------------------------------------------------------------


def _make_impacted_node(severity: str = "Tinggi", causal_chain: list[str] | None = None) -> ImpactedNode:
    return ImpactedNode(
        node_id="src/lib/auth.ts::loginHandler",
        node_type="Function",
        file_path="src/lib/auth.ts",
        severity=severity,  # type: ignore[arg-type]
        causal_chain=causal_chain or [],
        structural_justification="Directly implements the OAuth login flow.",
        traceability_backlinks=["srs__functional_requirements"],
    )


def _make_impact_report() -> ImpactReport:
    return ImpactReport(
        executive_summary="Introducing OAuth login affects the auth module and login UI.",
        impacted_nodes=[_make_impacted_node()],
        documentation_conflicts=["srs__security_requirements"],
        estimated_scope="menengah",
    )


def test_impacted_node_round_trip() -> None:
    node = _make_impacted_node()
    assert ImpactedNode.model_validate_json(node.model_dump_json()) == node


def test_impact_report_round_trip() -> None:
    """ImpactReport -> JSON -> ImpactReport is identity."""
    report = _make_impact_report()
    restored = ImpactReport.model_validate_json(report.model_dump_json())
    assert restored == report


def test_impact_report_executive_summary_truncation() -> None:
    """executive_summary > 800 chars is silently truncated."""
    report = ImpactReport(
        executive_summary="Z" * 1000,
        impacted_nodes=[],
        estimated_scope="terlokalisasi",
    )
    assert len(report.executive_summary) == 800


# ---------------------------------------------------------------------------
# severity_for_chain
# ---------------------------------------------------------------------------


def test_severity_for_chain_empty_is_tinggi() -> None:
    """SIS seeds (empty chain) are Tinggi by convention (blueprint §3.4)."""
    assert severity_for_chain([]) == "Tinggi"


def test_severity_for_chain_contract_edges_are_tinggi() -> None:
    """Contract edges IMPLEMENTS, TYPED_BY, FIELDS_ACCESSED → Tinggi."""
    assert severity_for_chain(["IMPLEMENTS"]) == "Tinggi"
    assert severity_for_chain(["TYPED_BY"]) == "Tinggi"
    assert severity_for_chain(["FIELDS_ACCESSED"]) == "Tinggi"
    # Mixed: contract edge dominates
    assert severity_for_chain(["CALLS", "IMPLEMENTS"]) == "Tinggi"


def test_severity_for_chain_behavioral_only_is_menengah() -> None:
    """Behavioral-only chains → Menengah."""
    assert severity_for_chain(["CALLS"]) == "Menengah"
    assert severity_for_chain(["INHERITS"]) == "Menengah"
    assert severity_for_chain(["CALLS", "DEFINES_METHOD"]) == "Menengah"


def test_severity_for_chain_module_only_is_rendah() -> None:
    """Module-composition-only chains → Rendah."""
    assert severity_for_chain(["IMPORTS"]) == "Rendah"
    assert severity_for_chain(["RENDERS"]) == "Rendah"
    assert severity_for_chain(["IMPORTS", "DYNAMIC_IMPORT"]) == "Rendah"


def test_severity_for_chain_behavioral_beats_module() -> None:
    """Behavioral edge elevates a module-only chain to Menengah."""
    assert severity_for_chain(["IMPORTS", "CALLS"]) == "Menengah"


# ---------------------------------------------------------------------------
# layer_compat — representative cells
# ---------------------------------------------------------------------------


def test_layer_compat_representative_cells() -> None:
    """Three representative cells from LAYER_COMPAT (blueprint §3.7)."""
    assert layer_compat("API_ROUTE", "FR") == 1.0
    assert layer_compat("TYPE_DEFINITION", "Design") == 0.9
    assert layer_compat("UTILITY", "NFR") == 0.7


def test_layer_compat_none_classification() -> None:
    """None classification falls back to the None row."""
    assert layer_compat(None, "FR") == 0.8
    assert layer_compat(None, "Design") == 0.8


def test_layer_compat_unknown_classification() -> None:
    """Unknown classification also falls back to the None row."""
    assert layer_compat("UNKNOWN_CLASS", "FR") == 0.8


# ---------------------------------------------------------------------------
# Settings defaults
# ---------------------------------------------------------------------------


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings instantiates with expected defaults when API key is provided."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-abc")
    # Prevent loading a real .env file during tests
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.llm_model == "google/gemini-2.5-flash"
    assert s.llm_temperature == 0.0
    assert s.llm_seed == 42
    assert s.rrf_k == 60
    assert s.min_traceability_similarity == 0.60
    assert s.degenerate_embed_min_length == 50
    assert s.bfs_high_conf_top_n == 5
    assert s.min_reranker_score_for_validation == 0.01
    assert s.plausibility_gate_density_threshold == 0.35
    assert s.plausibility_gate_max_per_file == 2
