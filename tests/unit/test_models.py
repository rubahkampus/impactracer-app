"""Sprint 1 acceptance tests: schema round-trips, truncation, constants, settings."""

from __future__ import annotations

import os

import pytest
from pydantic import Field, ValidationError

from impactracer.shared.models import (
    CRInterpretation,
    CandidateVerdict,
    CISResult,
    ImpactReport,
    ImpactedNode,
    NodeTrace,
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
    """function_purpose > 200 chars is silently truncated.

    Crucible Fix 3: max_length raised 150 -> 200 to allow richer
    function purposes for distributed-justification rendering.
    """
    v = CandidateVerdict(
        node_id="x",
        function_purpose="X" * 300,
        mechanism_of_impact="",
        justification="Rejected.",
        confirmed=False,
    )
    assert len(v.function_purpose) == 200


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
    """ImpactReport -> JSON -> ImpactReport is identity at the serialized level.

    Crucible E2E Schema Alignment: ImpactedNode is a backward-compat
    subclass of ImpactedEntity. After round-trip, Pydantic re-instantiates
    using the field-typed class (ImpactedEntity), so we compare on the
    serialized JSON form, which is the contract that matters for the API.
    """
    report = _make_impact_report()
    restored = ImpactReport.model_validate_json(report.model_dump_json())
    assert restored.model_dump() == report.model_dump()


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
    """Contract edges IMPLEMENTS, TYPED_BY, FIELDS_ACCESSED → Tinggi (last-hop rule)."""
    assert severity_for_chain(["IMPLEMENTS"]) == "Tinggi"
    assert severity_for_chain(["TYPED_BY"]) == "Tinggi"
    assert severity_for_chain(["FIELDS_ACCESSED"]) == "Tinggi"
    # Phase 1 (F-NEW-4): last-hop rule — CALLS → IMPLEMENTS: last hop is
    # IMPLEMENTS → Tinggi (previously this was also Tinggi under min(), but
    # now the semantics are explicit: the *final* dependency type determines
    # severity, not the minimum across the entire chain).
    assert severity_for_chain(["CALLS", "IMPLEMENTS"]) == "Tinggi"


def test_severity_for_chain_behavioral_only_is_menengah() -> None:
    """Behavioral-only chains → Menengah (last-hop rule)."""
    assert severity_for_chain(["CALLS"]) == "Menengah"
    assert severity_for_chain(["INHERITS"]) == "Menengah"
    assert severity_for_chain(["CALLS", "DEFINES_METHOD"]) == "Menengah"


def test_severity_for_chain_module_only_is_rendah() -> None:
    """Module-composition-only chains → Rendah (last-hop rule)."""
    assert severity_for_chain(["IMPORTS"]) == "Rendah"
    assert severity_for_chain(["RENDERS"]) == "Rendah"
    assert severity_for_chain(["IMPORTS", "DYNAMIC_IMPORT"]) == "Rendah"


def test_severity_for_chain_last_hop_determines_severity() -> None:
    """Phase 1 (F-NEW-4): LAST edge in chain determines severity, not minimum.

    Under the old min() rule, IMPORTS→CALLS→IMPLEMENTS would yield Tinggi
    (minimum rank). Under the last-hop rule, the same chain yields Tinggi only
    because the last edge IS IMPLEMENTS. But IMPLEMENTS→CALLS→IMPORTS must
    yield Rendah (last hop = IMPORTS), NOT Tinggi.

    This test verifies the anti-laundering property: a high-severity first hop
    does NOT elevate a low-severity last hop.
    """
    # Last hop is IMPLEMENTS → Tinggi
    assert severity_for_chain(["IMPORTS", "CALLS", "IMPLEMENTS"]) == "Tinggi"
    # Last hop is IMPORTS → Rendah (not elevated by earlier IMPLEMENTS)
    assert severity_for_chain(["IMPLEMENTS", "CALLS", "IMPORTS"]) == "Rendah"
    # Last hop is CALLS → Menengah (not elevated by earlier IMPLEMENTS)
    assert severity_for_chain(["IMPLEMENTS", "CALLS"]) == "Menengah"


def test_severity_for_chain_behavioral_beats_module() -> None:
    """Last-hop CALLS after IMPORTS → Menengah (CALLS is the proximate dep)."""
    assert severity_for_chain(["IMPORTS", "CALLS"]) == "Menengah"


# ---------------------------------------------------------------------------
# layer_compat — representative cells
# ---------------------------------------------------------------------------


def test_layer_compat_representative_cells() -> None:
    """Representative cells from LAYER_COMPAT (Sprint 6.5 recalibration)."""
    assert layer_compat("API_ROUTE",       "FR")     == 1.0
    assert layer_compat("TYPE_DEFINITION", "Design") == 1.0   # models ↔ DB design = perfect match
    assert layer_compat("UTILITY",         "NFR")    == 0.7
    assert layer_compat("UTILITY",         "Design") == 1.0   # services ↔ component design
    assert layer_compat("UTILITY",         "General") == 0.8  # services appear in process flows
    assert layer_compat("UI_COMPONENT",    "General") == 0.7  # stores/hooks in process flows
    assert layer_compat("TYPE_DEFINITION", "FR")     == 0.8   # data structures implement FRs


def test_layer_compat_none_classification() -> None:
    """None classification falls back to the None row (conservative)."""
    assert layer_compat(None, "FR")     == 0.8
    assert layer_compat(None, "Design") == 0.8


def test_layer_compat_unknown_classification() -> None:
    """Unknown classification also falls back to the None row."""
    assert layer_compat("UNKNOWN_CLASS", "FR") == 0.8


# ---------------------------------------------------------------------------
# Settings defaults
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CISResult — Phase 1 invariants (E-4)
# ---------------------------------------------------------------------------


def _make_node_trace(depth: int, chain: list[str], seed: str = "s1") -> NodeTrace:
    return NodeTrace(
        depth=depth,
        causal_chain=chain,
        path=[seed] + [f"n{i}" for i in range(depth)],
        source_seed=seed,
    )


def test_combined_sis_overwrites_propagated() -> None:
    """Phase 1 (E-4): when same node_id appears in both dicts, SIS trace wins.

    Invariant: combined()[node_id] is sis_nodes[node_id], not the propagated trace.
    This matters because seeds have depth=0 / empty causal_chain; using the
    propagated trace would misrepresent them as structurally inferred nodes.
    """
    seed_trace = _make_node_trace(depth=0, chain=[], seed="A")
    propagated_trace = _make_node_trace(depth=2, chain=["CALLS", "IMPORTS"], seed="A")

    cis = CISResult(
        sis_nodes={"A": seed_trace},
        propagated_nodes={"A": propagated_trace, "B": _make_node_trace(1, ["CALLS"])},
    )
    combined = cis.combined()

    # A must use the SIS trace (depth=0), not the propagated one (depth=2).
    assert combined["A"].depth == 0
    assert combined["A"].causal_chain == []
    assert combined["A"] is seed_trace

    # B (propagated only) appears unchanged.
    assert combined["B"].depth == 1


def test_combined_all_node_ids_complete() -> None:
    """all_node_ids() returns every unique node_id in SIS + propagated."""
    cis = CISResult(
        sis_nodes={"s1": _make_node_trace(0, [])},
        propagated_nodes={
            "p1": _make_node_trace(1, ["CALLS"]),
            "p2": _make_node_trace(2, ["CALLS", "IMPLEMENTS"]),
        },
    )
    ids = cis.all_node_ids()
    assert set(ids) == {"s1", "p1", "p2"}


def test_combined_empty_cis() -> None:
    """Empty CIS combined() returns empty dict."""
    cis = CISResult()
    assert cis.combined() == {}
    assert cis.all_node_ids() == []


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings instantiates with expected defaults when API key is provided."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-abc")
    # Prevent loading a real .env file during tests
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.llm_model == "google/gemini-2.5-flash"
    assert s.llm_temperature == 0.0
    assert s.llm_seed == 42
    assert s.rrf_k == 60
    assert s.min_traceability_similarity == 0.40
    assert s.degenerate_embed_min_length == 50
    assert s.bfs_high_conf_top_n == 5
    # Crucible Fix 9: score floor demoted to a sanity-only gate (-2.0).
    assert s.min_reranker_score_for_validation == -2.0
    # Crucible Fix 9: density threshold raised from 0.35 -> 0.50.
    assert s.plausibility_gate_density_threshold == 0.50
    # Crucible Fix 9: plausibility_gate_max_per_file removed entirely.
    assert not hasattr(s, "plausibility_gate_max_per_file")
