"""Unit tests for pipeline/validator.py (FR-C5).

Blueprint: master_blueprint.md §4 Step 4.
"""

from __future__ import annotations

from unittest.mock import MagicMock


from impactracer.pipeline.validator import (
    _BATCH_SIZE,
    build_validator_prompt,
    chunk_candidates,
    validate_sis_candidates_batched,
)
from impactracer.shared.models import (
    Candidate,
    CandidateVerdict,
    CRInterpretation,
    SISValidationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cr(
    named_entry_points=None,
    out_of_scope_operations=None,
) -> CRInterpretation:
    return CRInterpretation(
        is_actionable=True,
        primary_intent="Add pin feature to commission listings",
        change_type="ADDITION",
        affected_layers=["requirement", "design", "code"],
        domain_concepts=["pin", "commission listing", "profile"],
        search_queries=["pin listing", "commission pin feature"],
        named_entry_points=named_entry_points or [],
        out_of_scope_operations=out_of_scope_operations or [],
    )


def _make_code_candidate(
    node_id="src/lib/services/commission.service.ts::pinListing",
    file_path="src/lib/services/commission.service.ts",
    node_type="Function",
    reranker_score=0.8,
    snippet="async function pinListing() {...}",
    abstraction=None,
) -> Candidate:
    return Candidate(
        node_id=node_id,
        node_type=node_type,
        collection="code_units",
        rrf_score=0.5,
        reranker_score=reranker_score,
        file_path=file_path,
        file_classification="UTILITY",
        name=node_id.split("::")[-1] if "::" in node_id else node_id,
        text_snippet=snippet,
        internal_logic_abstraction=abstraction,
    )


def _make_verdict(node_id: str, confirmed: bool) -> CandidateVerdict:
    return CandidateVerdict(
        node_id=node_id,
        function_purpose="test purpose",
        mechanism_of_impact="add field" if confirmed else "",
        justification="test reason",
        confirmed=confirmed,
    )


def _mock_llm(confirmed_ids: list[str]):
    """Return a mock LLMClient that confirms only the given IDs."""
    def _call(system, user, response_schema, call_name):
        # B6: node_ids are wrapped in <<NODE_ID_START>>...<<NODE_ID_END>> delimiters
        import re
        ids_in_prompt = re.findall(r"<<NODE_ID_START>>(.+?)<<NODE_ID_END>>", user)
        verdicts = [
            _make_verdict(nid, nid in confirmed_ids)
            for nid in ids_in_prompt
        ]
        return SISValidationResult(verdicts=verdicts)

    client = MagicMock()
    client.call.side_effect = _call
    return client


# ---------------------------------------------------------------------------
# chunk_candidates
# ---------------------------------------------------------------------------

def test_chunk_candidates_empty():
    assert chunk_candidates([]) == []


def test_chunk_candidates_fewer_than_batch_size():
    candidates = [_make_code_candidate(node_id=f"fn{i}") for i in range(3)]
    chunks = chunk_candidates(candidates)
    assert len(chunks) == 1
    assert len(chunks[0]) == 3


def test_chunk_candidates_exact_batch_size():
    candidates = [_make_code_candidate(node_id=f"fn{i}") for i in range(_BATCH_SIZE)]
    chunks = chunk_candidates(candidates)
    assert len(chunks) == 1
    assert len(chunks[0]) == _BATCH_SIZE


def test_chunk_candidates_splits_correctly():
    candidates = [_make_code_candidate(node_id=f"fn{i}") for i in range(12)]
    chunks = chunk_candidates(candidates, batch_size=5)
    assert len(chunks) == 3
    assert len(chunks[0]) == 5
    assert len(chunks[1]) == 5
    assert len(chunks[2]) == 2


def test_chunk_candidates_custom_batch_size():
    candidates = [_make_code_candidate(node_id=f"fn{i}") for i in range(7)]
    chunks = chunk_candidates(candidates, batch_size=3)
    assert len(chunks) == 3
    sizes = [len(c) for c in chunks]
    assert sizes == [3, 3, 1]


# ---------------------------------------------------------------------------
# build_validator_prompt — structural tests
# ---------------------------------------------------------------------------

def test_prompt_contains_cr_intent():
    cr = _make_cr()
    c = _make_code_candidate()
    prompt = build_validator_prompt(cr, [c])
    assert cr.primary_intent in prompt


def test_prompt_contains_change_type():
    cr = _make_cr()
    c = _make_code_candidate()
    prompt = build_validator_prompt(cr, [c])
    assert "ADDITION" in prompt


def test_prompt_contains_domain_concepts():
    cr = _make_cr()
    c = _make_code_candidate()
    prompt = build_validator_prompt(cr, [c])
    assert "pin" in prompt


def test_prompt_contains_node_id():
    cr = _make_cr()
    c = _make_code_candidate(node_id="src/lib/services/commission.service.ts::pinListing")
    prompt = build_validator_prompt(cr, [c])
    assert "src/lib/services/commission.service.ts::pinListing" in prompt


def test_prompt_contains_file_path():
    cr = _make_cr()
    c = _make_code_candidate(file_path="src/lib/services/commission.service.ts")
    prompt = build_validator_prompt(cr, [c])
    assert "src/lib/services/commission.service.ts" in prompt


def test_prompt_contains_node_type():
    cr = _make_cr()
    c = _make_code_candidate(node_type="Function")
    prompt = build_validator_prompt(cr, [c])
    assert "Function" in prompt


def test_prompt_prefers_internal_logic_abstraction_over_snippet():
    cr = _make_cr()
    c = _make_code_candidate(
        snippet="raw text_snippet content",
        abstraction="const pinned = await db.pin(id);",
    )
    prompt = build_validator_prompt(cr, [c])
    assert "const pinned = await db.pin(id);" in prompt
    assert "raw text_snippet content" not in prompt


def test_prompt_falls_back_to_text_snippet():
    cr = _make_cr()
    c = _make_code_candidate(snippet="fallback snippet", abstraction=None)
    prompt = build_validator_prompt(cr, [c])
    assert "fallback snippet" in prompt


def test_prompt_excludes_retrieval_scores():
    cr = _make_cr()
    c = _make_code_candidate(reranker_score=0.9876)
    prompt = build_validator_prompt(cr, [c])
    # Must not contain the exact score value (Anti-Circular Logic Mandate)
    assert "0.9876" not in prompt
    assert "reranker" not in prompt.lower()
    assert "rrf" not in prompt.lower()
    assert "cosine" not in prompt.lower()


def test_prompt_includes_out_of_scope_when_non_empty():
    cr = _make_cr(out_of_scope_operations=["delete listing", "update price"])
    c = _make_code_candidate()
    prompt = build_validator_prompt(cr, [c])
    assert "OUT-OF-SCOPE" in prompt
    assert "delete listing" in prompt
    assert "update price" in prompt


def test_prompt_includes_named_entry_points_when_non_empty():
    cr = _make_cr(named_entry_points=["pinListing", "ProfilePage"])
    c = _make_code_candidate()
    prompt = build_validator_prompt(cr, [c])
    assert "NAMED ENTRY POINTS" in prompt
    assert "pinListing" in prompt


def test_prompt_omits_out_of_scope_section_when_empty():
    cr = _make_cr(out_of_scope_operations=[])
    c = _make_code_candidate()
    prompt = build_validator_prompt(cr, [c])
    assert "OUT-OF-SCOPE" not in prompt


def test_prompt_sends_full_snippet_without_truncation():
    cr = _make_cr()
    long_snippet = "x" * 3000
    c = _make_code_candidate(snippet=long_snippet, abstraction=None)
    prompt = build_validator_prompt(cr, [c])
    # Full snippet sent — ILA is compact by design; truncation drops call sites
    assert "x" * 3000 in prompt


def test_prompt_numbers_candidates_starting_at_1():
    cr = _make_cr()
    c1 = _make_code_candidate(node_id="fn1")
    c2 = _make_code_candidate(node_id="fn2")
    prompt = build_validator_prompt(cr, [c1, c2])
    # B6: node_id is wrapped in unambiguous delimiters
    assert "[1] CODE NODE" in prompt
    assert "<<NODE_ID_START>>fn1<<NODE_ID_END>>" in prompt
    assert "[2] CODE NODE" in prompt
    assert "<<NODE_ID_START>>fn2<<NODE_ID_END>>" in prompt


def test_prompt_node_id_verbatim_delimiters():
    """B6: node_id is wrapped in <<NODE_ID_START>>...<<NODE_ID_END>> delimiters."""
    cr = _make_cr()
    c = _make_code_candidate(node_id="src/lib/services/commission.service.ts::pinListing")
    prompt = build_validator_prompt(cr, [c])
    assert "<<NODE_ID_START>>src/lib/services/commission.service.ts::pinListing<<NODE_ID_END>>" in prompt


def test_prompt_doc_chunk_has_documentation_section_header():
    """B5: doc chunk candidates use DOCUMENTATION SECTION header."""
    from impactracer.shared.models import Candidate
    cr = _make_cr()
    doc = Candidate(
        node_id="srs__v_1_pin",
        node_type="DocChunk",
        collection="doc_chunks",
        rrf_score=0.3,
        reranker_score=0.5,
        file_path="docs/srs.md",
        chunk_type="FR",
        name="srs__v_1_pin",
        text_snippet="Users can pin listings to their profile.",
    )
    prompt = build_validator_prompt(cr, [doc])
    assert "DOCUMENTATION SECTION" in prompt
    assert "<<NODE_ID_START>>srs__v_1_pin<<NODE_ID_END>>" in prompt
    assert "Confirmation criterion" in prompt


def test_prompt_business_context_injected_for_merged_doc():
    """B1: merged_doc_contexts are injected as Business Context blocks."""
    cr = _make_cr()
    c = _make_code_candidate()
    c.merged_doc_contexts = [("FR-P01 Pin Feature", "Users can pin listings.")]
    prompt = build_validator_prompt(cr, [c])
    assert "Business Context" in prompt
    assert "FR-P01 Pin Feature" in prompt
    assert "Users can pin listings." in prompt


def test_prompt_addition_cr_forward_looking_instruction():
    """N1: system prompt includes ADDITION-CR forward-looking instruction."""
    from impactracer.pipeline.validator import _SYSTEM_PROMPT
    assert "ADDITION" in _SYSTEM_PROMPT
    assert "forward" in _SYSTEM_PROMPT.lower() or "look forward" in _SYSTEM_PROMPT.lower() or "ADDITION CRs" in _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# validate_sis_candidates_batched
# ---------------------------------------------------------------------------

def test_validate_empty_candidates():
    cr = _make_cr()
    client = MagicMock()
    ids, justifications, degraded = validate_sis_candidates_batched(cr, [], client)
    assert ids == []
    assert justifications == {}
    assert degraded is False
    client.call.assert_not_called()


def test_validate_single_batch_all_confirmed():
    cr = _make_cr()
    candidates = [_make_code_candidate(node_id=f"fn{i}") for i in range(3)]
    all_ids = [c.node_id for c in candidates]
    client = _mock_llm(confirmed_ids=all_ids)

    ids, justifications, degraded = validate_sis_candidates_batched(cr, candidates, client)
    assert set(ids) == set(all_ids)
    assert client.call.call_count == 1
    assert all(nid in justifications for nid in ids)
    assert degraded is False


def test_validate_single_batch_partial_confirm():
    cr = _make_cr()
    c1 = _make_code_candidate(node_id="fn1")
    c2 = _make_code_candidate(node_id="fn2")
    c3 = _make_code_candidate(node_id="fn3")
    client = _mock_llm(confirmed_ids=["fn1", "fn3"])

    ids, _justs, _deg = validate_sis_candidates_batched(cr, [c1, c2, c3], client)
    assert "fn1" in ids
    assert "fn2" not in ids
    assert "fn3" in ids


def test_validate_multiple_batches():
    cr = _make_cr()
    candidates = [_make_code_candidate(node_id=f"fn{i}") for i in range(11)]
    all_ids = [c.node_id for c in candidates]
    client = _mock_llm(confirmed_ids=all_ids)

    ids, _justs, _deg = validate_sis_candidates_batched(cr, candidates, client)
    # 11 candidates -> 3 batches (5+5+1)
    assert client.call.call_count == 3
    assert set(ids) == set(all_ids)


def test_validate_call_name_is_validate_sis():
    cr = _make_cr()
    c = _make_code_candidate(node_id="fn1")
    client = MagicMock()
    client.call.return_value = SISValidationResult(
        verdicts=[_make_verdict("fn1", True)]
    )

    validate_sis_candidates_batched(cr, [c], client)
    call_kwargs = client.call.call_args
    assert call_kwargs[1]["call_name"] == "validate_sis" or call_kwargs[0][3] == "validate_sis"


def test_validate_fail_closed_on_empty_verdicts():
    """Crucible Fix 1 (FF-1): missing verdict -> DROP (was admit)."""
    cr = _make_cr()
    candidates = [_make_code_candidate(node_id="fn1")]
    client = MagicMock()
    client.call.return_value = SISValidationResult(verdicts=[])

    ids, _justs, _deg = validate_sis_candidates_batched(cr, candidates, client)
    # Fail-closed: candidate with no verdict is dropped, not admitted.
    assert ids == []


def test_validate_fail_closed_per_node_partial_coverage():
    """Crucible Fix 1 (FF-1): nodes with NO verdict are DROPPED (fail-closed)."""
    cr = _make_cr()
    c1 = _make_code_candidate(node_id="fn1")
    c2 = _make_code_candidate(node_id="fn2")
    c3 = _make_code_candidate(node_id="fn3")
    client = MagicMock()
    client.call.return_value = SISValidationResult(verdicts=[
        _make_verdict("fn1", True),
        _make_verdict("fn3", False),
    ])

    ids, _justs, _deg = validate_sis_candidates_batched(cr, [c1, c2, c3], client)
    assert "fn1" in ids       # explicitly confirmed
    assert "fn2" not in ids   # no verdict -> fail-CLOSED -> dropped
    assert "fn3" not in ids   # explicitly rejected


def test_validate_batch_exception_drops_batch_continues():
    """Crucible Amendment 1: batch exception -> drop batch, continue."""
    cr = _make_cr()
    # Two batches: 5 + 1. First raises, second succeeds.
    candidates = [_make_code_candidate(node_id=f"fn{i}") for i in range(6)]
    client = MagicMock()
    second_ok = SISValidationResult(verdicts=[_make_verdict("fn5", True)])
    client.call.side_effect = [RuntimeError("rate limit exhausted"), second_ok]

    ids, _justs, degraded = validate_sis_candidates_batched(cr, candidates, client)
    # First batch's 5 candidates dropped; second batch's fn5 admitted.
    assert ids == ["fn5"]
    assert degraded is True


def test_validate_rejects_verdicts_for_wrong_batch():
    """Verdicts referencing IDs not in the batch are silently ignored."""
    cr = _make_cr()
    c = _make_code_candidate(node_id="correct_fn")
    client = MagicMock()
    client.call.return_value = SISValidationResult(
        verdicts=[
            _make_verdict("wrong_fn", True),  # not in batch
            _make_verdict("correct_fn", True),
        ]
    )

    ids, _justs, _deg = validate_sis_candidates_batched(cr, [c], client)
    assert ids == ["correct_fn"]
    assert "wrong_fn" not in ids
