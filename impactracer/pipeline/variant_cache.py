"""Per-CR variant-chain cache.

The V0-V7 ablation is structurally additive. V_{n+1} extends V_n by
exactly one operation (cross-encoder, gate, LLM call, or BFS step).
Without caching, every variant re-runs LLM #1 (interpret) and any
LLM validators it shares with earlier variants. Each re-roll is a
non-deterministic Gemini draw, contaminating the V_{n+1} - V_n delta
with noise the ablation was not designed to measure.

This module caches the pure-function outputs of pipeline stages so
that V_{n+1} resumes from V_n's cached state at the appropriate
boundary. Inter-variant comparisons become paired (shared inputs)
instead of unpaired (independent draws).

Cache scope and isolation:
  - Per `run_tag` (UTC timestamp at harness start). Different runs
    NEVER share cache.
  - Per `cr_id`.
  - Per `anchor_priming` value (True or False). The two methodology
    branches produce different LLM #1 outputs and must NOT collide.
  - Per cache key (one boundary in the pipeline).

Cache shareability across variants (within one (run_tag, cr_id,
anchor_priming) scope):

  interp                  V0-V7 (LLM #1 output is variant-independent)
  retrieval_v0            V0 only (BM25-only)
  retrieval_v1            V1 only (dense-only)
  retrieval_v2plus        V2-V7 (BM25 + dense + RRF)
  rerank_gated            V3-V7 (cross-encoder + 3 gates output)
  sis_verdicts            V4-V7 (LLM #2 SIS verdicts)
  trace_verdicts          V5-V7 (LLM #3 trace verdicts + code seeds)
  bfs_cis                 V6-V7 (BFS CIS with sis_nodes and propagated)
  llm4_verdicts           V7 only

Deep-copy semantics: cache reads always return fresh objects via
Pydantic ``model_validate_json``. Pipeline code can mutate the
returned candidates in place (rerank score normalisation, pinning)
without corrupting the cache for later variants.

Reference: ``docs/evaluation_protocol.md``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from impactracer.shared.models import (
    CISResult,
    Candidate,
    CRInterpretation,
    NodeTrace,
)


class VariantCache:
    """File-backed cache for one CR within one (run_tag, anchor_priming) scope.

    Each cache key is one JSON file under
    ``<root_dir>/<run_tag>/<cr_id>/<priming_tag>/<key>.json``.

    Missing files yield ``None`` from getters. Putters overwrite. Getters
    always return deep copies via Pydantic ``model_validate``; pipeline
    code is free to mutate the returned objects without corrupting the
    cache.
    """

    def __init__(
        self,
        root_dir: Path | str,
        run_tag: str,
        cr_id: str,
        anchor_priming: bool,
    ) -> None:
        priming_tag = "anchor_on" if anchor_priming else "anchor_off"
        self._scope_dir = (
            Path(root_dir) / run_tag / cr_id / priming_tag
        )

    def _path(self, key: str) -> Path:
        return self._scope_dir / f"{key}.json"

    def _read_json(self, key: str) -> object | None:
        p = self._path(key)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def _write_json(self, key: str, data: object) -> None:
        self._scope_dir.mkdir(parents=True, exist_ok=True)
        self._path(key).write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    # ----------------------------------------------------------------
    # Step 1 — Interpret (LLM #1)
    # ----------------------------------------------------------------

    def get_interp(self) -> CRInterpretation | None:
        raw = self._read_json("interp")
        if raw is None:
            return None
        return CRInterpretation.model_validate(raw)

    def put_interp(self, ci: CRInterpretation) -> None:
        self._write_json("interp", ci.model_dump())

    # ----------------------------------------------------------------
    # Step 2 — Retrieval (variant-scoped)
    # ----------------------------------------------------------------

    def _get_candidates(self, key: str) -> list[Candidate] | None:
        raw = self._read_json(key)
        if raw is None:
            return None
        # Deep-copy by reconstructing each Candidate from dict.
        return [_candidate_from_dict(d) for d in raw]

    def _put_candidates(self, key: str, candidates: list[Candidate]) -> None:
        self._write_json(key, [_candidate_to_dict(c) for c in candidates])

    def get_retrieval_v0(self) -> list[Candidate] | None:
        return self._get_candidates("retrieval_v0")

    def put_retrieval_v0(self, candidates: list[Candidate]) -> None:
        self._put_candidates("retrieval_v0", candidates)

    def get_retrieval_v1(self) -> list[Candidate] | None:
        return self._get_candidates("retrieval_v1")

    def put_retrieval_v1(self, candidates: list[Candidate]) -> None:
        self._put_candidates("retrieval_v1", candidates)

    def get_retrieval_v2plus(self) -> list[Candidate] | None:
        return self._get_candidates("retrieval_v2plus")

    def put_retrieval_v2plus(self, candidates: list[Candidate]) -> None:
        self._put_candidates("retrieval_v2plus", candidates)

    # ----------------------------------------------------------------
    # Step 3 — Cross-encoder rerank + 3 gates (V3-V7 share)
    # ----------------------------------------------------------------

    def get_rerank_gated(self) -> list[Candidate] | None:
        return self._get_candidates("rerank_gated")

    def put_rerank_gated(self, candidates: list[Candidate]) -> None:
        self._put_candidates("rerank_gated", candidates)

    # ----------------------------------------------------------------
    # Step 4 — LLM #2 SIS validation (V4-V7 share)
    # ----------------------------------------------------------------

    def get_sis_verdicts(self) -> tuple[list[str], dict[str, dict[str, str]], bool] | None:
        raw = self._read_json("sis_verdicts")
        if raw is None:
            return None
        return (
            list(raw.get("confirmed_ids", [])),
            dict(raw.get("justifications", {})),
            bool(raw.get("degraded", False)),
        )

    def put_sis_verdicts(
        self,
        confirmed_ids: list[str],
        justifications: dict[str, dict[str, str]],
        degraded: bool,
    ) -> None:
        self._write_json(
            "sis_verdicts",
            {
                "confirmed_ids": list(confirmed_ids),
                "justifications": dict(justifications),
                "degraded": bool(degraded),
            },
        )

    # ----------------------------------------------------------------
    # Step 5b — LLM #3 trace validation + code seeds (V5-V7 share)
    # ----------------------------------------------------------------

    def get_trace_verdicts(
        self,
    ) -> tuple[list[str], dict[str, bool], dict[str, str], dict[str, str], bool] | None:
        raw = self._read_json("trace_verdicts")
        if raw is None:
            return None
        return (
            list(raw.get("validated_code_seeds", [])),
            {k: bool(v) for k, v in raw.get("low_confidence", {}).items()},
            dict(raw.get("justifications", {})),
            # mechanisms added when LLM #3 adopted the LLM #2-style two-standard
            # test. Older cache files (pre-fix) lack the key; default to {} so a
            # stale cache degrades gracefully to "no anchor-eligible resolved
            # seeds" rather than crashing the 5-tuple unpack.
            dict(raw.get("mechanisms", {})),
            bool(raw.get("degraded", False)),
        )

    def put_trace_verdicts(
        self,
        validated_code_seeds: list[str],
        low_confidence: dict[str, bool],
        justifications: dict[str, str],
        mechanisms: dict[str, str],
        degraded: bool,
    ) -> None:
        self._write_json(
            "trace_verdicts",
            {
                "validated_code_seeds": list(validated_code_seeds),
                "low_confidence": {k: bool(v) for k, v in low_confidence.items()},
                "justifications": dict(justifications),
                "mechanisms": dict(mechanisms),
                "degraded": bool(degraded),
            },
        )

    # ----------------------------------------------------------------
    # Step 6 — BFS CIS (V6-V7 share)
    # ----------------------------------------------------------------

    def get_bfs_cis(self) -> CISResult | None:
        raw = self._read_json("bfs_cis")
        if raw is None:
            return None
        return _cisresult_from_dict(raw)

    def put_bfs_cis(self, cis: CISResult) -> None:
        self._write_json("bfs_cis", _cisresult_to_dict(cis))

    # ----------------------------------------------------------------
    # Step 7 — LLM #4 propagation validation (V7 only)
    # ----------------------------------------------------------------

    def get_llm4_verdicts(self) -> tuple[CISResult, dict[str, str], bool] | None:
        raw = self._read_json("llm4_verdicts")
        if raw is None:
            return None
        return (
            _cisresult_from_dict(raw["cis"]),
            dict(raw.get("justifications", {})),
            bool(raw.get("degraded", False)),
        )

    def put_llm4_verdicts(
        self,
        cis: CISResult,
        justifications: dict[str, str],
        degraded: bool,
    ) -> None:
        self._write_json(
            "llm4_verdicts",
            {
                "cis": _cisresult_to_dict(cis),
                "justifications": dict(justifications),
                "degraded": bool(degraded),
            },
        )

    # ----------------------------------------------------------------
    # Step 7.5 — LLM #4 sibling promotion (V6-V7 shared after the
    # 2026-05-26 ablation-boundary fix; sibling promotion moved from
    # V7-only gating to V6+ gating to isolate the LLM #4 propagation
    # validation contribution at the V6->V7 boundary).
    #
    # Cached as the set of admitted sibling node_ids plus their LLM #4
    # justifications. The cache key is shared across V6 and V7 because
    # both variants read the same bfs_cis and run sibling promotion
    # against the same SIS-confirmed anchor set, so the LLM #4 sibling
    # call inputs are identical. Pairing this across V6 and V7 preserves
    # the Amendment-2 paired-clean comparison invariant.
    # ----------------------------------------------------------------

    def get_sibling_admissions(
        self,
    ) -> tuple[list[str], dict[str, str], int] | None:
        raw = self._read_json("sibling_admissions")
        if raw is None:
            return None
        return (
            list(raw.get("admitted_ids", [])),
            dict(raw.get("justifications", {})),
            int(raw.get("admitted_count", 0)),
        )

    def put_sibling_admissions(
        self,
        admitted_ids: list[str],
        justifications: dict[str, str],
        admitted_count: int,
    ) -> None:
        self._write_json(
            "sibling_admissions",
            {
                "admitted_ids": list(admitted_ids),
                "justifications": dict(justifications),
                "admitted_count": int(admitted_count),
            },
        )


# ----------------------------------------------------------------------
# Candidate serialisation helpers
# ----------------------------------------------------------------------


def _candidate_to_dict(c: Candidate) -> dict:
    """Serialise a Candidate dataclass to a JSON-safe dict.

    ``merged_doc_contexts`` is a list of tuples — JSON has no tuple type,
    so it round-trips as a list of two-element lists. The reader
    reconstructs tuples in :func:`_candidate_from_dict`.
    """
    return {
        "node_id": c.node_id,
        "node_type": c.node_type,
        "collection": c.collection,
        "rrf_score": c.rrf_score,
        "reranker_score": c.reranker_score,
        "raw_reranker_score": c.raw_reranker_score,
        "file_path": c.file_path,
        "file_classification": c.file_classification,
        "chunk_type": c.chunk_type,
        "name": c.name,
        "text_snippet": c.text_snippet,
        "internal_logic_abstraction": c.internal_logic_abstraction,
        "merged_doc_ids": list(c.merged_doc_ids),
        "merged_doc_contexts": [list(t) for t in c.merged_doc_contexts],
        "bm25_score": c.bm25_score,
        "cosine_score": c.cosine_score,
        "pinned_by_named_entry": c.pinned_by_named_entry,
        "anchor_boost_applied": c.anchor_boost_applied,
    }


def _candidate_from_dict(d: dict) -> Candidate:
    """Reconstruct a fresh Candidate dataclass from a dict.

    Returns a brand-new object so callers can mutate it without
    affecting the cache. ``merged_doc_contexts`` is restored as a list
    of two-tuples to match the dataclass type.
    """
    return Candidate(
        node_id=d["node_id"],
        node_type=d["node_type"],
        collection=d["collection"],
        rrf_score=float(d.get("rrf_score", 0.0)),
        reranker_score=float(d.get("reranker_score", 0.0)),
        raw_reranker_score=float(d.get("raw_reranker_score", 0.0)),
        file_path=d.get("file_path", ""),
        file_classification=d.get("file_classification"),
        chunk_type=d.get("chunk_type"),
        name=d.get("name", ""),
        text_snippet=d.get("text_snippet", ""),
        internal_logic_abstraction=d.get("internal_logic_abstraction"),
        merged_doc_ids=list(d.get("merged_doc_ids", [])),
        merged_doc_contexts=[
            (t[0], t[1]) for t in d.get("merged_doc_contexts", [])
        ],
        bm25_score=float(d.get("bm25_score", 0.0)),
        cosine_score=float(d.get("cosine_score", 0.0)),
        pinned_by_named_entry=bool(d.get("pinned_by_named_entry", False)),
        anchor_boost_applied=bool(d.get("anchor_boost_applied", False)),
    )


# ----------------------------------------------------------------------
# CISResult / NodeTrace serialisation helpers
# ----------------------------------------------------------------------


def _nodetrace_to_dict(nt: NodeTrace) -> dict:
    return asdict(nt)


def _nodetrace_from_dict(d: dict) -> NodeTrace:
    return NodeTrace(
        depth=int(d["depth"]),
        causal_chain=list(d.get("causal_chain", [])),
        path=list(d.get("path", [])),
        source_seed=d.get("source_seed", ""),
        low_confidence_seed=bool(d.get("low_confidence_seed", False)),
        collapsed_children=list(d.get("collapsed_children", [])),
        justification=d.get("justification", ""),
        justification_source=d.get("justification_source", ""),
        function_purpose=d.get("function_purpose", ""),
        mechanism_of_impact=d.get("mechanism_of_impact", ""),
    )


def _cisresult_to_dict(cis: CISResult) -> dict:
    return {
        "sis_nodes": {
            k: _nodetrace_to_dict(v) for k, v in cis.sis_nodes.items()
        },
        "propagated_nodes": {
            k: _nodetrace_to_dict(v) for k, v in cis.propagated_nodes.items()
        },
    }


def _cisresult_from_dict(d: dict) -> CISResult:
    """Reconstruct a fresh CISResult from a dict.

    Each NodeTrace is rebuilt to avoid aliasing the cached structure.
    """
    return CISResult(
        sis_nodes={
            k: _nodetrace_from_dict(v) for k, v in d.get("sis_nodes", {}).items()
        },
        propagated_nodes={
            k: _nodetrace_from_dict(v)
            for k, v in d.get("propagated_nodes", {}).items()
        },
    )
