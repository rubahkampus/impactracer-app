"""Dual-path hybrid search with Adaptive RRF (FR-C1, FR-C2).

Four ranked lists per query: dense-doc, bm25-doc, dense-code, bm25-code.
Fused via Adaptive RRF where path weights depend on ``change_type``.

Reference: master_blueprint.md §4 Step 2.
"""

from __future__ import annotations

import re
import sqlite3

from loguru import logger
from rank_bm25 import BM25Okapi

from impactracer.shared.constants import RRF_PATH_WEIGHTS
from impactracer.shared.models import Candidate, CRInterpretation


# ---------------------------------------------------------------------------
# BM25 tokenizer (ED-1: camelCase-aware, replaces naive whitespace split)
# ---------------------------------------------------------------------------


def _tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text for BM25 with camelCase decomposition.

    Three transformations (stdlib only, no new dependencies):
    1. camelCase / PascalCase split: commissionListingId → commission Listing Id
    2. Lowercase + split on non-alphanumeric (drops operators, punctuation)
    3. Filter tokens shorter than 2 chars and pure-numeric tokens (noise)

    Handles TypeScript identifiers, Indonesian words, and mixed corpus text.
    """
    # Split camelCase and PascalCase boundaries
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    # Lowercase and split on non-alphanumeric characters
    tokens = re.split(r"[^a-z0-9]+", text.lower())
    # Phase 2.7 (E-NEW-2): raised minimum token length from 2 → 3.
    # 2-char tokens admit high-frequency English stop-words ("to", "up",
    # "in", "of", "is", "at", "by", "be") that consume BM25 posting-list
    # budget without adding discriminative power. Indonesian 2-char tokens
    # ("di", "ke", "ya") are similarly non-discriminative. 3-char minimum
    # eliminates these while preserving meaningful short identifiers
    # (e.g. "get", "set", "api", "url", "jwt").
    return [t for t in tokens if len(t) >= 3 and not t.isdigit()]


# ---------------------------------------------------------------------------
# BM25 index builder
# ---------------------------------------------------------------------------


def build_bm25_from_chroma(collection: object) -> tuple[BM25Okapi, list[str]]:
    """Construct a BM25Okapi index from all documents in a ChromaDB collection.

    Returns ``(bm25, ids)`` where ``ids[i]`` corresponds to the i-th BM25 document.

    Raises RuntimeError if the collection is empty (ED-5).
    """
    result = collection.get(include=["documents", "metadatas"])
    ids: list[str] = result["ids"]
    documents: list[str] = result["documents"]

    if not documents:
        raise RuntimeError(
            f"ChromaDB collection '{collection.name}' is empty. "
            "Run `impactracer index <repo_path>` first."
        )

    tokenized = [_tokenize_for_bm25(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized)
    logger.debug("BM25 built: {} documents in collection '{}'", len(ids), collection.name)
    return bm25, ids


def build_metadata_cache(collection: object) -> dict[str, dict]:
    """Build a {node_id: metadata_dict} cache from a ChromaDB collection.

    Called once at context load time to avoid N+1 per-document metadata
    queries during BM25 chunk_type filtering (ED-3).
    """
    result = collection.get(include=["metadatas"])
    return {cid: meta for cid, meta in zip(result["ids"], result["metadatas"])}


# ---------------------------------------------------------------------------
# Adaptive RRF fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion_adaptive(
    ranked_lists: list[tuple[str, list[str]]],
    change_type: str,
    k: int = 60,
) -> dict[str, float]:
    """Weighted RRF fusion.

    Each entry is ``(path_label, ranked_ids)``.
    Weights come from :data:`impactracer.shared.constants.RRF_PATH_WEIGHTS`.

    Blueprint §4 Step 2:
        ARRF(d) = Σ_{p ∈ paths_present} W[change_type][p] / (rrf_k + rank_p(d) + 1)
    """
    weights = RRF_PATH_WEIGHTS.get(change_type, RRF_PATH_WEIGHTS["MODIFICATION"])
    scores: dict[str, float] = {}
    for path_label, ranked_ids in ranked_lists:
        w = weights.get(path_label, 1.0)
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank + 1)
    return scores


# ---------------------------------------------------------------------------
# Doc filter derivation
# ---------------------------------------------------------------------------


def _doc_filter_from_layers(affected_layers: list[str]) -> list[str]:
    """Map affected_layers to chunk_type values for ChromaDB $in filter.

    Blueprint §4 Step 2:
      "requirement" -> ["FR", "NFR"]
      "design"      -> ["Design"]
      "code"        -> ["General"]  (N5: General chunks describe process flows
                                    that code implements; dense search was
                                    silently excluding all 13 General chunks)

    When all three layers are present (the common case), all four chunk types
    are included, giving dense doc search full coverage.
    """
    types: list[str] = []
    for layer in affected_layers:
        if layer == "requirement":
            types.extend(["FR", "NFR"])
        elif layer == "design":
            types.append("Design")
        elif layer == "code":
            types.append("General")
    return list(dict.fromkeys(types))  # deduplicate, preserve order


# ---------------------------------------------------------------------------
# Candidate hydration from SQLite
# ---------------------------------------------------------------------------


def _hydrate_code_candidates(
    ids: list[str],
    scores: dict[str, float],
    cosine_scores: dict[str, float],
    bm25_scores: dict[str, float],
    conn: sqlite3.Connection,
    collection: object,
) -> list[Candidate]:
    """Build Candidate objects for code_units collection entries."""
    if not ids:
        return []

    # Fetch metadata from ChromaDB
    chroma_result = collection.get(ids=ids, include=["documents", "metadatas"])
    chroma_meta_map: dict[str, dict] = {}
    for cid, meta, doc in zip(
        chroma_result["ids"],
        chroma_result["metadatas"],
        chroma_result["documents"],
    ):
        chroma_meta_map[cid] = {"meta": meta, "doc": doc}

    # Fetch SQLite columns
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT node_id, node_type, file_path, file_classification, internal_logic_abstraction "
        f"FROM code_nodes WHERE node_id IN ({placeholders})",
        ids,
    ).fetchall()
    sql_map = {r[0]: r for r in rows}

    candidates: list[Candidate] = []
    for cid in ids:
        cm = chroma_meta_map.get(cid, {})
        meta = cm.get("meta", {})
        doc_text = cm.get("doc", "")
        sql = sql_map.get(cid)
        node_type = sql[1] if sql else meta.get("node_type", "Function")
        file_path = sql[2] if sql else meta.get("file_path", "")
        file_cls = sql[3] if sql else meta.get("file_classification")
        ila = sql[4] if sql else None
        name = cid.split("::")[-1] if "::" in cid else cid

        candidates.append(Candidate(
            node_id=cid,
            node_type=node_type,
            collection="code_units",
            rrf_score=scores.get(cid, 0.0),
            cosine_score=cosine_scores.get(cid, 0.0),
            bm25_score=bm25_scores.get(cid, 0.0),
            file_path=file_path,
            file_classification=file_cls,
            name=name,
            text_snippet=doc_text,   # full embed_text; context_builder caps at budget
            internal_logic_abstraction=ila,
        ))
    return candidates


def _hydrate_doc_candidates(
    ids: list[str],
    scores: dict[str, float],
    cosine_scores: dict[str, float],
    bm25_scores: dict[str, float],
    collection: object,
) -> list[Candidate]:
    """Build Candidate objects for doc_chunks collection entries.

    B3/FF-5: Populate file_path from ChromaDB metadata source_file so the
    density gate in Step 3.7 can distinguish srs.md from sdd.md correctly,
    and so the validator prompt shows a meaningful source file for doc chunks.
    """
    if not ids:
        return []

    chroma_result = collection.get(ids=ids, include=["documents", "metadatas"])
    candidates: list[Candidate] = []
    for cid, meta, doc in zip(
        chroma_result["ids"],
        chroma_result["metadatas"],
        chroma_result["documents"],
    ):
        candidates.append(Candidate(
            node_id=cid,
            node_type="DocChunk",
            collection="doc_chunks",
            rrf_score=scores.get(cid, 0.0),
            cosine_score=cosine_scores.get(cid, 0.0),
            bm25_score=bm25_scores.get(cid, 0.0),
            chunk_type=meta.get("chunk_type"),
            name=meta.get("section_title", cid),
            # B3/FF-5: populate from metadata — was always "" before, causing all
            # doc chunks to share file_path="" and be incorrectly density-gated.
            file_path=meta.get("source_file", ""),
            text_snippet=doc,   # full doc text — no truncation here; context_builder caps
        ))
    return candidates


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def hybrid_search(
    cr_interp: CRInterpretation,
    ctx: object,
    settings: object,
) -> list[Candidate]:
    """Execute dual-path search and return RRF-sorted candidates (FR-C1, FR-C2).

    Assembles up to four ranked lists depending on variant flags, fuses via
    Adaptive RRF, and returns the top-K RRF pool (settings.top_k_rrf_pool)
    for downstream reranking. The reranker then selects up to
    settings.max_admitted_seeds from this pool (FF-1: 50→15 funnel).
    """
    flags = ctx.variant_flags
    top_k = settings.top_k_per_query
    top_k_rrf_pool = settings.top_k_rrf_pool
    rrf_k = settings.rrf_k

    doc_filter = _doc_filter_from_layers(cr_interp.affected_layers)
    has_code_layer = "code" in cr_interp.affected_layers

    logger.info(
        "[retriever] affected_layers={} doc_filter={} has_code_layer={}",
        cr_interp.affected_layers, doc_filter, has_code_layer,
    )

    ranked_lists: list[tuple[str, list[str]]] = []
    cosine_scores: dict[str, float] = {}
    bm25_scores_map: dict[str, float] = {}

    queries = cr_interp.search_queries
    logger.info("[retriever] search_queries={}", queries)

    # FF-4: initialize unconditionally — prevents NameError if enable_dense=False
    # and a future code path references query_vecs outside the dense guard.
    query_vecs: list = []
    if flags.enable_dense:
        query_vecs = [ctx.embedder.embed_single(q) for q in queries]

    # -------------------------------------------------------------------
    # Path 1: Dense doc
    # -------------------------------------------------------------------
    dense_doc_ids: list[str] = []
    if flags.enable_dense and doc_filter:
        seen_dd: dict[str, float] = {}
        chroma_where = {"chunk_type": {"$in": doc_filter}}
        for qvec in query_vecs:
            try:
                res = ctx.doc_col.query(
                    query_embeddings=[qvec],
                    n_results=top_k,
                    where=chroma_where,
                    include=["distances"],
                )
            except Exception:
                res = {"ids": [[]], "distances": [[]]}
            for cid, dist in zip(res["ids"][0], res["distances"][0]):
                cos = 1.0 - dist
                if cid not in seen_dd or seen_dd[cid] < cos:
                    seen_dd[cid] = cos
        dense_doc_ids = sorted(seen_dd, key=seen_dd.__getitem__, reverse=True)[:top_k]
        for cid in dense_doc_ids:
            cosine_scores[cid] = max(cosine_scores.get(cid, 0.0), seen_dd[cid])
        logger.debug("[retriever] dense_doc: {} candidates", len(dense_doc_ids))

    # -------------------------------------------------------------------
    # Path 2: BM25 doc (ED-3: use pre-cached metadata, no N+1 queries)
    # -------------------------------------------------------------------
    bm25_doc_ids: list[str] = []
    if flags.enable_bm25 and ctx.doc_bm25_ids:
        seen_bd: dict[str, float] = {}
        doc_meta_cache: dict[str, dict] = getattr(ctx, "doc_meta_cache", {})
        for q in queries:
            tokens = _tokenize_for_bm25(q)
            raw_scores = ctx.doc_bm25.get_scores(tokens)
            for i, score in enumerate(raw_scores):
                if score <= 0:
                    continue
                cid = ctx.doc_bm25_ids[i]
                # ED-3: use cached metadata instead of per-doc ChromaDB get()
                if doc_filter:
                    chunk_type = doc_meta_cache.get(cid, {}).get("chunk_type", "")
                    if chunk_type not in doc_filter:
                        continue
                if cid not in seen_bd or seen_bd[cid] < score:
                    seen_bd[cid] = score
        bm25_doc_ids = sorted(seen_bd, key=seen_bd.__getitem__, reverse=True)[:top_k]
        for cid in bm25_doc_ids:
            bm25_scores_map[cid] = max(bm25_scores_map.get(cid, 0.0), seen_bd[cid])
        logger.debug("[retriever] bm25_doc: {} candidates", len(bm25_doc_ids))

    # -------------------------------------------------------------------
    # Path 3: Dense code
    # -------------------------------------------------------------------
    dense_code_ids: list[str] = []
    if flags.enable_dense and has_code_layer:
        seen_dc: dict[str, float] = {}
        for qvec in query_vecs:
            try:
                res = ctx.code_col.query(
                    query_embeddings=[qvec],
                    n_results=top_k,
                    include=["distances"],
                )
            except Exception:
                res = {"ids": [[]], "distances": [[]]}
            for cid, dist in zip(res["ids"][0], res["distances"][0]):
                cos = 1.0 - dist
                if cid not in seen_dc or seen_dc[cid] < cos:
                    seen_dc[cid] = cos
        dense_code_ids = sorted(seen_dc, key=seen_dc.__getitem__, reverse=True)[:top_k]
        for cid in dense_code_ids:
            cosine_scores[cid] = max(cosine_scores.get(cid, 0.0), seen_dc[cid])
        logger.debug("[retriever] dense_code: {} candidates", len(dense_code_ids))

    # -------------------------------------------------------------------
    # Path 4: BM25 code (ED-1: camelCase tokenizer applied to queries)
    # -------------------------------------------------------------------
    bm25_code_ids: list[str] = []
    if flags.enable_bm25 and has_code_layer and ctx.code_bm25_ids:
        seen_bc: dict[str, float] = {}
        for q in queries:
            tokens = _tokenize_for_bm25(q)
            raw_scores = ctx.code_bm25.get_scores(tokens)
            for i, score in enumerate(raw_scores):
                if score <= 0:
                    continue
                cid = ctx.code_bm25_ids[i]
                if cid not in seen_bc or seen_bc[cid] < score:
                    seen_bc[cid] = score
        bm25_code_ids = sorted(seen_bc, key=seen_bc.__getitem__, reverse=True)[:top_k]
        for cid in bm25_code_ids:
            bm25_scores_map[cid] = max(bm25_scores_map.get(cid, 0.0), seen_bc[cid])
        logger.debug("[retriever] bm25_code: {} candidates", len(bm25_code_ids))

    # -------------------------------------------------------------------
    # Assemble ranked lists for RRF
    # -------------------------------------------------------------------
    if dense_doc_ids:
        ranked_lists.append(("dense_doc", dense_doc_ids))
    if bm25_doc_ids:
        ranked_lists.append(("bm25_doc", bm25_doc_ids))
    if dense_code_ids:
        ranked_lists.append(("dense_code", dense_code_ids))
    if bm25_code_ids:
        ranked_lists.append(("bm25_code", bm25_code_ids))

    if not ranked_lists:
        logger.warning("[retriever] No ranked lists assembled — returning empty candidates")
        return []

    # -------------------------------------------------------------------
    # RRF fusion (or pass-through when only one list)
    # -------------------------------------------------------------------
    if flags.enable_rrf and len(ranked_lists) > 1:
        scores = reciprocal_rank_fusion_adaptive(
            ranked_lists, cr_interp.change_type, k=rrf_k
        )
    else:
        scores = {}
        for _label, ids_list in ranked_lists:
            for rank, cid in enumerate(ids_list):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

    # Top-K RRF pool (FF-1: top_k_rrf_pool=50, feeds cross-encoder which
    # selects max_admitted_seeds=15 from this broader pool)
    top_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k_rrf_pool]

    # AV-3: structured retrieval summary log for ablation variance verification
    logger.info(
        "[retriever] retrieval_summary variant={} rrf_pool_size={} "
        "dense_doc={} bm25_doc={} dense_code={} bm25_code={}",
        flags.variant_id,
        len(top_ids),
        len(dense_doc_ids),
        len(bm25_doc_ids),
        len(dense_code_ids),
        len(bm25_code_ids),
    )

    # -------------------------------------------------------------------
    # Identify which IDs belong to doc_chunks vs code_units
    # Nodes retrieved via dense_doc/bm25_doc → doc collection.
    # Nodes retrieved via dense_code/bm25_code → code collection.
    # If a node appears in both (edge case), code attribution wins.
    # -------------------------------------------------------------------
    doc_id_set = set(dense_doc_ids) | set(bm25_doc_ids)
    code_id_set = set(dense_code_ids) | set(bm25_code_ids)
    top_doc_ids = [cid for cid in top_ids if cid in doc_id_set and cid not in code_id_set]
    top_code_ids = [cid for cid in top_ids if cid in code_id_set]

    # -------------------------------------------------------------------
    # Hydrate candidates
    # -------------------------------------------------------------------
    doc_candidates = _hydrate_doc_candidates(
        top_doc_ids, scores, cosine_scores, bm25_scores_map, ctx.doc_col
    )
    code_candidates = _hydrate_code_candidates(
        top_code_ids, scores, cosine_scores, bm25_scores_map, ctx.conn, ctx.code_col
    )

    all_candidates = doc_candidates + code_candidates
    all_candidates.sort(key=lambda c: c.rrf_score, reverse=True)

    logger.info(
        "[retriever] post-RRF pool: {} candidates ({} doc, {} code)",
        len(all_candidates), len(doc_candidates), len(code_candidates),
    )
    return all_candidates
