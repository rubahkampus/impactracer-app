"""Offline indexer orchestrator.

Invoked by :func:`impactracer.cli.index`. Executes the full offline
pipeline:

1. Scan repository for Markdown and TypeScript/TSX source files.
2. Diff against file_hashes to determine work set (unless ``force``).
3. Chunk Markdown -> :mod:`doc_indexer`.
4. AST Pass 1 -> :mod:`code_indexer` -> skeletonize each node body.
5. AST Pass 2 -> :mod:`code_indexer` -> 13 edge types.
6. Embed all pending texts -> :mod:`embedder`.
7. Traceability precompute -> :mod:`traceability`.
8. Write index_metadata, print summary statistics.

Reference: master_blueprint.md §3.8
"""

from __future__ import annotations

import datetime as dt
import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from loguru import logger

from impactracer.indexer.code_indexer import extract_edges, extract_nodes
from impactracer.indexer.doc_indexer import chunk_markdown
from impactracer.indexer.embedder import Embedder
from impactracer.indexer.traceability import compute_and_store
from impactracer.persistence.chroma_client import get_client, init_collections
from impactracer.persistence.sqlite_client import connect, init_schema
from impactracer.shared.config import Settings


class IndexingStats(TypedDict):
    code_nodes: int
    doc_chunks: int
    edges: int
    files_scanned: int
    files_reindexed: int
    elapsed_seconds: float


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _rel_posix(path: Path) -> str:
    """Return the relative posix path stored in code_nodes.file_path.

    Mirrors the logic in code_indexer.extract_nodes exactly:
    - Find the last 'src' component and take from there.
    - If no 'src' found, fall back to just the filename (same as code_indexer).
    This is always non-None; callers no longer need None checks.
    """
    parts = path.parts
    try:
        src_idx = next(i for i in range(len(parts) - 1, -1, -1) if parts[i] == "src")
        return Path(*parts[src_idx:]).as_posix()
    except StopIteration:
        return path.name


def _scan_repo(repo_path: Path) -> tuple[list[Path], list[Path]]:
    """Return (md_files, ts_files) found under repo_path.

    Excludes files inside hidden directories (any path component starting
    with '.'). This filters .next/, .git/, etc.
    """
    docs_path = repo_path / "docs"
    md_files: list[Path] = []
    ts_files: list[Path] = []
    if docs_path.exists():
        md_files = sorted(
            p for p in docs_path.rglob("*.md")
            if not any(part.startswith(".") for part in p.parts)
        )
    ts_files = sorted(
        p for p in repo_path.rglob("*.ts")
        if not any(part.startswith(".") for part in p.parts)
    ) + sorted(
        p for p in repo_path.rglob("*.tsx")
        if not any(part.startswith(".") for part in p.parts)
    )
    return md_files, ts_files


def _load_file_hashes(conn: sqlite3.Connection) -> dict[str, str]:
    """Return {absolute_posix_path: sha256} from file_hashes."""
    rows = conn.execute("SELECT file_path, content_hash FROM file_hashes").fetchall()
    return {r[0]: r[1] for r in rows}


def _purge_ts_file(
    abs_posix: str,
    conn: sqlite3.Connection,
    code_col: Any,
) -> None:
    """Remove all SQLite and ChromaDB rows for a deleted/changed TS file.

    ``abs_posix`` is the absolute posix path as stored in file_hashes.
    ``code_nodes.file_path`` uses the src/... relative form (or just the
    filename for root-level files without a src/ ancestor).
    Does NOT commit — caller must commit.
    """
    rel = _rel_posix(Path(abs_posix))

    node_ids = [
        r[0]
        for r in conn.execute(
            "SELECT node_id FROM code_nodes WHERE file_path = ?", (rel,)
        ).fetchall()
    ]
    if node_ids:
        ph = ",".join("?" * len(node_ids))
        conn.execute(
            f"DELETE FROM structural_edges "
            f"WHERE source_id IN ({ph}) OR target_id IN ({ph})",
            node_ids + node_ids,
        )
        conn.execute(
            f"DELETE FROM doc_code_candidates WHERE code_id IN ({ph})",
            node_ids,
        )
        conn.execute(
            f"DELETE FROM code_nodes WHERE node_id IN ({ph})",
            node_ids,
        )
        try:
            code_col.delete(ids=node_ids)
        except Exception:
            pass

    conn.execute(
        "DELETE FROM file_dependencies WHERE dependent_file = ? OR target_file = ?",
        (rel, rel),
    )


def _purge_md_file(
    abs_posix: str,
    conn: sqlite3.Connection,
    doc_col: Any,
) -> None:
    """Remove ChromaDB doc_chunks and their candidate rows for a deleted MD file.

    ChromaDB metadata stores source_file as the absolute posix path.
    Does NOT commit — caller must commit.
    """
    try:
        existing = doc_col.get(where={"source_file": {"$eq": abs_posix}})
        doc_ids = existing.get("ids", [])
        if doc_ids:
            ph = ",".join("?" * len(doc_ids))
            conn.execute(
                f"DELETE FROM doc_code_candidates WHERE doc_id IN ({ph})",
                doc_ids,
            )
            doc_col.delete(ids=doc_ids)
    except Exception:
        pass


def _delete_stale_edges(rel_paths: set[str], conn: sqlite3.Connection) -> None:
    """Delete structural_edges and file_dependencies rows for the given relative paths.

    Called before Pass 2 re-extraction so that changed files get clean edges.
    Does NOT commit — caller must commit.
    """
    if not rel_paths:
        return
    ph = ",".join("?" * len(rel_paths))
    stale_node_ids = [
        r[0]
        for r in conn.execute(
            f"SELECT node_id FROM code_nodes WHERE file_path IN ({ph})",
            list(rel_paths),
        ).fetchall()
    ]
    if stale_node_ids:
        ph2 = ",".join("?" * len(stale_node_ids))
        conn.execute(
            f"DELETE FROM structural_edges WHERE source_id IN ({ph2})",
            stale_node_ids,
        )
    conn.execute(
        f"DELETE FROM file_dependencies WHERE dependent_file IN ({ph})",
        list(rel_paths),
    )


def _embed_code_nodes(
    conn: sqlite3.Connection,
    embedder: Embedder,
    code_col: Any,
    min_len: int,
    rel_paths: set[str] | None = None,
) -> None:
    """Embed non-degenerate code_nodes and upsert into ChromaDB.

    If ``rel_paths`` is None, embed ALL non-degenerate nodes (--force path).
    If ``rel_paths`` is a set of src/... relative paths, embed only those files.
    """
    if rel_paths is not None and not rel_paths:
        return

    if rel_paths is None:
        rows = conn.execute(
            "SELECT node_id, embed_text, file_classification, name, node_type "
            "FROM code_nodes WHERE embed_text IS NOT NULL AND length(embed_text) >= ?",
            (min_len,),
        ).fetchall()
    else:
        ph = ",".join("?" * len(rel_paths))
        rows = conn.execute(
            f"SELECT node_id, embed_text, file_classification, name, node_type "
            f"FROM code_nodes "
            f"WHERE embed_text IS NOT NULL AND length(embed_text) >= ? "
            f"AND file_path IN ({ph})",
            [min_len, *list(rel_paths)],
        ).fetchall()

    if not rows:
        return

    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    metas = [
        {
            "file_classification": r[2] or "None",
            "name": r[3],
            "node_type": r[4],
        }
        for r in rows
    ]

    all_vecs: list[np.ndarray] = []
    for start in range(0, len(texts), embedder.batch_size):
        batch = texts[start : start + embedder.batch_size]
        all_vecs.append(embedder.embed_batch(batch))
    vecs = np.concatenate(all_vecs, axis=0)

    code_col.upsert(
        ids=ids,
        embeddings=vecs.tolist(),
        documents=texts,
        metadatas=metas,
    )
    logger.debug("Upserted {} code vectors into ChromaDB", len(ids))


def _upsert_doc_chunks(
    chunks: list[dict],
    embedder: Embedder,
    doc_col: Any,
    min_len: int,
) -> None:
    """Embed and upsert doc chunks into ChromaDB."""
    non_degen = [c for c in chunks if len(c["text"]) >= min_len]
    if not non_degen:
        return

    ids = [c["chunk_id"] for c in non_degen]
    texts = [c["text"] for c in non_degen]
    metas = [
        {
            "chunk_id": c["chunk_id"],
            "source_file": c["source_file"],
            "section_title": c["section_title"],
            "chunk_type": c["chunk_type"],
        }
        for c in non_degen
    ]
    vecs = embedder.embed_batch(texts)
    doc_col.upsert(
        ids=ids,
        embeddings=vecs.tolist(),
        documents=texts,
        metadatas=metas,
    )
    logger.debug("Upserted {} doc chunk vectors into ChromaDB", len(ids))


def run_indexing(
    repo_path: Path,
    settings: Settings,
    force: bool = False,
) -> IndexingStats:
    """Execute the offline indexing pipeline end-to-end.

    Args:
        repo_path: Root of the target repository.
        settings: Runtime configuration.
        force: If True, invalidate all file_hashes and reindex everything.

    Returns:
        Statistics dictionary for CLI reporting.

    Raises:
        FileNotFoundError: If ``repo_path`` does not exist.
    """
    if not repo_path.exists():
        raise FileNotFoundError(f"repo_path does not exist: {repo_path}")

    t_start = time.perf_counter()

    # ── Step 0: connect persistence ─────────────────────────────────────────
    conn = connect(settings.db_path)
    init_schema(conn)
    chroma_client = get_client(settings.chroma_path)
    doc_col, code_col = init_collections(chroma_client)

    # ── Step 1: scan repo ───────────────────────────────────────────────────
    md_files, ts_files = _scan_repo(repo_path)
    all_files = md_files + ts_files
    logger.info(
        "Scanned repo: {} Markdown, {} TypeScript/TSX files",
        len(md_files), len(ts_files),
    )

    # Build lookup: relative path → absolute Path (for Pass 2 re-extraction)
    rel_to_abs: dict[str, Path] = {_rel_posix(p): p for p in ts_files}

    # ── Step 2: hash diff ───────────────────────────────────────────────────
    if force:
        conn.execute("DELETE FROM file_hashes")
        conn.commit()
        logger.info("--force: cleared file_hashes, full reindex")

    # file_hashes stores absolute posix paths as keys.
    known_hashes = _load_file_hashes(conn)
    # current_posix: absolute posix → absolute Path
    current_posix: dict[str, Path] = {p.as_posix(): p for p in all_files}

    deleted_posix = set(known_hashes.keys()) - set(current_posix.keys())
    work_set: list[Path] = []
    for posix, path in current_posix.items():
        h = _sha256(path)
        if known_hashes.get(posix) != h:
            work_set.append(path)

    logger.info(
        "Work set: {} new/changed files, {} deleted files",
        len(work_set), len(deleted_posix),
    )

    # ── Step 3: purge deleted files ─────────────────────────────────────────
    for posix in deleted_posix:
        logger.debug("Purging deleted file: {}", posix)
        if posix.endswith(".md"):
            _purge_md_file(posix, conn, doc_col)
        else:
            _purge_ts_file(posix, conn, code_col)
        conn.execute("DELETE FROM file_hashes WHERE file_path = ?", (posix,))
    if deleted_posix:
        conn.commit()

    # Split work set by type
    work_md = [p for p in work_set if p.suffix == ".md"]
    work_ts = [p for p in work_set if p.suffix in (".ts", ".tsx")]

    # Relative paths for the changed TS files (src/... or just filename)
    work_ts_rel: set[str] = {_rel_posix(p) for p in work_ts}

    # ── Step 4: Markdown chunking ────────────────────────────────────────────
    all_chunks: list[dict] = []
    for md_path in work_md:
        chunks = chunk_markdown(md_path)
        all_chunks.extend(chunks)
        logger.debug("Chunked {} → {} chunks", md_path.name, len(chunks))

    # ── Step 5: AST Pass 1 ───────────────────────────────────────────────────
    if work_ts_rel:
        ph = ",".join("?" * len(work_ts_rel))
        reverse_dep_rows = conn.execute(
            f"SELECT DISTINCT dependent_file FROM file_dependencies "
            f"WHERE target_file IN ({ph})",
            list(work_ts_rel),
        ).fetchall()
        # reverse_dep_posix: src/... relative paths from file_dependencies.dependent_file
        reverse_dep_rel: set[str] = {r[0] for r in reverse_dep_rows} - work_ts_rel
        if reverse_dep_rel:
            logger.info(
                "Reverse-dep expansion: {} additional files need edge re-extraction",
                len(reverse_dep_rel),
            )
    else:
        reverse_dep_rel: set[str] = set()

    # Pass 1: re-extract nodes for changed TS files
    for ts_path in work_ts:
        source_bytes = ts_path.read_bytes()
        extract_nodes(ts_path, source_bytes, conn)
        logger.debug("Pass 1: {}", ts_path.name)

    # ── Step 6: AST Pass 2 ─────────────────────────────────────────────────
    # Collect ALL known node IDs for cross-file resolution AFTER Pass 1.
    known_node_ids: set[str] = {
        r[0] for r in conn.execute("SELECT node_id FROM code_nodes").fetchall()
    }

    # Delete stale edges for the work set AND reverse-dep files before re-extraction.
    # All paths here are src/... relative.
    all_edge_rel = work_ts_rel | reverse_dep_rel
    _delete_stale_edges(all_edge_rel, conn)
    conn.commit()

    # Pass 2: re-extract edges for changed TS files
    for ts_path in work_ts:
        source_bytes = ts_path.read_bytes()
        n = extract_edges(ts_path, source_bytes, known_node_ids, conn)
        logger.debug("Pass 2: {} → {} edges", ts_path.name, n)

    # Pass 2: re-extract edges for reverse-dep files
    for rel in reverse_dep_rel:
        rpath = rel_to_abs.get(rel)
        if rpath is None:
            logger.warning("Reverse-dep file not found on disk: {}", rel)
            continue
        source_bytes = rpath.read_bytes()
        n = extract_edges(rpath, source_bytes, known_node_ids, conn)
        logger.debug("Pass 2 (rev-dep): {} → {} edges", rpath.name, n)

    conn.commit()

    # ── Step 7: embed and upsert ─────────────────────────────────────────────
    logger.info("Loading embedder: {}", settings.embedding_model)
    embedder = Embedder(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        max_length=settings.embedding_max_length,
    )
    min_len = settings.degenerate_embed_min_length

    logger.info("Embedding {} doc chunks...", len(all_chunks))
    _upsert_doc_chunks(all_chunks, embedder, doc_col, min_len)

    # Embed: on --force embed everything; on incremental embed only changed files.
    embed_rel = None if force else work_ts_rel
    logger.info(
        "Embedding code nodes ({})...",
        "all" if embed_rel is None else f"{len(embed_rel)} changed files",
    )
    _embed_code_nodes(conn, embedder, code_col, min_len, rel_paths=embed_rel)

    # ── Step 8: traceability (full recompute) ────────────────────────────────
    # Blueprint §3.8 step 8: "full recompute is correct; layer-weighted scores
    # depend on the population." Fetch ALL vecs from ChromaDB.
    all_code = code_col.get(include=["embeddings", "metadatas"])
    all_doc = doc_col.get(include=["embeddings", "metadatas"])

    full_code_vecs: dict[str, np.ndarray] = {}
    full_code_meta: dict[str, dict] = {}
    for cid, emb, meta in zip(
        all_code["ids"], all_code["embeddings"], all_code["metadatas"]
    ):
        full_code_vecs[cid] = np.array(emb, dtype=np.float32)
        full_code_meta[cid] = meta

    full_doc_vecs: dict[str, np.ndarray] = {}
    full_doc_meta: dict[str, dict] = {}
    for did, emb, meta in zip(
        all_doc["ids"], all_doc["embeddings"], all_doc["metadatas"]
    ):
        full_doc_vecs[did] = np.array(emb, dtype=np.float32)
        full_doc_meta[did] = meta

    logger.info(
        "Traceability: {} code vecs × {} doc vecs",
        len(full_code_vecs), len(full_doc_vecs),
    )
    pairs_stored = compute_and_store(
        code_vecs=full_code_vecs,
        doc_vecs=full_doc_vecs,
        code_meta=full_code_meta,
        doc_meta=full_doc_meta,
        top_k=settings.top_k_traceability,
        min_similarity=settings.min_traceability_similarity,
        conn=conn,
    )
    logger.info("Traceability pairs stored: {}", pairs_stored)

    # ── Step 9: update file_hashes ────────────────────────────────────────────
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    for path in work_set:
        posix = path.as_posix()
        h = _sha256(path)
        conn.execute(
            "INSERT OR REPLACE INTO file_hashes (file_path, content_hash, indexed_at) "
            "VALUES (?, ?, ?)",
            (posix, h, now),
        )
    conn.commit()

    # ── Step 10: index_metadata ──────────────────────────────────────────────
    total_code_nodes = conn.execute("SELECT COUNT(*) FROM code_nodes").fetchone()[0]
    total_edges = conn.execute("SELECT COUNT(*) FROM structural_edges").fetchone()[0]
    total_doc_chunks = doc_col.count()
    total_candidates = conn.execute(
        "SELECT COUNT(*) FROM doc_code_candidates"
    ).fetchone()[0]

    elapsed = time.perf_counter() - t_start

    metadata: dict[str, str] = {
        "edge_schema_version": "4.0",
        "embedding_model_name": settings.embedding_model,
        "traceability_k_parameter": str(settings.top_k_traceability),
        "min_traceability_similarity": str(settings.min_traceability_similarity),
        "indexing_timestamp": now,
        "total_code_nodes": str(total_code_nodes),
        "total_doc_chunks": str(total_doc_chunks),
        "total_structural_edges": str(total_edges),
        "total_doc_code_candidates": str(total_candidates),
        "skeletonization_enabled": "true",
        "incremental_indexing_enabled": "true",
    }
    conn.executemany(
        "INSERT OR REPLACE INTO index_metadata (key, value) VALUES (?, ?)",
        metadata.items(),
    )
    conn.commit()
    conn.close()

    stats: IndexingStats = {
        "code_nodes": total_code_nodes,
        "doc_chunks": total_doc_chunks,
        "edges": total_edges,
        "files_scanned": len(all_files),
        "files_reindexed": len(work_set),
        "elapsed_seconds": round(elapsed, 1),
    }
    return stats
