"""Integration tests for the offline indexing pipeline (Sprint 7).

Tests run against:
1. A small synthetic fixture repo (fast, no GPU needed — uses a stub embedder).
2. The real citrakara repo (slow, GPU embedder required; skipped if not present).

Reference: master_blueprint.md §3.8.
"""

from __future__ import annotations

import shutil
import sqlite3
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from impactracer.indexer.runner import run_indexing
from impactracer.shared.config import Settings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CITRAKARA_PATH = Path(r"C:\Users\Haidar\Documents\thesis\citrakara")


def _make_settings(tmp_path: Path) -> Settings:
    """Settings pointing at tmp storage; stub embedding model name."""
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        db_path=str(tmp_path / "impactracer.db"),
        chroma_path=str(tmp_path / "chroma_store"),
        embedding_model="BAAI/bge-m3",
        embedding_batch_size=4,
        embedding_max_length=64,
        top_k_traceability=3,
        min_traceability_similarity=0.0,  # keep all pairs in fixture tests
        degenerate_embed_min_length=5,
    )


def _make_fixture_repo(base: Path) -> Path:
    """Create a minimal synthetic repo with docs/ and src/."""
    repo = base / "fixture_repo"
    docs = repo / "docs"
    docs.mkdir(parents=True)
    src_app_api = repo / "src" / "app" / "api" / "items"
    src_app_api.mkdir(parents=True)
    src_lib = repo / "src" / "lib"
    src_lib.mkdir(parents=True)

    # Markdown doc
    (docs / "srs.md").write_text(textwrap.dedent("""\
        # SRS

        ## Kebutuhan Fungsional Item

        Users can create and view items.

        ## Perancangan Sistem

        The system uses a REST API.
    """), encoding="utf-8")

    # TypeScript API route
    (src_app_api / "route.ts").write_text(textwrap.dedent("""\
        import { createItem } from '../../../lib/item.service';

        export async function POST(req: Request) {
          const body = await req.json();
          return createItem(body);
        }
    """), encoding="utf-8")

    # TypeScript service
    (src_lib / "item.service.ts").write_text(textwrap.dedent("""\
        export function createItem(data: any) {
          return { id: 1, ...data };
        }
    """), encoding="utf-8")

    return repo


def _stub_embedder(model_name: str, batch_size: int = 32, max_length: int = 512):
    """Return a mock Embedder that yields deterministic random vectors."""
    from impactracer.indexer.embedder import Embedder

    rng = np.random.RandomState(42)

    mock = MagicMock(spec=Embedder)
    mock.batch_size = batch_size
    mock.max_length = max_length

    def embed_batch(texts):
        return rng.rand(len(texts), 1024).astype(np.float32)

    mock.embed_batch.side_effect = embed_batch
    mock.embed_single.side_effect = lambda t: embed_batch([t])[0].tolist()
    return mock


# ---------------------------------------------------------------------------
# Helper: patch Embedder construction in runner module
# ---------------------------------------------------------------------------

def _patch_embedder():
    return patch(
        "impactracer.indexer.runner.Embedder",
        side_effect=_stub_embedder,
    )


# ---------------------------------------------------------------------------
# Test 1: Initial run creates DB and ChromaDB
# ---------------------------------------------------------------------------

def test_initial_run_creates_stores(tmp_path):
    repo = _make_fixture_repo(tmp_path)
    settings = _make_settings(tmp_path)

    with _patch_embedder():
        stats = run_indexing(repo, settings, force=False)

    assert Path(settings.db_path).exists(), "SQLite DB not created"
    assert Path(settings.chroma_path).exists(), "ChromaDB dir not created"
    assert stats["files_scanned"] > 0
    assert stats["code_nodes"] > 0
    assert stats["doc_chunks"] > 0
    assert stats["edges"] > 0
    assert stats["elapsed_seconds"] > 0


# ---------------------------------------------------------------------------
# Test 2: Re-run without --force skips unchanged files
# ---------------------------------------------------------------------------

def test_incremental_skip_unchanged(tmp_path):
    repo = _make_fixture_repo(tmp_path)
    settings = _make_settings(tmp_path)

    with _patch_embedder():
        stats1 = run_indexing(repo, settings, force=False)

    # Second run — no files changed
    with _patch_embedder():
        stats2 = run_indexing(repo, settings, force=False)

    assert stats2["files_reindexed"] == 0, (
        f"Expected 0 reindexed on unchanged repo, got {stats2['files_reindexed']}"
    )
    # Totals stable
    assert stats2["code_nodes"] == stats1["code_nodes"]
    assert stats2["doc_chunks"] == stats1["doc_chunks"]


# ---------------------------------------------------------------------------
# Test 3: --force reindexes everything
# ---------------------------------------------------------------------------

def test_force_reindexes_all(tmp_path):
    repo = _make_fixture_repo(tmp_path)
    settings = _make_settings(tmp_path)

    with _patch_embedder():
        stats1 = run_indexing(repo, settings, force=False)

    all_files = stats1["files_scanned"]

    with _patch_embedder():
        stats2 = run_indexing(repo, settings, force=True)

    assert stats2["files_reindexed"] == all_files, (
        f"--force should reindex all {all_files} files, got {stats2['files_reindexed']}"
    )


# ---------------------------------------------------------------------------
# Test 4: Deleted file purges its rows
# ---------------------------------------------------------------------------

def test_deleted_file_purged(tmp_path):
    repo = _make_fixture_repo(tmp_path)
    settings = _make_settings(tmp_path)

    with _patch_embedder():
        run_indexing(repo, settings, force=False)

    # Delete the service file
    service_file = repo / "src" / "lib" / "item.service.ts"
    service_file.unlink()

    with _patch_embedder():
        run_indexing(repo, settings, force=False)

    conn = sqlite3.connect(settings.db_path)
    rows = conn.execute(
        "SELECT COUNT(*) FROM code_nodes WHERE file_path LIKE '%item.service%'"
    ).fetchone()[0]
    conn.close()
    assert rows == 0, f"Expected 0 rows for deleted file, got {rows}"


# ---------------------------------------------------------------------------
# Test 5: Modified file triggers re-extraction of its edges
# ---------------------------------------------------------------------------

def test_modified_file_reindexed(tmp_path):
    repo = _make_fixture_repo(tmp_path)
    settings = _make_settings(tmp_path)

    with _patch_embedder():
        run_indexing(repo, settings, force=False)

    # Modify the service file
    service_file = repo / "src" / "lib" / "item.service.ts"
    original = service_file.read_text(encoding="utf-8")
    service_file.write_text(
        original + "\nexport function deleteItem(id: number) { return true; }\n",
        encoding="utf-8",
    )

    with _patch_embedder():
        stats2 = run_indexing(repo, settings, force=False)

    # At least the modified file was reindexed
    assert stats2["files_reindexed"] >= 1


# ---------------------------------------------------------------------------
# Test 6: index_metadata has all required keys
# ---------------------------------------------------------------------------

def test_index_metadata_keys(tmp_path):
    repo = _make_fixture_repo(tmp_path)
    settings = _make_settings(tmp_path)

    with _patch_embedder():
        run_indexing(repo, settings, force=False)

    conn = sqlite3.connect(settings.db_path)
    rows = {r[0]: r[1] for r in conn.execute("SELECT key, value FROM index_metadata").fetchall()}
    conn.close()

    required = {
        "edge_schema_version",
        "embedding_model_name",
        "traceability_k_parameter",
        "min_traceability_similarity",
        "indexing_timestamp",
        "total_code_nodes",
        "total_doc_chunks",
        "total_structural_edges",
        "skeletonization_enabled",
        "incremental_indexing_enabled",
    }
    missing = required - rows.keys()
    assert not missing, f"index_metadata missing keys: {missing}"
    assert rows["edge_schema_version"] == "4.0"
    assert rows["skeletonization_enabled"] == "true"
    assert rows["incremental_indexing_enabled"] == "true"


# ---------------------------------------------------------------------------
# Test 7: citrakara full run (slow, requires real GPU + model)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not CITRAKARA_PATH.exists(),
    reason="citrakara repo not present",
)
def test_citrakara_full_run(tmp_path):
    """Run the full pipeline against the real citrakara repo."""
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        db_path=str(tmp_path / "impactracer.db"),
        chroma_path=str(tmp_path / "chroma_store"),
        embedding_model="BAAI/bge-m3",
        embedding_batch_size=32,
        embedding_max_length=512,
        top_k_traceability=5,
        min_traceability_similarity=0.40,
        degenerate_embed_min_length=50,
    )

    stats = run_indexing(CITRAKARA_PATH, settings, force=True)

    assert stats["code_nodes"] > 1000, f"Expected >1000 code_nodes, got {stats['code_nodes']}"
    assert stats["doc_chunks"] > 50, f"Expected >50 doc_chunks, got {stats['doc_chunks']}"
    assert stats["edges"] > 500, f"Expected >500 edges, got {stats['edges']}"
    assert stats["elapsed_seconds"] < 600, "Indexing took >10 minutes — unexpected"

    conn = sqlite3.connect(settings.db_path)
    metadata = {r[0]: r[1] for r in conn.execute("SELECT key, value FROM index_metadata").fetchall()}
    conn.close()

    assert metadata["edge_schema_version"] == "4.0"
    print("\n[citrakara] Stats:", stats)
    print("[citrakara] Metadata:", metadata)
