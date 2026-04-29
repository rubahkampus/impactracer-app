"""Acceptance tests for Sprint 3 — doc_indexer.py (FR-A1, FR-A2).

Blueprint reference: master_blueprint.md §3.1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from impactracer.indexer.doc_indexer import DocChunk, chunk_markdown, classify_chunk


# ---------------------------------------------------------------------------
# classify_chunk (FR-A2)
# ---------------------------------------------------------------------------

def test_classify_fr_kebutuhan_fungsional():
    assert classify_chunk("II. Kebutuhan Fungsional") == "FR"

def test_classify_fr_functional_requirement():
    assert classify_chunk("Functional Requirements Overview") == "FR"

def test_classify_fr_use_case():
    assert classify_chunk("IV. Use Case") == "FR"

def test_classify_nfr_non_fungsional():
    assert classify_chunk("III. Kebutuhan Non-Fungsional") == "NFR"

def test_classify_nfr_non_functional():
    assert classify_chunk("Non-Functional Requirements") == "NFR"

def test_classify_design_perancangan():
    assert classify_chunk("II. Perancangan Arsitektur") == "Design"

def test_classify_design_desain():
    assert classify_chunk("Desain Sistem") == "Design"

def test_classify_design_arsitektur():
    assert classify_chunk("Arsitektur Komponen") == "Design"

def test_classify_design_architecture():
    assert classify_chunk("System Architecture") == "Design"

def test_classify_general_pendahuluan():
    assert classify_chunk("I. Pendahuluan") == "General"

def test_classify_general_deskripsi_proses():
    assert classify_chunk("V. Deskripsi Proses Bisnis") == "General"

def test_classify_case_insensitive():
    assert classify_chunk("KEBUTUHAN FUNGSIONAL") == "FR"
    assert classify_chunk("NON-FUNGSIONAL") == "NFR"
    assert classify_chunk("PERANCANGAN") == "Design"


# ---------------------------------------------------------------------------
# chunk_markdown helpers — use tmp_path fixture for real file I/O
# ---------------------------------------------------------------------------

SAMPLE_MD = """\
# Document Title

Intro text before any H2 (should NOT become its own chunk).

## Section One

Content of section one.

### Sub Section A

Sub content A.

#### Deep Level

Deep content absorbs into Sub Section A chunk.

### Sub Section B

Sub content B.

## Section Two

Content of section two.
"""


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# FR-A1: Boundary splitting
# ---------------------------------------------------------------------------

def test_h1_is_not_a_boundary(tmp_path):
    """H1 must not produce a chunk boundary."""
    p = _write(tmp_path, "sample.md", SAMPLE_MD)
    chunks = chunk_markdown(p)
    # H1 "Document Title" must not appear as a section_title
    titles = [c["section_title"] for c in chunks]
    assert "Document Title" not in titles


def test_h2_creates_boundary(tmp_path):
    p = _write(tmp_path, "sample.md", SAMPLE_MD)
    chunks = chunk_markdown(p)
    titles = [c["section_title"] for c in chunks]
    assert "Section One" in titles
    assert "Section Two" in titles


def test_h3_creates_boundary(tmp_path):
    p = _write(tmp_path, "sample.md", SAMPLE_MD)
    chunks = chunk_markdown(p)
    titles = [c["section_title"] for c in chunks]
    assert "Sub Section A" in titles
    assert "Sub Section B" in titles


def test_h4_absorbs_into_enclosing_chunk(tmp_path):
    p = _write(tmp_path, "sample.md", SAMPLE_MD)
    chunks = chunk_markdown(p)
    titles = [c["section_title"] for c in chunks]
    assert "Deep Level" not in titles
    # But the H4 content must appear in the enclosing Sub Section A chunk.
    sub_a = next(c for c in chunks if c["section_title"] == "Sub Section A")
    assert "Deep content absorbs" in sub_a["text"]


def test_chunk_count(tmp_path):
    """SAMPLE_MD has 2 H2s and 2 H3s → 4 chunks total."""
    p = _write(tmp_path, "sample.md", SAMPLE_MD)
    chunks = chunk_markdown(p)
    assert len(chunks) == 4


# ---------------------------------------------------------------------------
# FR-A1: Chunk ID determinism and format
# ---------------------------------------------------------------------------

def test_chunk_id_format(tmp_path):
    """{file_stem}__{slugified_title}, lowercase, non-alnum → _."""
    p = _write(tmp_path, "sample.md", SAMPLE_MD)
    chunks = chunk_markdown(p)
    ids = {c["chunk_id"] for c in chunks}
    assert "sample__section_one" in ids
    assert "sample__section_two" in ids
    assert "sample__sub_section_a" in ids
    assert "sample__sub_section_b" in ids


def test_chunk_id_deterministic(tmp_path):
    """Running twice on the same file produces identical chunk_ids."""
    p = _write(tmp_path, "sample.md", SAMPLE_MD)
    ids_first = [c["chunk_id"] for c in chunk_markdown(p)]
    ids_second = [c["chunk_id"] for c in chunk_markdown(p)]
    assert ids_first == ids_second


def test_source_file_uses_posix_path(tmp_path):
    p = _write(tmp_path, "sample.md", SAMPLE_MD)
    chunks = chunk_markdown(p)
    for c in chunks:
        assert "\\" not in c["source_file"], "source_file must use forward slashes"


# ---------------------------------------------------------------------------
# FR-A2: Classification correctness on chunked output
# ---------------------------------------------------------------------------

CLASSIFIED_MD = """\
# Doc

## Kebutuhan Fungsional

FR content.

## Kebutuhan Non-Fungsional

NFR content.

## Perancangan Arsitektur

Design content.

## Pendahuluan

General content.
"""


def test_classification_fr_chunk(tmp_path):
    p = _write(tmp_path, "doc.md", CLASSIFIED_MD)
    chunks = chunk_markdown(p)
    fr_chunk = next(c for c in chunks if "Fungsional" in c["section_title"] and "Non" not in c["section_title"])
    assert fr_chunk["chunk_type"] == "FR"


def test_classification_nfr_chunk(tmp_path):
    p = _write(tmp_path, "doc.md", CLASSIFIED_MD)
    chunks = chunk_markdown(p)
    nfr_chunk = next(c for c in chunks if "Non-Fungsional" in c["section_title"])
    assert nfr_chunk["chunk_type"] == "NFR"


def test_classification_design_chunk(tmp_path):
    p = _write(tmp_path, "doc.md", CLASSIFIED_MD)
    chunks = chunk_markdown(p)
    design_chunk = next(c for c in chunks if "Perancangan" in c["section_title"])
    assert design_chunk["chunk_type"] == "Design"


def test_classification_general_chunk(tmp_path):
    p = _write(tmp_path, "doc.md", CLASSIFIED_MD)
    chunks = chunk_markdown(p)
    general_chunk = next(c for c in chunks if c["section_title"] == "Pendahuluan")
    assert general_chunk["chunk_type"] == "General"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_no_h2_h3_returns_single_chunk(tmp_path):
    """A file with only H1 and body text → one chunk with chunk_id ending __document."""
    content = "# Only Title\n\nSome content here with no subheadings.\n"
    p = _write(tmp_path, "flat.md", content)
    chunks = chunk_markdown(p)
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "flat__document"


def test_chunk_text_includes_heading_line(tmp_path):
    """Each chunk's text starts with its own heading."""
    p = _write(tmp_path, "sample.md", SAMPLE_MD)
    chunks = chunk_markdown(p)
    for c in chunks:
        # The heading marker must appear in the text
        assert c["section_title"] in c["text"]


def test_docchunk_has_all_keys(tmp_path):
    p = _write(tmp_path, "sample.md", SAMPLE_MD)
    chunks = chunk_markdown(p)
    required = {"chunk_id", "source_file", "section_title", "chunk_type", "text"}
    for c in chunks:
        assert required.issubset(c.keys())
