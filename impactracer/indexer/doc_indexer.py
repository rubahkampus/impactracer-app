"""Markdown chunking and chunk-type classification.

Implements FR-A1 (boundary-based chunking at H2/H3) and FR-A2
(deterministic substring classification into FR | NFR | Design | General).

Reference: master_blueprint.md §3.1.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

import mistune


# NFR must be checked before FR: "non-functional" contains "functional",
# so FR's keyword "functional requirement" would match NFR titles first otherwise.
CHUNK_TYPE_RULES: dict[str, list[str]] = {
    "NFR":    ["non-fungsional", "non-functional", "kebutuhan non"],
    "FR":     ["kebutuhan fungsional", "functional requirement", "use case"],
    "Design": ["perancangan", "desain", "arsitektur", "design", "architecture"],
}


class DocChunk(TypedDict):
    chunk_id: str
    source_file: str
    section_title: str
    chunk_type: str
    text: str


def classify_chunk(section_title: str) -> str:
    """Return one of FR, NFR, Design, or General."""
    t = section_title.lower()
    for ctype, keywords in CHUNK_TYPE_RULES.items():
        if any(kw in t for kw in keywords):
            return ctype
    return "General"


def _slugify(title: str) -> str:
    """Lowercase + replace non-alphanumeric runs with single underscore."""
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower())
    return slug.strip("_")


def _extract_heading_text(token: dict) -> str:
    """Recursively extract plain text from a mistune heading token."""
    parts: list[str] = []
    for child in token.get("children", []):
        if child.get("type") == "text":
            parts.append(child.get("raw", ""))
        elif "children" in child:
            parts.append(_extract_heading_text(child))
    return "".join(parts)


def chunk_markdown(filepath: Path) -> list[DocChunk]:
    """Split a Markdown file at H2/H3 boundaries.

    H1 is NOT a boundary. H4+ content absorbs into the enclosing H2/H3 chunk.
    Chunk IDs follow ``{file_stem}__{slugified_title}``. Deterministic across runs.
    """
    source = filepath.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    file_stem = filepath.stem

    # Parse AST to locate boundary headings (level 2 or 3).
    parse = mistune.create_markdown(renderer=None)
    tokens: list[dict] = parse(source)  # type: ignore[assignment]

    # Find (line_number_0indexed, title) for each H2/H3.
    # mistune 3.x does not annotate line numbers in the AST directly,
    # so we locate boundaries by scanning raw lines for ATX headings.
    boundary_positions: list[tuple[int, str]] = []  # (line_index, title)

    for token in tokens:
        if token.get("type") != "heading":
            continue
        level = token["attrs"]["level"]
        if level not in (2, 3):
            continue
        title = _extract_heading_text(token)
        boundary_positions.append((None, title))  # line index filled below

    # Walk source lines to assign exact line indices to boundaries in order.
    # We match each boundary in sequence to handle duplicate titles correctly.
    remaining = list(boundary_positions)
    resolved: list[tuple[int, str]] = []
    bi = 0
    for li, line in enumerate(lines):
        if bi >= len(remaining):
            break
        stripped = line.rstrip("\n\r")
        m = re.match(r"^(#{2,3})\s+(.+)$", stripped)
        if m and len(m.group(1)) in (2, 3):
            candidate_title = m.group(2).strip()
            _, expected_title = remaining[bi]
            if candidate_title == expected_title:
                resolved.append((li, expected_title))
                bi += 1

    if not resolved:
        # No H2/H3 found — single chunk for the entire file.
        return [DocChunk(
            chunk_id=f"{file_stem}__document",
            source_file=filepath.as_posix(),
            section_title=file_stem,
            chunk_type=classify_chunk(file_stem),
            text=source.strip(),
        )]

    chunks: list[DocChunk] = []
    for idx, (start_line, title) in enumerate(resolved):
        end_line = resolved[idx + 1][0] if idx + 1 < len(resolved) else len(lines)
        # Include the heading line itself in the chunk text.
        text = "".join(lines[start_line:end_line]).strip()
        slug = _slugify(title)
        chunk_id = f"{file_stem}__{slug}"
        chunks.append(DocChunk(
            chunk_id=chunk_id,
            source_file=filepath.as_posix(),
            section_title=title,
            chunk_type=classify_chunk(title),
            text=text,
        ))

    return chunks
