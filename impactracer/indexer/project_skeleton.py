"""Project-skeleton extraction (consumed by LLM #1, Step 1).

Produces a token-budgeted textual summary of the indexed codebase that
LLM #1 can use to ground its anchor-candidate guesses in real project
vocabulary instead of generic Next.js intuitions.

The skeleton is computed ONCE at index time and cached as a file under
``data/project_skeleton.txt``. The online pipeline reads it into the
interpreter prompt; the variant cache memoises LLM #1 outputs that
include it, so the file is read at most once per CR.

The skeleton contains four sections, in order:

  1. Top-level directories with file counts.
  2. Common path patterns by file_classification, with concrete examples.
  3. Naming conventions inferred from the index.
  4. Top-N most-incoming-referenced exported symbols (the "domain
     vocabulary" any new feature is likely to call or modify).

Section 4 is the most valuable for anchor extraction; section 3
disambiguates camelCase functions from PascalCase classes.

Reference: master_blueprint.md §7 §4.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from loguru import logger


# Target output size: ~1500 tokens. Token rule-of-thumb is ~4 chars/token,
# so the file should stay under ~6000 chars. The constants below are
# tuned to land in that envelope on the citrakara index.
_MAX_TOP_DIRS = 12
_MAX_EXAMPLES_PER_PATTERN = 6
_MAX_DOMAIN_VOCABULARY = 40


def build_project_skeleton(conn: sqlite3.Connection) -> str:
    """Build the project-skeleton text from the indexed database.

    The output is deterministic given the index state. Re-running the
    indexer on an unchanged repo produces a bit-identical skeleton.
    """
    sections: list[str] = []
    sections.append(_section_top_dirs(conn))
    sections.append(_section_path_patterns(conn))
    sections.append(_section_naming_conventions(conn))
    sections.append(_section_domain_vocabulary(conn))
    return "\n\n".join(sections)


def write_project_skeleton(conn: sqlite3.Connection, path: Path) -> int:
    """Compute the skeleton and write it to ``path``.

    Returns the byte length so the caller can log it. Creates parent
    directories as needed.
    """
    skeleton = build_project_skeleton(conn)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(skeleton, encoding="utf-8")
    n = len(skeleton.encode("utf-8"))
    logger.info(
        "[project_skeleton] Wrote {} bytes (~{} tokens) to {}",
        n, n // 4, path,
    )
    return n


def read_project_skeleton(path: Path) -> str | None:
    """Read a previously written skeleton, or return None on miss.

    Callers MUST tolerate a missing file (older indexes may
    lack one). The online pipeline degrades gracefully by falling
    back to the prior single-stage interpreter.
    """
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# Section builders
# ----------------------------------------------------------------------


def _section_top_dirs(conn: sqlite3.Connection) -> str:
    """List the top-level directories that hold most of the code.

    Computed at depth 2 (so ``src/components`` is one bucket, not
    every ``src/components/X``). Filters out tests and mock-only
    sub-trees so the skeleton highlights production code.
    """
    rows = conn.execute(
        """
        SELECT file_path FROM code_nodes
        WHERE node_type = 'File'
          AND file_path NOT LIKE '%__tests__%'
          AND file_path NOT LIKE '%__mocks__%'
          AND file_path NOT LIKE '%.test.%'
        """
    ).fetchall()

    counts: Counter[str] = Counter()
    for (fp,) in rows:
        # Take the first two path segments as the bucket.
        parts = fp.split("/")
        if len(parts) < 2:
            continue
        bucket = "/".join(parts[:2])
        counts[bucket] += 1

    top = counts.most_common(_MAX_TOP_DIRS)
    lines = ["TOP-LEVEL DIRECTORIES (file counts, production code):"]
    for bucket, n in top:
        lines.append(f"  {bucket:30s} ({n} files)")
    return "\n".join(lines)


def _section_path_patterns(conn: sqlite3.Connection) -> str:
    """Enumerate path patterns by file_classification with examples.

    The five classifications are API_ROUTE, PAGE_COMPONENT, UI_COMPONENT,
    UTILITY, TYPE_DEFINITION. For each, harvest representative
    file paths to show LLM #1 the actual layout.
    """
    lines = ["COMMON PATH PATTERNS BY ARCHITECTURAL LAYER:"]

    # Group services / repositories / models separately since they
    # all classify as UTILITY but have distinct conventions worth
    # surfacing to LLM #1.
    subgroups = [
        ("services", "src/lib/services/", "%/services/%"),
        ("repositories", "src/lib/db/repositories/", "%/repositories/%"),
        ("models", "src/lib/db/models/", "%/models/%"),
    ]
    for label, prefix, like in subgroups:
        rows = conn.execute(
            f"""
            SELECT DISTINCT file_path FROM code_nodes
            WHERE node_type = 'File'
              AND file_path LIKE ?
              AND file_path NOT LIKE '%__tests__%'
              AND file_path NOT LIKE '%__mocks__%'
              AND file_path NOT LIKE '%.test.%'
            ORDER BY file_path
            LIMIT ?
            """,
            (like, _MAX_EXAMPLES_PER_PATTERN),
        ).fetchall()
        if not rows:
            continue
        examples = ", ".join(Path(r[0]).stem for r in rows)
        lines.append(f"  {label:12s} ({prefix}<name>.ts): {examples}")

    # The five classifications.
    for cls in ["API_ROUTE", "PAGE_COMPONENT", "UI_COMPONENT", "TYPE_DEFINITION"]:
        rows = conn.execute(
            """
            SELECT DISTINCT file_path FROM code_nodes
            WHERE file_classification = ?
              AND file_path NOT LIKE '%__tests__%'
              AND file_path NOT LIKE '%__mocks__%'
              AND file_path NOT LIKE '%.test.%'
            ORDER BY file_path
            LIMIT ?
            """,
            (cls, _MAX_EXAMPLES_PER_PATTERN),
        ).fetchall()
        if not rows:
            continue
        # API routes and pages are all named route.ts/page.tsx, so use the
        # parent directory chain for differentiation instead of the basename.
        if cls in ("API_ROUTE", "PAGE_COMPONENT"):
            examples = ", ".join(_short_route_label(r[0]) for r in rows)
        else:
            examples = ", ".join(Path(r[0]).stem for r in rows)
        lines.append(f"  {cls:15s}: {examples}")
    return "\n".join(lines)


def _short_route_label(file_path: str) -> str:
    """Convert ``src/app/api/user/bookmarks/route.ts`` into
    ``api/user/bookmarks``. Falls back to the basename on unexpected
    paths so we never return an empty string.
    """
    p = Path(file_path)
    parts = p.parts
    # Strip leading 'src/app' and trailing 'route.ts'/'page.tsx'.
    cleaned = [x for x in parts if x not in ("src", "app", "route.ts", "page.tsx", "route.tsx", "page.ts")]
    return "/".join(cleaned) or p.stem


def _section_naming_conventions(conn: sqlite3.Connection) -> str:
    """Infer naming conventions from observed identifiers.

    The goal is to tell LLM #1 the case style used for each kind of
    symbol so its anchor candidates match the real index (avoiding
    the C3 PascalCase-vs-camelCase mismatch).
    """
    # Sample function names by file_classification to expose verb
    # prefixes per architectural layer.
    def _top_prefixes(node_type: str, like: str | None, limit: int = 8) -> list[str]:
        if like is None:
            rows = conn.execute(
                """
                SELECT name FROM code_nodes
                WHERE node_type = ? AND exported = 1
                  AND file_path NOT LIKE '%__tests__%'
                """,
                (node_type,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT name FROM code_nodes
                WHERE node_type = ? AND exported = 1
                  AND file_path LIKE ?
                  AND file_path NOT LIKE '%__tests__%'
                """,
                (node_type, like),
            ).fetchall()
        prefixes: Counter[str] = Counter()
        for (n,) in rows:
            # Strip the leading verb prefix (lowercase run).
            m = re.match(r"^([a-z]+)", n or "")
            if m:
                prefixes[m.group(1)] += 1
        return [p for p, _ in prefixes.most_common(limit)]

    service_prefixes = _top_prefixes("Function", "%/services/%")
    repo_prefixes = _top_prefixes("Function", "%/repositories/%")

    lines = ["NAMING CONVENTIONS:"]
    lines.append(
        "  Functions: camelCase. Services use verbs like "
        f"{', '.join(service_prefixes[:6]) or '(see code)'}. "
        f"Repositories use {', '.join(repo_prefixes[:4]) or '(see code)'}."
    )
    lines.append(
        "  Interfaces and types: PascalCase with leading I (e.g. ICommissionListing, "
        "IUser, IContract). Payload/Input types end with the suffix Payload."
    )
    lines.append(
        "  Components: PascalCase. Page components end with Page (e.g. "
        "CommissionFormPage), section blocks end with Section."
    )
    lines.append(
        "  Mongoose schemas: PascalCase ending in Schema (e.g. UserSchema, "
        "CommissionListingSchema). Field changes affect both the I-interface "
        "AND the *Schema literal in the same model.ts file."
    )
    lines.append(
        "  API route handler functions: ALL CAPS HTTP verbs (GET, POST, "
        "PATCH, PUT, DELETE) inside route.ts files."
    )
    return "\n".join(lines)


def _section_domain_vocabulary(conn: sqlite3.Connection) -> str:
    """The N most-incoming-referenced exported symbols.

    These are the symbols any new feature is most likely to call,
    extend, or modify. LLM #1 uses this section to ground its
    anchor_candidates in real names rather than inventing them.

    Excludes auth/connection plumbing (connectDB, handleError,
    toObjectId, getAuthSession, withAuth, rotateToken) since those
    are infrastructure rather than domain anchors.
    """
    plumbing = {
        "connectDB", "handleError", "toObjectId", "getAuthSession",
        "withAuth", "rotateToken", "isUserOwner",
    }
    rows = conn.execute(
        """
        SELECT cn.name, cn.node_type, cn.file_path, COUNT(se.source_id) AS in_degree
        FROM code_nodes cn
        LEFT JOIN structural_edges se ON cn.node_id = se.target_id
        WHERE cn.node_type IN ('Function', 'Method', 'Class', 'Interface',
                               'TypeAlias', 'Variable')
          AND cn.exported = 1
          AND cn.file_path NOT LIKE '%__tests__%'
          AND cn.file_path NOT LIKE '%__mocks__%'
          AND cn.file_path NOT LIKE '%.test.%'
        GROUP BY cn.node_id
        HAVING in_degree > 0
        ORDER BY in_degree DESC
        """
    ).fetchall()

    by_file: dict[str, list[str]] = defaultdict(list)
    selected: list[tuple[str, str, str]] = []
    for name, ntype, fp, _deg in rows:
        if name in plumbing:
            continue
        if len(selected) >= _MAX_DOMAIN_VOCABULARY:
            break
        # Cap per-file to 3 so one repo doesn't dominate.
        if len(by_file[fp]) >= 3:
            continue
        by_file[fp].append(name)
        selected.append((name, ntype, fp))

    lines = ["DOMAIN VOCABULARY (most-called/imported exported symbols):"]
    # Group by host file for readability.
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for name, ntype, fp in selected:
        grouped[fp].append((name, ntype))
    for fp in sorted(grouped):
        symbols = ", ".join(n for n, _ in grouped[fp])
        lines.append(f"  {fp}: {symbols}")
    return "\n".join(lines)
