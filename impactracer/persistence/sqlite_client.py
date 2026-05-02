"""SQLite connection factory and schema initialization.

Schema reference: 04_database_schema.md §1.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS code_nodes (
    node_id                     TEXT    PRIMARY KEY,
    node_type                   TEXT    NOT NULL
        CHECK (node_type IN (
            'File', 'Class', 'Function', 'Method',
            'Interface', 'TypeAlias', 'Enum',
            'ExternalPackage', 'InterfaceField'
        )),
    name                        TEXT    NOT NULL,
    file_path                   TEXT,
    line_start                  INTEGER,
    line_end                    INTEGER,
    signature                   TEXT,
    docstring                   TEXT,
    source_code                 TEXT,
    internal_logic_abstraction  TEXT,
    route_path                  TEXT,
    file_classification         TEXT
        CHECK (file_classification IN (
            'API_ROUTE', 'PAGE_COMPONENT', 'UI_COMPONENT',
            'UTILITY', 'TYPE_DEFINITION'
        ) OR file_classification IS NULL),
    exported                    INTEGER DEFAULT 0,
    embed_text                  TEXT,
    client_directive            TEXT
        CHECK (client_directive IN ('client', 'server') OR client_directive IS NULL)
);

CREATE INDEX IF NOT EXISTS idx_code_nodes_file_path
    ON code_nodes(file_path);
CREATE INDEX IF NOT EXISTS idx_code_nodes_classification
    ON code_nodes(file_classification);

CREATE TABLE IF NOT EXISTS structural_edges (
    source_id   TEXT    NOT NULL,
    target_id   TEXT    NOT NULL,
    edge_type   TEXT    NOT NULL
        CHECK (edge_type IN (
            'CALLS', 'INHERITS', 'IMPLEMENTS', 'TYPED_BY', 'FIELDS_ACCESSED',
            'DEFINES_METHOD', 'HOOK_DEPENDS_ON', 'PASSES_CALLBACK',
            'IMPORTS', 'RENDERS', 'DEPENDS_ON_EXTERNAL',
            'CLIENT_API_CALLS', 'DYNAMIC_IMPORT', 'CONTAINS'
        )),
    PRIMARY KEY (source_id, target_id, edge_type),
    FOREIGN KEY (source_id) REFERENCES code_nodes(node_id),
    FOREIGN KEY (target_id) REFERENCES code_nodes(node_id)
);

CREATE INDEX IF NOT EXISTS idx_edges_target_type
    ON structural_edges(target_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_edges_source_type
    ON structural_edges(source_id, edge_type);

CREATE TABLE IF NOT EXISTS doc_code_candidates (
    code_id                      TEXT    NOT NULL,
    doc_id                       TEXT    NOT NULL,
    weighted_similarity_score    REAL    NOT NULL,
    PRIMARY KEY (code_id, doc_id),
    FOREIGN KEY (code_id) REFERENCES code_nodes(node_id)
);

CREATE INDEX IF NOT EXISTS idx_dcc_doc_score
    ON doc_code_candidates(doc_id, weighted_similarity_score DESC);
CREATE INDEX IF NOT EXISTS idx_dcc_code_score
    ON doc_code_candidates(code_id, weighted_similarity_score DESC);

CREATE TABLE IF NOT EXISTS file_hashes (
    file_path       TEXT    PRIMARY KEY,
    content_hash    TEXT    NOT NULL,
    indexed_at      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS file_dependencies (
    dependent_file  TEXT    NOT NULL,
    target_file     TEXT    NOT NULL,
    PRIMARY KEY (dependent_file, target_file)
);

CREATE TABLE IF NOT EXISTS index_metadata (
    key     TEXT    PRIMARY KEY,
    value   TEXT    NOT NULL
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection with pragmas applied."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes. Idempotent.

    Also applies forward migrations for columns/constraints added after the
    initial schema (safe to run repeatedly — ALTER TABLE IF NOT EXISTS
    silently no-ops if the column already exists via the try/except guard).
    """
    conn.executescript(SCHEMA_DDL)
    # Sprint 7.75 migration: client_directive column (nullable TEXT).
    # SQLite does not support ALTER TABLE … ADD COLUMN … CHECK, but the
    # constraint in the DDL above applies to freshly-created tables; for
    # existing DBs we just add the bare column (application-level constraint).
    try:
        conn.execute("ALTER TABLE code_nodes ADD COLUMN client_directive TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists
