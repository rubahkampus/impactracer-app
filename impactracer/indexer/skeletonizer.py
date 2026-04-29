"""Two-pass tag-and-fold AST reduction (FR-A6).

Produces the ``internal_logic_abstraction`` for Function and Method nodes.
Preserves high-signal nodes (calls, returns, throws, imports) and their
ancestors while folding boilerplate (long JSX, large arrays/objects,
long string literals, comments).

Reference: 06_offline_indexer.md §3.4.
"""

from __future__ import annotations

from tree_sitter import Node


HIGH_SIGNAL_NODE_TYPES: frozenset[str] = frozenset({
    "call_expression",
    "return_statement",
    "throw_statement",
    "import_declaration",
})


def skeletonize_node(root: Node, source_bytes: bytes) -> str:
    """Return the reduced skeleton of a function or component body.

    Implementation outline:
      1. Pass 1: walk the AST and tag every :data:`HIGH_SIGNAL_NODE_TYPES`
         node plus all its ancestors as ``DO_NOT_ERASE``.
      2. Pass 2: walk again, emitting source bytes per node. Tagged nodes
         recurse into children; untagged nodes apply fold rules.

    Fold rules (applied in order, first match wins): see 06_offline_indexer.md §3.4.
    """
    raise NotImplementedError("Sprint 4")
