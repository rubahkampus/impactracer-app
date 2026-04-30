"""Two-pass tag-and-fold AST reduction (FR-A6).

Produces the ``internal_logic_abstraction`` for Function and Method nodes.
Preserves high-signal nodes (calls, returns, throws, imports) and their
ancestors while folding boilerplate (long JSX, large arrays/objects,
long string literals, comments).

Reference: master_blueprint.md §3.3
"""

from __future__ import annotations

from tree_sitter import Node


HIGH_SIGNAL_NODE_TYPES: frozenset[str] = frozenset({
    "call_expression",
    "return_statement",
    "throw_statement",
    "import_declaration",
})

_HOOK_NAMES: frozenset[str] = frozenset({
    "useEffect", "useCallback", "useMemo", "useLayoutEffect",
})


def skeletonize_node(root: Node, source_bytes: bytes) -> str:
    """Return the reduced skeleton of a function or component body.

    Pass 1: tag every HIGH_SIGNAL_NODE_TYPES node and all its ancestors
    as DO_NOT_ERASE.

    Pass 2: walk the root, tracking source position to preserve whitespace
    between tokens. Tagged nodes recurse; untagged nodes apply fold rules.
    """
    tagged: set[int] = set()
    _tag_high_signal(root, tagged)

    parts: list[str] = []
    cursor = [root.start_byte]
    _emit(root, source_bytes, tagged, parts, cursor, is_root=True)
    return "".join(parts).strip()


# ---------------------------------------------------------------------------
# Pass 1 — tag ancestors of high-signal nodes
# ---------------------------------------------------------------------------

def _tag_high_signal(node: Node, tagged: set[int]) -> bool:
    is_signal = node.type in HIGH_SIGNAL_NODE_TYPES
    for child in node.children:
        if _tag_high_signal(child, tagged):
            is_signal = True
    if is_signal:
        tagged.add(id(node))
    return is_signal


# ---------------------------------------------------------------------------
# Pass 2 — position-tracked emit with fold rules
# ---------------------------------------------------------------------------

def _emit(node: Node, src: bytes, tagged: set[int],
          parts: list[str], cursor: list[int], is_root: bool = False) -> None:
    """Emit text for a node, preserving whitespace gaps between tokens."""
    if is_root:
        # Emit the opening brace of the function body, then children, then closing
        cursor[0] = node.start_byte
        for child in node.children:
            _emit(child, src, tagged, parts, cursor)
        # Flush any trailing bytes up to node end
        if cursor[0] < node.end_byte:
            parts.append(src[cursor[0]:node.end_byte].decode(errors="replace"))
        return

    # Fill whitespace gap before this node
    if cursor[0] < node.start_byte:
        parts.append(src[cursor[0]:node.start_byte].decode(errors="replace"))
    cursor[0] = node.start_byte

    if id(node) in tagged:
        # Tagged: recurse verbatim, advancing cursor through children
        for child in node.children:
            _emit(child, src, tagged, parts, cursor)
        cursor[0] = node.end_byte
        return

    # --- Fold rules (first match wins) ---

    if node.type in ("jsx_element", "jsx_self_closing_element", "jsx_fragment"):
        count = _count_jsx_elements(node)
        parts.append(f"/* [JSX: {count} elements] */")
        cursor[0] = node.end_byte
        return

    if node.type == "array":
        items = [c for c in node.children if c.type not in ("[", "]", ",")]
        if len(items) > 3 and not _is_hook_dep_array(node, src):
            parts.append(f"/* [array: {len(items)} items] */")
            cursor[0] = node.end_byte
            return

    if node.type == "object":
        props = [c for c in node.children if c.type not in ("{", "}", ",")]
        if len(props) > 4:
            parts.append(f"/* [object: {len(props)} props] */")
            cursor[0] = node.end_byte
            return

    if node.type in ("if_statement", "switch_statement"):
        if not _has_high_signal_descendant(node, tagged):
            parts.append("/* [logic block] */")
            cursor[0] = node.end_byte
            return

    if node.type == "template_string":
        raw = src[node.start_byte:node.end_byte].decode(errors="replace")
        if len(raw) > 100:
            parts.append(f"`/* [template: {len(raw)} chars] */`")
            cursor[0] = node.end_byte
            return

    if node.type == "string":
        raw = src[node.start_byte:node.end_byte].decode(errors="replace")
        if len(raw) > 80:
            parts.append(f'"/* [string: {len(raw)} chars] */"')
            cursor[0] = node.end_byte
            return

    if node.type == "comment":
        cursor[0] = node.end_byte
        return

    # Default: leaf nodes emit their raw bytes; inner nodes recurse
    if node.child_count == 0:
        parts.append(src[node.start_byte:node.end_byte].decode(errors="replace"))
        cursor[0] = node.end_byte
    else:
        for child in node.children:
            _emit(child, src, tagged, parts, cursor)
        cursor[0] = node.end_byte


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_high_signal_descendant(node: Node, tagged: set[int]) -> bool:
    if id(node) in tagged:
        return True
    for child in node.children:
        if _has_high_signal_descendant(child, tagged):
            return True
    return False


def _count_jsx_elements(node: Node) -> int:
    count = 1 if node.type in ("jsx_element", "jsx_self_closing_element", "jsx_fragment") else 0
    for child in node.children:
        count += _count_jsx_elements(child)
    return count


def _is_hook_dep_array(node: Node, src: bytes) -> bool:
    """Return True if this array is the last argument of a React hook call."""
    parent = node.parent
    if parent is None or parent.type != "arguments":
        return False
    call = parent.parent
    if call is None or call.type != "call_expression":
        return False
    fn_node = call.child_by_field_name("function")
    if fn_node is None:
        return False
    fn_name = src[fn_node.start_byte:fn_node.end_byte].decode(errors="replace")
    if fn_name not in _HOOK_NAMES:
        return False
    arg_children = [c for c in parent.children if c.type not in ("(", ")", ",")]
    return bool(arg_children) and arg_children[-1].id == node.id
