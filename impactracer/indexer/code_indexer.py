"""TypeScript/TSX AST parser and edge extractor (FR-A3, FR-A4).

Two-pass design:

- Pass 1 (:func:`extract_nodes`): walks the AST to produce :class:`File`,
  :class:`Class`, :class:`Function`, :class:`Method`, :class:`Interface`,
  :class:`TypeAlias`, :class:`Enum`, and :class:`InterfaceField` nodes.
  Populates ``internal_logic_abstraction`` via :func:`skeletonize_node`.

- Pass 2 (:func:`extract_edges`): walks the AST again, with the full
  set of node IDs from Pass 1 available, and emits all 13 edge types.
  Populates ``file_dependencies`` for incremental reindex.

Reference: master_blueprint.md §3.2 (Pass 1), §3.4 (Pass 2)
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from tree_sitter import Node, Parser
from tree_sitter_languages import get_parser as _get_ts_parser_impl

from impactracer.indexer.skeletonizer import skeletonize_node


# ---------------------------------------------------------------------------
# Parser factory  (blueprint §3.2)
# ---------------------------------------------------------------------------

def get_ts_parser(file_path: Path) -> Parser:
    """Return the correct parser for a .ts or .tsx file."""
    if file_path.suffix == ".tsx":
        return _get_ts_parser_impl("tsx")
    return _get_ts_parser_impl("typescript")


# ---------------------------------------------------------------------------
# File classification  (blueprint §3.2)
# ---------------------------------------------------------------------------

# Patterns checked in order, first match wins.
# Tuples of (match_function, classification).
def _make_classifier():
    _page_names  = {"page.ts", "page.tsx", "layout.ts", "layout.tsx"}
    _route_names = {"route.ts", "route.tsx"}

    def classify(rel_path: Path) -> str | None:
        p        = rel_path.as_posix()
        filename = rel_path.name

        # ── src/app/** ──────────────────────────────────────────────────
        if p.startswith("src/app/"):
            if filename in _route_names:
                return "API_ROUTE"
            if filename in _page_names:
                return "PAGE_COMPONENT"
            return None   # other app/ files (e.g. globals.css shims) unclassified

        # ── React UI components ─────────────────────────────────────────
        if p.startswith("src/components/"):
            return "UI_COMPONENT"

        if p.startswith("src/hooks/"):
            return "UI_COMPONENT"

        if p.startswith("src/lib/stores/"):
            return "UI_COMPONENT"

        # Test/mock exclusions must precede the models/ rule.
        if (p.startswith("src/lib/test/")
                or p.startswith("src/lib/db/models/__mocks__/")
                or p.startswith("src/lib/db/repositories/__tests__/")
                or p.startswith("src/lib/services/__tests__/")
                or p.startswith("src/lib/utils/__mocks__/")):
            return None

        if p.startswith("src/lib/db/models/"):
            return "TYPE_DEFINITION"

        if p.startswith("src/lib/"):
            return "UTILITY"

        # ── Standalone TypeScript type declaration files ────────────────
        if p.startswith("src/types/"):
            return "TYPE_DEFINITION"

        # ── Anything else (src/config.ts, src/middleware.ts, src/theme/)──
        return None

    return classify


classify_file = _make_classifier()


# ---------------------------------------------------------------------------
# Route path derivation  (blueprint §3.2)
# ---------------------------------------------------------------------------

def derive_route_path(rel_path: Path) -> str | None:
    """Derive the route_path for API_ROUTE files.

    src/app/api/commissions/[id]/route.ts -> /api/commissions/{id}
    """
    p = rel_path.as_posix()
    # Strip src/app prefix and trailing /route.{ts,tsx} or /page.{ts,tsx} etc.
    if not p.startswith("src/app/"):
        return None
    after_app = p[len("src/app"):]  # e.g. /api/commissions/[id]/route.ts
    # Remove the filename segment (route.ts, page.tsx, layout.tsx…)
    parts = after_app.split("/")
    # parts[0] is '', parts[-1] is filename
    route_parts = parts[:-1]  # drop the filename
    route = "/".join(route_parts)  # e.g. /api/commissions/[id]
    # Convert [param] -> {param}
    route = re.sub(r"\[([^\]]+)\]", r"{\1}", route)
    return route if route else "/"


# ---------------------------------------------------------------------------
# Node ID helper
# ---------------------------------------------------------------------------

def _make_node_id(file_posix: str, name: str) -> str:
    """Construct a node ID: <file_posix>::<name>. Forward slashes always."""
    return f"{file_posix}::{name}"


# ---------------------------------------------------------------------------
# JSDoc extraction
# ---------------------------------------------------------------------------

def _extract_jsdoc(node: Node, src: bytes) -> str | None:
    """Return the preceding /** ... */ comment text (stripped), or None."""
    parent = node.parent
    if parent is None:
        return None
    siblings = list(parent.children)
    idx = siblings.index(node)
    if idx > 0:
        prev = siblings[idx - 1]
        if prev.type == "comment":
            text = src[prev.start_byte:prev.end_byte].decode(errors="replace").strip()
            if text.startswith("/**"):
                # Strip /** ... */ wrapper and leading * on each line
                inner = text[3:-2] if text.endswith("*/") else text[3:]
                lines = [ln.strip().lstrip("*").strip() for ln in inner.splitlines()]
                return " ".join(ln for ln in lines if ln)
    return None


# ---------------------------------------------------------------------------
# Signature extraction
# ---------------------------------------------------------------------------

def _fn_signature(node: Node, name: str, src: bytes) -> str:
    """Build a concise signature: name(params): return_type"""
    params_node = node.child_by_field_name("parameters")
    return_type_node = node.child_by_field_name("return_type")
    params = src[params_node.start_byte:params_node.end_byte].decode(errors="replace") if params_node else "()"
    ret = src[return_type_node.start_byte:return_type_node.end_byte].decode(errors="replace") if return_type_node else ""
    return f"{name}{params}{ret}"


def _method_signature(node: Node, name: str, src: bytes) -> str:
    """Build method signature including optional async/static/visibility modifiers."""
    params_node = node.child_by_field_name("parameters")
    return_type_node = node.child_by_field_name("return_type")
    params = src[params_node.start_byte:params_node.end_byte].decode(errors="replace") if params_node else "()"
    ret = src[return_type_node.start_byte:return_type_node.end_byte].decode(errors="replace") if return_type_node else ""
    return f"{name}{params}{ret}"


# ---------------------------------------------------------------------------
# Export detection
# ---------------------------------------------------------------------------

def _is_exported(node: Node) -> tuple[bool, bool]:
    """Return (is_exported, is_default) by walking up parent chain."""
    parent = node.parent
    if parent is None:
        return False, False
    if parent.type == "export_statement":
        is_default = any(c.type == "default" for c in parent.children)
        return True, is_default
    # Handle nested: export_statement > lexical_declaration > variable_declarator > arrow_function
    if parent.type in ("variable_declarator", "lexical_declaration"):
        return _is_exported(parent)
    return False, False


# ---------------------------------------------------------------------------
# Has-JSX check
# ---------------------------------------------------------------------------

def _has_jsx(node: Node) -> bool:
    if node.type in ("jsx_element", "jsx_self_closing_element", "jsx_fragment"):
        return True
    for child in node.children:
        if _has_jsx(child):
            return True
    return False


# ---------------------------------------------------------------------------
# CamelCase splitting for synthetic UI docstrings
# ---------------------------------------------------------------------------

def _camel_to_readable(name: str) -> str:
    """Split CamelCase or camelCase into space-separated words."""
    return " ".join(re.sub(r"([A-Z])", r" \1", name).split())


# ---------------------------------------------------------------------------
# Exported names collector (for File embed_text)
# ---------------------------------------------------------------------------

def _collect_exported_names(root: Node, src: bytes) -> list[str]:
    """Return sorted unique names of exported declarations in this file."""
    names: list[str] = []
    for child in root.children:
        if child.type != "export_statement":
            continue
        for grandchild in child.children:
            if grandchild.type == "function_declaration":
                n = grandchild.child_by_field_name("name")
                if n:
                    names.append(src[n.start_byte:n.end_byte].decode(errors="replace"))
            elif grandchild.type in ("class_declaration", "interface_declaration",
                                     "type_alias_declaration", "enum_declaration"):
                n = grandchild.child_by_field_name("name")
                if n:
                    names.append(src[n.start_byte:n.end_byte].decode(errors="replace"))
            elif grandchild.type == "lexical_declaration":
                for vd in grandchild.children:
                    if vd.type == "variable_declarator":
                        n = vd.child_by_field_name("name")
                        if n:
                            names.append(src[n.start_byte:n.end_byte].decode(errors="replace"))
            elif grandchild.type == "export_clause":
                for spec in grandchild.children:
                    if spec.type == "export_specifier":
                        n = spec.children[0] if spec.children else None
                        if n:
                            names.append(src[n.start_byte:n.end_byte].decode(errors="replace"))
    return sorted(set(n for n in names if n))


# ---------------------------------------------------------------------------
# compose_embed_text / compose_file_embed_text / synthesize_ui_docstring
# ---------------------------------------------------------------------------

def synthesize_ui_docstring(name: str, signature: str) -> str:
    """Synthesize a docstring for an exported UI component without JSDoc.

    Blueprint §3.2: {readable_name} UI component. Props: {readable_prop_types}
    """
    readable_name = _camel_to_readable(name)
    # Extract prop type names from signature (simplified: pull type_identifiers)
    prop_types = re.findall(r"\b([A-Z][A-Za-z0-9]*(?:Props?|Type|Interface))\b", signature)
    if prop_types:
        readable_props = ", ".join(_camel_to_readable(p) for p in prop_types)
    else:
        # Fall back to showing the raw params if no typed props found
        readable_props = signature
    return f"{readable_name} UI component. Props: {readable_props}"


def compose_embed_text(node: dict[str, Any]) -> str:
    """Compose the BGE-M3 input text for a non-File, non-degenerate node.

    Layout: docstring\\nsignature (empty lines omitted). Falls back to name.
    Blueprint §3.2.
    """
    parts = []
    docstring = (node.get("docstring") or "").strip()
    signature = (node.get("signature") or "").strip()
    if docstring:
        parts.append(docstring)
    if signature:
        parts.append(signature)
    result = "\n".join(parts).strip()
    return result if result else (node.get("name") or "")


def compose_file_embed_text(
    file_node: dict[str, Any],
    exported_names: list[str],
    rel_dir: str,
) -> str:
    """Compose the enriched BGE-M3 input text for a File node.

    Blueprint §3.2: {filename} [{classification}] ({rel_dir})\\nexports: {sorted exports}
    Extended: path components are emitted as space-separated keywords so the
    embedding model can latch onto domain tokens like "auth", "wallet", "login".
    """
    filename = file_node.get("name") or ""
    classification = file_node.get("file_classification") or "NULL"
    exports_str = ", ".join(exported_names) if exported_names else "(none)"
    # Expand directory path into individual keyword tokens (e.g. "auth login api")
    path_keywords = " ".join(
        p for p in rel_dir.replace("\\", "/").split("/")
        if p and p not in ("src", "app", "lib", ".")
    )
    base = f"{filename} [{classification}] ({rel_dir})\nexports: {exports_str}"
    if path_keywords:
        return f"{base}\npath keywords: {path_keywords}"
    return base


# ---------------------------------------------------------------------------
# Main Pass 1: extract_nodes
# ---------------------------------------------------------------------------

def extract_nodes(
    file_path: Path,
    source_bytes: bytes,
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Pass 1: extract all nodes from a source file.

    Returns list of node dicts. Also inserts into code_nodes via the
    provided connection (INSERT OR REPLACE).
    Blueprint §3.2.
    """
    parser = get_ts_parser(file_path)
    tree = parser.parse(source_bytes)
    root = tree.root_node

    # Resolve relative path from repo root (strip leading components)
    # file_path is absolute; we derive rel_path as posix relative to src/
    # Find 'src' in the parts and take from there
    parts = file_path.parts
    try:
        src_idx = next(i for i in range(len(parts) - 1, -1, -1) if parts[i] == "src")
        rel_path = Path(*parts[src_idx:])
    except StopIteration:
        rel_path = Path(file_path.name)

    rel_posix = rel_path.as_posix()
    file_classification = classify_file(rel_path)
    route_path = derive_route_path(rel_path) if file_classification in ("API_ROUTE", "PAGE_COMPONENT") else None
    rel_dir = str(rel_path.parent.as_posix())

    # Detect Next.js 'use client' / 'use server' directive (must be first statement).
    client_directive: str | None = None
    for top_stmt in root.children:
        if top_stmt.type == "expression_statement":
            for sc in top_stmt.children:
                if sc.type == "string":
                    val = source_bytes[sc.start_byte:sc.end_byte].decode(errors="replace").strip("'\" ")
                    if val == "use client":
                        client_directive = "client"
                    elif val == "use server":
                        client_directive = "server"
            break  # directive must be the very first statement if present
        if top_stmt.type not in ("comment", "hash_bang_line"):
            break  # non-directive first statement — stop scanning

    # Collect exported names for the File embed_text
    exported_names = _collect_exported_names(root, source_bytes)

    nodes: list[dict[str, Any]] = []

    # --- File node (always index 0) ---
    file_node_id = rel_posix
    file_node: dict[str, Any] = {
        "node_id": file_node_id,
        "node_type": "File",
        "name": file_path.name,
        "file_path": rel_posix,
        "file_classification": file_classification,
        "route_path": route_path,
        "signature": None,
        "docstring": None,
        "internal_logic_abstraction": None,
        "source_code": None,
        "start_line": 1,
        "end_line": source_bytes.count(b"\n") + 1,
        "is_exported": True,
        "client_directive": client_directive,
    }
    file_node["embed_text"] = compose_file_embed_text(file_node, exported_names, rel_dir)
    nodes.append(file_node)

    # --- Walk top-level declarations ---
    # Track external packages seen (to emit one ExternalPackage per unique specifier)
    external_packages: dict[str, dict[str, Any]] = {}

    # We need to look at every top-level and nested declaration.
    # Strategy: recursive walk collecting nodes for each declaration type.
    _walk_declarations(
        root, source_bytes, rel_posix, file_path.name,
        file_classification, nodes, external_packages,
        parent_class_id=None,
    )

    # --- ExternalPackage nodes ---
    for pkg_id, pkg_node in external_packages.items():
        nodes.append(pkg_node)

    # --- Insert all nodes into SQLite ---
    _insert_nodes(nodes, conn)

    return nodes


def _walk_declarations(
    node: Node,
    src: bytes,
    file_posix: str,
    filename: str,
    file_classification: str | None,
    nodes: list[dict[str, Any]],
    external_packages: dict[str, dict[str, Any]],
    parent_class_id: str | None,
) -> None:
    """Recursive walker for all declaration types."""
    for child in node.children:
        # Unwrap export_statement to get the actual declaration
        decl = child
        if child.type == "export_statement":
            # Find the actual declaration inside the export
            inner = _unwrap_export(child)
            if inner is not None:
                decl = inner
            else:
                # Could be export * from ... or export { } from ... — handle imports
                _handle_import_for_external(child, src, file_posix, external_packages)
                continue

        if decl.type == "import_statement":
            _handle_import_for_external(decl, src, file_posix, external_packages)
            continue

        if decl.type == "function_declaration":
            fn_node = _build_function_node(decl, child, src, file_posix, file_classification)
            if fn_node:
                nodes.append(fn_node)
            continue

        if decl.type == "lexical_declaration":
            # Could contain arrow-function variable declarators OR non-arrow
            # canonical const declarations (schemas, constant data, factories).
            for vd in decl.children:
                if vd.type != "variable_declarator":
                    continue
                val = vd.child_by_field_name("value")
                if val is None:
                    continue
                if val.type == "arrow_function":
                    fn_node = _build_arrow_function_node(vd, val, child, src, file_posix, file_classification)
                    if fn_node:
                        nodes.append(fn_node)
                elif val.type in _VARIABLE_VALUE_TYPES:
                    var_node = _build_variable_node(vd, val, child, src, file_posix, file_classification)
                    if var_node:
                        nodes.append(var_node)
            continue

        if decl.type == "class_declaration":
            class_node, method_nodes = _build_class_nodes(decl, child, src, file_posix, file_classification)
            if class_node:
                nodes.append(class_node)
                nodes.extend(method_nodes)
            continue

        if decl.type == "interface_declaration":
            iface_node, field_nodes = _build_interface_nodes(decl, child, src, file_posix)
            if iface_node:
                nodes.append(iface_node)
                nodes.extend(field_nodes)
            continue

        if decl.type == "type_alias_declaration":
            ta_node, field_nodes = _build_type_alias_nodes(decl, child, src, file_posix)
            if ta_node:
                nodes.append(ta_node)
                nodes.extend(field_nodes)
            continue

        if decl.type == "enum_declaration":
            en_node = _build_enum_node(decl, child, src, file_posix)
            if en_node:
                nodes.append(en_node)
            continue

        # Recurse into other statement containers (e.g., module_declaration)
        if child.child_count > 0:
            _walk_declarations(
                child, src, file_posix, filename, file_classification,
                nodes, external_packages, parent_class_id,
            )


def _unwrap_export(export_node: Node) -> Node | None:
    """Return the declaration child of an export_statement, or None."""
    for child in export_node.children:
        if child.type in (
            "function_declaration",
            "class_declaration",
            "interface_declaration",
            "type_alias_declaration",
            "enum_declaration",
            "lexical_declaration",
        ):
            return child
    return None


def _handle_import_for_external(
    node: Node,
    src: bytes,
    file_posix: str,
    external_packages: dict[str, dict[str, Any]],
) -> None:
    """Extract non-relative imports and register ExternalPackage nodes."""
    # Works for both import_statement and export_statement (re-export from)
    specifier_node = None
    for child in node.children:
        if child.type == "string":
            specifier_node = child
            break
    if specifier_node is None:
        return
    raw = src[specifier_node.start_byte:specifier_node.end_byte].decode(errors="replace")
    # Strip quotes
    specifier = raw.strip("'\"")
    # Relative imports are handled in Pass 2 (IMPORTS edge); skip here
    if specifier.startswith("."):
        return
    # @/ path-alias imports are intra-repo — Pass 2 resolves them as IMPORTS edges.
    # Creating an ExternalPackage node here would leave a rogue unreachable node.
    if specifier.startswith("@/"):
        return
    if specifier in external_packages:
        return
    pkg_id = f"ext::{specifier}"
    external_packages[specifier] = {
        "node_id": pkg_id,
        "node_type": "ExternalPackage",
        "name": specifier,
        "file_path": file_posix,
        "file_classification": None,
        "route_path": None,
        "signature": None,
        "docstring": None,
        "internal_logic_abstraction": None,
        "source_code": None,
        "start_line": node.start_point[0] + 1,
        "end_line": node.end_point[0] + 1,
        "is_exported": False,
        "embed_text": "",  # Degenerate: not embedded (blueprint §3.2)
    }


# ---------------------------------------------------------------------------
# Node builders
# ---------------------------------------------------------------------------

_DEGENERATE_MIN_LEN = 50  # blueprint §3.2 / §7


def _build_function_node(
    decl: Node,
    raw_child: Node,
    src: bytes,
    file_posix: str,
    file_classification: str | None,
) -> dict[str, Any] | None:
    name_node = decl.child_by_field_name("name")
    if name_node is None:
        return None
    name = src[name_node.start_byte:name_node.end_byte].decode(errors="replace")
    is_exported, _ = _is_exported(decl)
    docstring = _extract_jsdoc(raw_child, src)
    signature = _fn_signature(decl, name, src)

    # React component flag: uppercase first char + JSX in body
    body_node = decl.child_by_field_name("body")
    has_jsx_body = body_node is not None and _has_jsx(body_node)
    is_component = name[0].isupper() and has_jsx_body

    # Synthetic UI docstring (blueprint §3.2)
    if (docstring is None and is_component and is_exported
            and file_classification == "UI_COMPONENT"):
        docstring = synthesize_ui_docstring(name, signature)

    embed_text = compose_embed_text({
        "name": name, "docstring": docstring, "signature": signature,
    })

    # Skeletonize
    skeleton = None
    if body_node is not None:
        try:
            skeleton = skeletonize_node(decl, src)
        except Exception:
            skeleton = None

    source_code = src[decl.start_byte:decl.end_byte].decode(errors="replace")

    return {
        "node_id": _make_node_id(file_posix, name),
        "node_type": "Function",
        "name": name,
        "file_path": file_posix,
        "file_classification": file_classification,
        "route_path": None,
        "signature": signature,
        "docstring": docstring,
        "internal_logic_abstraction": skeleton,
        "source_code": source_code,
        "start_line": decl.start_point[0] + 1,
        "end_line": decl.end_point[0] + 1,
        "is_exported": is_exported,
        "embed_text": embed_text,
        "is_component": is_component,
    }


def _build_arrow_function_node(
    vd: Node,
    arrow: Node,
    raw_child: Node,
    src: bytes,
    file_posix: str,
    file_classification: str | None,
) -> dict[str, Any] | None:
    name_node = vd.child_by_field_name("name")
    if name_node is None:
        return None
    name = src[name_node.start_byte:name_node.end_byte].decode(errors="replace")
    is_exported, _ = _is_exported(vd)
    docstring = _extract_jsdoc(raw_child, src)

    # Build signature from the arrow function's parameters and return type
    params_node = arrow.child_by_field_name("parameters")
    return_type_node = arrow.child_by_field_name("return_type")
    params = src[params_node.start_byte:params_node.end_byte].decode(errors="replace") if params_node else "()"
    ret = src[return_type_node.start_byte:return_type_node.end_byte].decode(errors="replace") if return_type_node else ""
    signature = f"{name}{params}{ret}"

    # React component flag
    body = arrow.child_by_field_name("body")
    has_jsx_body = _has_jsx(arrow)
    is_component = name[0].isupper() and has_jsx_body

    if (docstring is None and is_component and is_exported
            and file_classification == "UI_COMPONENT"):
        docstring = synthesize_ui_docstring(name, signature)

    embed_text = compose_embed_text({
        "name": name, "docstring": docstring, "signature": signature,
    })

    skeleton = None
    if body is not None:
        try:
            skeleton = skeletonize_node(arrow, src)
        except Exception:
            skeleton = None

    source_code = src[vd.start_byte:vd.end_byte].decode(errors="replace")

    return {
        "node_id": _make_node_id(file_posix, name),
        "node_type": "Function",
        "name": name,
        "file_path": file_posix,
        "file_classification": file_classification,
        "route_path": None,
        "signature": signature,
        "docstring": docstring,
        "internal_logic_abstraction": skeleton,
        "source_code": source_code,
        "start_line": vd.start_point[0] + 1,
        "end_line": vd.end_point[0] + 1,
        "is_exported": is_exported,
        "embed_text": embed_text,
        "is_component": is_component,
    }


_VARIABLE_VALUE_TYPES: frozenset[str] = frozenset({
    "new_expression",
    "object",
    "array",
    "call_expression",
})
"""Tree-sitter RHS types that qualify a ``const NAME = <RHS>`` as a Variable.

Arrow-function RHS is handled by ``_build_arrow_function_node`` (emits a
``Function`` node). Anything else with one of these RHS shapes becomes a
``Variable`` node: Mongoose schemas (``new Schema(...)``), constant arrays /
objects, and factory calls such as ``Object.freeze({...})`` or
``createStore(...)``.
"""


def _variable_name_is_canonical(name: str) -> bool:
    """Heuristic for whether a ``const NAME = ...`` is worth indexing.

    Accepts PascalCase (typical for schemas, components, type-shaped constants),
    SCREAMING_SNAKE_CASE (typical for module-level constants/templates), and
    any name with internal capital letters. Rejects single-lowercase locals
    like ``i``, ``j``, ``tmp`` to keep the index lean.
    """
    if not name or not name[0].isalpha():
        return False
    if name[0].isupper():
        return True  # PascalCase or SCREAMING
    return any(c.isupper() for c in name[1:])


def _summarize_variable_value(val: Node, src: bytes, cap: int = 400) -> str:
    """Return a short, embed-friendly summary of a variable's RHS.

    For ``new Schema<T>({...fields...})``: collect top-level keys of the first
    object argument and emit them space-separated — these are the schema's
    field names and are exactly the tokens BGE-M3 / BM25 needs to match
    Indonesian CR descriptions that mention them indirectly.

    For array / object literals: collect top-level keys / element identifiers.

    Falls back to the raw source up to ``cap`` characters.
    """
    try:
        if val.type == "new_expression":
            for arg_list in val.children:
                if arg_list.type != "arguments":
                    continue
                for arg in arg_list.children:
                    if arg.type == "object":
                        keys: list[str] = []
                        for prop in arg.children:
                            if prop.type == "pair":
                                k = prop.child_by_field_name("key")
                                if k is not None:
                                    keys.append(
                                        src[k.start_byte:k.end_byte]
                                        .decode(errors="replace")
                                        .strip("\"'`")
                                    )
                        if keys:
                            return " ".join(keys)[:cap]
                        break
                break
        if val.type == "object":
            keys = []
            for prop in val.children:
                if prop.type == "pair":
                    k = prop.child_by_field_name("key")
                    if k is not None:
                        keys.append(
                            src[k.start_byte:k.end_byte]
                            .decode(errors="replace")
                            .strip("\"'`")
                        )
            if keys:
                return " ".join(keys)[:cap]
        if val.type == "array":
            # Pull identifier tokens from the first ~5 elements for a sniff.
            tokens: list[str] = []
            for elem in val.children[:20]:
                if elem.type == "object":
                    for prop in elem.children:
                        if prop.type == "pair":
                            k = prop.child_by_field_name("key")
                            if k is not None:
                                tokens.append(
                                    src[k.start_byte:k.end_byte]
                                    .decode(errors="replace")
                                    .strip("\"'`")
                                )
            if tokens:
                return " ".join(tokens)[:cap]
    except Exception:
        pass
    raw = src[val.start_byte:val.end_byte].decode(errors="replace")
    return raw[:cap]


def _build_variable_node(
    vd: Node,
    val: Node,
    raw_child: Node,
    src: bytes,
    file_posix: str,
    file_classification: str | None,
) -> dict[str, Any] | None:
    """Build a ``Variable`` node for a top-level ``const NAME = <non-arrow RHS>``.

    Returns None for unnameable, non-canonical, or unsupported RHS shapes.
    """
    name_node = vd.child_by_field_name("name")
    if name_node is None:
        return None
    name = src[name_node.start_byte:name_node.end_byte].decode(errors="replace")
    if not _variable_name_is_canonical(name):
        return None
    if val.type not in _VARIABLE_VALUE_TYPES:
        return None

    is_exported, _ = _is_exported(vd)
    docstring = _extract_jsdoc(raw_child, src)

    # Signature: identifier + (optional) type annotation + RHS-kind hint.
    type_ann_node = vd.child_by_field_name("type")
    type_ann = (
        src[type_ann_node.start_byte:type_ann_node.end_byte].decode(errors="replace")
        if type_ann_node is not None
        else ""
    )
    rhs_kind = {
        "new_expression": "new",
        "object": "object literal",
        "array": "array literal",
        "call_expression": "factory call",
    }.get(val.type, val.type)
    signature = f"const {name}{type_ann} = <{rhs_kind}>"

    rhs_summary = _summarize_variable_value(val, src)
    embed_parts = [name, _camel_to_readable(name), signature]
    if docstring:
        embed_parts.append(docstring)
    if rhs_summary:
        embed_parts.append(rhs_summary)
    embed_text = " | ".join(p for p in embed_parts if p)

    source_code = src[vd.start_byte:vd.end_byte].decode(errors="replace")
    return {
        "node_id": _make_node_id(file_posix, name),
        "node_type": "Variable",
        "name": name,
        "file_path": file_posix,
        "file_classification": file_classification,
        "route_path": None,
        "signature": signature,
        "docstring": docstring,
        "internal_logic_abstraction": None,
        "source_code": source_code,
        "start_line": vd.start_point[0] + 1,
        "end_line": vd.end_point[0] + 1,
        "is_exported": is_exported,
        "embed_text": embed_text,
    }


def _build_class_nodes(
    decl: Node,
    raw_child: Node,
    src: bytes,
    file_posix: str,
    file_classification: str | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    name_node = decl.child_by_field_name("name")
    if name_node is None:
        return None, []
    name = src[name_node.start_byte:name_node.end_byte].decode(errors="replace")
    is_exported, _ = _is_exported(decl)
    docstring = _extract_jsdoc(raw_child, src)

    # Extract heritage
    heritage = decl.child_by_field_name("body")
    heritage_node = None
    for c in decl.children:
        if c.type == "class_heritage":
            heritage_node = c
            break

    extends_name: str | None = None
    implements_names: list[str] = []
    if heritage_node:
        for c in heritage_node.children:
            if c.type == "extends_clause":
                for ic in c.children:
                    if ic.type in ("type_identifier", "identifier"):
                        extends_name = src[ic.start_byte:ic.end_byte].decode(errors="replace")
                        break
                    if ic.type == "member_expression":
                        extends_name = src[ic.start_byte:ic.end_byte].decode(errors="replace")
                        break
            elif c.type == "implements_clause":
                for ic in c.children:
                    if ic.type in ("type_identifier", "identifier", "generic_type"):
                        type_id = ic.child_by_field_name("name") if ic.type == "generic_type" else ic
                        if type_id:
                            implements_names.append(
                                src[type_id.start_byte:type_id.end_byte].decode(errors="replace")
                            )

    source_code = src[decl.start_byte:decl.end_byte].decode(errors="replace")
    # Signature: class name [extends X] [implements Y, Z]
    sig_parts = [f"class {name}"]
    if extends_name:
        sig_parts.append(f"extends {extends_name}")
    if implements_names:
        sig_parts.append(f"implements {', '.join(implements_names)}")
    signature = " ".join(sig_parts)

    embed_text = compose_embed_text({
        "name": name, "docstring": docstring, "signature": signature,
    })

    class_node_id = _make_node_id(file_posix, name)
    class_node: dict[str, Any] = {
        "node_id": class_node_id,
        "node_type": "Class",
        "name": name,
        "file_path": file_posix,
        "file_classification": file_classification,
        "route_path": None,
        "signature": signature,
        "docstring": docstring,
        "internal_logic_abstraction": None,
        "source_code": source_code,
        "start_line": decl.start_point[0] + 1,
        "end_line": decl.end_point[0] + 1,
        "is_exported": is_exported,
        "embed_text": embed_text,
        "extends_name": extends_name,
        "implements_names": implements_names,
    }

    # --- Methods ---
    method_nodes: list[dict[str, Any]] = []
    body_node = decl.child_by_field_name("body")
    if body_node:
        for i, mc in enumerate(body_node.children):
            if mc.type == "method_definition":
                m_node = _build_method_node(mc, body_node, i, src, file_posix, class_node_id, file_classification)
                if m_node:
                    method_nodes.append(m_node)

    return class_node, method_nodes


def _build_method_node(
    decl: Node,
    parent_body: Node,
    idx: int,
    src: bytes,
    file_posix: str,
    class_node_id: str,
    file_classification: str | None,
) -> dict[str, Any] | None:
    name_node = decl.child_by_field_name("name")
    if name_node is None:
        return None
    name = src[name_node.start_byte:name_node.end_byte].decode(errors="replace")
    # Skip computed property names (e.g., [Symbol.iterator])
    if name.startswith("["):
        return None

    # Preceding sibling comment
    docstring = None
    if idx > 0:
        prev = parent_body.children[idx - 1]
        if prev.type == "comment":
            text = src[prev.start_byte:prev.end_byte].decode(errors="replace").strip()
            if text.startswith("/**"):
                inner = text[3:-2] if text.endswith("*/") else text[3:]
                lines = [ln.strip().lstrip("*").strip() for ln in inner.splitlines()]
                docstring = " ".join(ln for ln in lines if ln)

    signature = _method_signature(decl, name, src)
    embed_text = compose_embed_text({
        "name": name, "docstring": docstring, "signature": signature,
    })

    body_node = decl.child_by_field_name("body")
    skeleton = None
    if body_node is not None:
        try:
            skeleton = skeletonize_node(decl, src)
        except Exception:
            skeleton = None

    source_code = src[decl.start_byte:decl.end_byte].decode(errors="replace")

    # Qualified method id: ClassName.methodName
    class_name = class_node_id.split("::")[-1]
    method_node_id = _make_node_id(file_posix, f"{class_name}.{name}")

    return {
        "node_id": method_node_id,
        "node_type": "Method",
        "name": name,
        "file_path": file_posix,
        "file_classification": file_classification,
        "route_path": None,
        "signature": signature,
        "docstring": docstring,
        "internal_logic_abstraction": skeleton,
        "source_code": source_code,
        "start_line": decl.start_point[0] + 1,
        "end_line": decl.end_point[0] + 1,
        "is_exported": True,  # methods inherit class export status
        "embed_text": embed_text,
        "parent_class_id": class_node_id,
    }


def _build_interface_nodes(
    decl: Node,
    raw_child: Node,
    src: bytes,
    file_posix: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    name_node = decl.child_by_field_name("name")
    if name_node is None:
        return None, []
    name = src[name_node.start_byte:name_node.end_byte].decode(errors="replace")
    is_exported, _ = _is_exported(decl)
    docstring = _extract_jsdoc(raw_child, src)

    # Collect extends
    extends_names: list[str] = []
    for c in decl.children:
        if c.type == "extends_type_clause":
            for ic in c.children:
                if ic.type in ("type_identifier", "identifier"):
                    extends_names.append(src[ic.start_byte:ic.end_byte].decode(errors="replace"))

    sig_parts = [f"interface {name}"]
    if extends_names:
        sig_parts.append(f"extends {', '.join(extends_names)}")
    signature = " ".join(sig_parts)

    source_code = src[decl.start_byte:decl.end_byte].decode(errors="replace")
    embed_text = compose_embed_text({
        "name": name, "docstring": docstring, "signature": signature,
    })

    iface_node_id = _make_node_id(file_posix, name)
    iface_node: dict[str, Any] = {
        "node_id": iface_node_id,
        "node_type": "Interface",
        "name": name,
        "file_path": file_posix,
        "file_classification": None,
        "route_path": None,
        "signature": signature,
        "docstring": docstring,
        "internal_logic_abstraction": None,
        "source_code": source_code,
        "start_line": decl.start_point[0] + 1,
        "end_line": decl.end_point[0] + 1,
        "is_exported": is_exported,
        "embed_text": embed_text,
    }

    # --- InterfaceField nodes (one per property_signature in object_type) ---
    field_nodes: list[dict[str, Any]] = []
    body_node = None
    for c in decl.children:
        if c.type == "object_type":
            body_node = c
            break
    if body_node:
        for prop in body_node.children:
            if prop.type == "property_signature":
                prop_name_node = prop.child_by_field_name("name")
                if prop_name_node is None:
                    # Some property_signatures have the name as first identifier child
                    for cc in prop.children:
                        if cc.type in ("property_identifier", "identifier"):
                            prop_name_node = cc
                            break
                if prop_name_node is None:
                    continue
                prop_name = src[prop_name_node.start_byte:prop_name_node.end_byte].decode(errors="replace")
                field_id = _make_node_id(file_posix, f"{name}.{prop_name}")
                field_nodes.append({
                    "node_id": field_id,
                    "node_type": "InterfaceField",
                    "name": prop_name,
                    "file_path": file_posix,
                    "file_classification": None,
                    "route_path": None,
                    "signature": src[prop.start_byte:prop.end_byte].decode(errors="replace"),
                    "docstring": None,
                    "internal_logic_abstraction": None,
                    "source_code": None,
                    "start_line": prop.start_point[0] + 1,
                    "end_line": prop.end_point[0] + 1,
                    "is_exported": False,
                    "embed_text": "",  # Degenerate by blueprint §3.2
                    "parent_interface_id": iface_node_id,
                })

    return iface_node, field_nodes


def _build_type_alias_nodes(
    decl: Node,
    raw_child: Node,
    src: bytes,
    file_posix: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    name_node = decl.child_by_field_name("name")
    if name_node is None:
        return None, []
    name = src[name_node.start_byte:name_node.end_byte].decode(errors="replace")
    is_exported, _ = _is_exported(decl)
    docstring = _extract_jsdoc(raw_child, src)
    source_code = src[decl.start_byte:decl.end_byte].decode(errors="replace")
    signature = f"type {name} = ..."
    embed_text = compose_embed_text({
        "name": name, "docstring": docstring, "signature": signature,
    })

    ta_node_id = _make_node_id(file_posix, name)
    ta_node: dict[str, Any] = {
        "node_id": ta_node_id,
        "node_type": "TypeAlias",
        "name": name,
        "file_path": file_posix,
        "file_classification": None,
        "route_path": None,
        "signature": signature,
        "docstring": docstring,
        "internal_logic_abstraction": None,
        "source_code": source_code,
        "start_line": decl.start_point[0] + 1,
        "end_line": decl.end_point[0] + 1,
        "is_exported": is_exported,
        "embed_text": embed_text,
    }

    # InterfaceField children for object-shape TypeAliases
    field_nodes: list[dict[str, Any]] = []
    # Find the value of the type alias — if it's an object_type, emit fields
    value_node = None
    for c in decl.children:
        if c.type == "object_type":
            value_node = c
            break
    if value_node:
        for prop in value_node.children:
            if prop.type == "property_signature":
                prop_name_node = prop.child_by_field_name("name")
                if prop_name_node is None:
                    for cc in prop.children:
                        if cc.type in ("property_identifier", "identifier"):
                            prop_name_node = cc
                            break
                if prop_name_node is None:
                    continue
                prop_name = src[prop_name_node.start_byte:prop_name_node.end_byte].decode(errors="replace")
                field_id = _make_node_id(file_posix, f"{name}.{prop_name}")
                field_nodes.append({
                    "node_id": field_id,
                    "node_type": "InterfaceField",
                    "name": prop_name,
                    "file_path": file_posix,
                    "file_classification": None,
                    "route_path": None,
                    "signature": src[prop.start_byte:prop.end_byte].decode(errors="replace"),
                    "docstring": None,
                    "internal_logic_abstraction": None,
                    "source_code": None,
                    "start_line": prop.start_point[0] + 1,
                    "end_line": prop.end_point[0] + 1,
                    "is_exported": False,
                    "embed_text": "",
                    "parent_interface_id": ta_node_id,
                })

    return ta_node, field_nodes


def _build_enum_node(
    decl: Node,
    raw_child: Node,
    src: bytes,
    file_posix: str,
) -> dict[str, Any] | None:
    name_node = decl.child_by_field_name("name")
    if name_node is None:
        return None
    name = src[name_node.start_byte:name_node.end_byte].decode(errors="replace")
    is_exported, _ = _is_exported(decl)
    docstring = _extract_jsdoc(raw_child, src)
    source_code = src[decl.start_byte:decl.end_byte].decode(errors="replace")
    signature = f"enum {name}"
    embed_text = compose_embed_text({
        "name": name, "docstring": docstring, "signature": signature,
    })

    return {
        "node_id": _make_node_id(file_posix, name),
        "node_type": "Enum",
        "name": name,
        "file_path": file_posix,
        "file_classification": None,
        "route_path": None,
        "signature": signature,
        "docstring": docstring,
        "internal_logic_abstraction": None,
        "source_code": source_code,
        "start_line": decl.start_point[0] + 1,
        "end_line": decl.end_point[0] + 1,
        "is_exported": is_exported,
        "embed_text": embed_text,
    }


# ---------------------------------------------------------------------------
# SQLite insertion
# ---------------------------------------------------------------------------

def _insert_nodes(nodes: list[dict[str, Any]], conn: sqlite3.Connection) -> None:
    """INSERT OR REPLACE all node dicts into code_nodes."""
    sql = """
        INSERT OR REPLACE INTO code_nodes (
            node_id, node_type, name, file_path, file_classification,
            route_path, signature, docstring, internal_logic_abstraction,
            source_code, embed_text, line_start, line_end, exported,
            client_directive
        ) VALUES (
            :node_id, :node_type, :name, :file_path, :file_classification,
            :route_path, :signature, :docstring, :internal_logic_abstraction,
            :source_code, :embed_text, :line_start, :line_end, :exported,
            :client_directive
        )
    """
    for n in nodes:
        row = {
            "node_id": n["node_id"],
            "node_type": n["node_type"],
            "name": n["name"],
            "file_path": n["file_path"],
            "file_classification": n.get("file_classification"),
            "route_path": n.get("route_path"),
            "signature": n.get("signature"),
            "docstring": n.get("docstring"),
            "internal_logic_abstraction": n.get("internal_logic_abstraction"),
            "source_code": n.get("source_code"),
            "embed_text": n.get("embed_text", ""),
            "line_start": n.get("start_line", 1),
            "line_end": n.get("end_line", 1),
            "exported": 1 if n.get("is_exported") else 0,
            "client_directive": n.get("client_directive"),
        }
        conn.execute(sql, row)
    conn.commit()


# ---------------------------------------------------------------------------
# Pass 2: helpers
# ---------------------------------------------------------------------------

# Regex for CLIENT_API_CALLS: matches /api/... path strings
_API_PATH_RE = re.compile(r"^/api/[^\s\"'`?#]+")
# Template literal expression placeholder
_TEMPLATE_EXPR_RE = re.compile(r"\$\{[^}]*\}")
# Dynamic segment :param or [param] -> [id]
_DYN_SEG_RE = re.compile(r":[a-zA-Z_][a-zA-Z0-9_]*|\[[^\]]+\]")


def _resolve_rel_import(
    specifier: str,
    file_posix: str,
    known_node_ids: set[str],
) -> str | None:
    """Resolve a relative import specifier to a File node_id.

    Tries extensions .ts, .tsx, /index.ts, /index.tsx in order.
    Returns None if the resolved file is not in known_node_ids.
    Blueprint §3.4.
    """
    # file_posix is e.g. "src/components/home/HomePage.tsx"
    base_dir = file_posix.rsplit("/", 1)[0] if "/" in file_posix else ""

    # Normalize: resolve ./ and ../
    parts = (base_dir + "/" + specifier).split("/")
    resolved: list[str] = []
    for p in parts:
        if p == "..":
            if resolved:
                resolved.pop()
        elif p and p != ".":
            resolved.append(p)
    base = "/".join(resolved)

    # Try exact, then with extensions
    candidates = [
        base,
        base + ".ts",
        base + ".tsx",
        base + "/index.ts",
        base + "/index.tsx",
    ]
    for c in candidates:
        if c in known_node_ids:
            return c
    return None


def resolve_call_target(
    callee_name: str,
    import_map: dict[str, str],
    file_posix: str,
    known_node_ids: set[str],
) -> str | None:
    """Resolve a bare function/variable name to a node_id.

    Checks import_map first (cross-file), then same-file scope.
    Blueprint §3.4 CALLS rule.
    """
    # Cross-file via import_map
    if callee_name in import_map:
        target = import_map[callee_name]
        if target in known_node_ids:
            return target
    # Same-file: try <file>::<name>
    same_file = f"{file_posix}::{callee_name}"
    if same_file in known_node_ids:
        return same_file
    return None


_HTTP_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")


def _route_segments_match(url_segs: list[str], disk_segs: list[str]) -> bool:
    """Return True if disk route path segments positionally match url segments.

    A disk segment like ``[paramName]`` (any bracket-wrapped name) is treated
    as a wildcard and matches any single url segment.  Literal segments must
    match exactly (case-sensitive).
    """
    if len(url_segs) != len(disk_segs):
        return False
    for u, d in zip(url_segs, disk_segs):
        if d.startswith("[") and d.endswith("]"):
            continue  # wildcard — matches any value
        if u != d:
            return False
    return True


# Pre-built per-call-site cache: maps normalized segment lists to file_ids.
# This is intentionally NOT a module-level cache — it is rebuilt for each
# extract_edges() invocation which is cheap (O(n_routes)) and correct for
# incremental re-index scenarios.

def resolve_api_route(
    raw_path: str,
    known_node_ids: set[str],
) -> list[str]:
    """Resolve a /api/... string to API_ROUTE node_id(s).

    Prefers HTTP-method Function nodes (route.ts::GET, route.ts::POST) when
    they exist; falls back to the File node.  Returns [] when unresolvable.

    Uses positional wildcard matching: a disk segment ``[paramName]`` matches
    any single URL segment; selects the most-specific match (fewest wildcards).

    Blueprint §3.4 CLIENT_API_CALLS.
    """
    # Strip query string / fragment
    path = raw_path.split("?")[0].split("#")[0]
    # Replace ${...} template expressions with a placeholder segment so we can
    # split cleanly.  We do NOT collapse [param] → [id] here.
    path = _TEMPLATE_EXPR_RE.sub("[id]", path)

    path_no_slash = path.lstrip("/")          # e.g. "api/proposal/[id]/respond"
    url_segs = path_no_slash.split("/")

    # Collect candidate route file_ids from known_node_ids.
    # A route file looks like "src/app/.../route.ts" or "src/pages/...".
    # We pre-filter by segment count to keep the inner loop fast.
    n = len(url_segs)

    best_file_id: str | None = None
    best_wildcards = n + 1  # lower is better (more specific match)

    for nid in known_node_ids:
        # Only consider File nodes (no "::" separator)
        if "::" in nid:
            continue
        # Must be a route file under src/app/ or src/pages/api/
        if not (nid.startswith("src/app/") or nid.startswith("src/pages/")):
            continue
        if not (nid.endswith("/route.ts") or nid.endswith("/route.tsx")
                or (nid.startswith("src/pages/") and (nid.endswith(".ts") or nid.endswith(".tsx")))):
            continue

        # Build disk segments: strip "src/app/" prefix + "/route.ts" suffix
        if nid.startswith("src/app/"):
            disk_path = nid[len("src/app/"):]              # "api/.../route.ts"
            if disk_path.endswith("/route.ts"):
                disk_path = disk_path[:-len("/route.ts")]  # "api/..."
            elif disk_path.endswith("/route.tsx"):
                disk_path = disk_path[:-len("/route.tsx")]
            else:
                continue  # not a route.ts file under src/app/
        else:
            # src/pages/api/...
            disk_path = nid[len("src/pages/"):]            # "api/....ts"
            disk_path = disk_path.rsplit(".", 1)[0]        # strip .ts/.tsx

        disk_segs = disk_path.split("/")

        if not _route_segments_match(url_segs, disk_segs):
            continue

        # Count wildcards in disk route to prefer most-specific match
        wildcards = sum(1 for s in disk_segs if s.startswith("[") and s.endswith("]"))
        if wildcards < best_wildcards:
            best_wildcards = wildcards
            best_file_id = nid

    if best_file_id is None:
        return []

    # Fan out to HTTP-method Function nodes when present.
    fn_targets = [
        f"{best_file_id}::{m}" for m in _HTTP_METHODS
        if f"{best_file_id}::{m}" in known_node_ids
    ]
    if fn_targets:
        return fn_targets
    return [best_file_id]


def _emit_edge(
    source_id: str,
    target_id: str,
    edge_type: str,
    conn: sqlite3.Connection,
    counter: list[int],
) -> None:
    """INSERT OR IGNORE one structural edge. Increments counter[0]."""
    conn.execute(
        "INSERT OR IGNORE INTO structural_edges (source_id, target_id, edge_type) "
        "VALUES (?, ?, ?)",
        (source_id, target_id, edge_type),
    )
    counter[0] += 1


def _extract_string_specifier(node: Node, src: bytes) -> str | None:
    """Return the string value from a `string` AST node.

    Prefers the `string_fragment` child (avoids quote stripping edge cases).
    """
    for c in node.children:
        if c.type == "string_fragment":
            return src[c.start_byte:c.end_byte].decode(errors="replace")
    # Fallback: strip surrounding quotes from raw text
    raw = src[node.start_byte:node.end_byte].decode(errors="replace")
    return raw.strip("'\"")


def _resolve_alias_import(
    specifier: str,
    known_node_ids: set[str],
) -> str | None:
    """Resolve a TypeScript @/ path-alias specifier to a File node_id.

    @/ maps to src/ by tsconfig.json paths convention in citrakara.
    Probes .ts / .tsx / /index.ts / /index.tsx in order.
    Returns None if no indexed node matches (falls through to DEPENDS_ON_EXTERNAL).
    Blueprint §3.4 — alias imports are intra-repo and must resolve to real IMPORTS edges.
    """
    # @/lib/services/auth.service -> src/lib/services/auth.service
    base = "src/" + specifier[len("@/"):]
    candidates = [
        base,
        base + ".ts",
        base + ".tsx",
        base + "/index.ts",
        base + "/index.tsx",
    ]
    for c in candidates:
        if c in known_node_ids:
            return c
    return None


def _build_import_map(
    root: Node,
    src: bytes,
    file_posix: str,
    known_node_ids: set[str],
    conn: sqlite3.Connection,
    counter: list[int],
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Parse all import_statement nodes.

    Returns:
        import_map: local_name -> target node_id (for cross-file resolution)
        dep_pairs: list of (dependent_file, target_file) for file_dependencies
    Blueprint §3.4 steps 1-2.

    @/ path aliases (TypeScript tsconfig paths) are treated as intra-repo imports:
    resolved to src/ File nodes and emitted as IMPORTS edges (not DEPENDS_ON_EXTERNAL).
    This ensures the full CIA chain API_ROUTE -> service -> repository is traversable by BFS.
    """
    import_map: dict[str, str] = {}
    dep_pairs: list[tuple[str, str]] = []

    for child in root.children:
        stmt = child
        # tree-sitter TypeScript uses `import_statement`, not `import_declaration`
        if child.type not in ("import_statement", "export_statement"):
            continue

        # Find the string specifier node
        specifier_node = None
        for c in stmt.children:
            if c.type == "string":
                specifier_node = c
                break
        if specifier_node is None:
            continue

        specifier = _extract_string_specifier(specifier_node, src)
        if specifier is None:
            continue

        if specifier.startswith("."):
            # Relative import -> IMPORTS edge + file_dependencies
            target_file_id = _resolve_rel_import(specifier, file_posix, known_node_ids)
            if target_file_id is not None:
                _emit_edge(file_posix, target_file_id, "IMPORTS", conn, counter)
                dep_pairs.append((file_posix, target_file_id))
                # Map imported names to their target node_ids
                _map_import_names(stmt, src, target_file_id, import_map, known_node_ids)
        elif specifier.startswith("@/"):
            # TypeScript path alias (@/ -> src/) — intra-repo, resolve as IMPORTS
            target_file_id = _resolve_alias_import(specifier, known_node_ids)
            if target_file_id is not None:
                _emit_edge(file_posix, target_file_id, "IMPORTS", conn, counter)
                dep_pairs.append((file_posix, target_file_id))
                _map_import_names(stmt, src, target_file_id, import_map, known_node_ids)
            else:
                # Alias not resolvable (e.g. generated type file) -> DEPENDS_ON_EXTERNAL
                pkg_id = f"ext::{specifier}"
                if pkg_id in known_node_ids:
                    _emit_edge(file_posix, pkg_id, "DEPENDS_ON_EXTERNAL", conn, counter)
        else:
            # Non-relative, non-alias -> DEPENDS_ON_EXTERNAL edge (real npm package)
            pkg_id = f"ext::{specifier}"
            if pkg_id in known_node_ids:
                _emit_edge(file_posix, pkg_id, "DEPENDS_ON_EXTERNAL", conn, counter)
            # Also map namespace imports: import * as X from 'pkg'
            _map_namespace_import(stmt, src, pkg_id, import_map, known_node_ids)

    return import_map, dep_pairs


def _map_import_names(
    stmt: Node,
    src: bytes,
    target_file_id: str,
    import_map: dict[str, str],
    known_node_ids: set[str],
) -> None:
    """Populate import_map with {local_name -> target_node_id} from an import stmt."""
    for c in stmt.children:
        if c.type == "import_clause":
            _parse_import_clause(c, src, target_file_id, import_map, known_node_ids)
        elif c.type == "namespace_import":
            # import * as X from '...' — top-level (some tree-sitter versions)
            for nc in c.children:
                if nc.type == "identifier":
                    name = src[nc.start_byte:nc.end_byte].decode(errors="replace")
                    if target_file_id in known_node_ids:
                        import_map[name] = target_file_id
        elif c.type == "named_exports":
            # re-export clause: extract specifier names
            for spec in c.children:
                if spec.type == "export_specifier":
                    name_node = spec.children[0] if spec.children else None
                    if name_node:
                        name = src[name_node.start_byte:name_node.end_byte].decode(errors="replace")
                        candidate = f"{target_file_id}::{name}"
                        if candidate in known_node_ids:
                            import_map[name] = candidate


def _parse_import_clause(
    clause: Node,
    src: bytes,
    target_file_id: str,
    import_map: dict[str, str],
    known_node_ids: set[str],
) -> None:
    """Handle default, named, and namespace imports inside an import_clause."""
    for c in clause.children:
        if c.type in ("identifier", "type_identifier"):
            # Default import: import Foo from './foo'
            name = src[c.start_byte:c.end_byte].decode(errors="replace")
            # Target is the file node itself for default imports
            if target_file_id in known_node_ids:
                import_map[name] = target_file_id
            # Also try same-named export in target file
            fn_candidate = f"{target_file_id}::{name}"
            if fn_candidate in known_node_ids:
                import_map[name] = fn_candidate
        elif c.type == "named_imports":
            for spec in c.children:
                if spec.type == "import_specifier":
                    # import { A as B } from './x' — use original name for lookup
                    orig_node = spec.children[0] if spec.children else None
                    alias_node = spec.children[-1] if len(spec.children) > 1 else orig_node
                    if orig_node:
                        orig = src[orig_node.start_byte:orig_node.end_byte].decode(errors="replace")
                        alias = src[alias_node.start_byte:alias_node.end_byte].decode(errors="replace") if alias_node else orig
                        candidate = f"{target_file_id}::{orig}"
                        if candidate in known_node_ids:
                            import_map[alias] = candidate
        elif c.type == "namespace_import":
            # import * as X from './y'
            for nc in c.children:
                if nc.type == "identifier":
                    name = src[nc.start_byte:nc.end_byte].decode(errors="replace")
                    if target_file_id in known_node_ids:
                        import_map[name] = target_file_id


def _map_namespace_import(
    stmt: Node,
    src: bytes,
    pkg_id: str,
    import_map: dict[str, str],
    known_node_ids: set[str],
) -> None:
    """Map `import * as X from 'pkg'` -> import_map[X] = pkg_id."""
    for c in stmt.children:
        if c.type == "import_clause":
            for ic in c.children:
                if ic.type == "namespace_import":
                    for nc in ic.children:
                        if nc.type == "identifier":
                            name = src[nc.start_byte:nc.end_byte].decode(errors="replace")
                            if pkg_id in known_node_ids:
                                import_map[name] = pkg_id


def _emit_class_edges(
    root: Node,
    src: bytes,
    file_posix: str,
    import_map: dict[str, str],
    known_node_ids: set[str],
    conn: sqlite3.Connection,
    counter: list[int],
) -> None:
    """Emit INHERITS, IMPLEMENTS, DEFINES_METHOD for each class.
    Blueprint §3.4 step 3.
    """
    for child in root.children:
        decl = child
        if child.type == "export_statement":
            inner = _unwrap_export(child)
            if inner is not None:
                decl = inner

        if decl.type != "class_declaration":
            continue

        name_node = decl.child_by_field_name("name")
        if name_node is None:
            continue
        class_name = src[name_node.start_byte:name_node.end_byte].decode(errors="replace")
        class_id = f"{file_posix}::{class_name}"

        if class_id not in known_node_ids:
            continue

        # Heritage clause
        for c in decl.children:
            if c.type == "class_heritage":
                for hc in c.children:
                    if hc.type == "extends_clause":
                        # INHERITS — one parent
                        for ic in hc.children:
                            if ic.type in ("type_identifier", "identifier"):
                                parent_name = src[ic.start_byte:ic.end_byte].decode(errors="replace")
                                target = resolve_call_target(parent_name, import_map, file_posix, known_node_ids)
                                if target:
                                    _emit_edge(class_id, target, "INHERITS", conn, counter)
                                break
                    elif hc.type == "implements_clause":
                        # IMPLEMENTS — possibly multiple
                        for ic in hc.children:
                            if ic.type in ("type_identifier", "identifier"):
                                iface_name = src[ic.start_byte:ic.end_byte].decode(errors="replace")
                                target = resolve_call_target(iface_name, import_map, file_posix, known_node_ids)
                                if target:
                                    _emit_edge(class_id, target, "IMPLEMENTS", conn, counter)
                            elif ic.type == "generic_type":
                                name_n = ic.child_by_field_name("name")
                                if name_n:
                                    iface_name = src[name_n.start_byte:name_n.end_byte].decode(errors="replace")
                                    target = resolve_call_target(iface_name, import_map, file_posix, known_node_ids)
                                    if target:
                                        _emit_edge(class_id, target, "IMPLEMENTS", conn, counter)

        # DEFINES_METHOD: class -> each method
        body_node = decl.child_by_field_name("body")
        if body_node:
            for mc in body_node.children:
                if mc.type == "method_definition":
                    mn = mc.child_by_field_name("name")
                    if mn:
                        method_name = src[mn.start_byte:mn.end_byte].decode(errors="replace")
                        method_id = f"{file_posix}::{class_name}.{method_name}"
                        if method_id in known_node_ids:
                            _emit_edge(class_id, method_id, "DEFINES_METHOD", conn, counter)


def _collect_type_refs(type_node: Node, src: bytes) -> list[str]:
    """Collect all type_identifier names from a type annotation node.

    Handles plain types, generic types (Array<T>), union/intersection.
    Blueprint §3.4 TYPED_BY rule.
    """
    names: list[str] = []
    _collect_type_refs_recursive(type_node, src, names)
    return names


def _collect_type_refs_recursive(node: Node, src: bytes, out: list[str]) -> None:
    if node.type == "type_identifier":
        out.append(src[node.start_byte:node.end_byte].decode(errors="replace"))
    elif node.type == "generic_type":
        # Generic: Array<T> -> emit name + type args
        name_n = node.child_by_field_name("name")
        if name_n:
            out.append(src[name_n.start_byte:name_n.end_byte].decode(errors="replace"))
        args = node.child_by_field_name("type_arguments")
        if args:
            for arg in args.children:
                _collect_type_refs_recursive(arg, src, out)
    else:
        for c in node.children:
            _collect_type_refs_recursive(c, src, out)


def _is_hook_dep_array(node: Node, src: bytes) -> bool:
    """Return True if node is the dep-array argument of a React hook call.
    Blueprint §3.3 (skeletonizer) and §3.4 HOOK_DEPENDS_ON.
    """
    from impactracer.shared.constants import HOOK_NAMES
    parent = node.parent
    if parent is None or parent.type != "arguments":
        return False
    call = parent.parent
    if call is None or call.type != "call_expression":
        return False
    fn = call.child_by_field_name("function")
    if fn is None:
        return False
    fn_text = src[fn.start_byte:fn.end_byte].decode(errors="replace")
    return fn_text in HOOK_NAMES


def _extract_string_value(node: Node, src: bytes) -> str | None:
    """Return the string value of a string or template_string literal, or None."""
    if node.type == "string":
        # Prefer string_fragment child (avoids quote-stripping issues)
        for c in node.children:
            if c.type == "string_fragment":
                return src[c.start_byte:c.end_byte].decode(errors="replace")
        raw = src[node.start_byte:node.end_byte].decode(errors="replace")
        return raw.strip("'\"")
    if node.type == "template_string":
        # Reconstruct by walking byte gaps between children.
        # tree-sitter may NOT emit template_chars as named nodes;
        # the literal text appears in gaps between child nodes.
        parts: list[str] = []
        cursor = node.start_byte
        for c in node.children:
            gap = src[cursor:c.start_byte]
            if gap and gap not in (b"`",):
                parts.append(gap.decode(errors="replace"))
            if c.type == "template_substitution":
                parts.append("[id]")
            elif c.type == "template_chars":
                parts.append(src[c.start_byte:c.end_byte].decode(errors="replace"))
            cursor = c.end_byte
        # trailing gap (closing backtick already excluded by not appending '`')
        return "".join(parts)
    return None


def _get_root_identifier(node: Node, src: bytes) -> str | None:
    """Return the root-level identifier of a call expression's function.

    For `foo(...)` -> 'foo'
    For `foo.bar(...)` -> 'foo'
    For `await foo(...)` -> recurse into foo(...)
    """
    if node.type in ("identifier", "type_identifier"):
        return src[node.start_byte:node.end_byte].decode(errors="replace")
    if node.type == "member_expression":
        obj = node.child_by_field_name("object")
        if obj:
            return _get_root_identifier(obj, src)
    if node.type == "call_expression":
        fn = node.child_by_field_name("function")
        if fn:
            return _get_root_identifier(fn, src)
    return None


def _get_direct_name(node: Node, src: bytes) -> str | None:
    """Return the direct name of a call/identifier (not the root, the whole chain)."""
    if node.type in ("identifier", "type_identifier"):
        return src[node.start_byte:node.end_byte].decode(errors="replace")
    if node.type == "member_expression":
        prop = node.child_by_field_name("property")
        if prop:
            return src[prop.start_byte:prop.end_byte].decode(errors="replace")
    return None


def _emit_body_edges(
    body_root: Node,
    src: bytes,
    source_id: str,
    file_posix: str,
    import_map: dict[str, str],
    known_node_ids: set[str],
    conn: sqlite3.Connection,
    counter: list[int],
) -> None:
    """Recursively walk a function/method body emitting all body-level edges.

    Handles: CALLS, TYPED_BY, FIELDS_ACCESSED, RENDERS, PASSES_CALLBACK,
             HOOK_DEPENDS_ON, DYNAMIC_IMPORT, CLIENT_API_CALLS.
    Blueprint §3.4 step 4.
    """
    from impactracer.shared.constants import BUILTIN_PATTERNS, PRIMITIVE_TYPES, HOOK_NAMES

    _walk_body(
        body_root, src, source_id, file_posix,
        import_map, known_node_ids, conn, counter,
        BUILTIN_PATTERNS, PRIMITIVE_TYPES, HOOK_NAMES,
    )


def _walk_body(
    node: Node,
    src: bytes,
    source_id: str,
    file_posix: str,
    import_map: dict[str, str],
    known_node_ids: set[str],
    conn: sqlite3.Connection,
    counter: list[int],
    builtins: frozenset[str],
    primitives: frozenset[str],
    hook_names: frozenset[str],
) -> None:
    """Recursive visitor for body-level edge types."""

    if node.type == "call_expression":
        fn_node = node.child_by_field_name("function")
        fn_text = src[fn_node.start_byte:fn_node.end_byte].decode(errors="replace") if fn_node else ""

        # --- DYNAMIC_IMPORT: dynamic(() => import('./X')) or React.lazy(() => import('./X')) ---
        if fn_text in ("dynamic", "React.lazy") or fn_text.endswith(".lazy"):
            args = node.child_by_field_name("arguments")
            if args:
                for arg in args.children:
                    _find_dynamic_imports(arg, src, source_id, file_posix, import_map, known_node_ids, conn, counter)

        # --- CLIENT_API_CALLS: scan fetch() or axiosClient.X() args ---
        # Must run BEFORE the builtins check (fetch is a builtin but still relevant here)
        fn_node2 = fn_node
        if fn_node2 is not None:
            root_name = _get_root_identifier(fn_node2, src)
            is_http_call = (
                root_name == "fetch"
                or (fn_node2.type == "member_expression" and _is_http_method_call(fn_node2, src))
            )
            if is_http_call:
                args_node = node.child_by_field_name("arguments")
                if args_node:
                    first_arg = _first_real_child(args_node)
                    if first_arg is not None:
                        val = _extract_string_value(first_arg, src)
                        if val and _API_PATH_RE.match(val):
                            for target in resolve_api_route(val, known_node_ids):
                                _emit_edge(source_id, target, "CLIENT_API_CALLS", conn, counter)

        # --- CALLS ---
        if fn_node2 is not None:
            root_id_str = _get_root_identifier(fn_node2, src)
            if root_id_str and root_id_str not in builtins:
                callee_name = src[fn_node2.start_byte:fn_node2.end_byte].decode(errors="replace")
                if fn_node2.type == "member_expression":
                    obj = fn_node2.child_by_field_name("object")
                    obj_name = src[obj.start_byte:obj.end_byte].decode(errors="replace") if obj else ""
                    prop = fn_node2.child_by_field_name("property")
                    prop_name = src[prop.start_byte:prop.end_byte].decode(errors="replace") if prop else ""
                    if obj_name in import_map:
                        ns_target = import_map[obj_name]
                        fn_target = f"{ns_target}::{prop_name}"
                        if fn_target in known_node_ids:
                            _emit_edge(source_id, fn_target, "CALLS", conn, counter)
                    else:
                        target = resolve_call_target(callee_name, import_map, file_posix, known_node_ids)
                        if target:
                            _emit_edge(source_id, target, "CALLS", conn, counter)
                else:
                    target = resolve_call_target(callee_name, import_map, file_posix, known_node_ids)
                    if target:
                        _emit_edge(source_id, target, "CALLS", conn, counter)

        # --- HOOK_DEPENDS_ON: useEffect/useCallback/useMemo/useLayoutEffect dep arrays ---
        if fn_node2 is not None:
            fn_text2 = src[fn_node2.start_byte:fn_node2.end_byte].decode(errors="replace")
            if fn_text2 in hook_names:
                args_node = node.child_by_field_name("arguments")
                if args_node:
                    dep_array = _find_dep_array(args_node)
                    if dep_array is not None:
                        for elem in dep_array.children:
                            if elem.type == "identifier":
                                dep_name = src[elem.start_byte:elem.end_byte].decode(errors="replace")
                                target = resolve_call_target(dep_name, import_map, file_posix, known_node_ids)
                                if target:
                                    _emit_edge(source_id, target, "HOOK_DEPENDS_ON", conn, counter)
                            elif elem.type == "member_expression":
                                obj = elem.child_by_field_name("object")
                                if obj and obj.type == "identifier":
                                    dep_name = src[obj.start_byte:obj.end_byte].decode(errors="replace")
                                    target = resolve_call_target(dep_name, import_map, file_posix, known_node_ids)
                                    if target:
                                        _emit_edge(source_id, target, "HOOK_DEPENDS_ON", conn, counter)

    # --- TYPED_BY: type annotations on parameters / variable declarations ---
    if node.type in ("required_parameter", "optional_parameter",
                     "variable_declarator", "lexical_declaration"):
        type_ann = node.child_by_field_name("type")
        if type_ann:
            for type_name in _collect_type_refs(type_ann, src):
                if type_name not in primitives:
                    target = resolve_call_target(type_name, import_map, file_posix, known_node_ids)
                    if target:
                        _emit_edge(source_id, target, "TYPED_BY", conn, counter)

    # --- FIELDS_ACCESSED: member_expression where object type is known Interface ---
    if node.type == "member_expression":
        obj = node.child_by_field_name("object")
        prop = node.child_by_field_name("property")
        if obj and prop:
            obj_text = src[obj.start_byte:obj.end_byte].decode(errors="replace")
            prop_text = src[prop.start_byte:prop.end_byte].decode(errors="replace")
            # Look up obj in import_map to see if it maps to an Interface
            _try_emit_fields_accessed(
                obj_text, prop_text, source_id, import_map, file_posix,
                known_node_ids, conn, counter,
            )

    # --- JSX elements: RENDERS + PASSES_CALLBACK ---
    if node.type in ("jsx_element", "jsx_self_closing_element"):
        _emit_jsx_edges(node, src, source_id, file_posix, import_map, known_node_ids, conn, counter)
        # Do NOT recurse into JSX children here — handled inside _emit_jsx_edges
        return

    # Recurse
    for child in node.children:
        _walk_body(
            child, src, source_id, file_posix,
            import_map, known_node_ids, conn, counter,
            builtins, primitives, hook_names,
        )


def _is_http_method_call(member_expr: Node, src: bytes) -> bool:
    """Return True if a member_expression ends in .get/.post/.put/.delete/.patch."""
    prop = member_expr.child_by_field_name("property")
    if prop is None:
        return False
    name = src[prop.start_byte:prop.end_byte].decode(errors="replace")
    return name in ("get", "post", "put", "delete", "patch", "request")


def _first_real_child(args_node: Node) -> Node | None:
    """Return the first non-punctuation child of an arguments node."""
    for c in args_node.children:
        if c.type not in ("(", ")", ","):
            return c
    return None


def _find_dep_array(args_node: Node) -> Node | None:
    """Find the last array literal argument in an arguments node."""
    last_array: Node | None = None
    for c in args_node.children:
        if c.type == "array":
            last_array = c
    return last_array


def _find_dynamic_imports(
    node: Node,
    src: bytes,
    source_id: str,
    file_posix: str,
    import_map: dict[str, str],
    known_node_ids: set[str],
    conn: sqlite3.Connection,
    counter: list[int],
) -> None:
    """Recurse to find import() calls and emit DYNAMIC_IMPORT edges."""
    if node.type == "call_expression":
        fn = node.child_by_field_name("function")
        if fn and src[fn.start_byte:fn.end_byte].decode(errors="replace") == "import":
            args = node.child_by_field_name("arguments")
            if args:
                for arg in args.children:
                    if arg.type == "string":
                        specifier = src[arg.start_byte:arg.end_byte].decode(errors="replace").strip("'\"")
                        if specifier.startswith("."):
                            target = _resolve_rel_import(specifier, file_posix, known_node_ids)
                            if target:
                                _emit_edge(source_id, target, "DYNAMIC_IMPORT", conn, counter)
    for child in node.children:
        _find_dynamic_imports(child, src, source_id, file_posix, import_map, known_node_ids, conn, counter)


def _try_emit_fields_accessed(
    obj_name: str,
    prop_name: str,
    source_id: str,
    import_map: dict[str, str],
    file_posix: str,
    known_node_ids: set[str],
    conn: sqlite3.Connection,
    counter: list[int],
) -> None:
    """Emit FIELDS_ACCESSED if obj_name resolves to an Interface and prop is a known field.

    Blueprint §3.4: member_expression where object's annotated type is a known
    Interface and the property exists as an InterfaceField node.
    We approximate by checking if import_map[obj_name] is an Interface node and
    obj_name::prop_name (or the resolved type::prop_name) exists as InterfaceField.
    """
    if obj_name not in import_map:
        return
    iface_id = import_map[obj_name]
    # iface_id might be the Interface node itself, or a File node
    # Try: iface_id is "src/foo.ts::IUser" pattern => check "src/foo.ts::IUser.propName"
    field_candidate = f"{iface_id}.{prop_name}"
    if field_candidate in known_node_ids:
        _emit_edge(source_id, field_candidate, "FIELDS_ACCESSED", conn, counter)


def _emit_jsx_edges(
    jsx_node: Node,
    src: bytes,
    source_id: str,
    file_posix: str,
    import_map: dict[str, str],
    known_node_ids: set[str],
    conn: sqlite3.Connection,
    counter: list[int],
) -> None:
    """Emit RENDERS and PASSES_CALLBACK for a JSX element node.
    Blueprint §3.4.
    """
    # Opening element
    if jsx_node.type == "jsx_element":
        open_elem = jsx_node.child_by_field_name("open_tag")
        attrs_parent = open_elem
    else:
        attrs_parent = jsx_node  # jsx_self_closing_element

    if attrs_parent is None:
        return

    # Tag name
    tag_name_node = None
    for c in (attrs_parent.children if attrs_parent else []):
        if c.type in ("jsx_opening_element", "jsx_self_closing_element"):
            # shouldn't happen at this level, but guard
            break
        if c.type in ("identifier", "jsx_namespace_name", "member_expression"):
            tag_name_node = c
            break
        if c.type == "type_identifier":
            tag_name_node = c
            break

    # For jsx_element, the open_tag is a child
    # tree-sitter TSX: jsx_element -> jsx_opening_element + children + jsx_closing_element
    # jsx_opening_element -> "<" + identifier/member + attributes + ">"
    open_tag = None
    if jsx_node.type == "jsx_element":
        for c in jsx_node.children:
            if c.type == "jsx_opening_element":
                open_tag = c
                break
    else:
        open_tag = jsx_node  # jsx_self_closing_element

    if open_tag is None:
        return

    # Extract tag name from the opening element
    tag_name = None
    for c in open_tag.children:
        if c.type in ("identifier", "type_identifier"):
            tag_name = src[c.start_byte:c.end_byte].decode(errors="replace")
            break
        if c.type == "member_expression":
            tag_name = src[c.start_byte:c.end_byte].decode(errors="replace")
            break

    if tag_name and tag_name[0].isupper():
        # RENDERS: resolve component name
        target = resolve_call_target(tag_name, import_map, file_posix, known_node_ids)
        if target:
            _emit_edge(source_id, target, "RENDERS", conn, counter)

    # PASSES_CALLBACK: scan JSX attributes for onX={handler}
    for c in open_tag.children:
        if c.type == "jsx_attribute":
            _emit_passes_callback(c, src, source_id, file_posix, import_map, known_node_ids, conn, counter)

    # Recurse into JSX children for nested elements
    if jsx_node.type == "jsx_element":
        for child in jsx_node.children:
            if child.type in ("jsx_element", "jsx_self_closing_element"):
                _emit_jsx_edges(child, src, source_id, file_posix, import_map, known_node_ids, conn, counter)
            elif child.type == "jsx_expression":
                # Walk expression content for nested JSX
                for ec in child.children:
                    _walk_jsx_expression(ec, src, source_id, file_posix, import_map, known_node_ids, conn, counter)


def _walk_jsx_expression(
    node: Node,
    src: bytes,
    source_id: str,
    file_posix: str,
    import_map: dict[str, str],
    known_node_ids: set[str],
    conn: sqlite3.Connection,
    counter: list[int],
) -> None:
    """Walk a JSX expression for nested JSX elements."""
    if node.type in ("jsx_element", "jsx_self_closing_element"):
        _emit_jsx_edges(node, src, source_id, file_posix, import_map, known_node_ids, conn, counter)
    else:
        for child in node.children:
            _walk_jsx_expression(child, src, source_id, file_posix, import_map, known_node_ids, conn, counter)


def _emit_passes_callback(
    attr_node: Node,
    src: bytes,
    source_id: str,
    file_posix: str,
    import_map: dict[str, str],
    known_node_ids: set[str],
    conn: sqlite3.Connection,
    counter: list[int],
) -> None:
    """Emit PASSES_CALLBACK / transitive CALLS for JSX attribute `onX={...}`.

    Three handler forms:
      1. ``onX={importedFn}``   → PASSES_CALLBACK to the imported node.
      2. ``onX={localHandler}`` → local arrow, not a known node; walk body for CALLS.
      3. ``onX={() => doFn()}`` → inline arrow; walk body for CALLS.

    Blueprint §3.4.
    """
    from impactracer.shared.constants import BUILTIN_PATTERNS, PRIMITIVE_TYPES, HOOK_NAMES

    # jsx_attribute -> attr_name, "=", jsx_expression | string
    attr_name_node = None
    attr_value_node = None
    for c in attr_node.children:
        if c.type in ("property_identifier", "identifier", "jsx_namespace_name"):
            attr_name_node = c
        elif c.type == "jsx_expression":
            attr_value_node = c

    if attr_name_node is None or attr_value_node is None:
        return

    attr_name = src[attr_name_node.start_byte:attr_name_node.end_byte].decode(errors="replace")
    if not re.match(r"^on[A-Z]", attr_name):
        return

    for c in attr_value_node.children:
        if c.type == "identifier":
            fn_name = src[c.start_byte:c.end_byte].decode(errors="replace")
            target = resolve_call_target(fn_name, import_map, file_posix, known_node_ids)
            if target:
                # Case 1: known imported function → PASSES_CALLBACK
                _emit_edge(source_id, target, "PASSES_CALLBACK", conn, counter)
            # Case 2: local handler identifier — body already walked by _emit_body_edges;
            # transitive CALLS edges are emitted there.  Nothing extra needed here.
            break
        if c.type == "member_expression":
            # e.g. props.onClick -> skip (not a resolvable local ref)
            break
        if c.type in ("arrow_function", "function"):
            # Case 3: inline handler — walk the function body for CALLS/CLIENT_API_CALLS
            fn_body = c.child_by_field_name("body")
            if fn_body is None:
                # For concise arrow: the whole node IS the body
                fn_body = c
            _walk_body(
                fn_body, src, source_id, file_posix,
                import_map, known_node_ids, conn, counter,
                BUILTIN_PATTERNS, PRIMITIVE_TYPES, HOOK_NAMES,
            )
            break


def _get_function_bodies(
    root: Node,
    src: bytes,
    file_posix: str,
    known_node_ids: set[str],
) -> list[tuple[str, Node]]:
    """Return list of (source_node_id, body_node) for all Function/Method nodes in file.

    Used to dispatch body edge walking.
    """
    results: list[tuple[str, Node]] = []
    _collect_fn_bodies(root, src, file_posix, known_node_ids, results)
    return results


def _collect_fn_bodies(
    node: Node,
    src: bytes,
    file_posix: str,
    known_node_ids: set[str],
    results: list[tuple[str, Node]],
    parent_class_name: str | None = None,
) -> None:
    """Recursively collect function/method body nodes with their source node_ids."""
    for child in node.children:
        decl = child
        if child.type == "export_statement":
            inner = _unwrap_export(child)
            if inner is not None:
                decl = inner

        if decl.type == "function_declaration":
            name_n = decl.child_by_field_name("name")
            if name_n:
                name = src[name_n.start_byte:name_n.end_byte].decode(errors="replace")
                node_id = f"{file_posix}::{name}"
                body = decl.child_by_field_name("body")
                if body and node_id in known_node_ids:
                    results.append((node_id, decl))
                    # Also recurse into nested function declarations
                    _collect_nested_fn_bodies(body, src, file_posix, known_node_ids, results)

        elif decl.type == "lexical_declaration":
            for vd in decl.children:
                if vd.type == "variable_declarator":
                    val = vd.child_by_field_name("value")
                    if val and val.type == "arrow_function":
                        name_n = vd.child_by_field_name("name")
                        if name_n:
                            name = src[name_n.start_byte:name_n.end_byte].decode(errors="replace")
                            node_id = f"{file_posix}::{name}"
                            if node_id in known_node_ids:
                                results.append((node_id, val))
                                body = val.child_by_field_name("body")
                                if body:
                                    _collect_nested_fn_bodies(body, src, file_posix, known_node_ids, results)

        elif decl.type == "class_declaration":
            name_n = decl.child_by_field_name("name")
            if name_n:
                class_name = src[name_n.start_byte:name_n.end_byte].decode(errors="replace")
                body_node = decl.child_by_field_name("body")
                if body_node:
                    for mc in body_node.children:
                        if mc.type == "method_definition":
                            mn = mc.child_by_field_name("name")
                            if mn:
                                method_name = src[mn.start_byte:mn.end_byte].decode(errors="replace")
                                method_id = f"{file_posix}::{class_name}.{method_name}"
                                body = mc.child_by_field_name("body")
                                if body and method_id in known_node_ids:
                                    results.append((method_id, mc))


def _collect_nested_fn_bodies(
    body: Node,
    src: bytes,
    file_posix: str,
    known_node_ids: set[str],
    results: list[tuple[str, Node]],
) -> None:
    """Collect any inner arrow-function or function declarations inside a body.

    These are local sub-functions (not top-level nodes), so we skip them
    (they are not in known_node_ids and we don't emit edges from them separately).
    """
    pass  # Local inner functions inherit edges from their enclosing top-level function


# ---------------------------------------------------------------------------
# CONTAINS edge emitter
# ---------------------------------------------------------------------------

def _emit_contains_edges(
    file_posix: str,
    known_node_ids: set[str],
    conn: sqlite3.Connection,
    counter: list[int],
) -> None:
    """Emit CONTAINS edges for every node owned by this file.

    Two passes:
      1. File → {Function, Method, Interface, TypeAlias, Class, Enum, InterfaceField}
      2. Interface → InterfaceField  (derived from ``file::Interface.field`` id convention)

    Blueprint §3.4.
    """
    # --- Pass 1: File → direct children (including InterfaceField + Variable) ---
    cur = conn.execute(
        "SELECT node_id FROM code_nodes WHERE file_path = ? AND node_type IN "
        "('Function','Method','Interface','TypeAlias','Class','Enum','InterfaceField','Variable')",
        (file_posix,),
    )
    for row in cur:
        child_id = row[0]
        if child_id in known_node_ids:
            _emit_edge(file_posix, child_id, "CONTAINS", conn, counter)

    # --- Pass 2: Interface → InterfaceField ---
    # InterfaceField node_ids have the form  "src/…/file.ts::InterfaceName.fieldName"
    # The parent Interface node_id is        "src/…/file.ts::InterfaceName"
    cur2 = conn.execute(
        "SELECT node_id FROM code_nodes WHERE file_path = ? AND node_type = 'InterfaceField'",
        (file_posix,),
    )
    for row in cur2:
        field_id: str = row[0]
        if field_id not in known_node_ids:
            continue
        # Derive interface node_id by stripping the ".fieldName" suffix
        # node_id format: "path/to/file.ts::InterfaceName.fieldName"
        separator = "::"
        sep_idx = field_id.rfind(separator)
        if sep_idx == -1:
            continue
        symbol_part = field_id[sep_idx + len(separator):]  # "InterfaceName.fieldName"
        dot_idx = symbol_part.rfind(".")
        if dot_idx == -1:
            continue
        iface_symbol = symbol_part[:dot_idx]               # "InterfaceName"
        iface_id = field_id[:sep_idx + len(separator)] + iface_symbol
        if iface_id in known_node_ids:
            _emit_edge(iface_id, field_id, "CONTAINS", conn, counter)


# ---------------------------------------------------------------------------
# Pass 2: main entry point
# ---------------------------------------------------------------------------

def extract_edges(
    file_path: Path,
    source_bytes: bytes,
    known_node_ids: set[str],
    conn: sqlite3.Connection,
) -> int:
    """Pass 2: extract all 13 edge types.

    Returns the number of edges emitted. Also populates
    ``file_dependencies`` for incremental reindex support.
    Blueprint §3.4.
    """
    parser = get_ts_parser(file_path)
    tree = parser.parse(source_bytes)
    root = tree.root_node

    # Resolve rel_path (same logic as Pass 1)
    parts = file_path.parts
    try:
        src_idx = next(i for i in range(len(parts) - 1, -1, -1) if parts[i] == "src")
        rel_path = Path(*parts[src_idx:])
    except StopIteration:
        rel_path = Path(file_path.name)
    file_posix = rel_path.as_posix()

    counter: list[int] = [0]

    # Step 1-2: build import_map, emit IMPORTS + DEPENDS_ON_EXTERNAL
    import_map, dep_pairs = _build_import_map(
        root, source_bytes, file_posix, known_node_ids, conn, counter,
    )

    # Populate file_dependencies
    if dep_pairs:
        conn.executemany(
            "INSERT OR IGNORE INTO file_dependencies (dependent_file, target_file) VALUES (?, ?)",
            dep_pairs,
        )

    # Step 2b: CONTAINS edges — bridges the File ↔ symbol membrane for BFS
    _emit_contains_edges(file_posix, known_node_ids, conn, counter)

    # Step 3: class edges — INHERITS, IMPLEMENTS, DEFINES_METHOD
    _emit_class_edges(
        root, source_bytes, file_posix, import_map, known_node_ids, conn, counter,
    )

    # Step 4: body edges for every Function/Method in this file
    fn_bodies = _get_function_bodies(root, source_bytes, file_posix, known_node_ids)
    for node_id, fn_node in fn_bodies:
        _emit_body_edges(
            fn_node, source_bytes, node_id, file_posix,
            import_map, known_node_ids, conn, counter,
        )

    # Step 4b: scan module-level statements for DYNAMIC_IMPORT
    # (dynamic() / React.lazy() calls often appear at module scope, not inside a function)
    _scan_module_dynamic_imports(
        root, source_bytes, file_posix, import_map, known_node_ids, conn, counter,
    )

    # Step 4c: Mongoose entity references — model<IFoo>() calls and ref: 'ModelName' literals
    _emit_mongoose_edges(
        root, source_bytes, file_posix, import_map, known_node_ids, conn, counter,
    )

    # Step 4d: Middleware edges — parse config.matcher and emit CALLS to guarded routes
    if file_path.name in ("middleware.ts", "middleware.tsx"):
        _emit_middleware_edges(
            root, source_bytes, file_posix, known_node_ids, conn, counter,
        )

    conn.commit()
    return counter[0]


def _emit_middleware_edges(
    root: Node,
    src: bytes,
    file_posix: str,
    known_node_ids: set[str],
    conn: sqlite3.Connection,
    counter: list[int],
) -> None:
    """Emit synthetic CALLS edges from middleware to the API routes it guards.

    Next.js middleware.ts exports a ``config`` object with a ``matcher`` array of
    path patterns (e.g. ``"/api/:path*"``).  Because middleware has no static
    import relationship to the route handlers it protects, it is otherwise a BFS
    dead-end.  This pass:
      1. Walks the AST to find the exported ``config.matcher`` array literal.
      2. For each string pattern, enumerates all API_ROUTE file nodes whose
         file_posix path matches the pattern (treating ``*`` and ``:param``
         wildcards as segment wildcards).
      3. Emits CALLS edges: middleware_fn → api_route_file (and its HTTP-method
         Function children when present).

    The source node is ``file_posix::middleware`` when present in known_node_ids,
    otherwise the file node itself (file_posix).

    Blueprint §3.4.
    """
    # Identify the source node: prefer the exported "middleware" Function
    middleware_fn_id = f"{file_posix}::middleware"
    source_id = middleware_fn_id if middleware_fn_id in known_node_ids else file_posix
    if source_id not in known_node_ids:
        return

    # --- Step 1: find config.matcher array in AST ---
    matcher_patterns: list[str] = []
    _collect_matcher_patterns(root, src, matcher_patterns)

    if not matcher_patterns:
        return

    # --- Step 2: for each pattern, find matching API_ROUTE files ---
    for pattern in matcher_patterns:
        # Normalise pattern: strip leading "/" and trailing wildcards for matching
        # Supported forms: "/api/:path*", "/api/users/:id", "/api/**"
        # Strategy: convert pattern to a list of segment "matchers"
        clean = pattern.split("?")[0].split("#")[0].lstrip("/")
        # Replace ":segment*" (zero-or-more) with a trailing wildcard flag
        trailing_wildcard = clean.endswith(":path*") or clean.endswith("/**") or clean.endswith("*")
        # Normalise: remove trailing wildcard glob
        clean = re.sub(r"/:path\*$", "", clean)
        clean = re.sub(r"/\*\*$", "", clean)
        clean = re.sub(r"/\*$", "", clean)
        clean = clean.rstrip("*")
        pat_segs = [s for s in clean.split("/") if s]

        for nid in known_node_ids:
            if "::" in nid:
                continue
            if not nid.startswith("src/app/") and not nid.startswith("src/pages/"):
                continue
            if not (nid.endswith("/route.ts") or nid.endswith("/route.tsx")
                    or (nid.startswith("src/pages/")
                        and (nid.endswith(".ts") or nid.endswith(".tsx")))):
                continue

            # Build disk segments (same logic as resolve_api_route)
            if nid.startswith("src/app/"):
                disk_path = nid[len("src/app/"):]
                if disk_path.endswith("/route.ts"):
                    disk_path = disk_path[:-len("/route.ts")]
                elif disk_path.endswith("/route.tsx"):
                    disk_path = disk_path[:-len("/route.tsx")]
                else:
                    continue
            else:
                disk_path = nid[len("src/pages/"):]
                disk_path = disk_path.rsplit(".", 1)[0]

            disk_segs = [s for s in disk_path.split("/") if s]

            # Match: if trailing_wildcard, disk must START with pat_segs prefix
            # Otherwise, positional match with wildcard segments
            if trailing_wildcard:
                if len(disk_segs) < len(pat_segs):
                    continue
                # Check prefix
                match = True
                for p, d in zip(pat_segs, disk_segs):
                    if p.startswith(":"):
                        continue  # wildcard segment
                    if p != d:
                        match = False
                        break
                if not match:
                    continue
            else:
                if not _route_segments_match(
                    pat_segs,
                    [s for s in disk_segs[:len(pat_segs)]] if len(disk_segs) >= len(pat_segs) else disk_segs,
                ):
                    # Fall back to prefix-only if lengths differ slightly
                    if len(disk_segs) != len(pat_segs):
                        continue
                    if not _route_segments_match(pat_segs, disk_segs):
                        continue

            # Emit CALLS: middleware → route file (or HTTP-method Functions)
            fn_targets = [
                f"{nid}::{m}" for m in _HTTP_METHODS
                if f"{nid}::{m}" in known_node_ids
            ]
            if fn_targets:
                for ft in fn_targets:
                    _emit_edge(source_id, ft, "CALLS", conn, counter)
            else:
                _emit_edge(source_id, nid, "CALLS", conn, counter)


def _collect_matcher_patterns(node: Node, src: bytes, patterns: list[str]) -> None:
    """Walk AST to find the ``config`` export's ``matcher`` array and collect string literals.

    Handles:
        export const config = { matcher: ['/api/:path*', '/dashboard/:path*'] }
        export const config = { matcher: '/api/:path*' }
    """
    # Look for: export_statement → lexical_declaration → variable_declarator
    # where the variable name is "config" and value is an object_expression
    # with a "matcher" property.
    if node.type in ("export_statement", "lexical_declaration", "variable_declaration"):
        _try_extract_config_matcher(node, src, patterns)

    for child in node.children:
        _collect_matcher_patterns(child, src, patterns)


def _try_extract_config_matcher(node: Node, src: bytes, patterns: list[str]) -> None:
    """Attempt to extract matcher strings from a variable declaration node."""
    # Navigate: export_statement → declaration (lexical) → variable_declarator → value
    decl = node
    if node.type == "export_statement":
        for c in node.children:
            if c.type in ("lexical_declaration", "variable_declaration"):
                decl = c
                break

    for vd in decl.children:
        if vd.type != "variable_declarator":
            continue
        # Check name == "config"
        name_node = vd.child_by_field_name("name")
        if name_node is None:
            continue
        name_text = src[name_node.start_byte:name_node.end_byte].decode(errors="replace")
        if name_text != "config":
            continue
        # Value should be an object_expression
        val = vd.child_by_field_name("value")
        if val is None or val.type != "object":
            continue
        # Find "matcher" property
        for prop in val.children:
            if prop.type not in ("pair", "property_identifier", "shorthand_property_identifier"):
                continue
            key = prop.child_by_field_name("key") or prop
            key_text = src[key.start_byte:key.end_byte].decode(errors="replace").strip("'\"")
            if key_text != "matcher":
                continue
            value = prop.child_by_field_name("value")
            if value is None:
                continue
            if value.type == "array":
                for elem in value.children:
                    if elem.type in ("string", "template_string"):
                        raw = src[elem.start_byte:elem.end_byte].decode(errors="replace").strip("'\"` ")
                        if raw:
                            patterns.append(raw)
            elif value.type in ("string", "template_string"):
                raw = src[value.start_byte:value.end_byte].decode(errors="replace").strip("'\"` ")
                if raw:
                    patterns.append(raw)


def _scan_module_dynamic_imports(
    root: Node,
    src: bytes,
    file_posix: str,
    import_map: dict[str, str],
    known_node_ids: set[str],
    conn: sqlite3.Connection,
    counter: list[int],
) -> None:
    """Scan top-level and immediately-nested statements for dynamic() / React.lazy() calls.

    These are module-scope variable declarations like:
        const Chart = dynamic(() => import('./HeavyChart'))
    We attribute the DYNAMIC_IMPORT edge to the File node (file_posix itself).
    Blueprint §3.4 DYNAMIC_IMPORT rule.
    """
    file_node_id = file_posix  # File node has node_id == file_posix
    if file_node_id not in known_node_ids:
        return
    for child in root.children:
        decl = child
        if child.type == "export_statement":
            inner = _unwrap_export(child)
            if inner:
                decl = inner
        if decl.type == "lexical_declaration":
            for vd in decl.children:
                if vd.type == "variable_declarator":
                    val = vd.child_by_field_name("value")
                    if val and val.type == "call_expression":
                        fn = val.child_by_field_name("function")
                        if fn:
                            fn_text = src[fn.start_byte:fn.end_byte].decode(errors="replace")
                            if fn_text in ("dynamic", "React.lazy") or fn_text.endswith(".lazy"):
                                args = val.child_by_field_name("arguments")
                                if args:
                                    for arg in args.children:
                                        _find_dynamic_imports(
                                            arg, src, file_node_id, file_posix,
                                            import_map, known_node_ids, conn, counter,
                                        )


# ---------------------------------------------------------------------------
# Mongoose entity edge emitter
# ---------------------------------------------------------------------------

# Regex to match `model('ModelName', ...)` or `model<IFoo>('ModelName', ...)`
# capturing the string literal model name.
_MONGOOSE_REF_RE = re.compile(r"""['"]([A-Z][A-Za-z0-9]*)['"]""")


def _emit_mongoose_edges(
    root: Node,
    src: bytes,
    file_posix: str,
    import_map: dict[str, str],
    known_node_ids: set[str],
    conn: sqlite3.Connection,
    counter: list[int],
) -> None:
    """Emit TYPED_BY edges for Mongoose patterns missed by the standard annotation walk.

    Patterns captured:
    1. `model<IFoo>(...)` or `model<IFoo, IFoo>(...)`  — generic call expression
       where the function is `model` (imported from mongoose).  The type argument
       is an Interface name; resolve it and emit TYPED_BY from the File node.
    2. `ref: 'ModelName'` — string literal inside a Schema object.  Resolve
       'ModelName' to a TYPE_DEFINITION File node via the import_map / name lookup
       and emit TYPED_BY from the File node.

    Both edges are attributed to the File node (file_posix) since the schema/model
    definition is a file-level construct, not inside any single function.
    Blueprint §3.4.
    """
    file_node_id = file_posix
    if file_node_id not in known_node_ids:
        return
    _walk_mongoose(root, src, file_node_id, file_posix, import_map, known_node_ids, conn, counter)


def _walk_mongoose(
    node: Node,
    src: bytes,
    source_id: str,
    file_posix: str,
    import_map: dict[str, str],
    known_node_ids: set[str],
    conn: sqlite3.Connection,
    counter: list[int],
) -> None:
    """Recursive walk for Mongoose-specific TYPED_BY patterns."""
    # Pattern 1: call_expression where function is 'model' with type_arguments
    if node.type == "call_expression":
        fn = node.child_by_field_name("function")
        if fn is not None:
            fn_text = src[fn.start_byte:fn.end_byte].decode(errors="replace")
            # mongoose `model<IUser>(...)` — bare `model` or `mongoose.model`
            if fn_text == "model" or fn_text.endswith(".model"):
                type_args = node.child_by_field_name("type_arguments")
                if type_args:
                    for arg in type_args.children:
                        if arg.type in ("type_identifier", "predefined_type"):
                            type_name = src[arg.start_byte:arg.end_byte].decode(errors="replace")
                            target = resolve_call_target(type_name, import_map, file_posix, known_node_ids)
                            if target:
                                _emit_edge(source_id, target, "TYPED_BY", conn, counter)
                        elif arg.type == "generic_type":
                            name_n = arg.child_by_field_name("name")
                            if name_n:
                                type_name = src[name_n.start_byte:name_n.end_byte].decode(errors="replace")
                                target = resolve_call_target(type_name, import_map, file_posix, known_node_ids)
                                if target:
                                    _emit_edge(source_id, target, "TYPED_BY", conn, counter)

    # Pattern 2: pair node `ref: 'ModelName'` inside a Schema object
    if node.type == "pair":
        key_node = node.child_by_field_name("key")
        val_node = node.child_by_field_name("value")
        if key_node and val_node:
            key_text = src[key_node.start_byte:key_node.end_byte].decode(errors="replace").strip("'\"")
            if key_text == "ref" and val_node.type == "string":
                ref_name = src[val_node.start_byte:val_node.end_byte].decode(errors="replace").strip("'\"")
                if ref_name and ref_name[0].isupper():
                    # Try to resolve as a TYPE_DEFINITION File: src/lib/db/models/<name>.model.ts
                    # or as an Interface node imported in this file.
                    target = resolve_call_target(ref_name, import_map, file_posix, known_node_ids)
                    if target:
                        _emit_edge(source_id, target, "TYPED_BY", conn, counter)
                    else:
                        # Probe common model file names
                        lower = ref_name[0].lower() + ref_name[1:]
                        model_candidates = [
                            f"src/lib/db/models/{lower}.model.ts",
                            f"src/lib/db/models/{ref_name}.model.ts",
                        ]
                        for mc in model_candidates:
                            if mc in known_node_ids:
                                _emit_edge(source_id, mc, "TYPED_BY", conn, counter)
                                break

    for child in node.children:
        _walk_mongoose(child, src, source_id, file_posix, import_map, known_node_ids, conn, counter)
