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
    import fnmatch

    def _match(rel_posix: str, pattern: str) -> bool:
        return fnmatch.fnmatch(rel_posix, pattern)

    _page_names = {"page.ts", "page.tsx", "layout.ts", "layout.tsx"}
    _route_names = {"route.ts", "route.tsx"}

    def classify(rel_path: Path) -> str | None:
        p = rel_path.as_posix()
        filename = rel_path.name
        # Check by filename first for files directly inside src/app/ tree
        if p.startswith("src/app/") and filename in _route_names:
            return "API_ROUTE"
        if p.startswith("src/app/") and filename in _page_names:
            return "PAGE_COMPONENT"
        # src/components/** (any .ts/.tsx under components/)
        if p.startswith("src/components/"):
            return "UI_COMPONENT"
        if p.startswith("src/lib/") or p.startswith("src/utils/"):
            return "UTILITY"
        if p.startswith("src/types/"):
            return "TYPE_DEFINITION"
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
    """
    filename = file_node.get("name") or ""
    classification = file_node.get("file_classification") or "NULL"
    exports_str = ", ".join(exported_names) if exported_names else "(none)"
    return f"{filename} [{classification}] ({rel_dir})\nexports: {exports_str}"


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
            # Could contain arrow-function variable declarators
            for vd in decl.children:
                if vd.type == "variable_declarator":
                    val = vd.child_by_field_name("value")
                    if val and val.type == "arrow_function":
                        fn_node = _build_arrow_function_node(vd, val, child, src, file_posix, file_classification)
                        if fn_node:
                            nodes.append(fn_node)
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
            source_code, embed_text, line_start, line_end, exported
        ) VALUES (
            :node_id, :node_type, :name, :file_path, :file_classification,
            :route_path, :signature, :docstring, :internal_logic_abstraction,
            :source_code, :embed_text, :line_start, :line_end, :exported
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
        }
        conn.execute(sql, row)
    conn.commit()


# ---------------------------------------------------------------------------
# Pass 2 stub (Sprint 5)
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
    """
    raise NotImplementedError("Sprint 5")
