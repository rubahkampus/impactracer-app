"""Tests for code_indexer.py Pass 1 — acceptance criteria.

Sprint 4. Blueprint §3.2.
"""

import sqlite3
from pathlib import Path

import pytest

from impactracer.persistence.sqlite_client import init_schema
from impactracer.indexer.code_indexer import (
    classify_file,
    compose_embed_text,
    compose_file_embed_text,
    derive_route_path,
    extract_edges,
    extract_nodes,
    get_ts_parser,
    synthesize_ui_docstring,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    init_schema(c)
    yield c
    c.close()


def _make_file_path(rel: str) -> Path:
    """Create an absolute-like path that looks like the citrakara repo."""
    return Path("C:/Users/Haidar/Documents/thesis/citrakara") / rel


# ---------------------------------------------------------------------------
# get_ts_parser
# ---------------------------------------------------------------------------

def test_get_ts_parser_returns_tsx_for_tsx():
    parser = get_ts_parser(Path("src/components/Foo.tsx"))
    # Parse a JSX snippet — should not error
    tree = parser.parse(b"const x = <div/>;")
    assert tree.root_node.type == "program"


def test_get_ts_parser_returns_ts_for_ts():
    parser = get_ts_parser(Path("src/lib/helper.ts"))
    tree = parser.parse(b"const x: number = 1;")
    assert tree.root_node.type == "program"


# ---------------------------------------------------------------------------
# classify_file — all NEXTJS_ROUTE_PATTERNS rows
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rel,expected", [
    # API routes
    ("src/app/api/auth/login/route.ts",             "API_ROUTE"),
    ("src/app/api/commissions/[id]/route.ts",       "API_ROUTE"),
    ("src/app/api/chats/route.tsx",                 "API_ROUTE"),
    # Page components
    ("src/app/[username]/dashboard/page.tsx",       "PAGE_COMPONENT"),
    ("src/app/[username]/dashboard/layout.tsx",     "PAGE_COMPONENT"),
    ("src/app/page.ts",                             "PAGE_COMPONENT"),
    # UI components
    ("src/components/auth/LoginForm.tsx",           "UI_COMPONENT"),
    ("src/components/SomeDeep/Nested.ts",           "UI_COMPONENT"),
    # Hooks → UI_COMPONENT (client-side, not server logic)
    ("src/hooks/useStartConversation.ts",           "UI_COMPONENT"),
    # Zustand stores → UI_COMPONENT (client-state, not backend)
    ("src/lib/stores/websocketStore.ts",            "UI_COMPONENT"),
    ("src/lib/stores/uiStore.ts",                   "UI_COMPONENT"),
    # Mongoose models → TYPE_DEFINITION (must match before UTILITY catch-all)
    ("src/lib/db/models/user.model.ts",             "TYPE_DEFINITION"),
    ("src/lib/db/models/wallet.model.ts",           "TYPE_DEFINITION"),
    ("src/lib/db/models/contract.model.ts",         "TYPE_DEFINITION"),
    # Model mocks → None (test infrastructure, excluded)
    ("src/lib/db/models/__mocks__/wallet.mock.ts",  None),
    # Repository test files → None
    ("src/lib/db/repositories/__tests__/wallet.repository.test.ts", None),
    # Services test files → None
    ("src/lib/services/__tests__/gallery.wallet.test.ts", None),
    # Backend UTILITY (services, repositories, utils, api helpers)
    ("src/lib/services/auth.service.ts",            "UTILITY"),
    ("src/lib/db/repositories/user.repository.ts",  "UTILITY"),
    ("src/lib/utils/jwt.ts",                        "UTILITY"),
    ("src/lib/api/withAuth.ts",                     "UTILITY"),
    ("src/lib/db/connection.ts",                    "UTILITY"),
    # TypeScript type files
    ("src/types/common.ts",                         "TYPE_DEFINITION"),
    ("src/types/proposal.ts",                       "TYPE_DEFINITION"),
    # Unclassified
    ("middleware.ts",                               None),
    ("src/config.ts",                               None),
])
def test_classify_file(rel, expected):
    assert classify_file(Path(rel)) == expected


def test_classify_file_null_for_unmatched():
    assert classify_file(Path("config/jest.config.ts")) is None


# ---------------------------------------------------------------------------
# derive_route_path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rel,expected", [
    ("src/app/api/commissions/[id]/route.ts",       "/api/commissions/{id}"),
    ("src/app/api/auth/login/route.ts",             "/api/auth/login"),
    ("src/app/api/chats/[id]/messages/route.ts",    "/api/chats/{id}/messages"),
    ("src/app/[username]/dashboard/page.tsx",        "/{username}/dashboard"),
    ("src/app/[username]/dashboard/layout.tsx",      "/{username}/dashboard"),
])
def test_derive_route_path(rel, expected):
    assert derive_route_path(Path(rel)) == expected


def test_derive_route_path_non_app_returns_none():
    assert derive_route_path(Path("src/lib/helper.ts")) is None


# ---------------------------------------------------------------------------
# extract_nodes — File node is always at index 0
# ---------------------------------------------------------------------------

def test_file_node_is_first(conn):
    src = b"export function foo() { return 1; }"
    nodes = extract_nodes(_make_file_path("src/lib/helper.ts"), src, conn)
    assert nodes[0]["node_type"] == "File"


def test_file_node_id_is_rel_posix(conn):
    src = b"export function foo() { return 1; }"
    nodes = extract_nodes(_make_file_path("src/lib/helper.ts"), src, conn)
    assert nodes[0]["node_id"] == "src/lib/helper.ts"


def test_file_node_classification(conn):
    src = b"export function foo() { return 1; }"
    nodes = extract_nodes(_make_file_path("src/app/api/auth/login/route.ts"), src, conn)
    assert nodes[0]["file_classification"] == "API_ROUTE"


def test_file_node_route_path_for_api_route(conn):
    src = b"export async function POST(req) { return null; }"
    nodes = extract_nodes(_make_file_path("src/app/api/commissions/[id]/route.ts"), src, conn)
    assert nodes[0]["route_path"] == "/api/commissions/{id}"


def test_file_node_no_route_path_for_utility(conn):
    src = b"export function foo() { return 1; }"
    nodes = extract_nodes(_make_file_path("src/lib/helper.ts"), src, conn)
    assert nodes[0]["route_path"] is None


# ---------------------------------------------------------------------------
# extract_nodes — function declarations
# ---------------------------------------------------------------------------

def test_function_declaration_extracted(conn):
    src = b"export function loginUser(req) { return null; }"
    nodes = extract_nodes(_make_file_path("src/lib/services/auth.service.ts"), src, conn)
    types = {n["node_type"] for n in nodes}
    assert "Function" in types


def test_function_node_id(conn):
    src = b"export function loginUser(req) { return null; }"
    nodes = extract_nodes(_make_file_path("src/lib/services/auth.service.ts"), src, conn)
    ids = {n["node_id"] for n in nodes}
    assert "src/lib/services/auth.service.ts::loginUser" in ids


def test_function_is_exported(conn):
    src = b"export function loginUser(req) { return null; }"
    nodes = extract_nodes(_make_file_path("src/lib/services/auth.service.ts"), src, conn)
    fn = next(n for n in nodes if n["node_type"] == "Function")
    assert fn["is_exported"] is True


def test_unexported_function_not_exported(conn):
    src = b"function loginUser(req) { return null; }"
    nodes = extract_nodes(_make_file_path("src/lib/services/auth.service.ts"), src, conn)
    fn = next(n for n in nodes if n["node_type"] == "Function")
    assert fn["is_exported"] is False


def test_function_jsdoc_extracted(conn):
    src = b"""
/** Logs in the user. */
export function loginUser(req) { return null; }
"""
    nodes = extract_nodes(_make_file_path("src/lib/services/auth.service.ts"), src, conn)
    fn = next(n for n in nodes if n["node_type"] == "Function")
    assert fn["docstring"] == "Logs in the user."


def test_function_has_skeleton(conn):
    src = b"export function f(x) { doWork(x); return x; }"
    nodes = extract_nodes(_make_file_path("src/lib/helper.ts"), src, conn)
    fn = next(n for n in nodes if n["node_type"] == "Function")
    assert fn["internal_logic_abstraction"] is not None
    assert "doWork" in fn["internal_logic_abstraction"]


# ---------------------------------------------------------------------------
# extract_nodes — arrow-function variable_declarators
# ---------------------------------------------------------------------------

def test_arrow_function_extracted(conn):
    src = b"export const helper = (x) => { return x * 2; };"
    nodes = extract_nodes(_make_file_path("src/lib/helper.ts"), src, conn)
    ids = {n["node_id"] for n in nodes}
    assert "src/lib/helper.ts::helper" in ids


def test_arrow_function_node_type(conn):
    src = b"export const helper = (x) => { return x * 2; };"
    nodes = extract_nodes(_make_file_path("src/lib/helper.ts"), src, conn)
    fn = next(n for n in nodes if n.get("name") == "helper")
    assert fn["node_type"] == "Function"


# ---------------------------------------------------------------------------
# extract_nodes — class nodes
# ---------------------------------------------------------------------------

def test_class_extracted(conn):
    src = b"export class UserService extends BaseService { getUser() { return null; } }"
    nodes = extract_nodes(_make_file_path("src/lib/services/user.service.ts"), src, conn)
    types = [n["node_type"] for n in nodes]
    assert "Class" in types


def test_class_extends_info(conn):
    src = b"export class Foo extends Bar { }"
    nodes = extract_nodes(_make_file_path("src/lib/foo.ts"), src, conn)
    cls = next(n for n in nodes if n["node_type"] == "Class")
    assert cls["extends_name"] == "Bar"


def test_class_implements_info(conn):
    src = b"export class Foo extends Bar implements IBaz, IQux { }"
    nodes = extract_nodes(_make_file_path("src/lib/foo.ts"), src, conn)
    cls = next(n for n in nodes if n["node_type"] == "Class")
    assert "IBaz" in cls["implements_names"]
    assert "IQux" in cls["implements_names"]


def test_class_method_extracted(conn):
    src = b"export class Foo { myMethod(x) { return x; } }"
    nodes = extract_nodes(_make_file_path("src/lib/foo.ts"), src, conn)
    types = [n["node_type"] for n in nodes]
    assert "Method" in types


def test_class_method_node_id(conn):
    src = b"export class Foo { myMethod(x) { return x; } }"
    nodes = extract_nodes(_make_file_path("src/lib/foo.ts"), src, conn)
    ids = {n["node_id"] for n in nodes}
    assert "src/lib/foo.ts::Foo.myMethod" in ids


def test_class_method_has_skeleton(conn):
    src = b"export class Foo { async myMethod(x) { doWork(x); return x; } }"
    nodes = extract_nodes(_make_file_path("src/lib/foo.ts"), src, conn)
    m = next(n for n in nodes if n["node_type"] == "Method")
    assert m["internal_logic_abstraction"] is not None


# ---------------------------------------------------------------------------
# extract_nodes — interface nodes
# ---------------------------------------------------------------------------

def test_interface_extracted(conn):
    src = b"export interface IUser { name: string; age: number; }"
    nodes = extract_nodes(_make_file_path("src/types/common.ts"), src, conn)
    types = [n["node_type"] for n in nodes]
    assert "Interface" in types


def test_interface_fields_extracted(conn):
    src = b"export interface IUser { name: string; age: number; }"
    nodes = extract_nodes(_make_file_path("src/types/common.ts"), src, conn)
    field_nodes = [n for n in nodes if n["node_type"] == "InterfaceField"]
    names = {n["name"] for n in field_nodes}
    assert names == {"name", "age"}


def test_interface_field_node_ids(conn):
    src = b"export interface IUser { name: string; age: number; }"
    nodes = extract_nodes(_make_file_path("src/types/common.ts"), src, conn)
    ids = {n["node_id"] for n in nodes}
    assert "src/types/common.ts::IUser.name" in ids
    assert "src/types/common.ts::IUser.age" in ids


def test_interface_field_embed_text_is_empty(conn):
    """InterfaceField nodes are degenerate by blueprint §3.2 — embed_text == ''."""
    src = b"export interface IUser { name: string; }"
    nodes = extract_nodes(_make_file_path("src/types/common.ts"), src, conn)
    field = next(n for n in nodes if n["node_type"] == "InterfaceField")
    assert field["embed_text"] == ""


# ---------------------------------------------------------------------------
# extract_nodes — TypeAlias nodes (object shape → InterfaceField children)
# ---------------------------------------------------------------------------

def test_type_alias_extracted(conn):
    src = b"export type Status = 'active' | 'inactive';"
    nodes = extract_nodes(_make_file_path("src/types/common.ts"), src, conn)
    types = [n["node_type"] for n in nodes]
    assert "TypeAlias" in types


def test_type_alias_object_shape_emits_fields(conn):
    src = b"export type MyAlias = { x: number; y: string; };"
    nodes = extract_nodes(_make_file_path("src/types/common.ts"), src, conn)
    fields = [n for n in nodes if n["node_type"] == "InterfaceField"]
    assert {f["name"] for f in fields} == {"x", "y"}


# ---------------------------------------------------------------------------
# extract_nodes — Enum nodes
# ---------------------------------------------------------------------------

def test_enum_extracted(conn):
    src = b"export enum Direction { Up = 1, Down = 2 }"
    nodes = extract_nodes(_make_file_path("src/types/common.ts"), src, conn)
    types = [n["node_type"] for n in nodes]
    assert "Enum" in types


def test_enum_node_id(conn):
    src = b"export enum Direction { Up = 1, Down = 2 }"
    nodes = extract_nodes(_make_file_path("src/types/common.ts"), src, conn)
    ids = {n["node_id"] for n in nodes}
    assert "src/types/common.ts::Direction" in ids


# ---------------------------------------------------------------------------
# extract_nodes — ExternalPackage nodes
# ---------------------------------------------------------------------------

def test_external_package_extracted(conn):
    src = b"import { useState } from 'react';\nexport function f() {}"
    nodes = extract_nodes(_make_file_path("src/components/Foo.tsx"), src, conn)
    types = [n["node_type"] for n in nodes]
    assert "ExternalPackage" in types


def test_external_package_node_id(conn):
    src = b"import axios from 'axios';\nexport function f() {}"
    nodes = extract_nodes(_make_file_path("src/lib/helper.ts"), src, conn)
    ids = {n["node_id"] for n in nodes}
    assert "ext::axios" in ids


def test_external_package_embed_text_is_empty(conn):
    """ExternalPackage nodes are degenerate — embed_text == ''."""
    src = b"import axios from 'axios';\nexport function f() {}"
    nodes = extract_nodes(_make_file_path("src/lib/helper.ts"), src, conn)
    ext = next(n for n in nodes if n["node_type"] == "ExternalPackage")
    assert ext["embed_text"] == ""


def test_relative_import_does_not_create_external_package(conn):
    src = b"import { foo } from './utils';\nexport function f() {}"
    nodes = extract_nodes(_make_file_path("src/lib/helper.ts"), src, conn)
    ext_nodes = [n for n in nodes if n["node_type"] == "ExternalPackage"]
    assert ext_nodes == []


def test_external_package_deduplicated(conn):
    """Same external package imported twice → only one ExternalPackage node."""
    src = b"""
import { useState } from 'react';
import { useEffect } from 'react';
export function f() {}
"""
    nodes = extract_nodes(_make_file_path("src/components/Foo.tsx"), src, conn)
    ext_nodes = [n for n in nodes if n["node_type"] == "ExternalPackage"]
    react_nodes = [n for n in ext_nodes if n["name"] == "react"]
    assert len(react_nodes) == 1


# ---------------------------------------------------------------------------
# extract_nodes — React component detection and synthetic UI docstring
# ---------------------------------------------------------------------------

def test_react_component_is_component_flag(conn):
    src = b"""
export function ChatDashboard({ profile }) {
  return <div>{profile}</div>;
}
"""
    nodes = extract_nodes(_make_file_path("src/components/ChatDashboard.tsx"), src, conn)
    fn = next(n for n in nodes if n["node_type"] == "Function")
    assert fn.get("is_component") is True


def test_lowercase_function_not_component(conn):
    src = b"""
export function helper(x) {
  return x * 2;
}
"""
    nodes = extract_nodes(_make_file_path("src/components/helper.tsx"), src, conn)
    fn = next(n for n in nodes if n["node_type"] == "Function")
    assert fn.get("is_component") is False


def test_synthetic_ui_docstring_generated(conn):
    """Exported UI_COMPONENT Function without JSDoc gets synthetic docstring."""
    src = b"""
export function ChatDashboard({ profile }) {
  return <div/>;
}
"""
    nodes = extract_nodes(_make_file_path("src/components/ChatDashboard.tsx"), src, conn)
    fn = next(n for n in nodes if n["node_type"] == "Function")
    assert fn["docstring"] is not None
    assert "Chat Dashboard" in fn["docstring"]
    assert "UI component" in fn["docstring"]


def test_synthetic_docstring_not_for_non_ui_component(conn):
    """UI component in API_ROUTE file should NOT get synthetic docstring."""
    src = b"""
export function MyComp() {
  return <div/>;
}
"""
    # File classified as API_ROUTE, not UI_COMPONENT
    nodes = extract_nodes(_make_file_path("src/app/api/auth/login/route.tsx"), src, conn)
    fn = next(n for n in nodes if n["node_type"] == "Function" and n["name"] == "MyComp")
    assert fn["docstring"] is None


def test_jsdoc_not_overridden_by_synthetic(conn):
    """Real JSDoc must NOT be replaced by synthetic docstring."""
    src = b"""
/** My real docstring. */
export function ChatDashboard({ profile }) {
  return <div/>;
}
"""
    nodes = extract_nodes(_make_file_path("src/components/ChatDashboard.tsx"), src, conn)
    fn = next(n for n in nodes if n["node_type"] == "Function")
    assert fn["docstring"] == "My real docstring."


# ---------------------------------------------------------------------------
# compose_embed_text
# ---------------------------------------------------------------------------

def test_compose_embed_text_docstring_and_signature():
    node = {"name": "foo", "docstring": "Does foo.", "signature": "foo(x: number): void"}
    result = compose_embed_text(node)
    assert result == "Does foo.\nfoo(x: number): void"


def test_compose_embed_text_signature_only():
    node = {"name": "foo", "docstring": None, "signature": "foo(x: number): void"}
    result = compose_embed_text(node)
    assert result == "foo(x: number): void"


def test_compose_embed_text_fallback_to_name():
    node = {"name": "foo", "docstring": None, "signature": None}
    result = compose_embed_text(node)
    assert result == "foo"


def test_compose_embed_text_empty_docstring_ignored():
    node = {"name": "foo", "docstring": "", "signature": "foo()"}
    result = compose_embed_text(node)
    assert result == "foo()"


# ---------------------------------------------------------------------------
# compose_file_embed_text
# ---------------------------------------------------------------------------

def test_compose_file_embed_text_format():
    file_node = {"name": "LoginForm.tsx", "file_classification": "UI_COMPONENT"}
    result = compose_file_embed_text(file_node, ["LoginForm", "ArrowHelper"], "src/components/auth")
    assert "LoginForm.tsx" in result
    assert "[UI_COMPONENT]" in result
    assert "(src/components/auth)" in result
    assert "LoginForm" in result
    assert "ArrowHelper" in result


def test_compose_file_embed_text_no_exports():
    file_node = {"name": "empty.ts", "file_classification": None}
    result = compose_file_embed_text(file_node, [], "src/lib")
    assert "(none)" in result


# ---------------------------------------------------------------------------
# synthesize_ui_docstring
# ---------------------------------------------------------------------------

def test_synthesize_ui_docstring_readable_name():
    result = synthesize_ui_docstring("ChatDashboard", "ChatDashboard()")
    assert "Chat Dashboard" in result
    assert "UI component" in result


def test_synthesize_ui_docstring_with_props_type():
    result = synthesize_ui_docstring("LoginForm", "LoginForm({ onSuccess }: LoginFormProps)")
    assert "Login Form" in result
    assert "Login Form Props" in result


# ---------------------------------------------------------------------------
# extract_nodes — nodes persisted to SQLite
# ---------------------------------------------------------------------------

def test_nodes_persisted_to_sqlite(conn):
    src = b"export function foo() { return 1; }"
    extract_nodes(_make_file_path("src/lib/helper.ts"), src, conn)
    rows = conn.execute("SELECT node_id FROM code_nodes").fetchall()
    ids = {r[0] for r in rows}
    assert "src/lib/helper.ts" in ids
    assert "src/lib/helper.ts::foo" in ids


def test_insert_idempotent(conn):
    src = b"export function foo() { return 1; }"
    fp = _make_file_path("src/lib/helper.ts")
    extract_nodes(fp, src, conn)
    extract_nodes(fp, src, conn)  # second call — INSERT OR REPLACE
    count = conn.execute(
        "SELECT COUNT(*) FROM code_nodes WHERE file_path = 'src/lib/helper.ts'"
    ).fetchone()[0]
    # File node + foo node = 2 rows
    assert count == 2


# ---------------------------------------------------------------------------
# Variable node extraction (Sprint 13-W1) — const FOO = <non-arrow RHS>
# ---------------------------------------------------------------------------

def test_variable_extracts_mongoose_schema_with_field_names_in_embed_text(conn):
    """A `const X = new Schema({...})` is emitted as a Variable node and the
    embed_text contains the schema's top-level field names so retrieval can
    surface the schema via field-name queries."""
    src = (
        b"import { Schema, model } from 'mongoose';\n"
        b"interface IUser { email: string; username: string; }\n"
        b"const UserSchema = new Schema<IUser>({\n"
        b"  email: { type: String, required: true },\n"
        b"  username: { type: String, required: true },\n"
        b"  pinnedCommissions: [{ type: Schema.Types.ObjectId }],\n"
        b"});\n"
        b"export default model<IUser>('User', UserSchema);\n"
    )
    nodes = extract_nodes(_make_file_path("src/lib/db/models/user.model.ts"), src, conn)
    by_id = {n["node_id"]: n for n in nodes}
    nid = "src/lib/db/models/user.model.ts::UserSchema"
    assert nid in by_id, f"UserSchema not extracted; got {list(by_id)}"
    var = by_id[nid]
    assert var["node_type"] == "Variable"
    assert var["name"] == "UserSchema"
    assert var["file_path"] == "src/lib/db/models/user.model.ts"
    # Field names from the schema must appear in embed_text (BGE-M3 / BM25 fodder).
    emb = var["embed_text"].lower()
    assert "email" in emb
    assert "username" in emb
    assert "pinnedcommissions" in emb


def test_variable_extracts_const_array_literal(conn):
    """A `const TEMPLATES = [...]` is emitted with array-element keys in embed_text."""
    src = (
        b"interface TemplateData { id: string; title: string; basePrice: number; }\n"
        b"const TEMPLATES: TemplateData[] = [\n"
        b"  { id: 'a', title: 'Sketch', basePrice: 10 },\n"
        b"  { id: 'b', title: 'Lineart', basePrice: 25 },\n"
        b"];\n"
    )
    nodes = extract_nodes(
        _make_file_path("src/components/dashboard/commissions/form/TemplateSection.tsx"),
        src, conn,
    )
    by_id = {n["node_id"]: n for n in nodes}
    nid = "src/components/dashboard/commissions/form/TemplateSection.tsx::TEMPLATES"
    assert nid in by_id, f"TEMPLATES not extracted; got {list(by_id)}"
    var = by_id[nid]
    assert var["node_type"] == "Variable"
    emb = var["embed_text"].lower()
    assert "id" in emb
    assert "title" in emb
    assert "baseprice" in emb


def test_variable_skips_lowercase_locals(conn):
    """Plain lowercase locals like `const tmp = ...` are not emitted (heuristic)."""
    src = b"const tmp = { foo: 1 };\nconst i = 0;\n"
    nodes = extract_nodes(_make_file_path("src/lib/helper.ts"), src, conn)
    types = {n["node_type"] for n in nodes}
    assert "Variable" not in types


def test_variable_node_emits_contains_edge_from_file(conn):
    """The File node must CONTAIN any Variable extracted from it."""
    from impactracer.indexer.code_indexer import extract_edges

    src = (
        b"import { Schema } from 'mongoose';\n"
        b"const UserSchema = new Schema({ email: String });\n"
    )
    fp = _make_file_path("src/lib/db/models/user.model.ts")
    nodes = extract_nodes(fp, src, conn)
    known = {n["node_id"] for n in nodes}
    extract_edges(fp, src, known, conn)
    file_id = "src/lib/db/models/user.model.ts"
    var_id = "src/lib/db/models/user.model.ts::UserSchema"
    row = conn.execute(
        "SELECT 1 FROM structural_edges WHERE source_id=? AND target_id=? AND edge_type='CONTAINS'",
        (file_id, var_id),
    ).fetchone()
    assert row is not None, "Expected File -> Variable CONTAINS edge"


def test_variable_node_type_accepted_by_sqlite_check_constraint(conn):
    """Smoke test: the DDL's CHECK constraint allows 'Variable' as a node_type."""
    conn.execute(
        "INSERT INTO code_nodes (node_id, node_type, name) VALUES (?, ?, ?)",
        ("src/x.ts::Y", "Variable", "Y"),
    )


# ---------------------------------------------------------------------------
# Node IDs use forward slashes (blueprint §2 invariant 8)
# ---------------------------------------------------------------------------

def test_node_ids_use_forward_slashes(conn):
    src = b"export function foo() { return 1; }"
    nodes = extract_nodes(_make_file_path("src/lib/helper.ts"), src, conn)
    for n in nodes:
        assert "\\" not in n["node_id"], f"Backslash in node_id: {n['node_id']}"
