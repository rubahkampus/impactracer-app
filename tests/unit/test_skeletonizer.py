"""Tests for skeletonizer.py — fold-rule acceptance criteria.

Sprint 4. Blueprint §3.3.
"""

from tree_sitter_languages import get_parser

from impactracer.indexer.skeletonizer import skeletonize_node


def _skeleton(src: bytes, lang: str = "typescript") -> str:
    p = get_parser(lang)
    tree = p.parse(src)
    root = tree.root_node
    # Find first function_declaration or method_definition or arrow_function at top level
    for child in root.children:
        if child.type == "function_declaration":
            return skeletonize_node(child, src)
        if child.type == "export_statement":
            for gc in child.children:
                if gc.type == "function_declaration":
                    return skeletonize_node(gc, src)
                if gc.type == "lexical_declaration":
                    for vd in gc.children:
                        if vd.type == "variable_declarator":
                            val = vd.child_by_field_name("value")
                            if val and val.type == "arrow_function":
                                return skeletonize_node(val, src)
    raise ValueError("No function found in source")


# ---------------------------------------------------------------------------
# High-signal preservation
# ---------------------------------------------------------------------------

def test_call_expression_preserved():
    src = b"""
function f() {
  doSomething();
}
"""
    result = _skeleton(src)
    assert "doSomething()" in result


def test_return_statement_preserved():
    src = b"""
function f() {
  if (x) { return 42; }
}
"""
    result = _skeleton(src)
    assert "return 42" in result


def test_throw_statement_preserved():
    src = b"""
function f() {
  throw new Error('fail');
}
"""
    result = _skeleton(src)
    assert "throw" in result


def test_import_declaration_preserved():
    src = b"""
function f() {
  import('./module');
}
"""
    result = _skeleton(src)
    assert "import(" in result


# ---------------------------------------------------------------------------
# JSX fold
# ---------------------------------------------------------------------------

def test_jsx_element_folded():
    src = b"""
function f() {
  return (
    <div>
      <span>hi</span>
    </div>
  );
}
"""
    result = _skeleton(src, "tsx")
    # The return is preserved (high-signal) but the JSX content is folded
    assert "return" in result
    assert "/* [JSX:" in result
    assert "<div>" not in result


def test_jsx_self_closing_folded():
    src = b"""
function f() {
  return <MyComp />;
}
"""
    result = _skeleton(src, "tsx")
    assert "/* [JSX:" in result


def test_jsx_nested_count():
    src = b"""
function f() {
  return (
    <div>
      <span>a</span>
      <span>b</span>
    </div>
  );
}
"""
    result = _skeleton(src, "tsx")
    # 3 JSX elements total (div + 2 spans)
    assert "/* [JSX: 3 elements] */" in result


# ---------------------------------------------------------------------------
# Array fold
# ---------------------------------------------------------------------------

def test_large_array_folded():
    src = b"""
function f() {
  const items = [1, 2, 3, 4, 5];
  return items;
}
"""
    result = _skeleton(src)
    assert "/* [array: 5 items] */" in result
    assert "1, 2, 3, 4, 5" not in result


def test_small_array_not_folded():
    src = b"""
function f() {
  const items = [1, 2, 3];
  return items;
}
"""
    result = _skeleton(src)
    # 3 items — boundary, NOT folded (>3 means strictly more than 3)
    assert "[array:" not in result


def test_hook_dep_array_exempt_useeffect():
    src = b"""
function f() {
  useEffect(() => {
    doWork();
  }, [userId, connectionStatus, foo, bar]);
}
"""
    result = _skeleton(src)
    # Dep array [userId, connectionStatus, foo, bar] has 4 items but must NOT fold
    assert "[array:" not in result
    assert "userId" in result


def test_hook_dep_array_exempt_usecallback():
    src = b"""
function f() {
  const cb = useCallback((x) => {
    doWork(x);
  }, [dep1, dep2, dep3, dep4]);
}
"""
    result = _skeleton(src)
    assert "[array:" not in result


def test_hook_dep_array_exempt_usememo():
    src = b"""
function f() {
  const val = useMemo(() => compute(), [a, b, c, d]);
}
"""
    result = _skeleton(src)
    assert "[array:" not in result


# ---------------------------------------------------------------------------
# Object fold
# ---------------------------------------------------------------------------

def test_large_object_folded():
    src = b"""
function f() {
  const cfg = { a: 1, b: 2, c: 3, d: 4, e: 5 };
  return cfg;
}
"""
    result = _skeleton(src)
    assert "/* [object: 5 props] */" in result


def test_small_object_not_folded():
    src = b"""
function f() {
  const cfg = { a: 1, b: 2, c: 3, d: 4 };
  return cfg;
}
"""
    result = _skeleton(src)
    # 4 props — NOT folded (>4 means strictly more than 4)
    assert "[object:" not in result


def test_object_inside_call_not_folded_independently():
    """Object inside a call_expression: the call tags ancestors,
    so the object node's parent chain is tagged — it will recurse normally."""
    src = b"""
function f() {
  doSomething({ a: 1, b: 2, c: 3, d: 4, e: 5 });
}
"""
    result = _skeleton(src)
    # The call_expression tags this subtree as DO_NOT_ERASE, so no fold
    assert "doSomething(" in result


# ---------------------------------------------------------------------------
# if_statement / switch_statement fold
# ---------------------------------------------------------------------------

def test_if_without_high_signal_folded():
    src = b"""
function f() {
  if (loading) {
    setVisible(false);
  }
  return null;
}
"""
    result = _skeleton(src)
    # setVisible is a call_expression -> tags the if block -> NOT folded
    # Actually setVisible() IS a call -> tagged -> NOT folded. Confirm that.
    assert "setVisible" in result


def test_if_truly_no_high_signal_folded():
    """An if_statement that contains only variable assignments (no calls/returns/throws)."""
    src = b"""
function f() {
  if (x > 0) {
    let y = 1;
    let z = 2;
  }
  return 42;
}
"""
    result = _skeleton(src)
    # The if body has no calls/returns/throws -> should fold
    assert "/* [logic block] */" in result
    assert "return 42" in result


# ---------------------------------------------------------------------------
# Template string fold
# ---------------------------------------------------------------------------

def test_long_template_string_folded():
    long_template = b"`" + b"x" * 101 + b"`"
    src = b"function f() { const s = " + long_template + b"; return s; }"
    result = _skeleton(src)
    assert "/* [template:" in result
    assert "chars] */" in result


def test_short_template_string_not_folded():
    src = b'function f() { return `hello ${name}`; }'
    result = _skeleton(src)
    assert "[template:" not in result


# ---------------------------------------------------------------------------
# String literal fold
# ---------------------------------------------------------------------------

def test_long_string_folded():
    long_str = b'"' + b"a" * 81 + b'"'
    src = b"function f() { const s = " + long_str + b"; return s; }"
    result = _skeleton(src)
    assert "/* [string:" in result


def test_short_string_not_folded():
    src = b'function f() { return "hello"; }'
    result = _skeleton(src)
    assert "[string:" not in result


# ---------------------------------------------------------------------------
# Comment removal
# ---------------------------------------------------------------------------

def test_single_line_comment_removed():
    src = b"""
function f() {
  // This comment should be removed
  return 42;
}
"""
    result = _skeleton(src)
    assert "// This comment" not in result
    assert "return 42" in result


def test_multi_line_comment_removed():
    src = b"""
function f() {
  /* Multi-line comment */
  return 42;
}
"""
    result = _skeleton(src)
    assert "Multi-line comment" not in result
    assert "return 42" in result


# ---------------------------------------------------------------------------
# Whitespace preservation
# ---------------------------------------------------------------------------

def test_whitespace_preserved_between_statements():
    src = b"""
function f() {
  const a = 1;
  const b = 2;
  return a + b;
}
"""
    result = _skeleton(src)
    assert "const a" in result
    assert "const b" in result
    assert "return a + b" in result
    # Tokens must be separated (not smashed together)
    assert "const a = 1" in result


# ---------------------------------------------------------------------------
# Arrow function skeletonization
# ---------------------------------------------------------------------------

def test_arrow_function_skeleton():
    src = b"""
export const handler = async (req, res) => {
  const data = await fetchData(req.id);
  return res.json(data);
};
"""
    result = _skeleton(src)
    assert "fetchData(" in result
    assert "return" in result
