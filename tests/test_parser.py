"""Tests for parser.py — symbol name extraction across languages."""

from __future__ import annotations

from code_rag.indexer.parser import extract_symbols, parse_file


def _parse_and_extract(source: str, language: str = "cpp"):
    """Helper: parse source and return {name: (kind, start_line)} dict."""
    source_bytes = source.encode()
    tree = parse_file(source_bytes, language)
    symbols = extract_symbols(tree, source_bytes, language)
    return {s.name: (s.kind, s.start_line) for s in symbols}


# ── C++ ────────────────────────────────────────────────────────────────────


def test_plain_function():
    """Free function returns the function name."""
    syms = _parse_and_extract("void free_function() {}")
    assert "free_function" in syms
    assert syms["free_function"][0] == "function"


def test_class_specifier():
    """Class name is extracted correctly."""
    syms = _parse_and_extract("class AActor { void Tick(); };")
    assert "AActor" in syms
    assert syms["AActor"][0] == "class"


def test_qualified_method_definition():
    """AActor::MethodName → MethodName, NOT AActor."""
    src = "bool AActor::CheckDefaultSubobjectsInternal() const { return true; }\n"
    syms = _parse_and_extract(src)
    assert "CheckDefaultSubobjectsInternal" in syms
    assert syms["CheckDefaultSubobjectsInternal"][0] == "function"
    assert "AActor" not in syms  # class qualifier must not become a symbol


def test_multiple_qualified_methods():
    """Multiple methods from the same class each get their own name."""
    src = """
bool AActor::CheckDefaultSubobjectsInternal() const { return true; }
void AActor::PostInitProperties() {}
void AActor::SetOwner(AActor* NewOwner) {}
"""
    syms = _parse_and_extract(src)
    assert "CheckDefaultSubobjectsInternal" in syms
    assert "PostInitProperties" in syms
    assert "SetOwner" in syms
    assert "AActor" not in syms


def test_operator_overload():
    """AActor::operator= → operator=."""
    src = "AActor& AActor::operator=() { return *this; }\n"
    syms = _parse_and_extract(src)
    assert "operator=" in syms
    assert syms["operator="][0] == "function"


def test_template_member_function():
    """template<T> void FMath::Sqrt(T) → Sqrt, not the parameter name."""
    src = "template<typename T> void FMath::Sqrt(T val) {}\n"
    syms = _parse_and_extract(src)
    assert "Sqrt" in syms
    assert syms["Sqrt"][0] == "function"


def test_nested_namespace():
    """Foo::Bar::Baz() → Baz (innermost name)."""
    src = "int Foo::Bar::Baz() { return 0; }\n"
    syms = _parse_and_extract(src)
    assert "Baz" in syms
    assert syms["Baz"][0] == "function"


def test_inline_qualified_method():
    """FRotator AActor::K2_GetActorRotation() const → K2_GetActorRotation."""
    src = "FRotator AActor::K2_GetActorRotation() const { return FRotator(); }\n"
    syms = _parse_and_extract(src)
    assert "K2_GetActorRotation" in syms
    assert syms["K2_GetActorRotation"][0] == "function"


def test_struct_specifier():
    """Struct name is extracted."""
    syms = _parse_and_extract("struct FVector { float X, Y, Z; };")
    assert "FVector" in syms
    assert syms["FVector"][0] == "class"


# ── Python ──────────────────────────────────────────────────────────────────


def test_python_function():
    syms = _parse_and_extract("def foo(): pass", "python")
    assert "foo" in syms
    assert syms["foo"][0] == "function"


def test_python_class():
    syms = _parse_and_extract("class MyClass:\n    pass", "python")
    assert "MyClass" in syms
    assert syms["MyClass"][0] == "class"


# ── JavaScript ──────────────────────────────────────────────────────────────


def test_js_function():
    syms = _parse_and_extract("function hello() {}", "javascript")
    assert "hello" in syms
    assert syms["hello"][0] == "function"


def test_js_class():
    syms = _parse_and_extract("class Widget {}", "javascript")
    assert "Widget" in syms
    assert syms["Widget"][0] == "class"


# ── TypeScript ──────────────────────────────────────────────────────────────


def test_ts_interface():
    syms = _parse_and_extract("interface IProps { name: string }", "typescript")
    assert "IProps" in syms
    assert syms["IProps"][0] == "interface"
