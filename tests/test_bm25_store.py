"""Unit tests for BM25Store and code-aware tokenizer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from code_rag.storage.bm25_store import BM25Store, _split_identifier, _tokenize_code
from tests.conftest import make_chunk


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------


class TestSplitIdentifier:
    def test_snake_case(self):
        parts = _split_identifier("get_file_context")
        assert "get" in parts
        assert "file" in parts
        assert "context" in parts

    def test_camel_case(self):
        parts = _split_identifier("handleAuthCallback")
        assert "handle" in parts
        assert "auth" in parts
        assert "callback" in parts

    def test_acronym(self):
        parts = _split_identifier("HTTPResponse")
        assert "http" in parts
        assert "response" in parts

    def test_full_original_preserved(self):
        # The full lowercased token should always appear first
        parts = _split_identifier("MyClass")
        assert parts[0] == "myclass"

    def test_single_word(self):
        parts = _split_identifier("index")
        assert "index" in parts


class TestTokenizeCode:
    def test_basic(self):
        tokens = _tokenize_code("def search_code(query, top_k):")
        assert "search" in tokens
        assert "code" in tokens
        assert "query" in tokens
        assert "top" in tokens

    def test_stop_words_removed(self):
        tokens = _tokenize_code("the quick brown fox")
        # "the" is a stop word; short tokens (<2 chars) are dropped
        assert "the" not in tokens

    def test_empty(self):
        assert _tokenize_code("") == []

    def test_single_char_tokens_dropped(self):
        tokens = _tokenize_code("a b c def")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "def" in tokens


# ---------------------------------------------------------------------------
# BM25Store tests
# ---------------------------------------------------------------------------


class TestBM25Store:
    @pytest.fixture
    def store(self, tmp_path):
        return BM25Store(tmp_path / "bm25.pkl")

    def test_empty_search_returns_nothing(self, store):
        results = store.search("anything")
        assert results == []

    def test_add_and_search(self, store):
        chunk = make_chunk(
            text="def authenticate_user(username, password): pass",
            file_path="auth.py",
            symbol_name="authenticate_user",
        )
        store.add(0, chunk)
        results = store.search("authenticate user")
        assert len(results) > 0
        assert results[0][0].file_path == "auth.py"

    def test_relevance_ordering(self, store):
        """More relevant chunk should rank higher."""
        chunk_a = make_chunk(
            text="def render_frame(delta): pass",
            file_path="render.py",
            symbol_name="render_frame",
        )
        chunk_b = make_chunk(
            text="def authenticate_user(pw): pass",
            file_path="auth.py",
            symbol_name="authenticate_user",
        )
        store.add(0, chunk_a)
        store.add(1, chunk_b)

        results = store.search("authenticate user")
        assert results[0][0].file_path == "auth.py"

    def test_remove_by_file(self, store):
        chunk = make_chunk(file_path="remove_me.py")
        store.add(0, chunk)
        assert store.count() == 1
        store.remove_by_file("remove_me.py")
        assert store.count() == 0
        results = store.search("foo")
        assert results == []

    def test_save_and_reload(self, tmp_path):
        path = tmp_path / "bm25.pkl"
        store = BM25Store(path)
        chunk = make_chunk(text="def compute_hash(data): return sha256(data)")
        store.add(0, chunk)
        store.save()

        store2 = BM25Store(path)
        results = store2.search("compute hash")
        assert len(results) > 0

    def test_add_batch(self, store):
        chunks = [
            make_chunk(text="class PlayerController:", file_path="player.py"),
            make_chunk(text="class EnemyAI:", file_path="enemy.py"),
        ]
        store.add_batch([0, 1], chunks)
        assert store.count() == 2
        r = store.search("player controller")
        assert r[0][0].file_path == "player.py"

    def test_empty_query(self, store):
        store.add(0, make_chunk())
        assert store.search("") == []
        assert store.search("  ") == []
