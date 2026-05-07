"""Unit tests for VectorStore."""

from __future__ import annotations

import numpy as np
import pytest

from code_rag.storage.vector_store import VectorStore
from tests.conftest import make_chunk

DIM = 8  # small dimension for fast in-memory tests


class TestVectorStore:
    @pytest.fixture
    def store(self, tmp_path):
        return VectorStore(tmp_path / "vectors.db", dimension=DIM)

    def _random_embed(self, n: int = 1) -> np.ndarray:
        vecs = np.random.randn(n, DIM).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def test_available(self, store):
        assert store.available is True

    def test_empty_count(self, store):
        assert store.count() == 0

    def test_insert_and_count(self, store):
        chunks = [make_chunk(file_path="a.py"), make_chunk(file_path="b.py")]
        embeddings = self._random_embed(2)
        ids = store.insert(chunks, embeddings)
        assert len(ids) == 2
        assert store.count() == 2

    def test_search_returns_results(self, store):
        chunk = make_chunk(file_path="search_me.py")
        emb = self._random_embed(1)
        store.insert([chunk], emb)

        # Query with the same vector — should get a near-perfect score
        results = store.search(emb[0], top_k=5)
        assert len(results) == 1
        assert results[0]["file_path"] == "search_me.py"
        assert results[0]["score"] > 0.99

    def test_search_ordering(self, store):
        """Vector closest to the query should rank first."""
        np.random.seed(42)
        target = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        close = target.copy()
        close[1] = 0.01
        close /= np.linalg.norm(close)
        far = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        chunk_close = make_chunk(file_path="close.py")
        chunk_far = make_chunk(file_path="far.py")
        store.insert([chunk_close, chunk_far], np.stack([close, far]))

        results = store.search(target, top_k=2)
        assert results[0]["file_path"] == "close.py"

    def test_delete_by_file(self, store):
        chunks = [
            make_chunk(file_path="keep.py"),
            make_chunk(file_path="delete.py"),
        ]
        embs = self._random_embed(2)
        store.insert(chunks, embs)
        assert store.count() == 2

        store.delete_by_file("delete.py")
        assert store.count() == 1
        results = store.get_by_file("delete.py")
        assert results == []

    def test_get_by_file(self, store):
        chunk = make_chunk(file_path="specific.py", symbol_name="my_func")
        store.insert([chunk], self._random_embed(1))
        results = store.get_by_file("specific.py")
        assert len(results) == 1
        assert results[0]["file_path"] == "specific.py"

    def test_filter_by_language(self, store):
        py_chunk = make_chunk(file_path="a.py", language="python")
        cs_chunk = make_chunk(file_path="b.cs", language="csharp")
        embs = self._random_embed(2)
        store.insert([py_chunk, cs_chunk], embs)

        query = self._random_embed(1)[0]
        results = store.search(query, top_k=10, filter_expr='language == "python"')
        assert all(r["language"] == "python" for r in results)

    def test_parse_filter_single(self):
        result = VectorStore._parse_filter('language == "python"')
        assert result == {"language": "python"}

    def test_parse_filter_compound(self):
        result = VectorStore._parse_filter(
            'language == "python" and chunk_type == "code"'
        )
        assert result == {"$and": [{"language": "python"}, {"chunk_type": "code"}]}

    def test_parse_filter_invalid_returns_none(self):
        assert VectorStore._parse_filter("garbage") is None
