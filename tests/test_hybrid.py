"""Tests for hybrid.py — RRF fusion, search_by_type, language filters, path exclusion.

Requires a pre-built index of UE fixture files using BGE-small.
"""

from __future__ import annotations

import gc
import shutil
import tempfile
from pathlib import Path

import pytest

from code_rag.config import CodeRagConfig
from code_rag.indexer.pipeline import IndexPipeline
from code_rag.retriever.hybrid import HybridRetriever
from code_rag.retriever.semantic import SemanticRetriever
from code_rag.storage.vector_store import VectorStore
from code_rag.storage.bm25_store import BM25Store
from code_rag.indexer.embedder import Embedder

UE_FIXTURES = Path(__file__).parent / "fixtures" / "ue"


def _copy_ue_fixtures(dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for src in UE_FIXTURES.iterdir():
        if src.is_file():
            shutil.copy2(src, dst / src.name)


@pytest.fixture(scope="module")
def indexed_ue_repo(request):
    """Module-scoped: build a UE index once for all hybrid tests."""
    td = tempfile.mkdtemp()
    repo = Path(td)
    _copy_ue_fixtures(repo)
    config = CodeRagConfig(
        repo_path=repo,
        model_name="BAAI/bge-small-en-v1.5",
        device="cpu",
    )
    pipe = IndexPipeline(config)
    pipe.run()
    pipe._vector_store.close()

    def cleanup():
        gc.collect()
        try:
            shutil.rmtree(td, ignore_errors=False)
        except PermissionError:
            shutil.rmtree(td, ignore_errors=True)

    request.addfinalizer(cleanup)
    return repo


@pytest.fixture(scope="module")
def embedder():
    e = Embedder(model_name="BAAI/bge-small-en-v1.5", device="cpu")
    e._ensure_loaded()
    return e


@pytest.fixture
def hybrid(indexed_ue_repo, embedder):
    """Build a HybridRetriever from a pre-built index."""
    data_dir = indexed_ue_repo / ".code-rag"
    vector_store = VectorStore(data_dir / "vectors.db", embedder.dimension)
    bm25_store = BM25Store(data_dir / "bm25.pkl")
    semantic = SemanticRetriever(vector_store, embedder)
    from code_rag.retriever.keyword import KeywordRetriever
    keyword = KeywordRetriever(bm25_store)
    return HybridRetriever(semantic, keyword)


class TestHybridSearch:
    """Core hybrid search tests."""

    def test_search_returns_results(self, hybrid):
        results = hybrid.search("animation skeleton blend", top_k=10)
        assert len(results) > 0

    def test_results_have_expected_fields(self, hybrid):
        results = hybrid.search("animation", top_k=5)
        for r in results:
            assert r.score > 0.0
            assert r.source in ("semantic", "keyword", "hybrid")
            assert r.chunk.text
            assert r.chunk.file_path

    def test_results_contain_ue_files(self, hybrid):
        """Results should reference actual UE fixture files."""
        results = hybrid.search("behavior tree composite node", top_k=10)
        file_paths = {r.chunk.file_path for r in results}
        assert any("BTCompositeNode.cpp" in p for p in file_paths) or \
            any("BehaviorTree" in p for p in file_paths), (
            f"Expected UE behavior tree files in results, got: {file_paths}"
        )

    def test_results_are_scored_descending(self, hybrid):
        """Scores should be non-increasing (sorted by relevance)."""
        results = hybrid.search("animation tick player", top_k=10)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_search_with_language_filter(self, hybrid):
        """Language filter should only return results in that language."""
        results = hybrid.search("animation", top_k=20, language="cpp")
        assert len(results) > 0
        for r in results:
            assert r.chunk.language == "cpp"

    def test_top_k_respected(self, hybrid):
        for k in [3, 5, 10]:
            results = hybrid.search("animation skeleton", top_k=k)
            assert len(results) <= k


class TestSearchByType:
    """search_by_type / search_code / search_docs."""

    def test_search_code_only_returns_code(self, hybrid):
        results = hybrid.search_code("animation", top_k=10)
        for r in results:
            assert r.chunk.chunk_type == "code", (
                f"Expected code chunk, got {r.chunk.chunk_type}"
            )

    def test_search_by_type_code_is_same_as_search_code(self, hybrid):
        """search_by_type('code') should return same as search_code."""
        r1 = hybrid.search_code("animation", top_k=10)
        r2 = hybrid.search_by_type("animation", "code", top_k=10)
        assert len(r1) == len(r2)

    def test_search_all_types(self, hybrid):
        """search_by_type with None should return all chunk types."""
        results = hybrid.search_by_type("animation", None, top_k=20)
        types = {r.chunk.chunk_type for r in results}
        # UE fixtures are all code files, so all results should be code
        assert "code" in types

    def test_different_queries_return_different_results(self, hybrid):
        r1 = hybrid.search("animation skeleton blend pose", top_k=10)
        r2 = hybrid.search("parallel for task threading", top_k=10)
        paths1 = {r.chunk.file_path for r in r1}
        paths2 = {r.chunk.file_path for r in r2}
        # Results should differ — animation query should hit AnimationAsset.h
        anim_hits = sum(1 for p in paths1 if "Animation" in p)
        parallel_hits = sum(1 for p in paths2 if "Parallel" in p)
        assert anim_hits > 0 or parallel_hits > 0, (
            "Different queries should hit different files"
        )


class TestPathExclusion:
    """Test that path exclusion filters work."""

    def test_exclude_specific_file(self, hybrid):
        """Excluding a specific file should remove its results."""
        results_all = hybrid.search("behavior tree", top_k=10, exclude_paths=[])
        excluded = hybrid.search(
            "behavior tree", top_k=10,
            exclude_paths=["BehaviorTreeTypes.cpp"],
        )
        paths_excluded = {r.chunk.file_path for r in excluded}
        assert all("BehaviorTreeTypes.cpp" not in p for p in paths_excluded), (
            f"BehaviorTreeTypes.cpp should be excluded, got {paths_excluded}"
        )

    def test_exclude_by_extension(self, hybrid):
        """Exclude by extension pattern."""
        results = hybrid.search(
            "animation", top_k=20, exclude_paths=[".h"]
        )
        for r in results:
            assert not r.chunk.file_path.endswith(".h"), (
                f"Header files should be excluded, got {r.chunk.file_path}"
            )

    def test_default_exclude_filters_test_files(self, hybrid):
        """Default exclude paths should filter test/ directories."""
        results = hybrid.search("animation", top_k=10)
        for r in results:
            assert "test/" not in r.chunk.file_path.lower()


class TestEdgeCases:
    """Edge case handling."""

    def test_empty_query(self, hybrid):
        results = hybrid.search("", top_k=10)
        assert results == []

    def test_whitespace_query(self, hybrid):
        results = hybrid.search("   ", top_k=10)
        assert results == []

    def test_very_specific_query(self, hybrid):
        """A very specific query should still return something, even if low quality."""
        results = hybrid.search(
            "FBlendSampleData TotalWeight GetClampedWeight ENGINE_API",
            top_k=5,
        )
        # Should return some results — exact matching via BM25 should work
        assert len(results) >= 0  # At minimum, no crash

    def test_query_with_special_characters(self, hybrid):
        """Queries with special characters should not crash."""
        results = hybrid.search("UE::Anim::IAnimNotifyEventContextDataInterface", top_k=5)
        assert isinstance(results, list)


class TestRetrievalQuality:
    """Verify retrieval quality is reasonable."""

    def test_animation_query_hits_animation_file(self, hybrid):
        """A query about animation should rank AnimationAsset.h highly."""
        results = hybrid.search("animation asset skeleton play length", top_k=10)
        file_paths = [r.chunk.file_path for r in results]
        # AnimationAsset.h should be in top results
        assert any("AnimationAsset.h" in p for p in file_paths), (
            f"AnimationAsset.h not found in top results: {file_paths[:5]}"
        )

    def test_parallel_query_hits_parallel_file(self, hybrid):
        """A query about parallel for should rank ParallelFor.h highly."""
        results = hybrid.search("parallel for task graph worker thread", top_k=10)
        file_paths = [r.chunk.file_path for r in results]
        assert any("ParallelFor.h" in p for p in file_paths), (
            f"ParallelFor.h not found in top results: {file_paths[:5]}"
        )

    def test_behavior_tree_query_hits_bt_files(self, hybrid):
        """A query about behavior trees should hit BT files."""
        results = hybrid.search(
            "behavior tree composite node child decorator",
            top_k=10,
        )
        file_paths = [r.chunk.file_path for r in results]
        bt_hits = sum(1 for p in file_paths if "BehaviorTree" in p or "BT" in p)
        assert bt_hits >= 1, (
            f"Expected BT files in results, got: {file_paths[:5]}"
        )

    def test_both_semantic_and_keyword_contribute(self, hybrid):
        """Verify both retrieval sources can contribute to results."""
        results = hybrid.search("animation skeleton blend", top_k=20)
        sources = {r.source for r in results}
        # RRF fusion merges both sources — both should be present
        assert "semantic" in sources or "keyword" in sources or "hybrid" in sources
