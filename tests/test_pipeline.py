"""Tests for pipeline.py — end-to-end indexing, incremental updates, crash recovery.

Uses real UE engine source files as fixtures with BGE-small embeddings.
"""

from __future__ import annotations

import gc
import shutil
import tempfile
from pathlib import Path

import pytest

from code_rag.config import CodeRagConfig
from code_rag.indexer.pipeline import IndexPipeline

UE_FIXTURES = Path(__file__).parent / "fixtures" / "ue"


def _copy_ue_fixtures(dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for src in UE_FIXTURES.iterdir():
        if src.is_file():
            shutil.copy2(src, dst / src.name)


@pytest.fixture
def ue_repo(request):
    """Temporary repo with UE fixtures.

    Windows note: ChromaDB SQLite locks may prevent some files from being
    deleted.  ``shutil.rmtree(…, ignore_errors=True)`` suppresses those
    errors — leftover temp directories are cleaned by the OS eventually.
    """
    td = tempfile.mkdtemp()
    repo = Path(td)
    _copy_ue_fixtures(repo)

    def cleanup():
        gc.collect()
        try:
            shutil.rmtree(td, ignore_errors=False)
        except PermissionError:
            # ChromaDB SQLite WAL/SHM files may still be locked on Windows
            shutil.rmtree(td, ignore_errors=True)

    request.addfinalizer(cleanup)
    return repo


def _make_config(repo: Path) -> CodeRagConfig:
    return CodeRagConfig(
        repo_path=repo,
        model_name="BAAI/bge-small-en-v1.5",
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Basic indexing
# ---------------------------------------------------------------------------


class TestBasicIndexing:
    def test_pipeline_initializes_without_error(self, ue_repo):
        pipe = IndexPipeline(_make_config(ue_repo))
        assert pipe is not None
        assert not pipe.is_indexed
        pipe._vector_store.close()

    def test_pipeline_run_discovers_ue_files(self, ue_repo):
        pipe = IndexPipeline(_make_config(ue_repo))
        stats = pipe.run()
        assert stats.files_processed >= 4
        assert stats.files_skipped == 0
        assert stats.files_deleted == 0
        assert stats.chunks_created > 0
        assert stats.elapsed_seconds > 0
        pipe._vector_store.close()

    def test_pipeline_produces_chunks_for_all_files(self, ue_repo):
        pipe = IndexPipeline(_make_config(ue_repo))
        pipe.run()
        for name in [
            "AnimationAsset.h",
            "BehaviorTreeTypes.cpp",
            "BTCompositeNode.cpp",
            "ParallelFor.h",
        ]:
            info = pipe._metadata.get_file_info(name)
            assert info is not None, f"No metadata for {name}"
            assert info.chunk_count > 0, f"{name} has 0 chunks"
            assert info.language == "cpp", f"{name} language={info.language}"
            assert info.size > 0
        pipe._vector_store.close()

    def test_pipeline_stores_symbols(self, ue_repo):
        pipe = IndexPipeline(_make_config(ue_repo))
        pipe.run()
        symbols = pipe._metadata.get_symbols("AnimationAsset.h")
        assert len(symbols) > 0
        kinds = {s.kind for s in symbols}
        assert "class" in kinds
        assert len([s for s in symbols if s.kind == "class"]) >= 1
        pipe._vector_store.close()

    def test_pipeline_is_indexed_after_run(self, ue_repo):
        pipe = IndexPipeline(_make_config(ue_repo))
        assert not pipe.is_indexed
        pipe.run()
        assert pipe.is_indexed
        assert pipe._metadata.count_files() >= 4
        pipe._vector_store.close()

    def test_pipeline_vector_store_has_entries(self, ue_repo):
        pipe = IndexPipeline(_make_config(ue_repo))
        stats = pipe.run()
        assert pipe._vector_store.count() == stats.chunks_created
        pipe._vector_store.close()

    def test_pipeline_bm25_store_has_entries(self, ue_repo):
        pipe = IndexPipeline(_make_config(ue_repo))
        stats = pipe.run()
        assert pipe._bm25_store.count() == stats.chunks_created
        pipe._vector_store.close()


# ---------------------------------------------------------------------------
# Incremental indexing — DO NOT close pipeline1's VectorStore between runs
# because pipeline2 must re-open the same ChromaDB database.
# ---------------------------------------------------------------------------


class TestIncrementalIndexing:
    def test_second_run_skips_all_unchanged_files(self, ue_repo):
        config = _make_config(ue_repo)
        pipe1 = IndexPipeline(config)
        pipe1.run()
        # Don't close — pipeline2 needs the ChromaDB database

        pipe2 = IndexPipeline(config)
        stats2 = pipe2.run()
        assert stats2.files_processed == 0
        assert stats2.files_skipped >= 4
        assert stats2.files_deleted == 0
        assert stats2.chunks_created == 0
        pipe1._vector_store.close()
        pipe2._vector_store.close()

    def test_modified_file_is_reprocessed(self, ue_repo):
        config = _make_config(ue_repo)
        pipe1 = IndexPipeline(config)
        pipe1.run()

        target = ue_repo / "ParallelFor.h"
        original = target.read_bytes()
        target.write_bytes(original + b"\n// Modified for test\n")

        pipe2 = IndexPipeline(config)
        stats2 = pipe2.run()
        assert stats2.files_processed == 1
        assert stats2.files_skipped == 3
        assert stats2.files_deleted == 0
        assert stats2.chunks_created > 0
        pipe1._vector_store.close()
        pipe2._vector_store.close()

    def test_deleted_file_is_removed(self, ue_repo):
        config = _make_config(ue_repo)
        pipe1 = IndexPipeline(config)
        pipe1.run()
        original_count = pipe1._metadata.count_files()

        (ue_repo / "BehaviorTreeTypes.cpp").unlink()

        pipe2 = IndexPipeline(config)
        stats2 = pipe2.run()
        assert stats2.files_deleted == 1
        assert stats2.files_processed == 0
        assert pipe2._metadata.count_files() == original_count - 1
        assert pipe2._metadata.get_file_info("BehaviorTreeTypes.cpp") is None
        pipe1._vector_store.close()
        pipe2._vector_store.close()

    def test_new_file_is_indexed(self, ue_repo):
        config = _make_config(ue_repo)
        pipe1 = IndexPipeline(config)
        pipe1.run()

        (ue_repo / "NewClass.h").write_text(
            """class UNewClass {
public:
    void DoSomething();
    int32 GetValue() const { return Value; }
private:
    int32 Value = 0;
};
"""
        )

        pipe2 = IndexPipeline(config)
        stats2 = pipe2.run()
        assert stats2.files_processed == 1
        assert stats2.files_skipped == 4
        assert stats2.chunks_created > 0
        assert pipe2._metadata.get_file_info("NewClass.h") is not None
        pipe1._vector_store.close()
        pipe2._vector_store.close()

    def test_modifying_and_adding_then_reindexing(self, ue_repo):
        config = _make_config(ue_repo)
        pipe1 = IndexPipeline(config)
        pipe1.run()

        target = ue_repo / "AnimationAsset.h"
        original = target.read_bytes()
        target.write_bytes(original + b"\n// Modified header\n")

        (ue_repo / "NewUtils.h").write_text(
            """#pragma once
namespace Utils {
    inline float Clamp01(float x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
}
"""
        )
        (ue_repo / "BTCompositeNode.cpp").unlink()

        pipe2 = IndexPipeline(config)
        stats2 = pipe2.run()
        assert stats2.files_processed == 2
        assert stats2.files_deleted == 1
        assert pipe2._metadata.get_file_info("NewUtils.h") is not None
        assert pipe2._metadata.get_file_info("BTCompositeNode.cpp") is None
        assert pipe2._metadata.get_file_info("AnimationAsset.h") is not None
        pipe1._vector_store.close()
        pipe2._vector_store.close()


# ---------------------------------------------------------------------------
# Chunk quality
# ---------------------------------------------------------------------------


class TestChunkQuality:
    def test_gap_chunks_capture_includes(self, ue_repo):
        pipe = IndexPipeline(_make_config(ue_repo))
        pipe.run()
        for fp in pipe._metadata.get_all_files():
            chunk_ids = pipe._metadata.get_chunk_ids(fp)
            assert chunk_ids, f"No chunks for {fp}"
        pipe._vector_store.close()

    def test_chunks_not_empty(self, ue_repo):
        pipe = IndexPipeline(_make_config(ue_repo))
        stats = pipe.run()
        assert stats.chunks_created >= 10
        pipe._vector_store.close()

    def test_large_file_produces_more_chunks(self, ue_repo):
        pipe = IndexPipeline(_make_config(ue_repo))
        pipe.run()
        anim_info = pipe._metadata.get_file_info("AnimationAsset.h")
        assert anim_info.chunk_count >= 15, (
            f"AnimationAsset.h has {anim_info.chunk_count} chunks, expected >= 15"
        )
        pipe._vector_store.close()


# ---------------------------------------------------------------------------
# BM25 rebuild (crash recovery)
# ---------------------------------------------------------------------------


class TestBM25Rebuild:
    def test_rebuild_after_corruption(self, ue_repo):
        pipe = IndexPipeline(_make_config(ue_repo))
        pipe.run()
        bm25_count_before = pipe._bm25_store.count()
        assert bm25_count_before > 0

        pipe._bm25_store._file_map.clear()
        pipe._bm25_store._corpus.clear()
        pipe._bm25_store._index = None
        assert pipe._bm25_store.count() == 0

        pipe._rebuild_bm25_if_needed(pipe._config)
        assert pipe._bm25_store.count() == bm25_count_before
        pipe._vector_store.close()


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------


class TestConfigPersistence:
    def test_config_saved_on_init(self, ue_repo):
        IndexPipeline(_make_config(ue_repo))
        assert (ue_repo / ".code-rag" / "config.json").exists()

    def test_config_round_trips(self, ue_repo):
        config = _make_config(ue_repo)
        IndexPipeline(config)
        loaded = CodeRagConfig.load(ue_repo)
        assert loaded.model_name == "BAAI/bge-small-en-v1.5"
        assert loaded.device == "cpu"


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_stats_after_indexing(self, ue_repo):
        pipe = IndexPipeline(_make_config(ue_repo))
        pipe.run()
        s = pipe.get_stats()
        assert s["files"] >= 4
        assert s["chunks"] > 0
        assert s["symbols"] > 0
        pipe._vector_store.close()
