"""Unit tests for MetadataStore."""

from __future__ import annotations

import json

import pytest

from code_rag.models import FileInfo
from code_rag.storage.metadata import MetadataStore
from tests.conftest import make_symbol


class TestMetadataStore:
    @pytest.fixture
    def store(self, tmp_path):
        return MetadataStore(tmp_path / "metadata.json")

    def test_empty_store(self, store):
        assert store.count_files() == 0
        assert store.count_symbols() == 0
        assert store.get_all_files() == []

    def test_set_and_get_file_info(self, store):
        info = FileInfo(
            path="src/foo.py",
            sha256="abc123",
            language="python",
            size=100,
            chunk_count=3,
        )
        store.set_file_info("src/foo.py", info)
        result = store.get_file_info("src/foo.py")
        assert result is not None
        assert result.sha256 == "abc123"
        assert result.language == "python"
        assert result.chunk_count == 3

    def test_get_missing_file_returns_none(self, store):
        assert store.get_file_info("does_not_exist.py") is None

    def test_get_fingerprint(self, store):
        info = FileInfo(
            path="a.py", sha256="deadbeef", language="python", size=50, chunk_count=1
        )
        store.set_file_info("a.py", info)
        assert store.get_fingerprint("a.py") == "deadbeef"
        assert store.get_fingerprint("missing.py") is None

    def test_set_and_find_symbol_exact(self, store):
        sym = make_symbol(name="PlayerController", kind="class", file_path="player.py")
        store.set_symbols("player.py", [sym])
        results = store.find_symbol("PlayerController")
        assert len(results) == 1
        assert results[0].name == "PlayerController"

    def test_find_symbol_case_insensitive(self, store):
        sym = make_symbol(name="GameManager")
        store.set_symbols("game.py", [sym])
        results = store.find_symbol("gamemanager")
        assert len(results) == 1
        results2 = store.find_symbol("GAMEMANAGER")
        assert len(results2) == 1

    def test_find_symbol_substring(self, store):
        sym = make_symbol(name="PlayerInputHandler")
        store.set_symbols("input.py", [sym])
        results = store.find_symbol("Input")
        assert any(s.name == "PlayerInputHandler" for s in results)

    def test_find_symbol_not_found(self, store):
        assert store.find_symbol("NonExistentSymbol") == []

    def test_remove_file(self, store):
        info = FileInfo(
            path="del.py", sha256="x", language="python", size=10, chunk_count=1
        )
        sym = make_symbol(name="ToDelete", file_path="del.py")
        store.set_file_info("del.py", info)
        store.set_symbols("del.py", [sym])
        store.remove_file("del.py")
        assert store.get_file_info("del.py") is None
        assert store.find_symbol("ToDelete") == []
        assert store.count_files() == 0

    def test_save_and_reload(self, tmp_path):
        path = tmp_path / "metadata.json"
        store = MetadataStore(path)
        info = FileInfo(
            path="foo.cs", sha256="cafebabe", language="csharp", size=200, chunk_count=5
        )
        sym = make_symbol(
            name="MyClass", kind="class", file_path="foo.cs", language="csharp"
        )
        store.set_file_info("foo.cs", info)
        store.set_symbols("foo.cs", [sym])
        store.save()

        store2 = MetadataStore(path)
        assert store2.count_files() == 1
        assert store2.get_file_info("foo.cs").sha256 == "cafebabe"
        syms = store2.find_symbol("MyClass")
        assert len(syms) == 1
        assert syms[0].language == "csharp"

    def test_loads_legacy_windows_paths_as_posix(self, tmp_path):
        path = tmp_path / "metadata.json"
        path.write_text(
            json.dumps(
                {
                    "files": {
                        "src\\calc.py": {
                            "path": "src\\calc.py",
                            "sha256": "abc",
                            "language": "python",
                            "size": 120,
                            "chunk_count": 2,
                        }
                    },
                    "symbols": {
                        "src\\calc.py": [
                            {
                                "name": "add",
                                "kind": "function",
                                "file_path": "src\\calc.py",
                                "start_line": 1,
                                "end_line": 2,
                                "language": "python",
                            }
                        ]
                    },
                    "chunk_map": {"src\\calc.py": [0, 1]},
                }
            ),
            encoding="utf-8",
        )

        store = MetadataStore(path)

        assert store.get_file_info("src/calc.py") is not None
        assert store.get_symbols("src/calc.py")[0].name == "add"
        assert store.get_symbols("src/calc.py")[0].file_path == "src/calc.py"
        assert store.get_chunk_ids("src/calc.py") == [0, 1]

    def test_get_changed_files(self, store):
        # Seed with two files
        for name in ("a.py", "b.py"):
            store.set_file_info(
                name,
                FileInfo(
                    path=name, sha256="v1", language="python", size=10, chunk_count=1
                ),
            )

        current = {
            "a.py": "v2",  # modified
            "c.py": "v1",  # new
            # b.py is deleted
        }
        new, modified, deleted = store.get_changed_files(current)
        assert new == ["c.py"]
        assert modified == ["a.py"]
        assert deleted == ["b.py"]

    def test_chunk_ids(self, store):
        store.set_chunk_ids("foo.py", [10, 11, 12])
        assert store.get_chunk_ids("foo.py") == [10, 11, 12]
        assert store.get_chunk_ids("missing.py") == []
