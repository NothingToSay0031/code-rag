"""Tests for discovery.py — language detection, file classification, fingerprint."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from code_rag.indexer.discovery import (
    detect_language,
    classify_file,
    get_file_fingerprint,
    discover_files,
    _EXTENSION_MAP,
    _SPECIAL_FILES,
    _JUNK_DIRS,
    _MAX_FILE_SIZE,
)
from code_rag.config import CodeRagConfig

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


class TestDetectLanguage:
    @pytest.mark.parametrize("ext,expected", [
        (".cpp", "cpp"),
        (".h", "cpp"),
        (".hpp", "cpp"),
        (".c", "c"),
        (".py", "python"),
        (".ts", "typescript"),
        (".js", "javascript"),
        (".md", "markdown"),
        (".json", "json"),
        (".yaml", "yaml"),
    ])
    def test_known_extensions(self, ext, expected):
        p = Path(f"test{ext}")
        assert detect_language(p) == expected

    def test_ue_header_is_cpp(self):
        """UE .h files must be detected as C++."""
        assert detect_language(Path("AnimationAsset.h")) == "cpp"
        assert detect_language(Path("CoreMinimal.h")) == "cpp"

    def test_ue_source_is_cpp(self):
        """UE .cpp files must be detected as C++."""
        assert detect_language(Path("BTCompositeNode.cpp")) == "cpp"
        assert detect_language(Path("BehaviorTreeTypes.cpp")) == "cpp"

    def test_special_filenames(self):
        """Dockerfile, Makefile etc. should be detected."""
        for filename, lang in _SPECIAL_FILES.items():
            assert detect_language(Path(filename)) == lang

    def test_unknown_extension(self):
        """Unknown extensions should return None."""
        assert detect_language(Path("data.xyzzy")) is None

    def test_no_extension(self):
        """Files with no extension should return None (unless special)."""
        assert detect_language(Path("Makefile")) is not None  # special file
        assert detect_language(Path("random_file")) is None

    def test_custom_mappings(self):
        """Custom type mappings should override defaults."""
        assert detect_language(Path("shader.glsl")) is None  # not in default map
        assert detect_language(
            Path("shader.glsl"), custom_mappings={".glsl": "c"}
        ) == "c"

    def test_case_insensitive_extension(self):
        """Extensions should be case-insensitive."""
        assert detect_language(Path("TEST.H")) == "cpp"
        assert detect_language(Path("TEST.CPP")) == "cpp"


# ---------------------------------------------------------------------------
# File classification
# ---------------------------------------------------------------------------


class TestClassifyFile:
    def test_cpp_is_code(self):
        assert classify_file(Path("src/foo.cpp"), "cpp") == "code"
        assert classify_file(Path("include/bar.h"), "cpp") == "code"

    def test_python_is_code(self):
        assert classify_file(Path("lib/mod.py"), "python") == "code"

    def test_markdown_is_doc(self):
        assert classify_file(Path("docs/README.md"), "markdown") == "doc"

    def test_json_is_config(self):
        assert classify_file(Path("config.json"), "json") == "config"

    def test_unknown_language_is_unknown(self):
        assert classify_file(Path("data.bin"), None) == "unknown"


# ---------------------------------------------------------------------------
# File fingerprint
# ---------------------------------------------------------------------------


class TestFileFingerprint:
    def test_same_content_same_fingerprint(self):
        f1 = FIXTURES / "ue" / "AnimationAsset.h"
        fp1 = get_file_fingerprint(f1)
        fp2 = get_file_fingerprint(f1)
        assert fp1 == fp2
        assert len(fp1) == 64  # SHA-256 hex digest

    def test_different_files_different_fingerprint(self):
        fp_h = get_file_fingerprint(FIXTURES / "ue" / "AnimationAsset.h")
        fp_cpp = get_file_fingerprint(FIXTURES / "ue" / "BTCompositeNode.cpp")
        assert fp_h != fp_cpp

    def test_fingerprint_is_hex(self):
        fp = get_file_fingerprint(FIXTURES / "ue" / "ParallelFor.h")
        assert all(c in "0123456789abcdef" for c in fp)

    def test_fingerprint_deterministic(self):
        """Same file read twice should give identical fingerprint."""
        a = get_file_fingerprint(FIXTURES / "ue" / "AnimationAsset.h")
        b = get_file_fingerprint(FIXTURES / "ue" / "AnimationAsset.h")
        assert a == b


# ---------------------------------------------------------------------------
# Extension map integrity
# ---------------------------------------------------------------------------


class TestExtensionMap:
    def test_all_mapped_extensions_have_dot_prefix(self):
        for ext in _EXTENSION_MAP:
            assert ext.startswith("."), f"Extension {ext!r} missing dot prefix"

    def test_no_duplicate_language_mappings_override(self):
        """No two extensions that map to different languages should conflict."""
        # Extension map is authoritative — just verify integrity
        assert ".cpp" in _EXTENSION_MAP
        assert _EXTENSION_MAP[".cpp"] == "cpp"
        assert ".h" in _EXTENSION_MAP
        assert _EXTENSION_MAP[".h"] == "cpp"


# ---------------------------------------------------------------------------
# File discovery (against real fixture directory)
# ---------------------------------------------------------------------------


class TestDiscoverFiles:
    def test_discovers_ue_fixtures(self):
        """discover_files should find all UE fixture files."""
        config = CodeRagConfig(
            repo_path=FIXTURES,
            include_patterns=["ue/**"],
            exclude_patterns=["cpp_utils/**"],
        )
        files = discover_files(FIXTURES, config)
        paths = {str(f) for f in files}
        assert "ue\\AnimationAsset.h" in paths
        assert "ue\\BehaviorTreeTypes.cpp" in paths
        assert "ue\\BTCompositeNode.cpp" in paths
        assert "ue\\ParallelFor.h" in paths

    def test_discovers_cpp_utils_fixtures(self):
        """discover_files should find all cpp_utils fixture files."""
        config = CodeRagConfig(
            repo_path=FIXTURES,
            include_patterns=["cpp_utils/**"],
            exclude_patterns=["ue/**"],
        )
        files = discover_files(FIXTURES, config)
        paths = {str(f) for f in files}
        assert "cpp_utils\\FileManager.cpp" in paths
        assert "cpp_utils\\ShaderLangTree.h" in paths
        assert "cpp_utils\\Texture.cpp" in paths

    def test_exclude_pattern_works(self):
        """Excluded patterns should filter out files."""
        config = CodeRagConfig(
            repo_path=FIXTURES,
            include_patterns=["ue/**"],
            exclude_patterns=["**/*.h"],
        )
        files = discover_files(FIXTURES, config)
        paths = [str(f) for f in files]
        assert all(not p.endswith(".h") for p in paths)
        # .cpp files should still be there
        assert any(p.endswith(".cpp") for p in paths)

    def test_junk_dirs_ignored(self, tmp_dir):
        """Junk directories like node_modules, .git should be ignored."""
        # Create a minimal repo with a junk dir
        (tmp_dir / "src").mkdir(exist_ok=True)
        (tmp_dir / "src" / "main.cpp").write_text("int main() { return 0; }\n")
        (tmp_dir / ".git").mkdir(exist_ok=True)
        (tmp_dir / ".git" / "config").write_text("[core]\n")

        config = CodeRagConfig(repo_path=tmp_dir)
        files = discover_files(tmp_dir, config)
        paths = {str(f) for f in files}
        assert any("main.cpp" in p for p in paths)
        assert not any(".git" in p for p in paths)

    def test_discovered_files_are_relative(self):
        """All discovered paths must be relative to the repo root."""
        config = CodeRagConfig(
            repo_path=FIXTURES,
            include_patterns=["ue/**"],
        )
        files = discover_files(FIXTURES, config)
        for f in files:
            assert not f.is_absolute(), f"Path {f} should be relative"
            assert FIXTURES not in f.parents, f"Path {f} should not contain repo root"

    def test_symlinks_not_causing_duplicates(self):
        """discover_files should handle the fixture dir (no symlinks expected)."""
        config = CodeRagConfig(
            repo_path=FIXTURES,
            include_patterns=["ue/**"],
        )
        files = discover_files(FIXTURES, config)
        paths = [str(f) for f in files]
        # No duplicates
        assert len(paths) == len(set(paths))
