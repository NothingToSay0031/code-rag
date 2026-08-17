"""Unit tests for server.py helpers.

Covers three regression fixes:

1. ``_normalize_cpp_path``: Browse.VC.db stores file paths in upper-case;
   we must restore the actual on-disk case using :class:`MetadataStore`.
2. Reference rendering must emit nothing when there are no references —
   no ``(no references found)`` placeholder.
3. ``_resolve_file`` path-containment: absolute/rooted/drive-qualified paths
   and parent-directory escapes must never resolve to files outside an
   indexed repository.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from code_rag.models import FileInfo
from code_rag.storage.metadata import MetadataStore


# ── Lightweight stand-in so we don't depend on heavy ML deps in unit tests ──


@dataclass
class _FakeIndex:
    repo_path: Path
    metadata: "MetadataStore | None"
    prefix: str = ""
    _path_case_map: dict[str, str] | None = None


# ---------------------------------------------------------------------------
# _normalize_cpp_path
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_index(tmp_path):
    metadata = MetadataStore(tmp_path / "metadata.json")
    actual_rel = "Engine/Sources/Runtime/Plugins/Mega/Private/SubclassBridge.cpp"
    metadata.set_file_info(
        actual_rel,
        FileInfo(
            path=actual_rel, sha256="x", language="cpp", size=1, chunk_count=1
        ),
    )
    return _FakeIndex(repo_path=tmp_path, metadata=metadata)


def _import_server_helpers():
    """Import server module lazily to avoid heavy imports at collection."""
    pytest.importorskip("chromadb")
    from code_rag import server as srv

    return srv


def test_normalize_cpp_path_restores_real_case(fake_index):
    srv = _import_server_helpers()
    upper = (
        str(fake_index.repo_path)
        + "/ENGINE/SOURCES/RUNTIME/PLUGINS/MEGA/PRIVATE/SUBCLASSBRIDGE.CPP"
    )

    result = srv._normalize_cpp_path(fake_index, upper)

    assert result == "Engine/Sources/Runtime/Plugins/Mega/Private/SubclassBridge.cpp"


def test_normalize_cpp_path_handles_backslashes(fake_index):
    srv = _import_server_helpers()
    upper = (
        str(fake_index.repo_path).replace("/", "\\")
        + "\\ENGINE\\SOURCES\\RUNTIME\\PLUGINS\\MEGA\\PRIVATE\\SUBCLASSBRIDGE.CPP"
    )

    result = srv._normalize_cpp_path(fake_index, upper)

    assert result == "Engine/Sources/Runtime/Plugins/Mega/Private/SubclassBridge.cpp"


def test_normalize_cpp_path_unindexed_file_falls_back(fake_index):
    """File not in metadata: keep best-effort relative path (still better than absolute)."""
    srv = _import_server_helpers()
    upper = str(fake_index.repo_path) + "/UNKNOWN/Some.cpp"

    result = srv._normalize_cpp_path(fake_index, upper)

    # Strip prefix succeeded → relative path returned (case-insensitive lookup miss).
    assert result == "UNKNOWN/Some.cpp"


def test_normalize_cpp_path_outside_repo_returns_empty(fake_index):
    """Paths outside the repo must not be passed through (they'd enable reads
    beyond the indexed repositories); callers skip empty results."""
    srv = _import_server_helpers()
    foreign = "C:/Other/Repo/file.cpp"

    result = srv._normalize_cpp_path(fake_index, foreign)

    assert result == ""


# ---------------------------------------------------------------------------
# _render_refs — empty list must not emit "(no references found)"
# ---------------------------------------------------------------------------


def test_render_refs_empty_returns_empty_string():
    srv = _import_server_helpers()

    assert srv._render_refs([]) == ""


def test_render_refs_non_empty_renders_lines():
    srv = _import_server_helpers()

    rendered = srv._render_refs(
        [
            {
                "file_path": "src/a.py",
                "start_line": 10,
                "end_line": 10,
                "text": "foo()",
            }
        ]
    )

    assert "src/a.py" in rendered
    assert "L10" in rendered
    assert "no references found" not in rendered.lower()


# ---------------------------------------------------------------------------
# _resolve_file — path containment
# ---------------------------------------------------------------------------


def test_resolve_file_returns_local_path_for_single_index(fake_index):
    srv = _import_server_helpers()

    result = srv._resolve_file([fake_index], "src/main.py")

    assert result == (fake_index, "src/main.py")


def test_resolve_file_rejects_parent_traversal(fake_index):
    srv = _import_server_helpers()

    assert srv._resolve_file([fake_index], "../../etc/passwd") is None
    assert srv._resolve_file([fake_index], "..") is None
    assert srv._resolve_file([fake_index], "sub/../../../escape.txt") is None


def test_resolve_file_rejects_absolute_path(fake_index):
    srv = _import_server_helpers()
    absolute = str(Path("/") / "etc" / "passwd")

    assert srv._resolve_file([fake_index], absolute) is None


def test_resolve_file_rejects_rooted_path(fake_index):
    srv = _import_server_helpers()
    rooted = os.sep + "etc" + os.sep + "passwd"

    assert srv._resolve_file([fake_index], rooted) is None


@pytest.mark.skipif(os.name != "nt", reason="drive-relative paths are Windows-specific")
def test_resolve_file_rejects_drive_relative_path(fake_index):
    srv = _import_server_helpers()

    assert srv._resolve_file([fake_index], "C:foo") is None


def test_resolve_file_matches_index_prefix(tmp_path):
    srv = _import_server_helpers()
    sub = _FakeIndex(prefix="sub1", repo_path=tmp_path / "sub1", metadata=None)

    result = srv._resolve_file([sub], "sub1/src/main.py")

    assert result == (sub, "src/main.py")


def test_resolve_file_prefix_escape_rejected(fake_index, tmp_path):
    srv = _import_server_helpers()
    sub = _FakeIndex(prefix="sub1", repo_path=tmp_path / "sub1", metadata=None)

    # "sub1/../../other/secret.py" normalizes to "../other/secret.py", which
    # escapes the repo — must be rejected even though the prefix matches.
    result = srv._resolve_file([fake_index, sub], "sub1/../../other/secret.py")

    assert result is None


def test_resolve_file_stripped_local_escape_rejected(tmp_path):
    """A path that is safe as a whole can still escape AFTER prefix strip:
    'proj/../foo.cpp' normalizes to 'foo.cpp' (passes the global check), but
    stripping the 'proj' prefix yields '../foo.cpp' — must be rejected."""
    srv = _import_server_helpers()
    sub = _FakeIndex(prefix="proj", repo_path=tmp_path / "proj", metadata=None)

    assert srv._resolve_file([sub], "proj/../foo.cpp") is None
    assert srv._resolve_file([sub], "proj/x/../../escape.txt") is None


@pytest.mark.skipif(os.name != "nt", reason="drive-qualified locals are Windows-specific")
def test_resolve_file_stripped_drive_escape_rejected(tmp_path):
    """'proj/D:/secret.txt' passes the global check (drive not at position 0),
    but the stripped local 'D:/secret.txt' would escape to another drive."""
    srv = _import_server_helpers()
    sub = _FakeIndex(prefix="proj", repo_path=tmp_path / "proj", metadata=None)

    assert srv._resolve_file([sub], "proj/D:/secret.txt") is None


def test_resolve_file_longest_prefix_wins(tmp_path):
    """Nested prefixes ('a' vs 'a/b') must resolve to the deeper index."""
    srv = _import_server_helpers()
    outer = _FakeIndex(prefix="a", repo_path=tmp_path / "a", metadata=None)
    inner = _FakeIndex(prefix="a/b", repo_path=tmp_path / "a" / "b", metadata=None)

    result = srv._resolve_file([outer, inner], "a/b/x.cpp")

    assert result == (inner, "x.cpp")


@pytest.mark.skipif(os.name != "nt", reason="backslash-qualified paths are Windows-specific")
def test_resolve_file_matches_prefix_with_backslashes(tmp_path):
    """Windows clients send backslashes; prefix matching must still work."""
    srv = _import_server_helpers()
    sub = _FakeIndex(prefix="sub1", repo_path=tmp_path / "sub1", metadata=None)

    result = srv._resolve_file([sub], r"sub1\src\main.py")

    assert result == (sub, "src/main.py")
