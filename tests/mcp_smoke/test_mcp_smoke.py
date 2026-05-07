from __future__ import annotations

import os

import pytest

from .artifacts import write_artifacts
from .fixture_repo import FIXTURE_FILES, build_fixture_repo
from .index_runtime import (
    ensure_index,
    resolve_artifact_root,
    resolve_fixture_repo_path,
)
from .tool_runner import run_all_tools


def _force_reindex() -> bool:
    return os.environ.get("MCP_SMOKE_FORCE_REINDEX", "").strip() == "1"


def test_fixture_repo_contains_expected_files():
    repo = build_fixture_repo(resolve_fixture_repo_path())
    for rel_path in FIXTURE_FILES:
        assert (repo / rel_path).exists(), f"Missing fixture file: {rel_path}"


def test_mcp_smoke_end_to_end():
    pytest.importorskip("chromadb")

    repo = build_fixture_repo(resolve_fixture_repo_path())
    ensure_index(repo, force_reindex=_force_reindex())

    results = run_all_tools(repo)
    run_dir = write_artifacts(results, artifact_root=resolve_artifact_root())

    assert (run_dir / "summary.json").exists()
    assert (run_dir / "stdout.txt").exists()
    for row in results:
        assert (run_dir / "calls" / f"{row['tool']}.json").exists()
        assert (run_dir / "calls" / f"{row['tool']}.txt").exists()

    failures = [r for r in results if not r["ok"]]
    assert not failures, f"Some MCP calls failed. See artifacts at: {run_dir}"

