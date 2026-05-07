from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "tests" / "artifacts" / "mcp-smoke"
DEFAULT_WORKSPACE_ROOT = DEFAULT_ARTIFACT_ROOT / "_workspace"
DEFAULT_FIXTURE_REPO = DEFAULT_WORKSPACE_ROOT / "fixture_repo"


def resolve_artifact_root() -> Path:
    override = os.environ.get("MCP_SMOKE_ARTIFACT_ROOT", "").strip()
    if override:
        return Path(override)
    return DEFAULT_ARTIFACT_ROOT


def resolve_fixture_repo_path() -> Path:
    return DEFAULT_FIXTURE_REPO


def ensure_index(repo_root: Path, force_reindex: bool) -> None:
    """Create index once and reuse by default unless force_reindex is enabled."""
    data_dir = repo_root / ".code-rag"
    if force_reindex and data_dir.exists():
        shutil.rmtree(data_dir)
    if data_dir.exists():
        return

    env = os.environ.copy()
    src_dir = str(PROJECT_ROOT / "src")
    current_py = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = src_dir if not current_py else f"{src_dir};{current_py}"

    cmd = [
        sys.executable,
        "-c",
        "from code_rag.cli import main; main()",
        "init",
        str(repo_root),
        "--device",
        "cpu",
    ]
    run = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    if run.returncode != 0:
        detail = (
            "code-rag init failed.\n"
            f"stdout:\n{run.stdout}\n"
            f"stderr:\n{run.stderr}\n"
        )
        raise RuntimeError(detail)

