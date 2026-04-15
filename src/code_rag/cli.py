from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import click


_CONFIG_FILE = Path.home() / ".config" / "code-rag" / "config.json"


# ---------------------------------------------------------------------------
# Project-level MCP config helpers
# ---------------------------------------------------------------------------

_MCP_CLIENTS = {
    "opencode": {
        "files": ["opencode.json"],
        "schema": "https://opencode.ai/config.json",
    },
}


def _build_mcp_entry(repo_path: str, executable: str = "code-rag") -> dict:
    """Build the MCP server entry for a given repo path."""
    return {
        "type": "local",
        "command": [executable, "serve", "--repo", repo_path],
        "enabled": True,
    }


def _merge_json_file(path: Path, updates: dict) -> None:
    """Deep-merge *updates* into the JSON file at *path* (creates if missing)."""
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    def _deep_merge(base: dict, override: dict) -> dict:
        result = dict(base)
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = _deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    merged = _deep_merge(existing, updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_project_mcp_configs(
    repo_path: Path,
    executable: str = "code-rag",
    clients: list[str] | None = None,
) -> list[Path]:
    """Write project-level MCP configs for all supported AI clients.

    Returns the list of files written.
    """
    if clients is None:
        clients = list(_MCP_CLIENTS.keys())

    repo_abs = str(repo_path.resolve())
    mcp_entry = _build_mcp_entry(repo_abs, executable)
    written: list[Path] = []

    for client in clients:
        info = _MCP_CLIENTS.get(client)
        if not info:
            continue
        for rel_file in info["files"]:
            target = repo_path / rel_file
            # Build the update payload matching each client's schema
            if client == "opencode":
                update = {"mcp": {"code-rag": mcp_entry}}
                if "schema" in info:
                    update["$schema"] = info["schema"]  # type: ignore[assignment]
            else:
                continue
            _merge_json_file(target, update)
            written.append(target)

    return written


def _detect_executable() -> str:
    """Return the best 'code-rag' executable path to embed in configs."""
    # Prefer the currently running executable
    current = Path(sys.executable).parent / "code-rag"
    if current.exists():
        return str(current)
    current_exe = Path(sys.executable).parent / "code-rag.exe"
    if current_exe.exists():
        return str(current_exe)
    # Fall back to bare name (assumes it's in PATH)
    return "code-rag"


def _ensure_environment():
    """Auto-setup: ensure dependencies are installed via uv on first run.

    If running outside a uv-managed environment, attempt to bootstrap
    by running `uv sync` in the project directory.
    """
    import sys

    # sys.prefix != sys.base_prefix is the canonical Python check for
    # "am I running inside a virtual environment?".  This covers all venv
    # types (venv, virtualenv, uv) regardless of whether the shell has
    # activated the venv (VIRTUAL_ENV / UV_VIRTUAL_ENV are only set on
    # activation, NOT when the venv's Scripts/bin executable is run directly).
    if sys.prefix != sys.base_prefix:
        return  # Already inside a venv — never touch its packages

    # Legacy fallback for edge cases where sys.prefix check doesn't catch it
    if os.environ.get("UV_VIRTUAL_ENV") or os.environ.get("VIRTUAL_ENV"):
        return

    # Check if uv is available
    uv_path = shutil.which("uv")
    if not uv_path:
        return  # No uv, assume deps are manually managed

    # Try to auto-install if we detect a pyproject.toml nearby
    project_root = Path(__file__).resolve().parent.parent.parent
    if (project_root / "pyproject.toml").exists():
        try:
            subprocess.run(
                [uv_path, "sync", "--quiet"],
                cwd=str(project_root),
                check=True,
                timeout=300,
            )
        except subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError:
            pass  # Best effort — don't block the user


def _save_last_repo(repo_path: str):
    _CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_FILE.write_text(json.dumps({"repo_path": repo_path}), encoding="utf-8")


def _load_last_repo() -> str | None:
    if _CONFIG_FILE.exists():
        try:
            data = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
            return data.get("repo_path")
        except Exception:
            pass
    return None


@click.group()
@click.version_option(version="0.1.0")
def main():
    """code-rag: Local code repository RAG with MCP server."""
    _ensure_environment()


@main.command()
@click.argument(
    "repo_path",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
)
@click.option(
    "--include", "include_patterns", multiple=True, help="Include file patterns"
)
@click.option(
    "--exclude", "exclude_patterns", multiple=True, help="Exclude file patterns"
)
@click.option("--device", default="auto", help="Device: auto, cpu, or cuda")
@click.option(
    "--model",
    "model_name",
    default=None,
    help=(
        "Embedding model to use for this index. "
        "Defaults to Qwen/Qwen3-Embedding-0.6B. "
        "Examples: BAAI/bge-large-en-v1.5, Qwen/Qwen3-Embedding-4B"
    ),
)
def init(
    repo_path: str,
    include_patterns: tuple,
    exclude_patterns: tuple,
    device: str,
    model_name: str | None,
):
    """Index a code repository for RAG search.

    REPO_PATH: Path to the repository to index.
    """
    from code_rag.config import CodeRagConfig
    from code_rag.indexer.pipeline import IndexPipeline

    kwargs: dict = dict(
        repo_path=Path(repo_path),
        device=device,
        include_patterns=[p.replace("\\", "/") for p in include_patterns],
        exclude_patterns=[p.replace("\\", "/") for p in exclude_patterns],
    )
    if model_name:
        kwargs["model_name"] = model_name

    config = CodeRagConfig(**kwargs)

    click.echo(f"Indexing repository: {repo_path}")
    click.echo(f"Device: {config.resolve_device()}")
    click.echo(f"Model: {config.resolve_model_name()}")

    pipeline = IndexPipeline(config)
    stats = pipeline.run()

    # Persist repo path for zero-config `serve`
    resolved_repo = Path(repo_path).resolve()
    _save_last_repo(str(resolved_repo))

    click.echo("\nIndexing complete!")
    click.echo(f"  Files processed: {stats.files_processed}")
    click.echo(f"  Chunks created: {stats.chunks_created}")
    click.echo(f"  Files skipped: {stats.files_skipped}")
    click.echo(f"  Time: {stats.elapsed_seconds:.1f}s")

    # Write project-level MCP configs so AI clients only see this MCP
    # when the repo is open (instead of a global always-on config).
    exe = _detect_executable()
    written = _write_project_mcp_configs(resolved_repo, executable=exe)
    if written:
        click.echo("\nProject-level MCP configs written:")
        for p in written:
            click.echo(f"  {p}")
        click.echo(
            "\nTip: remove the global 'code-rag' MCP entry from your AI client's\n"
            "     global config so the MCP is only active in indexed projects."
        )


@main.command()
@click.option(
    "--repo",
    "repo_path",
    envvar="CODE_RAG_REPO",
    default=".",
    help="Path to indexed repository",
)
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="MCP transport",
)
@click.option("--port", default=8000, type=int, help="Port for SSE transport")
def serve(repo_path: str, transport: str, port: int):
    """Start the MCP server.

    Reads CODE_RAG_REPO environment variable or --repo option.
    Falls back to the last initialized repository.
    Supports multi-index: if subdirectories contain .code-rag, they are
    all served as a combined index.
    """
    import os

    resolved = Path(repo_path).resolve()
    # If user didn't specify --repo (using default "."), check for indexed data
    if repo_path == "." and not (resolved / ".code-rag").exists():
        # Check subdirectories for indices before falling back
        sub_indices = list(resolved.rglob(".code-rag"))
        if not sub_indices:
            # No subdirectory indices — fall back to last saved repo
            saved = _load_last_repo()
            if saved:
                saved_path = Path(saved)
                has_root_index = (saved_path / ".code-rag").exists()
                has_sub_indices = (
                    not has_root_index
                    and saved_path.is_dir()
                    and any(saved_path.rglob(".code-rag"))
                )
                if has_root_index or has_sub_indices:
                    resolved = saved_path
        # else: subdirectory indices exist — let server.py discover them

    os.environ["CODE_RAG_REPO"] = str(resolved)

    from code_rag.server import mcp, _get_state

    # Log to stderr to avoid interfering with MCP stdio protocol
    click.echo("Starting CodeRAG MCP server...", err=True)
    click.echo(f"Repository: {resolved}", err=True)
    click.echo(f"Transport: {transport}", err=True)

    # Eagerly initialize: load model, open indices, build BM25.
    # This runs BEFORE the MCP server accepts requests, avoiding
    # timeouts on the first tool call.
    click.echo("Loading model and indices (this may take a moment)...", err=True)
    try:
        state = _get_state()
        n_indices = len(state.get("indices", []))
        click.echo(f"Ready: {n_indices} index(es) loaded.", err=True)
    except Exception as exc:
        click.echo(
            f"Warning: pre-load failed ({exc}). Will retry on first request.", err=True
        )

    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="sse", port=port)


@main.command("setup-mcp")
@click.option(
    "--repo",
    "repo_path",
    default=None,
    help="Repository path to embed in the MCP config (default: current directory)",
)
@click.option(
    "--global",
    "global_config",
    is_flag=True,
    default=False,
    help="Write to AI-client global config instead of project directory",
)
@click.option(
    "--client",
    "clients",
    multiple=True,
    type=click.Choice(["opencode"]),
    default=["opencode"],
    show_default=True,
    help="Which AI clients to configure (can repeat)",
)
def setup_mcp(repo_path: str | None, global_config: bool, clients: tuple):
    """Configure MCP for AI clients (project-level or global).

    By default writes project-level config files so the MCP is only
    active in repositories that have been indexed.

    Use --global to write to your AI client's global config file instead.
    """
    exe = _detect_executable()

    if global_config:
        _setup_global_mcp(exe, list(clients))
        return

    # Project-level
    target_repo = Path(repo_path).resolve() if repo_path else Path.cwd()
    written = _write_project_mcp_configs(
        target_repo, executable=exe, clients=list(clients)
    )
    if written:
        click.echo("Project-level MCP configs written:")
        for p in written:
            click.echo(f"  {p}")
    else:
        click.echo("No configs written.")


def _setup_global_mcp(executable: str, clients: list[str]) -> None:
    """Write MCP entry to global AI client configs."""
    home = Path.home()

    client_global_files = {
        "opencode": [
            home / ".config" / "opencode" / "opencode.json",
        ],
    }

    for client in clients:
        paths = client_global_files.get(client, [])
        for gpath in paths:
            if client == "opencode":
                update = {
                    "$schema": "https://opencode.ai/config.json",
                    "mcp": {
                        "code-rag": {
                            "type": "local",
                            "command": [executable, "serve"],
                            "enabled": True,
                        }
                    },
                }
            else:
                continue
            _merge_json_file(gpath, update)
            click.echo(f"  [{client}] {gpath}")
