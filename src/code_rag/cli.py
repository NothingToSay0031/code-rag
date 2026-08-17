from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import click


_CONFIG_FILE = Path.home() / ".config" / "code-rag" / "config.json"


# ---------------------------------------------------------------------------
# Project-level MCP config helpers
# ---------------------------------------------------------------------------

_MCP_CLIENTS = {
    "opencode": {
        "files": [".opencode/opencode.json"],
        "schema": "https://opencode.ai/config.json",
    },
    "claude": {
        "files": [".mcp.json"],
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
                mcp_entry_with_timeout = {**mcp_entry, "timeout": 180000}
                update = {"mcp": {"code-rag": mcp_entry_with_timeout}}
                if "schema" in info:
                    update["$schema"] = info["schema"]  # type: ignore[assignment]
            elif client == "claude":
                update = {
                    "mcpServers": {
                        "code-rag": {
                            "command": executable,
                            "args": ["serve", "--repo", repo_abs],
                        }
                    }
                }
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
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            OSError,
        ):
            pass  # Best effort — don't block the user


def _code_rag_git_root() -> Path | None:
    """Locate the git checkout root for this installation, if it exists.

    Walks upward from this package so editable installs and ``src/`` layouts
    resolve to the repository root; plain wheel installs under site-packages
    typically have no matching checkout.
    """
    here = Path(__file__).resolve().parent
    for d in [here, *here.parents]:
        git = d / ".git"
        pyproject = d / "pyproject.toml"
        if not git.exists() or not pyproject.is_file():
            continue
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except Exception:
            continue
        name = (data.get("project") or {}).get("name")
        if name == "code-rag":
            return d
    return None


def _git_pull_code_rag_repo() -> None:
    root = _code_rag_git_root()
    if root is None:
        raise click.ClickException(
            "Cannot find a git checkout for code-rag "
            "(need a clone with .git and pyproject.toml name code-rag)."
        )
    click.echo(f"Updating code-rag at {root} (git pull)...", err=True)
    try:
        subprocess.run(["git", "pull"], cwd=str(root), check=True)
    except FileNotFoundError as exc:
        raise click.ClickException(
            "git was not found in PATH; install Git or use git pull manually."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(f"git pull failed (exit {exc.returncode}).") from exc


_SKILL_CONTENT = """\
---
name: code-rag
description: "Use when the user asks to explore or search a codebase — locating where a feature, function, or concept is implemented, understanding an unfamiliar repo, or mapping a spec, PRD, or task description to the relevant code."
---

# Explore the Codebase

Read the user's request — a spec, PRD, task, or design doc in any format — then
use **code-rag** to find the code behind it. A local HTTP query server holds the
model in memory — start it once, then query on demand.

## 1. Ensure the Server Is Running

```
code-rag servers
```

If a row appears (`port=...`), continue to step 2. If not, start the server:

**Windows (PowerShell):**
```powershell
Start-Process -WindowStyle Hidden -FilePath code-rag -ArgumentList "query-server","--idle-timeout","600"
```

**macOS / Linux:**
```bash
code-rag query-server --idle-timeout 600 &
```

Model loading takes 10-60 seconds. Poll until the server shows up (120s budget):

**Windows:**
```powershell
$deadline = (Get-Date).AddSeconds(120)
while ((code-rag servers 2>&1 | Out-String) -notmatch 'port=') {
  if ((Get-Date) -gt $deadline) {
    throw "Server failed to start. Is code-rag on PATH? Is the repo indexed?"
  }
  Start-Sleep 3
}
```

**macOS / Linux:**
```bash
for i in $(seq 1 40); do
  code-rag servers 2>/dev/null | grep -q "port=" && break
  [ "$i" = 40 ] && { echo "Server failed to start. Is code-rag on PATH? Is the repo indexed?" >&2; exit 1; }
  sleep 3
done
```

The server auto-shuts down after 600 idle seconds. During long analyses, run
`code-rag health` (or any query) at least every 600s to keep it alive. If a
query returns `"No query server found"`, the server died — restart from this
step.

## 2. Read the Request

Read the user's request in full. Extract:
- **Key concepts**: domain terms, feature names, architectural patterns mentioned
- **Entities**: classes, functions, modules, APIs referenced
- **Relationships**: "X depends on Y", "Z implements W"
- **Unknowns**: phrases like "see code", "as implemented in"

## 3. Explore the Codebase

Commands auto-discover the server. If multiple servers are running,
**`code-rag servers` will show a warning and list them** — use `--port N`
to target a specific one.

Always start broad, then narrow:

| Step | Command | Purpose |
|------|---------|---------|
| 1 | `code-rag indices` | See which repos are indexed |
| 2 | `code-rag tree [path]` | Understand project layout |
| 3 | `code-rag search "concept"` | Broad concept/behavior search |
| 4 | `code-rag search-docs "concept"` | Search markdown/rst/doc files |
| 5 | `code-rag symbols <file>` | Inspect a file's functions/classes |
| 6 | `code-rag def <Name>` | Look up exact identifier definition |
| 7 | `code-rag read <file> --start N --end N` | Expand snippets to full context |

### Multi-Repository Indexes

`code-rag indices` shows each index as:
```
(root)  D:\\Code\\myproject  model=Qwen/...  (23343 files)
(sub1)  D:\\Code\\other      model=Qwen/...  (1240 files)
```

Use the prefix in parentheses with `--index` to scope queries:
```
code-rag search "auth" --index sub1
code-rag def LoginHandler --index root
```

Stop exploring when every key concept from step 2 has a best-match location
(file:line or symbol) or a confirmed gap.

- Batch independent queries as parallel shell calls.
- Don't re-search the same query — remember previous results.
- Start with `--top-k 5` to see signal; expand to 10-15 only if needed.
- Add `--language` when the request is language-specific.

## 4. Synthesize Findings

Present findings as a structured report:
- **What you searched** — queries you ran and why
- **What you found** — relevant files, symbols, patterns with line references
- **Gaps** — request mentions with no matching code
- **Suggestions** — surprising findings, potential issues, next steps

## Command Reference

| Command | Example |
|---------|---------|
| `code-rag servers` | List running servers (check this first) |
| `code-rag health` | Quick alive check (for scripts/keepalive) |
| `code-rag stop [--port N] [--yes]` | Stop a server gracefully |
| `code-rag indices` | List indexed repos |
| `code-rag tree [path]` | `code-rag tree src/auth --depth 4` |
| `code-rag search <query>` | `code-rag search "auth middleware" --top-k 5 --language python --index root` |
| `code-rag search-docs <query>` | `code-rag search-docs "API design"` |
| `code-rag symbols <file>` | `code-rag symbols src/auth/login.py` |
| `code-rag def <name>` | `code-rag def LoginHandler --include-code --include-refs --index root` |
| `code-rag read <file> --start N --end N` | `code-rag read src/main.py --start 42 --end 80 --context 5` |

## Prerequisites

- A repo indexed with `code-rag init <repo_path>`
- The `code-rag` CLI installed and on PATH

"""


def _write_project_skill(repo_path: Path) -> Path | None:
    """Write the code-rag skill to the project's .opencode/skills/ directory.

    Returns the path to the written SKILL.md, or None if it already exists.
    """
    skill_dir = repo_path / ".opencode" / "skills" / "code-rag"
    skill_file = skill_dir / "SKILL.md"
    if skill_file.exists():
        return None
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file.write_text(_SKILL_CONTENT, encoding="utf-8")
    return skill_file


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



@click.group(invoke_without_command=True)
@click.version_option(version="0.1.0")
@click.option(
    "--update",
    is_flag=True,
    help="Run git pull in the code-rag source repository (git clone / installer checkout).",
)
@click.pass_context
def main(ctx: click.Context, update: bool) -> None:
    """code-rag: Local code repository RAG with MCP server."""
    if update:
        _git_pull_code_rag_repo()
        ctx.exit(0)
    _ensure_environment()
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


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
    "--chunk-size",
    "max_chunk_tokens",
    type=int,
    default=None,
    help="Max tokens per chunk (default: 512)",
)
@click.option(
    "--model",
    "model_name",
    default=None,
    help=(
        "Embedding model to use for this index. "
        "Defaults to BAAI/bge-small-en-v1.5. "
        "Examples: Qwen/Qwen3-Embedding-0.6B, Qwen/Qwen3-Embedding-4B"
    ),
)
def init(
    repo_path: str,
    include_patterns: tuple,
    exclude_patterns: tuple,
    device: str,
    max_chunk_tokens: int | None,
    model_name: str | None,
):
    """Index a code repository for RAG search.

    REPO_PATH: Path to the repository to index.
    """
    from code_rag.config import CodeRagConfig
    from code_rag.indexer.pipeline import IndexPipeline

    repo = Path(repo_path)
    kwargs: dict = dict(
        repo_path=repo,
        include_patterns=[p.replace("\\", "/") for p in include_patterns],
        exclude_patterns=[p.replace("\\", "/") for p in exclude_patterns],
    )

    # Reuse existing index settings (model, chunk size, device)
    # so re-indexing is consistent with the previous run.
    existing = CodeRagConfig.load(repo)
    if existing:
        kwargs.setdefault("model_name", existing.model_name)
        kwargs.setdefault("max_chunk_tokens", existing.max_chunk_tokens)
        kwargs.setdefault("device", existing.device)

    # CLI overrides take precedence
    if model_name:
        kwargs["model_name"] = model_name
    if max_chunk_tokens is not None:
        kwargs["max_chunk_tokens"] = max_chunk_tokens
    if device != "auto":
        kwargs["device"] = device

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

    # Write project-level code-rag skill (default and recommended).
    # MCP is available via 'code-rag setup-mcp' for users who need it.
    skill_file = _write_project_skill(resolved_repo)
    if skill_file:
        click.echo(f"\nProject-level skill written: {skill_file}")
        click.echo("  The AI can now use the 'code-rag' skill in this project.")


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


@main.command("query-server")
@click.option(
    "--repo",
    "repo_paths",
    multiple=True,
    default=None,
    help=(
        "Path to indexed repository (repeat for multiple repos). "
        "Falls back to CODE_RAG_REPO environment variable, then the "
        "current directory."
    ),
)
@click.option("--port", default=9020, type=int, help="HTTP server port (default: 9020)")
@click.option(
    "--idle-timeout",
    default=1200,
    type=int,
    help="Auto-shutdown after N seconds of inactivity (default: 1200 = 20 min). 0 to disable.",
)
def query_server(repo_paths: tuple, port: int, idle_timeout: int):
    """Start a lightweight HTTP query server for code-rag.

    Loads the model and indices once, then accepts JSON queries via HTTP POST.
    Designed as a local daemon for AI skills/tools — no MCP transport needed.

    Auto-shuts down after IDLE_TIMEOUT seconds of inactivity (default 20 min).
    Set --idle-timeout=0 to disable auto-shutdown.

    Pass --repo multiple times to serve multiple independent repositories
    from a single server instance.
    """
    import os

    repos: list[str] = []
    # click's envvar support with multiple=True would split the value on
    # whitespace, breaking Windows paths with spaces — read it manually.
    if repo_paths:
        for repo_path in repo_paths:
            repos.extend(rp for rp in str(repo_path).split(os.pathsep) if rp)
    else:
        env_raw = os.environ.get("CODE_RAG_REPO", "")
        repos = [r for r in env_raw.split(os.pathsep) if r]
    if not repos:
        repos = ["."]

    resolved_paths: list[str] = []
    for repo_path in repos:
        resolved = Path(repo_path).resolve()
        if repo_path == "." and not (resolved / ".code-rag").exists():
            sub_indices = list(resolved.rglob(".code-rag"))
            if not sub_indices:
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
        resolved_paths.append(str(resolved))

    os.environ["CODE_RAG_REPO"] = os.pathsep.join(resolved_paths)

    from code_rag.query_server import run_query_server

    click.echo("Starting CodeRAG query server...", err=True)
    click.echo(f"Repositories: {', '.join(resolved_paths)}", err=True)
    click.echo(f"Port: {port}", err=True)
    click.echo(f"Idle timeout: {idle_timeout}s", err=True)

    run_query_server(port=port, idle_timeout=idle_timeout)


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
    type=click.Choice(["opencode", "claude"]),
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
        "claude": [
            home / ".claude.json",
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
            elif client == "claude":
                update = {
                    "mcpServers": {
                        "code-rag": {
                            "command": executable,
                            "args": ["serve"],
                        }
                    }
                }
            else:
                continue
            _merge_json_file(gpath, update)
            click.echo(f"  [{client}] {gpath}")


@main.command("setup-skill")
@click.option(
    "--repo",
    "repo_path",
    default=None,
    help="Repository path to write the skill to (default: current directory)",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing skill file",
)
def setup_skill(repo_path: str | None, force: bool):
    """Write the code-rag skill to a project's .opencode/skills/ directory.

    This enables the AI to explore code with code-rag using a local
    HTTP query server (no MCP required).
    """
    target_repo = Path(repo_path).resolve() if repo_path else Path.cwd()

    skill_dir = target_repo / ".opencode" / "skills" / "code-rag"
    skill_file = skill_dir / "SKILL.md"

    if not force:
        written = _write_project_skill(target_repo)
        if written is None:
            click.echo(f"Skill already exists: {skill_file}")
            click.echo("Use --force to overwrite.")
        else:
            click.echo(f"Skill written: {written}")
        return

    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file.write_text(_SKILL_CONTENT, encoding="utf-8")
    click.echo(f"Skill written: {skill_file}")


# ---------------------------------------------------------------------------
# Query client helpers
# ---------------------------------------------------------------------------

import urllib.request
import urllib.error


def _discover_server_port() -> int | None:
    """Find the port of a running query server by scanning state files.

    Scans all state files, collects live servers, then warns to stderr
    if multiple are found and returns the first one's port.
    """
    import json as _json
    from pathlib import Path as _Path

    server_dir = _Path.home() / ".code-rag" / "servers"
    if not server_dir.exists():
        return None

    live_ports: list[int] = []
    for f in sorted(server_dir.glob("*.json")):
        try:
            info = _json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            f.unlink(missing_ok=True)
            continue
        port = info.get("port")
        if not port:
            f.unlink(missing_ok=True)
            continue

        try:
            url = f"http://127.0.0.1:{port}/health"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2) as resp:
                body = _json.loads(resp.read().decode("utf-8"))
                if body.get("status") == "ok":
                    live_ports.append(port)
                else:
                    f.unlink(missing_ok=True)
        except Exception:
            f.unlink(missing_ok=True)

    if not live_ports:
        return None

    if len(live_ports) > 1:
        import sys
        print(
            f"Warning: {len(live_ports)} query servers running (ports: {live_ports}). "
            f"Using port {live_ports[0]}. Use --port to pick a specific one.",
            file=sys.stderr,
        )

    return live_ports[0]


def _query(endpoint: str, data: dict | None = None, port: int | None = None) -> str:
    """Send a request to the query server and return the response body.

    If *port* is None, auto-discovers the first running query server.
    """
    import json as _json
    from pathlib import Path as _Path

    if port is None:
        port = _discover_server_port()

    if port is None:
        raise click.ClickException(
            "No query server found. Start it first: code-rag query-server"
        )
    url = f"http://127.0.0.1:{port}{endpoint}"
    try:
        if data is not None:
            body = json.dumps(data).encode("utf-8")
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}
            )
        else:
            req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            err = json.loads(body)
            msg = err.get("error", body)
        except Exception:
            msg = body
        raise click.ClickException(f"Server error: {msg}")
    except urllib.error.URLError:
        raise click.ClickException(
            f"No query server found at http://127.0.0.1:{port}.\n"
            f"Start it first: code-rag query-server"
        )

    if isinstance(result, dict) and "error" in result:
        raise click.ClickException(result["error"])
    return result.get("result", "")


# ---------------------------------------------------------------------------
# Query commands (talk to a running query server)
# ---------------------------------------------------------------------------


@main.command("search")
@click.argument("query")
@click.option("--top-k", type=int, default=8, help="Number of results (default: 8)")
@click.option("--offset", type=int, default=0, help="Pagination offset")
@click.option("--language", default=None, help="Language filter (e.g. python, cpp)")
@click.option("--index", "index_name", default=None, help="Index name (for multi-repo servers)")
@click.option("--port", default=None, type=int, help="Query server port (auto-detected)")
def search_cmd(query: str, top_k: int, offset: int, language: str | None, index_name: str | None, port: int | None):
    """Search code with a natural language query."""
    data: dict = {"query": query, "top_k": top_k, "offset": offset, "language": language}
    if index_name:
        data["index_name"] = index_name
    result = _query("/search_code", data, port=port)
    click.echo(result)


@main.command("search-docs")
@click.argument("query")
@click.option("--top-k", type=int, default=8, help="Number of results (default: 8)")
@click.option("--offset", type=int, default=0, help="Pagination offset")
@click.option("--port", default=None, type=int, help="Query server port (auto-detected)")
def search_docs_cmd(query: str, top_k: int, offset: int, port: int | None):
    """Search documentation files."""
    result = _query(
        "/search_docs",
        {"query": query, "top_k": top_k, "offset": offset},
        port=port,
    )
    click.echo(result)


@main.command("symbols")
@click.argument("file_path")
@click.option("--port", default=None, type=int, help="Query server port (auto-detected)")
def symbols_cmd(file_path: str, port: int | None):
    """List symbols (functions, classes) in a file."""
    result = _query("/get_file_symbols", {"file_path": file_path}, port=port)
    click.echo(result)


@main.command("def")
@click.argument("name")
@click.option("--include-code", is_flag=True, help="Include source code")
@click.option("--include-refs", is_flag=True, help="Include references/call sites")
@click.option("--max-results", type=int, default=20, help="Max results (default: 20)")
@click.option("--index", "index_name", default=None, help="Index name (for multi-repo servers)")
@click.option("--port", default=None, type=int, help="Query server port (auto-detected)")
def def_cmd(name: str, include_code: bool, include_refs: bool, max_results: int, index_name: str | None, port: int | None):
    """Look up a symbol definition by name."""
    data: dict = {
        "symbol_name": name,
        "include_code": include_code,
        "include_references": include_refs,
        "max_results": max_results,
    }
    if index_name:
        data["index_name"] = index_name
    result = _query("/get_symbol_info", data, port=port)
    click.echo(result)


@main.command("read")
@click.argument("file_path")
@click.option("--start", type=int, required=True, help="Start line (1-based, inclusive)")
@click.option("--end", type=int, required=True, help="End line (1-based, inclusive)")
@click.option("--context", type=int, default=0, help="Extra context lines")
@click.option("--port", default=None, type=int, help="Query server port (auto-detected)")
def read_cmd(file_path: str, start: int, end: int, context: int, port: int | None):
    """Read a line range from a source file."""
    result = _query(
        "/read_code",
        {
            "file_path": file_path,
            "start_line": start,
            "end_line": end,
            "context_lines": context,
        },
        port=port,
    )
    click.echo(result)


@main.command("tree")
@click.argument("path", required=False, default=None)
@click.option("--depth", type=int, default=4, help="Max tree depth (default: 4)")
@click.option("--port", default=None, type=int, help="Query server port (auto-detected)")
def tree_cmd(path: str | None, depth: int, port: int | None):
    """Show directory tree of indexed files."""
    data: dict = {"depth": depth}
    if path:
        data["path"] = path
    result = _query("/get_repo_structure", data, port=port)
    click.echo(result)


@main.command("indices")
@click.option("--port", default=None, type=int, help="Query server port (auto-detected)")
def indices_cmd(port: int | None):
    """List available code indices (repositories)."""
    result = _query("/list_indices", port=port)
    click.echo(result)


@main.command("servers")
def servers_cmd():
    """List running code-rag query servers."""
    import json
    import urllib.error
    from pathlib import Path

    server_dir = Path.home() / ".code-rag" / "servers"
    if not server_dir.exists():
        click.echo("No query servers found.")
        return

    found = False
    for f in sorted(server_dir.glob("*.json")):
        try:
            info = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        port = info.get("port", 0)
        pid = info.get("pid", "?")
        timeout = info.get("idle_timeout", "?")
        repos = info.get("repos", [info.get("repo", "?")])
        repos_str = ", ".join(repos) if isinstance(repos, list) else str(repos)

        # Check if actually alive
        alive = False
        try:
            url = f"http://127.0.0.1:{port}/health"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                alive = body.get("status") == "ok"
        except Exception:
            alive = False

        if not alive:
            f.unlink(missing_ok=True)
            continue

        found = True
        click.echo(f"  port={port}  repos={repos_str}  pid={pid}  idle_timeout={timeout}s")

    if not found:
        click.echo("No running query servers found.")


@main.command("stop")
@click.option("--port", type=int, default=None, help="Port of the server to stop (auto-detect if only one)")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def stop_cmd(port: int | None, yes: bool):
    """Stop a running code-rag query server gracefully."""
    import json
    import urllib.error
    import urllib.request
    from pathlib import Path

    server_dir = Path.home() / ".code-rag" / "servers"
    if not server_dir.exists():
        click.echo("No query servers found.")
        return

    # Find alive servers
    alive_servers: list[dict] = []
    for f in sorted(server_dir.glob("*.json")):
        try:
            info = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        p = info.get("port", 0)
        try:
            url = f"http://127.0.0.1:{p}/health"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                if body.get("status") == "ok":
                    alive_servers.append(info)
        except Exception:
            f.unlink(missing_ok=True)

    if not alive_servers:
        click.echo("No running query servers found.")
        return

    if port is None:
        if len(alive_servers) == 1:
            port = alive_servers[0]["port"]
        else:
            click.echo("Multiple servers running. Specify --port:")
            for s in alive_servers:
                repos = s.get("repos", [])
                repos_str = ", ".join(repos) if isinstance(repos, list) else str(repos)
                click.echo(f"  port={s['port']}  repos={repos_str}")
            return

    target = next((s for s in alive_servers if s["port"] == port), None)
    if target is None:
        click.echo(f"No server found on port {port}.")
        return

    repos = target.get("repos", [])
    repos_str = ", ".join(repos) if isinstance(repos, list) else str(repos)

    if not yes:
        click.confirm(
            f"Stop server on port {port} (repos: {repos_str})?",
            abort=True,
        )

    click.echo(f"Stopping server on port {port}...")

    try:
        body = json.dumps({}).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/shutdown",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            if result.get("status") == "shutting_down":
                click.echo("Server stopped.")
            else:
                click.echo(f"Unexpected response: {result}")
    except urllib.error.URLError:
        click.echo("Server already stopped (connection refused).")
    except Exception as exc:
        click.echo(f"Failed to stop server: {exc}")


@main.command("health")
@click.option("--port", default=None, type=int, help="Query server port (auto-detected)")
def health_cmd(port: int | None):
    """Check if a query server is alive (exits 0 if healthy)."""
    import json
    import urllib.error
    import urllib.request
    import sys

    discovered = _discover_server_port()
    port = port or discovered
    if not port:
        click.echo("No query server found.", err=True)
        sys.exit(1)

    try:
        url = f"http://127.0.0.1:{port}/health"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            click.echo(f"port={port}  status={body.get('status', '?')}  n_indices={body.get('n_indices', '?')}")
    except urllib.error.URLError:
        click.echo(f"No server responding on port {port}.", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Health check failed: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

main.add_command(indices_cmd, name="list")
main.add_command(def_cmd, name="symbol")
main.add_command(search_docs_cmd, name="docs")
