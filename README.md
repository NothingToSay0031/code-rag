# code-rag

Local code repository RAG with MCP server — index your codebase and search it semantically.

## One-click install

### Windows

```powershell
irm https://raw.githubusercontent.com/NothingToSay0031/code-rag/master/install.ps1 | iex
```

### macOS / Linux (Not functionally tested yet, but should work)

```bash
curl -sSfL https://raw.githubusercontent.com/NothingToSay0031/code-rag/master/install.sh | bash
```

> **GPU detection**: the installer automatically upgrades torch to a CUDA build when an NVIDIA GPU is detected. No manual steps required.

---

## Quick start

```bash
# 1. Index a repository
code-rag init path/to/your/project

# 2. Open the project in your AI agent
#    → opencode.json are created automatically
#    → the MCP server is active only for this project
```

That's it. Your AI agent will now use `code-rag` tools to search the codebase.

---

## How project-level MCP works

Running `code-rag init` writes config files **inside the indexed repo**:

| File | Client |
|------|--------|
| `opencode.json` | OpenCode |

Each file points the MCP at the specific repo path, so the tools are only active when you open that project — not globally in every session.

**Recommended**: remove the `code-rag` entry from your AI client's *global* config file if you added it previously.

---

## CLI reference

```
code-rag --update                  Git-pull the code-rag installation (see below)
code-rag init <REPO_PATH>          Index a repository (creates .code-rag/ + MCP configs)
code-rag serve [--repo <path>]     Start the MCP server manually
code-rag setup-mcp [--global]      Write MCP config without re-indexing
                                   --global  -> write to AI client's global config
```

### Updating code-rag itself

If you installed with the one-click script (a git clone under e.g. `~/.code-rag` on Unix or `%USERPROFILE%\.code-rag` on Windows), you can pull the latest `code-rag` source and then rely on your existing venv or re-run the installer as needed:

```bash
code-rag --update
```

This runs `git pull` in the detected code-rag repository root (a checkout that contains both `.git` and this project’s `pyproject.toml` with `name = "code-rag"`). It does **not** update your indexed projects — for those, re-run `code-rag init` on the project path (see [Keeping the index up to date](#keeping-the-index-up-to-date)). Plain PyPI / wheel-only installs with no git checkout will report that no repository was found.

### Options for `init`

```
--include PATTERN   Include only matching files (glob, repeatable)
--exclude PATTERN   Exclude matching files (glob, repeatable)
--device auto|cpu|cuda
```

> **PowerShell users**: PowerShell expands glob patterns before passing them to `code-rag`, causing `--include` / `--exclude` to malfunction. Use a `.coderagfilter` file in the repo root instead — it is never touched by the shell:
>
> ```ini
> # .coderagfilter
> [include]
> Engine/Shaders/**
> Engine/Sources/**
>
> [exclude]
> Engine/Sources/External/**
> ```
>
> Then run `code-rag init` without any `--include` / `--exclude` flags.

---

## Keeping the index up to date

Re-run `init` on the same repository at any time:

```bash
code-rag init /path/to/your/project
```

`init` is idempotent — it compares a SHA-256 hash of every file against the stored index and only re-processes what changed:

| File state | Action |
|---|---|
| Unchanged | Skipped |
| Modified | Old chunks deleted, file re-indexed |
| Deleted | Removed from all indices |
| New | Indexed |

Unchanged files are skipped entirely, so incremental updates are fast even on large repositories.

---

## Testing

Use the project virtual environment on Windows:

```powershell
# MCP smoke suite
.\.venv\Scripts\python.exe -m pytest -q tests\mcp_smoke\test_mcp_smoke.py

# Unit tests
.\.venv\Scripts\python.exe -m pytest -q tests

# Force rebuild index during smoke run
$env:MCP_SMOKE_FORCE_REINDEX='1'; .\.venv\Scripts\python.exe -m pytest -q tests\mcp_smoke\test_mcp_smoke.py
```

Smoke artifacts are written to `tests\artifacts\mcp-smoke\` by default.

---

## Supported languages

Python, JavaScript, TypeScript, C, C++, Java, C#, Rust, Go, Lua — plus documentation (Markdown, RST, plain text).

## Requirements

- Python 3.14+
- ~500 MB disk per indexed repo (vectors + BM25 index)
- RAM: ~1 GB for the embedding model (CPU), ~2 GB (GPU)
