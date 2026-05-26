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
--model MODEL_NAME  Embedding model (see "Model selection" below)
```

### Model selection

Two recommended models — pick based on your priorities:

| Model | Dims | Strength | GPU |
|-------|------|----------|-----|
| **`BAAI/bge-small-en-v1.5`** (default) | 384 | Best retrieval quality | Auto-detect |
| `Qwen/Qwen3-Embedding-0.6B` | 1024 | Fastest indexing | CUDA recommended |

Other supported models are available but not generally recommended:

| Model | Dims | Notes |
|-------|------|-------|
| `Qwen/Qwen3-Embedding-4B` | 2560 | Much heavier, marginal quality gain over 0.6B |
| `Qwen/Qwen3-Embedding-8B` | 4096 | Workstation/server cards only |
| `BAAI/bge-large-en-v1.5` | 1024 | Heavier than bge-small, no practical benefit |
| `minishlab/potion-code-16M` | 256 | CPU-only, very fast but low retrieval quality |

**BGE-small** is the default — it runs on CPU or GPU and consistently delivers the best retrieval accuracy on code repositories.

**Qwen3-0.6B** is a decoder-transformer that indexes significantly faster on GPU, while maintaining strong retrieval quality. CUDA recommended; CPU inference is impractically slow.

**Switching models** per index:

```bash
# Default: best retrieval quality (CPU or GPU)
code-rag init /path/to/repo

# Faster GPU indexing
code-rag init /path/to/repo --model Qwen/Qwen3-Embedding-0.6B

# Re-index an existing repo with a different model
# (remove .code-rag/ first, or the old model's config will be reused)
rm -rf /path/to/repo/.code-rag
code-rag init /path/to/repo --model Qwen/Qwen3-Embedding-0.6B
```

Existing indices remember their model choice via `.code-rag/config.json`.  Re-running `init` without removing `.code-rag/` reuses the persisted model — this is by design so incremental updates don't silently switch models.

> **PowerShell users**: PowerShell expands glob patterns (`**`, `*`) against the current directory **before** passing them to `code-rag`. If the target repo contains directories that match the patterns, `--include` / `--exclude` will receive hundreds of expanded file paths instead of the intended glob patterns, breaking the filter logic.
>
> **Workarounds** (pick one):
>
> 1. **`--%` stop-parsing token (simplest)** — everything after `--%` is passed literally:
>    ```powershell
>    code-rag --% init "D:\Code" --include 'Engine/Shaders/**' --include 'Engine/Sources/**' --exclude 'Engine/Sources/External/**'
>    ```
>    Note: `--%` must come **before** the subcommand and arguments.
>
> 2. **Run from a directory that doesn't contain the named paths** — PowerShell can only expand patterns that match files on disk:
>    ```powershell
>    cd C:\ && code-rag init "D:\Code" --include 'Engine/Shaders/**' ...
>    ```
>
> 3. **Use a `.coderagfilter` file** in the repo root — it is never touched by the shell:
>    ```ini
>    # .coderagfilter
>    [include]
>    Engine/Shaders/**
>    Engine/Sources/**
>
>    [exclude]
>    Engine/Sources/External/**
>    Engine/Sources/Compiler/**
>    ```
>    Then run `code-rag init` without any `--include` / `--exclude` flags.

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
- RAM: ~500 MB with default bge-small (CPU/GPU); ~2 GB with Qwen3-0.6B (GPU)
