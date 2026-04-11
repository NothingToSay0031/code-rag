# code-rag

Local code repository RAG with MCP server — index your codebase and search it semantically from any AI editor.

## One-click install

### Windows

```powershell
irm https://raw.githubusercontent.com/NothingToSay0031/code-rag/master/install.ps1 | iex
```

### macOS / Linux

```bash
curl -sSfL https://raw.githubusercontent.com/NothingToSay0031/code-rag/master/install.sh | bash
```

> **GPU detection**: the installer automatically upgrades torch to a CUDA build when an NVIDIA GPU is detected. No manual steps required.

---

## Quick start

```bash
# 1. Index a repository
code-rag init /path/to/your/project

# 2. Open the project in your AI editor
#    → opencode.json / .cursor/mcp.json / .vscode/mcp.json are created automatically
#    → the MCP server is active only for this project
```

That's it. Your AI editor will now use `code-rag` tools to search the codebase.

---

## How project-level MCP works

Running `code-rag init` writes config files **inside the indexed repo**:

| File | Client |
|------|--------|
| `opencode.json` | OpenCode / Codemaker |

Each file points the MCP at the specific repo path, so the tools are only active when you open that project — not globally in every session.

**Recommended**: remove the `code-rag` entry from your AI client's *global* config file if you added it previously.

---

## CLI reference

```
code-rag init <REPO_PATH>          Index a repository (creates .code-rag/ + MCP configs)
code-rag serve [--repo <path>]     Start the MCP server manually
code-rag setup-mcp [--global]      Write MCP config without re-indexing
                                   --global  -> write to AI client's global config
```

### Options for `init`

```
--include PATTERN   Include only matching files (glob, repeatable)
--exclude PATTERN   Exclude matching files (glob, repeatable)
--device auto|cpu|cuda
```

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

## Manual install (advanced)

Requirements: **Python 3.11+**, **git**, [**uv**](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/NothingToSay0031/code-rag
cd code-rag
uv sync          # CPU torch (default)

# CUDA upgrade (optional, ~2 GB download):
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128 --upgrade
```

Add the `.venv/bin/code-rag` (or `.venv\Scripts\code-rag.exe` on Windows) to your `PATH`, or use `uv run code-rag`.

---

## Supported languages

Python, JavaScript, TypeScript, C, C++, Java, C#, Rust, Go, Lua — plus documentation (Markdown, RST, plain text).

## Requirements

- Python 3.11+
- ~500 MB disk per indexed repo (vectors + BM25 index)
- RAM: ~1 GB for the embedding model (CPU), ~2 GB (GPU)
