"""Ripgrep-accelerated text search with Python fallback.

Provides :func:`find_references` which tries ``rg`` (ripgrep) first for
~100× faster text search, and transparently falls back to a pure-Python
line-by-line scan when ``rg`` is not installed.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Ripgrep availability (checked once) ──────────────────────────────────

_rg_available: bool | None = None


def _has_ripgrep() -> bool:
    global _rg_available
    if _rg_available is None:
        _rg_available = shutil.which("rg") is not None
        if _rg_available:
            logger.info("ripgrep detected — will use for reference search")
        else:
            logger.info("ripgrep not found — using Python fallback")
    return _rg_available


# ── Maximum file size to scan (both rg and Python) ───────────────────────

MAX_FILE_SIZE = 512 * 1024  # 512 KB

# ── Code file extensions for Python fallback filtering ──────────────────
# When file_list isn't provided and rg is unavailable, the Python fallback
# only scans files with these suffixes.  Matches the _EXT_LANGUAGE_MAP in
# server.py so post-search filtering stays cheap.
_CODE_EXTENSIONS = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".mjs",
        ".cjs",
        ".cpp",
        ".cc",
        ".cxx",
        ".c",
        ".h",
        ".hpp",
        ".hxx",
        ".java",
        ".cs",
        ".rs",
        ".go",
        ".lua",
        ".swift",
        ".m",
        ".mm",
        ".kt",
        ".kts",
        ".rb",
        ".rake",
        ".gemspec",
        ".sh",
        ".bash",
        ".zsh",
        ".sql",
        ".psql",
        ".proto",
        ".cue",
        ".cmake",
    }
)


# ── Public API ───────────────────────────────────────────────────────────


def find_references(
    repo_path: Path,
    symbol_name: str,
    *,
    max_refs: int = 30,
    exclude_file: str | None = None,
    exclude_lines: tuple[int, int] | None = None,
    file_list: list[str] | None = None,
) -> tuple[list[dict], bool]:
    """Search for textual references to *symbol_name* under *repo_path*.

    Tries ripgrep first for performance; falls back to Python if ``rg`` is
    not installed.

    Args:
        repo_path: Root directory to search.
        symbol_name: Literal string to search for (fixed-string, not regex).
        max_refs: Stop after collecting this many matches.
        exclude_file: Relative file path to skip (e.g. the definition file).
        exclude_lines: ``(start, end)`` line range to skip within
            *exclude_file* (the definition itself).
        file_list: Optional whitelist of relative file paths to scan.
            When omitted, ripgrep searches all files (respects ``.gitignore``)
            and the Python fallback restricts to code extensions
            (see ``_CODE_EXTENSIONS``).

    Returns:
        ``(refs, timed_out)`` where *refs* is a list of dicts with
        ``file_path`` (relative), ``start_line``, ``end_line``, ``text``,
        ``kind``, and *timed_out* is True when the search was cut short by
        the deadline.
    """
    if _has_ripgrep():
        result = _find_references_rg(
            repo_path,
            symbol_name,
            max_refs=max_refs,
            exclude_file=exclude_file,
            exclude_lines=exclude_lines,
            file_list=file_list,
        )
        if result is not None:
            return result
        # rg failed (timeout, error) — fall through to Python

    return _find_references_python(
        repo_path,
        symbol_name,
        max_refs=max_refs,
        exclude_file=exclude_file,
        exclude_lines=exclude_lines,
        file_list=file_list,
    )


# ── Ripgrep implementation ───────────────────────────────────────────────


def _find_references_rg(
    repo_path: Path,
    symbol_name: str,
    *,
    max_refs: int = 30,
    exclude_file: str | None = None,
    exclude_lines: tuple[int, int] | None = None,
    file_list: list[str] | None = None,
) -> tuple[list[dict], bool] | None:
    """Search via ``rg --json``.  Returns *None* on failure (caller should
    fall back to Python).  On success returns ``(refs, False)`` (rg never
    times out internally — its subprocess timeout causes a ``None`` return)."""
    cmd = [
        "rg",
        "--json",
        "-F",  # fixed-string (not regex) — safe for any symbol name
        "--max-filesize",
        f"{MAX_FILE_SIZE}",
        symbol_name,
        str(repo_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None

    # Exit code 1 = no matches (not an error for rg)
    if result.returncode not in (0, 1):
        return None

    refs: list[dict] = []
    for line in result.stdout.splitlines():
        if len(refs) >= max_refs:
            break
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "match":
            continue

        data = obj["data"]
        # rg uses {"text": "..."} for UTF-8 paths/lines and {"bytes": "..."} for
        # binary or non-UTF-8 content.  Skip matches we cannot represent as text.
        abs_path = data["path"].get("text")
        if abs_path is None:
            continue
        line_number = data["line_number"]
        line_text = data["lines"].get("text")
        if line_text is None:
            continue  # binary file match — skip
        line_text = line_text.rstrip("\n")

        # Convert absolute path to relative
        try:
            rel_path = Path(abs_path).relative_to(repo_path).as_posix()
        except ValueError:
            rel_path = abs_path

        # Skip the definition itself
        if exclude_file and rel_path == exclude_file:
            if exclude_lines and exclude_lines[0] <= line_number <= exclude_lines[1]:
                continue

        refs.append(
            {
                "file_path": rel_path,
                "start_line": line_number,
                "end_line": line_number,
                "text": line_text.strip(),
                "kind": "usage",
            }
        )

    return refs, False


# ── Python fallback ──────────────────────────────────────────────────────


def _find_references_python(
    repo_path: Path,
    symbol_name: str,
    *,
    max_refs: int = 30,
    exclude_file: str | None = None,
    exclude_lines: tuple[int, int] | None = None,
    file_list: list[str] | None = None,
    deadline: float | None = None,
) -> tuple[list[dict], bool]:
    """Brute-force line-by-line scan.  Used when ``rg`` is unavailable.

    If *file_list* is provided, only those files are scanned (relative to
    *repo_path*).  Otherwise walks the directory tree, restricted to code
    file extensions (see ``_CODE_EXTENSIONS``).

    If *deadline* (``time.perf_counter()`` value) is reached, scanning
    stops early and returns whatever has been collected so far.
    """
    import time

    if deadline is None:
        deadline = time.perf_counter() + 15

    refs: list[dict] = []
    timed_out = False

    if file_list is not None:
        paths = ((f, repo_path / f) for f in file_list)
    else:
        paths = (
            (str(p.relative_to(repo_path).as_posix()), p)
            for p in repo_path.rglob("*")
            if p.is_file() and p.suffix.lower() in _CODE_EXTENSIONS
        )

    for rel_path, abs_path in paths:
        if len(refs) >= max_refs:
            break
        if time.perf_counter() > deadline:
            timed_out = True
            break
        if not abs_path.exists():
            continue
        try:
            size = abs_path.stat().st_size
        except OSError:
            continue
        if size > MAX_FILE_SIZE:
            continue
        try:
            file_lines = abs_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for line_no, line_text in enumerate(file_lines, 1):
            if symbol_name not in line_text:
                continue
            if exclude_file and rel_path == exclude_file:
                if exclude_lines and exclude_lines[0] <= line_no <= exclude_lines[1]:
                    continue
            refs.append(
                {
                    "file_path": rel_path,
                    "start_line": line_no,
                    "end_line": line_no,
                    "text": line_text.strip(),
                    "kind": "usage",
                }
            )
            if len(refs) >= max_refs:
                break

    return refs, timed_out
