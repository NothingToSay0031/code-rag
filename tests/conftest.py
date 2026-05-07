"""Shared fixtures for code-rag tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from code_rag.models import Chunk, FileInfo, SymbolInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chunk(
    text: str = "def foo(): pass",
    file_path: str = "src/foo.py",
    start_line: int = 1,
    end_line: int = 1,
    chunk_type: str = "code",
    language: str | None = "python",
    symbol_name: str | None = "foo",
    symbol_kind: str | None = "function",
) -> Chunk:
    return Chunk(
        text=text,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        chunk_type=chunk_type,
        language=language,
        symbol_name=symbol_name,
        symbol_kind=symbol_kind,
    )


def make_symbol(
    name: str = "Foo",
    kind: str = "class",
    file_path: str = "src/foo.py",
    start_line: int = 1,
    end_line: int = 10,
    language: str = "python",
) -> SymbolInfo:
    return SymbolInfo(
        name=name,
        kind=kind,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        language=language,
    )


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)
