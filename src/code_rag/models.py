from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str  # "code" | "doc" | "config"
    language: str | None
    symbol_name: str | None
    symbol_kind: str | None  # "function" | "class" | "method" | None
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    source: str  # "semantic" | "keyword" | "hybrid"


@dataclass
class SymbolInfo:
    name: str
    kind: str  # "function" | "class" | "method" | "struct" | "interface" | ...
    file_path: str
    start_line: int
    end_line: int
    language: str


@dataclass
class FileInfo:
    path: str
    sha256: str
    language: str | None
    size: int
    chunk_count: int
