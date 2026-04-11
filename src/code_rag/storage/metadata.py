import json
import os
from pathlib import Path
from code_rag.models import FileInfo, SymbolInfo


class MetadataStore:
    def __init__(self, store_path: Path):
        self._store_path = store_path
        self._files: dict[str, FileInfo] = {}
        self._symbols: dict[str, list[SymbolInfo]] = {}  # file_path → symbols
        self._chunk_map: dict[str, list[int]] = {}  # file_path → chunk IDs
        # Inverted index: lowered symbol name → list of SymbolInfo
        self._name_index: dict[str, list[SymbolInfo]] = {}
        if store_path.exists():
            self._load()

    def set_file_info(self, file_path: str, info: FileInfo):
        self._files[file_path] = info

    def get_file_info(self, file_path: str) -> FileInfo | None:
        return self._files.get(file_path)

    def get_fingerprint(self, file_path: str) -> str | None:
        info = self._files.get(file_path)
        return info.sha256 if info else None

    def set_symbols(self, file_path: str, symbols: list[SymbolInfo]):
        # Remove old entries from name index
        old = self._symbols.get(file_path, [])
        for sym in old:
            key = sym.name.lower()
            bucket = self._name_index.get(key)
            if bucket is not None:
                try:
                    bucket.remove(sym)
                except ValueError:
                    pass
                if not bucket:
                    del self._name_index[key]
        # Store new symbols
        self._symbols[file_path] = symbols
        # Add to name index
        for sym in symbols:
            self._name_index.setdefault(sym.name.lower(), []).append(sym)

    def get_symbols(self, file_path: str) -> list[SymbolInfo]:
        return self._symbols.get(file_path, [])

    def find_symbol(self, symbol_name: str, *, limit: int = 200) -> list[SymbolInfo]:
        """Search symbols by name (case-insensitive).

        Uses an inverted name index for O(1) exact match.  Falls back to
        linear scan only for substring matches.  Exact matches are returned
        first.
        """
        query = symbol_name.lower()

        # O(1) exact match via inverted index
        exact = list(self._name_index.get(query, []))

        if len(exact) >= limit:
            return exact[:limit]

        # O(N) substring scan for partial matches (skip exact dupes)
        remaining = limit - len(exact)
        exact_set = set(id(s) for s in exact)
        partial: list[SymbolInfo] = []
        for symbols in self._symbols.values():
            for sym in symbols:
                if id(sym) in exact_set:
                    continue
                if query in sym.name.lower():
                    partial.append(sym)
                    if len(partial) >= remaining:
                        return exact + partial
        return exact + partial

    def set_chunk_ids(self, file_path: str, chunk_ids: list[int]):
        self._chunk_map[file_path] = chunk_ids

    def get_chunk_ids(self, file_path: str) -> list[int]:
        return self._chunk_map.get(file_path, [])

    def remove_file(self, file_path: str):
        self._files.pop(file_path, None)
        old = self._symbols.pop(file_path, None)
        if old:
            for sym in old:
                key = sym.name.lower()
                bucket = self._name_index.get(key)
                if bucket is not None:
                    try:
                        bucket.remove(sym)
                    except ValueError:
                        pass
                    if not bucket:
                        del self._name_index[key]
        self._chunk_map.pop(file_path, None)

    def get_all_files(self) -> list[str]:
        return list(self._files.keys())

    def get_changed_files(
        self, current_fingerprints: dict[str, str]
    ) -> tuple[list[str], list[str], list[str]]:
        """Returns (new_files, modified_files, deleted_files)."""
        stored = set(self._files.keys())
        current = set(current_fingerprints.keys())
        new_files = sorted(current - stored)
        deleted_files = sorted(stored - current)
        modified_files = sorted(
            f
            for f in stored & current
            if self._files[f].sha256 != current_fingerprints[f]
        )
        return new_files, modified_files, deleted_files

    def save(self):
        """Persist metadata with atomic write (temp + rename)."""
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "files": {k: self._file_to_dict(v) for k, v in self._files.items()},
            "symbols": {
                k: [self._symbol_to_dict(s) for s in v]
                for k, v in self._symbols.items()
            },
            "chunk_map": self._chunk_map,
        }
        tmp_path = self._store_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        # Atomic rename
        if os.name == "nt" and self._store_path.exists():
            self._store_path.unlink()
        tmp_path.rename(self._store_path)

    def _load(self):
        data = json.loads(self._store_path.read_text(encoding="utf-8"))
        self._files = {
            k: self._dict_to_file(v) for k, v in data.get("files", {}).items()
        }
        self._symbols = {
            k: [self._dict_to_symbol(s) for s in v]
            for k, v in data.get("symbols", {}).items()
        }
        self._chunk_map = data.get("chunk_map", {})
        # Rebuild inverted name index
        self._name_index = {}
        for symbols in self._symbols.values():
            for sym in symbols:
                self._name_index.setdefault(sym.name.lower(), []).append(sym)

    @staticmethod
    def _file_to_dict(f: FileInfo) -> dict:
        return {
            "path": f.path,
            "sha256": f.sha256,
            "language": f.language,
            "size": f.size,
            "chunk_count": f.chunk_count,
        }

    @staticmethod
    def _dict_to_file(d: dict) -> FileInfo:
        return FileInfo(
            path=d["path"],
            sha256=d["sha256"],
            language=d["language"],
            size=d["size"],
            chunk_count=d["chunk_count"],
        )

    @staticmethod
    def _symbol_to_dict(s: SymbolInfo) -> dict:
        return {
            "name": s.name,
            "kind": s.kind,
            "file_path": s.file_path,
            "start_line": s.start_line,
            "end_line": s.end_line,
            "language": s.language,
        }

    @staticmethod
    def _dict_to_symbol(d: dict) -> SymbolInfo:
        return SymbolInfo(
            name=d["name"],
            kind=d["kind"],
            file_path=d["file_path"],
            start_line=d["start_line"],
            end_line=d["end_line"],
            language=d["language"],
        )

    def count_files(self) -> int:
        return len(self._files)

    def count_symbols(self) -> int:
        return sum(len(syms) for syms in self._symbols.values())
