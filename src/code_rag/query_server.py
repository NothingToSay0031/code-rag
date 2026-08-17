from __future__ import annotations

import json
import os
import signal
import socket
import sys
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from code_rag.server import (
    _get_state,
    _resolve_index,
    _resolve_file,
    _qualify_path,
    _check_vector_availability,
    _build_snippet,
    _render_results,
    _build_tree,
    _MAX_TREE_LINES,
    _SEP,
    _DECLARATION_KINDS,
    _MERGE_GAP,
    _MAX_GROUP_LINES,
    _read_line_range,
    _render_refs,
    _find_references,
    _EXT_LANGUAGE_MAP,
    _guess_language,
    _Index,
)

_SERVER_DIR = Path.home() / ".code-rag" / "servers"


class _ShutdownTimer:
    """Auto-shutdown after *timeout_seconds* of inactivity."""

    def __init__(self, timeout_seconds: float, port: int):
        self._timeout = timeout_seconds
        self._port = port
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._timeout, self._shutdown)
            self._timer.daemon = True
            self._timer.start()

    def cancel(self):
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def _shutdown(self):
        print("[CodeRAG] Idle timeout reached — shutting down.", file=sys.stderr)
        try:
            state_file = _SERVER_DIR / f"{self._port}.json"
            state_file.unlink(missing_ok=True)
        except Exception:
            pass
        os._exit(0)


class _Handler(BaseHTTPRequestHandler):
    """HTTP JSON handler wrapping code-rag queries.

    ThreadingHTTPServer creates one handler per request, all sharing the
    module-level ``_query_state`` (loaded once at startup).
    """

    shutdown_timer: _ShutdownTimer | None = None

    def do_GET(self):
        self._reset_timer()
        if self.path == "/health":
            self._reply_json({"status": "ok"})
        elif self.path == "/list_indices":
            self._handle_list_indices()
        else:
            self._reply_error(404, f"Not found: {self.path}")

    def do_POST(self):
        self._reset_timer()
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            params: dict[str, Any] = json.loads(body)
        except Exception:
            self._reply_error(400, "Invalid JSON body")
            return

        routes = {
            "/search_code": self._handle_search_code,
            "/search_docs": self._handle_search_docs,
            "/get_file_symbols": self._handle_get_file_symbols,
            "/get_repo_structure": self._handle_get_repo_structure,
            "/get_symbol_info": self._handle_get_symbol_info,
            "/read_code": self._handle_read_code,
            "/shutdown": self._handle_shutdown,
        }
        handler = routes.get(self.path)
        if handler is None:
            self._reply_error(404, f"Not found: {self.path}")
            return
        handler(params)

    def log_message(self, format, *args):
        """Suppress default access logs; we log to stderr manually."""
        pass

    def _reset_timer(self):
        if _Handler.shutdown_timer is not None:
            _Handler.shutdown_timer.reset()

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    def _handle_search_code(self, params: dict):
        query = params.get("query", "")
        top_k = int(params.get("top_k", 8))
        offset = int(params.get("offset", 0))
        language = params.get("language")
        index_name = params.get("index_name")
        exclude_paths = params.get("exclude_paths")

        state = _get_state()
        indices = state["indices"]
        try:
            idx = _resolve_index(indices, index_name)
        except ValueError as exc:
            self._reply_error(400, str(exc))
            return

        warning = _check_vector_availability(idx)
        results = idx.hybrid.search_code(
            query, top_k=top_k + offset, language=language, exclude_paths=exclude_paths
        )
        results = results[offset : offset + top_k]

        output = [
            {
                "file_path": _qualify_path(idx.prefix, r.chunk.file_path),
                "start_line": r.chunk.start_line,
                "end_line": r.chunk.end_line,
                "language": r.chunk.language,
                "symbol_name": r.chunk.symbol_name,
                "score": round(r.score, 4),
                "snippet": _build_snippet(r, idx, top_k, rank),
            }
            for rank, r in enumerate(results)
        ]
        text = _render_results(output, offset=offset, warning=warning)
        self._reply_json({"result": text, "count": len(output)})

    def _handle_search_docs(self, params: dict):
        query = params.get("query", "")
        top_k = int(params.get("top_k", 8))
        offset = int(params.get("offset", 0))
        index_name = params.get("index_name")
        exclude_paths = params.get("exclude_paths")

        state = _get_state()
        indices = state["indices"]
        try:
            idx = _resolve_index(indices, index_name)
        except ValueError as exc:
            self._reply_error(400, str(exc))
            return

        warning = _check_vector_availability(idx)
        results = idx.hybrid.search_docs(
            query, top_k=top_k + offset, exclude_paths=exclude_paths
        )
        results = results[offset : offset + top_k]

        output = [
            {
                "file_path": _qualify_path(idx.prefix, r.chunk.file_path),
                "start_line": r.chunk.start_line,
                "end_line": r.chunk.end_line,
                "score": round(r.score, 4),
                "snippet": _build_snippet(r, idx, top_k, rank),
            }
            for rank, r in enumerate(results)
        ]
        text = _render_results(output, offset=offset, warning=warning)
        self._reply_json({"result": text, "count": len(output)})

    def _handle_get_file_symbols(self, params: dict):
        file_path = params.get("file_path", "")

        state = _get_state()
        indices = state["indices"]

        resolved = _resolve_file(indices, file_path)
        if resolved is None:
            self._reply_error(404, f"File not found: {file_path}")
            return
        idx, local_path = resolved

        abs_path = idx.repo_path / local_path
        if not abs_path.exists():
            self._reply_error(404, f"File not found: {file_path}")
            return

        try:
            content = abs_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = abs_path.read_text(encoding="latin-1")

        symbols = idx.metadata.get_symbols(local_path)
        file_info = idx.metadata.get_file_info(local_path)
        lang = file_info.language if file_info else ""
        size = file_info.size if file_info else len(content.encode())

        lines = [f"{file_path}  {lang}  {size} bytes", "", f"Symbols ({len(symbols)}):"]
        for s in symbols:
            lines.append(f"  {s.kind}  {s.name}  L{s.start_line}–{s.end_line}")
        if not symbols:
            lines.append("  (none)")
        self._reply_json({"result": "\n".join(lines), "count": len(symbols)})

    def _handle_get_repo_structure(self, params: dict):
        depth = int(params.get("depth", 3))
        path = params.get("path")
        index_name = params.get("index_name")

        state = _get_state()
        indices = state["indices"]
        lines: list[str] = []

        if path is not None:
            if len(indices) == 1 and not indices[0].prefix:
                idx = indices[0]
                target = idx.repo_path / path
            else:
                idx = None
                target = None
                if index_name is not None:
                    try:
                        idx = _resolve_index(indices, index_name)
                    except ValueError as exc:
                        self._reply_error(400, str(exc))
                        return
                    target = idx.repo_path / path
                else:
                    for candidate in indices:
                        if candidate.prefix and path.startswith(candidate.prefix + "/"):
                            idx = candidate
                            local_path = path[len(candidate.prefix) + 1 :]
                            target = candidate.repo_path / local_path
                            break
                        elif candidate.prefix and path == candidate.prefix:
                            idx = candidate
                            target = candidate.repo_path
                            break
                    if idx is None:
                        for candidate in indices:
                            t = candidate.repo_path / path
                            if t.exists():
                                idx = candidate
                                target = t
                                break
                if idx is None or target is None:
                    self._reply_error(404, f"Path not found: {path}")
                    return
            if not target.exists():
                self._reply_error(404, f"Path not found: {path}")
                return
            if not target.is_dir():
                self._reply_error(400, f"Not a directory: {path}")
                return
            lines.append(f"{target.name}/  (subtree of {idx.prefix or idx.repo_path.name})")
            _build_tree(target, "", depth, lines, idx.metadata, idx.repo_path)
        elif len(indices) == 1 and not indices[0].prefix:
            idx = indices[0]
            lines.append(f"{idx.repo_path.name}/  ({idx.metadata.count_files()} files)")
            _build_tree(idx.repo_path, "", depth, lines, idx.metadata, idx.repo_path)
        else:
            root = state["root"]
            lines.append(f"{root.name}/ (multi-index, {len(indices)} repos)")
            for idx in indices:
                lines.append(f"├── [{idx.prefix}]  ({idx.metadata.count_files()} files)")
                _build_tree(idx.repo_path, "│   ", depth, lines, idx.metadata, idx.repo_path)

        if len(lines) > _MAX_TREE_LINES:
            lines = lines[:_MAX_TREE_LINES]
            lines.append(f"... (truncated at {_MAX_TREE_LINES} lines — specify a 'path' to see a subdirectory)")
        self._reply_json({"result": "\n".join(lines)})

    def _handle_get_symbol_info(self, params: dict):
        symbol_name = params.get("symbol_name", "")
        mode = params.get("mode", "grouped")
        max_results = int(params.get("max_results", 20))
        exact_match = bool(params.get("exact_match", True))
        include_code = bool(params.get("include_code", False))
        include_references = bool(params.get("include_references", False))
        max_refs_per_symbol = int(params.get("max_refs_per_symbol", 30))
        index_name = params.get("index_name")

        state = _get_state()
        indices = state["indices"]

        if index_name is not None:
            try:
                indices = [_resolve_index(indices, index_name)]
            except ValueError as exc:
                self._reply_error(400, str(exc))
                return

        raw_symbols: list[tuple[_Index, Any]] = []
        browse_db_extras = None
        from code_rag.models import SymbolInfo
        from code_rag.storage.browse_db import BrowseDBProvider

        for idx in indices:
            if idx.browse_db and idx.browse_db.available:
                db_results = idx.browse_db.find_symbol(symbol_name)
                if db_results:
                    for item in db_results:
                        from code_rag.server import _normalize_cpp_path
                        rel_path = _normalize_cpp_path(idx, item["file_path"])
                        raw_symbols.append(
                            (
                                idx,
                                SymbolInfo(
                                    name=item["name"],
                                    kind=item["kind"],
                                    file_path=rel_path,
                                    start_line=item["start_line"],
                                    end_line=item["end_line"],
                                    language=_guess_language(rel_path),
                                ),
                            )
                        )
                    class_items = [i for i in db_results if i["kind_id"] in (1, 2, 5)]
                    if class_items:
                        bases = idx.browse_db.find_base_classes(symbol_name)
                        derived = idx.browse_db.find_derived_classes(symbol_name)
                        if bases or derived:
                            browse_db_extras = {}
                            if bases:
                                browse_db_extras["base_classes"] = bases
                            if derived:
                                browse_db_extras["derived_classes"] = [d["name"] for d in derived]
                    continue

            for s in idx.metadata.find_symbol(symbol_name, exact_only=exact_match):
                raw_symbols.append((idx, s))

        if mode == "declaration":
            result = self._symbol_info_declaration(
                raw_symbols, max_results, include_code, include_references, max_refs_per_symbol,
            )
        elif mode == "grouped":
            result = self._symbol_info_grouped(
                raw_symbols, max_results, include_code, include_references, max_refs_per_symbol,
            )
        else:
            result = self._symbol_info_all(
                raw_symbols, max_results, include_code, include_references, max_refs_per_symbol,
            )

        if browse_db_extras:
            extra_lines: list[str] = []
            if "base_classes" in browse_db_extras:
                extra_lines.append(f"Base classes: {', '.join(browse_db_extras['base_classes'])}")
            if "derived_classes" in browse_db_extras:
                extra_lines.append(f"Derived classes: {', '.join(browse_db_extras['derived_classes'])}")
            if extra_lines:
                nl = result.find("\n")
                if nl != -1:
                    result = result[: nl + 1] + "\n".join(extra_lines) + "\n" + result[nl + 1 :]

        self._reply_json({"result": result})

    def _handle_read_code(self, params: dict):
        file_path = params.get("file_path", "")
        start_line = int(params.get("start_line", 1))
        end_line = int(params.get("end_line", 1))
        context_lines = int(params.get("context_lines", 0))
        index_name = params.get("index_name")

        state = _get_state()
        indices = state["indices"]

        resolved = _resolve_file(indices, file_path)
        if resolved is None:
            self._reply_error(404, f"File not found: {file_path}")
            return
        idx, local_path = resolved

        abs_path = idx.repo_path / local_path
        if not abs_path.exists():
            self._reply_error(404, f"File not found: {file_path}")
            return

        try:
            all_lines = abs_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            try:
                all_lines = abs_path.read_text(encoding="latin-1").splitlines()
            except Exception as exc:
                self._reply_error(500, f"Could not read file: {exc}")
                return
        except Exception as exc:
            self._reply_error(500, f"Could not read file: {exc}")
            return

        total = len(all_lines)
        effective_start = max(1, start_line - context_lines)
        effective_end = min(total, end_line + context_lines)

        selected = all_lines[effective_start - 1 : effective_end]
        code = "\n".join(
            f"{effective_start + i}: {line}" for i, line in enumerate(selected)
        )
        header = f"{file_path}  L{effective_start}–{effective_end}  (total: {total} lines)"
        self._reply_json({"result": f"{header}\n\n{code}"})

    def _handle_list_indices(self):
        state = _get_state()
        indices = state["indices"]
        result = "\n".join(
            f"{idx.prefix or '(root)'}  {idx.repo_path}  "
            f"model={idx.config.resolve_model_name()}  "
            f"({idx.metadata.count_files()} files)"
            for idx in indices
        )
        self._reply_json({"result": result, "count": len(indices)})

    def _handle_shutdown(self, params: dict[str, Any]):
        server = getattr(self.__class__, "server_instance", None)
        if server is None:
            self._reply_error(500, "Server instance not available")
            return
        self._reply_json({"status": "shutting_down"})
        if self.shutdown_timer:
            self.shutdown_timer.cancel()
        server.shutdown()

    # ------------------------------------------------------------------
    # Symbol info helpers (mirrored from server.py)
    # ------------------------------------------------------------------

    def _symbol_info_all(
        self, raw_symbols, max_results, include_code, include_references, max_refs_per_symbol,
    ) -> str:
        lines: list[str] = []
        count = 0
        for idx, s in raw_symbols:
            if count >= max_results:
                break
            qpath = _qualify_path(idx.prefix, s.file_path)
            meta = f"{s.kind} {s.name}  {qpath}  L{s.start_line}–{s.end_line}  {s.language}"
            lines.append(_SEP)
            lines.append(meta)
            if include_code:
                code = _read_line_range(idx.repo_path, s.file_path, s.start_line, s.end_line)
                if code is not None:
                    lines.append("")
                    lines.append(code)
            if include_references:
                refs = _find_references(idx, s, max_refs=max_refs_per_symbol)
                if refs:
                    lines.append(f"\nReferences ({len(refs)}):")
                    lines.append(_render_refs(refs))
            count += 1
        lines.append(_SEP)
        return "\n".join(lines) if lines else "(no results)"

    def _symbol_info_declaration(
        self, raw_symbols, max_results, include_code, include_references, max_refs_per_symbol,
    ) -> str:
        lines: list[str] = []
        count = 0
        for idx, s in raw_symbols:
            if count >= max_results:
                break
            if s.kind not in _DECLARATION_KINDS:
                continue
            qpath = _qualify_path(idx.prefix, s.file_path)
            meta = f"{s.kind} {s.name}  {qpath}  L{s.start_line}–{s.end_line}  {s.language}"
            lines.append(_SEP)
            lines.append(meta)
            if include_code:
                code = _read_line_range(idx.repo_path, s.file_path, s.start_line, s.end_line)
                if code is not None:
                    lines.append("")
                    lines.append(code)
            if include_references:
                refs = _find_references(idx, s, max_refs=max_refs_per_symbol)
                if refs:
                    lines.append(f"\nReferences ({len(refs)}):")
                    lines.append(_render_refs(refs))
            count += 1
        lines.append(_SEP)
        return "\n".join(lines) if lines else "(no results)"

    def _symbol_info_grouped(
        self, raw_symbols, max_results, include_code, include_references, max_refs_per_symbol,
    ) -> str:
        from collections import OrderedDict

        groups: OrderedDict[tuple[str, str], list[tuple[_Index, Any]]] = OrderedDict()
        for idx, s in raw_symbols:
            key = (idx.prefix, s.file_path)
            groups.setdefault(key, []).append((idx, s))

        out_lines: list[str] = []
        result_count = 0
        for (_prefix, _fpath), members in groups.items():
            if result_count >= max_results:
                break
            idx = members[0][0]
            symbols = [s for _, s in members]
            qpath = _qualify_path(idx.prefix, _fpath)
            symbols.sort(key=lambda s: s.start_line)

            merged_ranges: list[tuple[int, int, list[Any]]] = []
            for sym in symbols:
                if merged_ranges:
                    prev_start, prev_end, prev_syms = merged_ranges[-1]
                    if sym.start_line <= prev_end + _MERGE_GAP:
                        merged_ranges[-1] = (prev_start, max(prev_end, sym.end_line), prev_syms + [sym])
                        continue
                merged_ranges.append((sym.start_line, sym.end_line, [sym]))

            for range_start, range_end, range_syms in merged_ranges:
                if result_count >= max_results:
                    break
                effective_end = min(range_end, range_start + _MAX_GROUP_LINES - 1)
                truncated = range_end > effective_end
                sym_list = "  ".join(
                    f"{s.kind} {s.name} (L{s.start_line}–{s.end_line})" for s in range_syms
                )
                meta = f"{qpath}  L{range_start}–{effective_end}  {symbols[0].language}"
                out_lines.append(_SEP)
                out_lines.append(meta)
                out_lines.append(f"Symbols: {sym_list}")
                code = _read_line_range(idx.repo_path, _fpath, range_start, effective_end)
                if include_code and code is not None:
                    if truncated:
                        hidden_syms = [s for s in range_syms if s.start_line > effective_end]
                        if hidden_syms:
                            sig_list = ", ".join(
                                f"{s.kind} {s.name} (L{s.start_line})" for s in hidden_syms
                            )
                            code += f"\n... (truncated, {range_end - effective_end} more lines — {sig_list})"
                        else:
                            code += f"\n... (truncated, {range_end - effective_end} more lines)"
                    out_lines.append("")
                    out_lines.append(code)
                if include_references:
                    refs = _find_references(idx, range_syms[0], max_refs=max_refs_per_symbol)
                    if refs:
                        out_lines.append(f"\nReferences ({len(refs)}):")
                        out_lines.append(_render_refs(refs))
                result_count += 1
        out_lines.append(_SEP)
        return "\n".join(out_lines) if out_lines else "(no results)"

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _reply_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False, indent=2)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def _reply_error(self, status: int, message: str):
        self._reply_json({"error": message}, status=status)


def run_query_server(port: int, idle_timeout: int):
    """Start the HTTP query server.

    Loads the model and indices once, then listens for JSON queries.
    Auto-shuts down after *idle_timeout* seconds without a request.
    """
    # ── Duplicate startup detection ──
    try:
        probe = socket.create_connection(("127.0.0.1", port), timeout=2.0)
        probe.sendall(
            b"GET /health HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n"
        )
        probe.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        while True:
            chunk = probe.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
        probe.close()
        resp = b"".join(chunks).decode("utf-8", errors="replace")
        if '"status": "ok"' in resp or '"status":"ok"' in resp:
            print(
                f"[CodeRAG] Query server is already running on http://127.0.0.1:{port}",
                file=sys.stderr,
            )
            sys.exit(0)
        else:
            print(
                f"[CodeRAG] Port {port} is already in use by another process.",
                file=sys.stderr,
            )
            sys.exit(1)
    except (ConnectionRefusedError, socket.timeout, OSError):
        pass  # Port is free — proceed

    print(f"[CodeRAG] Loading model and indices (this may take a moment)...", file=sys.stderr)
    state = _get_state()
    n_indices = len(state.get("indices", []))
    if n_indices == 0:
        print("[CodeRAG] Error: No indexed repositories found.", file=sys.stderr)
        sys.exit(1)
    print(f"[CodeRAG] Ready: {n_indices} index(es) loaded.", file=sys.stderr)

    timer = _ShutdownTimer(idle_timeout, port)
    _Handler.shutdown_timer = timer
    timer.reset()

    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    except OSError as exc:
        print(
            f"[CodeRAG] Cannot bind to port {port}: {exc}",
            file=sys.stderr,
        )
        print(
            f"[CodeRAG] The port may be in use. Try a different port with --port.",
            file=sys.stderr,
        )
        sys.exit(1)

    def _sig_handler(signum, frame):
        timer.cancel()
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    # ── Save state file for "code-rag servers" to discover ──
    _SERVER_DIR.mkdir(parents=True, exist_ok=True)
    state_file = _SERVER_DIR / f"{port}.json"
    repo_raw = os.environ.get("CODE_RAG_REPO", os.getcwd())
    repos = [r for r in repo_raw.split(os.pathsep) if r.strip()]
    state_file.write_text(
        json.dumps({
            "pid": os.getpid(),
            "port": port,
            "repos": repos,
            "idle_timeout": idle_timeout,
        }),
        encoding="utf-8",
    )

    def _cleanup():
        try:
            state_file.unlink(missing_ok=True)
        except Exception:
            pass

    try:
        _Handler.server_instance = server
        print(f"[CodeRAG] Query server listening on http://127.0.0.1:{port}", file=sys.stderr)
        print(f"[CodeRAG] Auto-shutdown after {idle_timeout}s idle.", file=sys.stderr)
        server.serve_forever()
    finally:
        timer.cancel()
        _cleanup()
