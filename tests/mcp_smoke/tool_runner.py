from __future__ import annotations

import os
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

SERVER_PATH = Path(__file__).resolve().parents[2] / "src" / "code_rag" / "server.py"


def _discover_tool_names() -> list[str]:
    source = SERVER_PATH.read_text(encoding="utf-8")
    return re.findall(r"@mcp\.tool\(\)\s+def\s+([a-zA-Z_]\w*)\s*\(", source, re.MULTILINE)


@dataclass(frozen=True)
class ToolCall:
    tool: str
    params: dict[str, Any]
    invoke: Callable[[Any], str]


def _call_read_code(server: Any) -> str:
    return server.read_code("src/calc.py", 1, 20, context_lines=1)


def _call_search_code(server: Any) -> str:
    return server.search_code(
        query="add function",
        top_k=3,
        offset=0,
        language="python",
        exclude_paths=[],
    )


def _call_search_docs(server: Any) -> str:
    return server.search_docs(
        query="fixture repository",
        top_k=3,
        offset=0,
        exclude_paths=[],
    )


def _call_get_file_symbols(server: Any) -> str:
    return server.get_file_symbols("src/calc.py")


def _call_get_repo_structure(server: Any) -> str:
    return server.get_repo_structure(depth=4, path="src")


def _call_get_symbol_info(server: Any) -> str:
    return server.get_symbol_info(
        symbol_name="add",
        mode="all",
        max_results=20,
        exact_match=True,
        include_code=True,
        include_references=True,
        max_refs_per_symbol=30,
    )


DISPATCH: dict[str, ToolCall] = {
    "read_code": ToolCall(
        tool="read_code",
        params={"file_path": "src/calc.py", "start_line": 1, "end_line": 20, "context_lines": 1},
        invoke=_call_read_code,
    ),
    "list_indices": ToolCall(
        tool="list_indices",
        params={},
        invoke=lambda s: s.list_indices(),
    ),
    "search_code": ToolCall(
        tool="search_code",
        params={
            "query": "add function",
            "top_k": 3,
            "offset": 0,
            "language": "python",
            "exclude_paths": [],
        },
        invoke=_call_search_code,
    ),
    "search_docs": ToolCall(
        tool="search_docs",
        params={
            "query": "fixture repository",
            "top_k": 3,
            "offset": 0,
            "exclude_paths": [],
        },
        invoke=_call_search_docs,
    ),
    "get_file_symbols": ToolCall(
        tool="get_file_symbols",
        params={"file_path": "src/calc.py"},
        invoke=_call_get_file_symbols,
    ),
    "get_repo_structure": ToolCall(
        tool="get_repo_structure",
        params={"depth": 4, "path": "src"},
        invoke=_call_get_repo_structure,
    ),
    "get_symbol_info": ToolCall(
        tool="get_symbol_info",
        params={
            "symbol_name": "add",
            "mode": "all",
            "max_results": 20,
            "exact_match": True,
            "include_code": True,
            "include_references": True,
            "max_refs_per_symbol": 30,
        },
        invoke=_call_get_symbol_info,
    ),
}


def run_all_tools(repo_root: Path) -> list[dict[str, Any]]:
    """Invoke every discovered MCP tool exactly once and capture raw outputs."""
    os.environ["CODE_RAG_REPO"] = str(repo_root)
    import code_rag.server as server

    server._state.clear()
    results: list[dict[str, Any]] = []
    for tool_name in _discover_tool_names():
        started = time.perf_counter()
        call = DISPATCH.get(tool_name)
        if call is None:
            results.append(
                {
                    "tool": tool_name,
                    "params": {},
                    "ok": False,
                    "output": "",
                    "error": f"No invocation template configured for tool '{tool_name}'",
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                }
            )
            continue
        try:
            raw = call.invoke(server)
            results.append(
                {
                    "tool": tool_name,
                    "params": call.params,
                    "ok": True,
                    "output": raw,
                    "error": "",
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                }
            )
        except Exception:
            results.append(
                {
                    "tool": tool_name,
                    "params": call.params,
                    "ok": False,
                    "output": "",
                    "error": traceback.format_exc(),
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                }
            )
    return results

