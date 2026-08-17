"""Unit tests for query_server.py (HTTP handlers, shutdown timer).

Runs without model loading: a ThreadingHTTPServer serves the real
``_Handler`` against monkeypatched state (fake indices).
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from code_rag.models import Chunk, SearchResult
from code_rag.query_server import _Handler, _ShutdownTimer


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeConfig:
    def resolve_model_name(self) -> str:
        return "test-model"


class _FakeMetadata:
    def __init__(self, files):
        self._files = list(files)

    def get_all_files(self):
        return self._files

    def count_files(self):
        return len(self._files)

    def get_symbols(self, path):
        return []

    def get_file_info(self, path):
        return None

    def find_symbol(self, name, exact_only=True):
        return []


class _FakeHybrid:
    def __init__(self, results=None):
        self._results = results or []

    def search_code(self, query, top_k=8, language=None, exclude_paths=None):
        return self._results

    def search_docs(self, query, top_k=8, exclude_paths=None):
        return self._results


@dataclass
class _FakeIndex:
    prefix: str = ""
    repo_path: Path = field(default_factory=Path)
    config: _FakeConfig = field(default_factory=_FakeConfig)
    embedder: object = None
    metadata: _FakeMetadata = field(default_factory=lambda: _FakeMetadata([]))
    hybrid: _FakeHybrid = field(default_factory=_FakeHybrid)
    vector_store: object = None
    bm25_store: object = None
    browse_db: object = None
    _path_case_map: dict | None = None


# ---------------------------------------------------------------------------
# Shutdown timer
# ---------------------------------------------------------------------------


def test_shutdown_timer_zero_timeout_disables_timer():
    timer = _ShutdownTimer(0, 12345)
    timer.reset()
    assert timer._timer is None


def test_shutdown_timer_negative_timeout_disables_timer():
    timer = _ShutdownTimer(-5, 12345)
    timer.reset()
    assert timer._timer is None


def test_shutdown_timer_positive_timeout_starts_timer():
    timer = _ShutdownTimer(60, 12345)
    timer.reset()
    assert timer._timer is not None and timer._timer.is_alive()
    timer.cancel()
    assert timer._timer is None


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


@pytest.fixture
def query_http(monkeypatch, tmp_path):
    import code_rag.query_server as qs

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("print('hi')\n", encoding="utf-8")

    idx = _FakeIndex(prefix="", repo_path=repo, metadata=_FakeMetadata(["main.py"]))
    state = {"root": repo, "indices": [idx]}

    monkeypatch.setattr(qs, "_get_state", lambda: state)
    monkeypatch.setattr(qs._Handler, "shutdown_timer", None)

    server = ThreadingHTTPServer(("127.0.0.1", 0), qs._Handler)
    qs._Handler.server_instance = server
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield SimpleNamespace(
        base=f"http://127.0.0.1:{server.server_address[1]}",
        repo=repo,
        idx=idx,
    )
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _get(base, endpoint):
    with urllib.request.urlopen(base + endpoint, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post(base, endpoint, payload):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base + endpoint, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_error(base, endpoint, payload):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(base, endpoint, payload)
    return excinfo.value


def test_health_reports_n_indices(query_http):
    body = _get(query_http.base, "/health")
    assert body["status"] == "ok"
    assert body["n_indices"] == 1


def test_list_indices(query_http):
    body = _get(query_http.base, "/list_indices")
    assert body["count"] == 1
    assert "test-model" in body["result"]
    assert str(query_http.repo) in body["result"]


def test_search_code_rejects_bad_top_k(query_http):
    exc = _post_error(query_http.base, "/search_code", {"query": "x", "top_k": "abc"})
    assert exc.code == 400
    msg = json.loads(exc.read().decode("utf-8"))
    assert "Invalid value for 'top_k'" in msg["error"]


def test_search_code_rejects_out_of_range_top_k(query_http):
    exc = _post_error(query_http.base, "/search_code", {"query": "x", "top_k": 0})
    assert exc.code == 400


def test_search_code_returns_results(query_http):
    query_http.idx.hybrid._results = [
        SearchResult(
            chunk=Chunk(
                text="print('hi')",
                file_path="main.py",
                start_line=1,
                end_line=1,
                chunk_type="code",
                language="python",
                symbol_name=None,
                symbol_kind=None,
            ),
            score=0.9,
            source="hybrid",
        )
    ]
    body = _post(query_http.base, "/search_code", {"query": "print"})
    assert body["count"] == 1
    assert "main.py" in body["result"]
    assert "print('hi')" in body["result"]


def test_read_code_normal(query_http):
    body = _post(
        query_http.base,
        "/read_code",
        {"file_path": "main.py", "start_line": 1, "end_line": 1},
    )
    assert "print('hi')" in body["result"]
    assert "total: 1 lines" in body["result"]


def test_read_code_rejects_invalid_range(query_http):
    exc = _post_error(
        query_http.base,
        "/read_code",
        {"file_path": "main.py", "start_line": 5, "end_line": 3},
    )
    assert exc.code == 400


def test_read_code_rejects_parent_traversal(query_http):
    secret = query_http.repo.parent / "secret.txt"
    secret.write_text("top secret\n", encoding="utf-8")
    exc = _post_error(
        query_http.base,
        "/read_code",
        {"file_path": "../secret.txt", "start_line": 1, "end_line": 1},
    )
    assert exc.code == 404


def test_get_file_symbols_missing_file(query_http):
    exc = _post_error(
        query_http.base, "/get_file_symbols", {"file_path": "nope.py"}
    )
    assert exc.code == 404


def test_tree_rejects_parent_traversal(query_http):
    exc = _post_error(query_http.base, "/get_repo_structure", {"path": ".."})
    assert exc.code == 404


def test_tree_accepts_normal_path(query_http):
    body = _post(query_http.base, "/get_repo_structure", {"path": ".", "depth": 2})
    assert "repo/" in body["result"]


def test_symbol_info_string_false_is_not_truthy(query_http):
    body = _post(
        query_http.base,
        "/get_symbol_info",
        {
            "symbol_name": "x",
            "exact_match": "false",
            "include_code": "false",
            "include_references": "false",
        },
    )
    assert body["result"] == "(no results)"


def test_symbol_info_include_references_does_not_crash(query_http, monkeypatch):
    """_find_references returns (refs, timed_out); the helpers must unpack it
    instead of treating the tuple as a list (regression: crashed with
    AttributeError and dropped the connection)."""
    import code_rag.query_server as qs

    query_http.idx.metadata.find_symbol = lambda name, exact_only=True: [
        SimpleNamespace(
            name="foo",
            kind="function",
            file_path="main.py",
            start_line=1,
            end_line=1,
            language="python",
        )
    ]
    monkeypatch.setattr(
        qs,
        "_find_references",
        lambda *a, **k: (
            [
                {
                    "file_path": "other.py",
                    "start_line": 10,
                    "end_line": 10,
                    "text": "foo()",
                }
            ],
            False,
        ),
    )
    body = _post(
        query_http.base,
        "/get_symbol_info",
        {"symbol_name": "foo", "include_references": True, "mode": "all"},
    )
    assert "References (1)" in body["result"]
    assert "other.py" in body["result"]


def test_symbol_info_include_references_timed_out_label(query_http, monkeypatch):
    import code_rag.query_server as qs

    query_http.idx.metadata.find_symbol = lambda name, exact_only=True: [
        SimpleNamespace(
            name="foo",
            kind="function",
            file_path="main.py",
            start_line=1,
            end_line=1,
            language="python",
        )
    ]
    monkeypatch.setattr(qs, "_find_references", lambda *a, **k: ([], True))
    body = _post(
        query_http.base,
        "/get_symbol_info",
        {"symbol_name": "foo", "include_references": True, "mode": "all"},
    )
    assert "timed out" in body["result"]


def test_post_non_object_json_body_400(query_http):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(query_http.base, "/search_code", [1, 2, 3])
    assert excinfo.value.code == 400
    msg = json.loads(excinfo.value.read().decode("utf-8"))
    assert "must be an object" in msg["error"]


def test_post_invalid_content_length_400(query_http):
    import socket
    import urllib.parse

    host, port = urllib.parse.urlparse(query_http.base).netloc.split(":")
    s = socket.create_connection((host, int(port)), timeout=10)
    s.sendall(
        (
            "POST /search_code HTTP/1.0\r\n"
            "Host: 127.0.0.1\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: -5\r\n"
            "\r\n"
        ).encode()
    )
    resp = b""
    while True:
        chunk = s.recv(4096)
        if not chunk:
            break
        resp += chunk
    s.close()
    status_line = resp.split(b"\r\n", 1)[0]
    assert b"400" in status_line, status_line


def test_unknown_route_404(query_http):
    exc = _post_error(query_http.base, "/nope", {})
    assert exc.code == 404


def test_shutdown_endpoint_stops_server(query_http):
    body = _post(query_http.base, "/shutdown", {})
    assert body["status"] == "shutting_down"

    deadline = time.time() + 5
    while True:
        try:
            _get(query_http.base, "/health")
        except Exception:
            break
        if time.time() > deadline:
            pytest.fail("server did not shut down after /shutdown")
        time.sleep(0.1)
