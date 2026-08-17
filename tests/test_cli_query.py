"""Unit tests for query CLI commands (no running server, no model loading)."""

from __future__ import annotations

import os

import pytest
from click.testing import CliRunner

from code_rag import cli


@pytest.fixture(autouse=True)
def _no_env_setup(monkeypatch):
    monkeypatch.setattr(cli, "_ensure_environment", lambda: None)


@pytest.fixture
def runner():
    return CliRunner()


def test_search_cmd_builds_payload(monkeypatch, runner):
    captured = {}

    def fake_query(endpoint, data=None, port=None):
        captured["endpoint"] = endpoint
        captured["data"] = data
        captured["port"] = port
        return "OK"

    monkeypatch.setattr(cli, "_query", fake_query)
    result = runner.invoke(
        cli.main, ["search", "auth", "--top-k", "5", "--index", "sub1"]
    )

    assert result.exit_code == 0
    assert result.output == "OK\n"
    assert captured["endpoint"] == "/search_code"
    assert captured["data"]["query"] == "auth"
    assert captured["data"]["top_k"] == 5
    assert captured["data"]["index_name"] == "sub1"


def test_search_cmd_omits_index_when_unset(monkeypatch, runner):
    captured = {}

    def fake_query(endpoint, data=None, port=None):
        captured["data"] = data
        return "OK"

    monkeypatch.setattr(cli, "_query", fake_query)
    result = runner.invoke(cli.main, ["search", "auth"])

    assert result.exit_code == 0
    assert captured["data"] == {"query": "auth", "top_k": 8, "offset": 0, "language": None}


def test_def_cmd_builds_payload(monkeypatch, runner):
    captured = {}

    def fake_query(endpoint, data=None, port=None):
        captured["endpoint"] = endpoint
        captured["data"] = data
        return "OK"

    monkeypatch.setattr(cli, "_query", fake_query)
    result = runner.invoke(
        cli.main,
        ["def", "LoginHandler", "--include-code", "--include-refs", "--max-results", "5"],
    )

    assert result.exit_code == 0
    assert captured["endpoint"] == "/get_symbol_info"
    assert captured["data"]["symbol_name"] == "LoginHandler"
    assert captured["data"]["include_code"] is True
    assert captured["data"]["include_references"] is True
    assert captured["data"]["max_results"] == 5


def test_read_cmd_requires_start_and_end(monkeypatch, runner):
    result = runner.invoke(cli.main, ["read", "main.py"])

    assert result.exit_code != 0
    assert "start" in result.output


def test_search_no_server_raises_click_exception(monkeypatch, runner):
    monkeypatch.setattr(cli, "_discover_server_port", lambda: None)

    result = runner.invoke(cli.main, ["search", "auth"])

    assert result.exit_code != 0
    assert "No query server found" in result.output


def test_health_no_server_exits_1(monkeypatch, runner):
    monkeypatch.setattr(cli, "_discover_server_port", lambda: None)

    result = runner.invoke(cli.main, ["health"])

    assert result.exit_code == 1
    assert "No query server found" in result.output


def test_query_server_multiple_repos_echo(monkeypatch, runner, tmp_path):
    repo_a = tmp_path / "a"
    repo_b = tmp_path / "b"
    repo_a.mkdir()
    repo_b.mkdir()

    def fake_run(port=None, idle_timeout=None):
        pass

    import code_rag.query_server as qs

    monkeypatch.setattr(qs, "run_query_server", fake_run)

    result = runner.invoke(
        cli.main,
        ["query-server", "--repo", str(repo_a), "--repo", str(repo_b), "--port", "9998"],
    )

    assert result.exit_code == 0
    expected = f"Repositories: {repo_a.resolve()}, {repo_b.resolve()}"
    assert expected in result.output


def test_query_server_env_var_with_spaces(monkeypatch, runner, tmp_path):
    """CODE_RAG_REPO with spaces must be split on os.pathsep, not whitespace."""
    repo = tmp_path / "My Repo With Spaces"
    repo.mkdir()
    captured = {}

    def fake_run(port=None, idle_timeout=None):
        captured["port"] = port

    import code_rag.query_server as qs

    monkeypatch.setattr(qs, "run_query_server", fake_run)
    monkeypatch.setenv("CODE_RAG_REPO", str(repo))

    result = runner.invoke(cli.main, ["query-server", "--port", "9997"])

    assert result.exit_code == 0
    assert str(repo.resolve()) in result.output
    assert captured["port"] == 9997
    assert str(repo.resolve()) in os.environ.get("CODE_RAG_REPO", "")
