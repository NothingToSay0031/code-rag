from __future__ import annotations

from pathlib import Path

FIXTURE_FILES: dict[str, str] = {
    "README.md": "# Fixture repository\n\nUsed by MCP smoke tests.\n",
    "src/calc.py": (
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n\n"
        "def apply_twice(value: int) -> int:\n"
        "    return add(value, value)\n"
    ),
    "src/calc.ts": (
        "export function mul(a: number, b: number): number {\n"
        "  return a * b;\n"
        "}\n"
    ),
    "src/engine.cpp": (
        "int twice(int x) {\n"
        "    return x * 2;\n"
        "}\n"
    ),
    "docs/guide.md": (
        "# Guide\n\n"
        "This fixture repository demonstrates addition and multiplication.\n"
    ),
}


def build_fixture_repo(repo_root: Path) -> Path:
    """Create deterministic fixture files used by local MCP smoke tests."""
    repo_root.mkdir(parents=True, exist_ok=True)
    for rel_path, content in FIXTURE_FILES.items():
        target = repo_root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return repo_root

