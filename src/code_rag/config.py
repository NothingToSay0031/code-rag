from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FilterConfig:
    include_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)


@dataclass
class CodeRagConfig:
    repo_path: Path
    data_dir: str = ".code-rag"
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    device: str = "auto"
    max_chunk_tokens: int = 4096
    custom_type_mappings: dict[str, str] = field(
        default_factory=lambda: {".glsl": "c", ".hlsl": "c", ".fx": "c"}
    )
    include_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.repo_path, str):
            self.repo_path = Path(self.repo_path)

    @property
    def abs_data_dir(self) -> Path:
        return self.repo_path / self.data_dir

    def resolve_device(self) -> str:
        if self.device == "cpu":
            return "cpu"
        # "auto" or "cuda": try CUDA, fall back to CPU
        # torch.cuda.is_available() can return False on some Windows setups even
        # when CUDA works (DLL load-order issue), so probe with a tiny tensor too.
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            torch.zeros(1, device="cuda")
            return "cuda"
        except Exception:
            return "cpu"

    def resolve_model_name(self) -> str:
        device = self.resolve_device()
        if self.model_name == "BAAI/bge-small-en-v1.5" and device == "cuda":
            return "BAAI/bge-large-en-v1.5"
        return self.model_name


def parse_coderagfilter(path: Path) -> FilterConfig:
    """Parse a ``.coderagfilter`` file.

    Supports two formats (can be mixed in one file):

    **Legacy format** — bare patterns are excludes, ``!`` prefix means include::

        node_modules/
        *.log
        !src/important.log

    **Section format** — explicit ``[exclude]`` / ``[include]`` headers::

        [exclude]
        node_modules/
        *.log

        [include]
        src/important.log
    """
    if not path.exists():
        return FilterConfig()

    include_patterns: list[str] = []
    exclude_patterns: list[str] = []

    # "exclude" or "include"; None means legacy mode (no section header seen yet)
    section: str | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Section headers
        lower = line.lower()
        if lower == "[exclude]":
            section = "exclude"
            continue
        if lower == "[include]":
            section = "include"
            continue

        if section == "include":
            include_patterns.append(line)
        elif section == "exclude":
            exclude_patterns.append(line)
        else:
            # Legacy: ! prefix → include, bare → exclude
            if line.startswith("!"):
                include_patterns.append(line[1:])
            else:
                exclude_patterns.append(line)

    return FilterConfig(
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )
