from __future__ import annotations

import json
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

    # ------------------------------------------------------------------
    # Persistence: save/load model settings to .code-rag/config.json
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist model_name and device into ``<data_dir>/config.json``.

        Called by :class:`~code_rag.indexer.pipeline.IndexPipeline` after
        initialisation so the server can recreate the exact same embedder.
        """
        config_file = self.abs_data_dir / "config.json"
        self.abs_data_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "model_name": self.model_name,
            "device": self.device,
        }
        config_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, repo_path: Path) -> "CodeRagConfig":
        """Load a :class:`CodeRagConfig` from ``<repo_path>/.code-rag/config.json``.

        Falls back to default values when the file is absent or unreadable
        (e.g. indices built before this feature was added).
        """
        config_file = repo_path / ".code-rag" / "config.json"
        if config_file.exists():
            try:
                data = json.loads(config_file.read_text(encoding="utf-8"))
                return cls(
                    repo_path=repo_path,
                    model_name=data.get("model_name", "Qwen/Qwen3-Embedding-0.6B"),
                    device=data.get("device", "auto"),
                )
            except Exception:
                pass
        return cls(repo_path=repo_path)


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
