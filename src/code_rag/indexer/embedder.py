from __future__ import annotations

import numpy as np

MODEL_CONFIGS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-large-en-v1.5": 1024,
}

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    """Embedding engine with dual runtime paths:
    - CPU: ONNX Runtime + bge-small (fast, lightweight)
    - GPU: sentence-transformers + torch + bge-large (high quality)
    """

    def __init__(
        self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "auto"
    ):
        self._model_name = model_name
        self._device = device
        self._model = None  # LAZY LOADING
        self._resolved_device: str | None = None
        self._backend: str | None = None  # "onnx" or "sentence_transformers"
        self._tokenizer_obj = None  # Lazy-loaded tokenizer (lightweight, no model)

    def _resolve_device(self):
        """Resolve device and model name without loading model."""
        if self._resolved_device is not None:
            return
        if self._device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    # is_available() can return True even when the CUDA
                    # runtime fails to initialise (e.g. CPU-only torch build
                    # paired with an NVIDIA driver).  Do a cheap smoke-test
                    # that triggers the full lazy-init before committing.
                    try:
                        torch.zeros(1, device="cuda")
                        self._resolved_device = "cuda"
                    except Exception:
                        self._resolved_device = "cpu"
                else:
                    self._resolved_device = "cpu"
            except ImportError:
                self._resolved_device = "cpu"
        else:
            self._resolved_device = self._device
        # Auto-upgrade model for GPU
        if (
            self._resolved_device == "cuda"
            and self._model_name == "BAAI/bge-small-en-v1.5"
        ):
            self._model_name = "BAAI/bge-large-en-v1.5"

    @property
    def tokenizer(self):
        """Get the model's tokenizer without loading model weights.

        Useful for accurate token counting during chunking.
        """
        if self._tokenizer_obj is None:
            self._resolve_device()
            from transformers import AutoTokenizer

            self._tokenizer_obj = AutoTokenizer.from_pretrained(self._model_name)
        return self._tokenizer_obj

    def _ensure_loaded(self):
        if self._model is not None:
            return
        self._resolve_device()

        if self._resolved_device == "cpu":
            # Try ONNX Runtime first for CPU (faster inference)
            try:
                self._load_onnx()
                return
            except Exception:
                pass

        # Fallback / GPU path: sentence-transformers + torch
        self._load_sentence_transformers()

    def _load_onnx(self):
        """Load model with ONNX Runtime backend via sentence-transformers."""
        print(f"Loading {self._model_name} with ONNX Runtime (CPU)...")
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            self._model_name,
            device="cpu",
            backend="onnx",
        )
        self._model.max_seq_length = 512
        self._backend = "onnx"
        print(f"Model loaded (ONNX). Dimension: {self.dimension}")

    def _load_sentence_transformers(self):
        """Load model with standard sentence-transformers + torch."""
        print(f"Loading {self._model_name} on {self._resolved_device}...")
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            self._model_name,
            device=self._resolved_device,
        )
        self._model.max_seq_length = 512
        self._backend = "sentence_transformers"
        print(f"Model loaded (torch). Dimension: {self.dimension}")

    def _compute_batch_size(self) -> int:
        """Calculate optimal batch size based on GPU memory.

        Returns a conservative batch size that fits in available VRAM with
        generous safety margins.  Falls back to 64 on CPU or when detection
        fails.  Capped at 128 to avoid CUDA instability on some GPUs.
        """
        self._resolve_device()
        if self._resolved_device != "cuda":
            return 64  # CPU default

        try:
            import torch

            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / 1e9

            dim = self.dimension
            if dim >= 1024:
                # bge-large: ~1.2GB model weights, heavier per-sample cost
                model_gb = 1.5
                per_sample_mb = 1.5  # conservative for long sequences
            else:
                # bge-small: ~0.13GB model weights
                model_gb = 0.4
                per_sample_mb = 0.5

            available_gb = total_gb * 0.75 - model_gb  # 25% safety margin
            batch_size = max(8, int(available_gb * 1024 / per_sample_mb))
            return min(batch_size, 128)  # conservative cap for stability
        except Exception:
            return 64

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed passages. NO query instruction prefix."""
        self._ensure_loaded()
        batch_size = self._compute_batch_size()
        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query. PREPENDS BGE instruction prefix."""
        self._ensure_loaded()
        prefixed = BGE_QUERY_INSTRUCTION + query
        return self._model.encode(
            prefixed,
            normalize_embeddings=True,
        )

    @property
    def dimension(self) -> int:
        self._resolve_device()
        return MODEL_CONFIGS.get(self._model_name, 384)
