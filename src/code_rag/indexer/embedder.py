from __future__ import annotations

import os
import sys

import numpy as np

# Reduce CUDA allocator fragmentation.  Set before any torch import so the
# flag takes effect when the CUDA context is first created.
# expandable_segments is only supported on Linux; skip on other platforms to
# avoid the torch UserWarning about unsupported allocator options.
if sys.platform == "linux":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_CONFIGS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-large-en-v1.5": 1024,
    "Qwen/Qwen3-Embedding-0.6B": 1024,
    "Qwen/Qwen3-Embedding-4B": 2560,
    "Qwen/Qwen3-Embedding-8B": 4096,
}

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
QWEN3_QUERY_INSTRUCTION = (
    "Given a natural language query, retrieve relevant code snippets and documentation"
)


class Embedder:
    """Embedding engine with dual runtime paths:
    - CPU: ONNX Runtime + bge-small (fast, lightweight)
    - GPU: sentence-transformers + torch + bge-large (high quality)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "auto",
        max_seq_length: int | None = None,
    ):
        self._model_name = model_name
        self._device = device
        self._max_seq_length = max_seq_length
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

        if self._resolved_device == "cpu" and "qwen3" not in self._model_name.lower():
            # Try ONNX Runtime first for CPU (faster inference).
            # Qwen3 models are decoder-based and lack ONNX runtime support,
            # so skip directly to the sentence-transformers path for them.
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
        self._model.max_seq_length = self._max_seq_length or 512
        self._backend = "onnx"
        print(f"Model loaded (ONNX). Dimension: {self.dimension}")

    def _load_sentence_transformers(self):
        """Load model with standard sentence-transformers + torch."""
        print(f"Loading {self._model_name} on {self._resolved_device}...")
        from sentence_transformers import SentenceTransformer

        kwargs: dict = {}
        if "qwen3" in self._model_name.lower():
            # Qwen3 is a decoder (left-to-right) model; left-padding is required
            # so that the last real token is always at position [-1].
            kwargs["processor_kwargs"] = {"padding_side": "left"}

        self._model = SentenceTransformer(
            self._model_name,
            device=self._resolved_device,
            **kwargs,
        )
        # Cap the model's input length to match the chunking budget so that
        # embeddings cover the full chunk text without silent truncation.
        # Falls back to 512 for BGE models when no explicit limit is given.
        self._model.max_seq_length = self._max_seq_length or 512
        self._backend = "sentence_transformers"
        print(f"Model loaded (torch). Dimension: {self.dimension}")

    def _compute_batch_size(self) -> int:
        """Calculate optimal batch size from *actual* free VRAM after model load.

        Queries ``torch.cuda.mem_get_info()`` so that whatever VRAM the model
        already occupies is already subtracted — no guesswork about model size.

        Transformer self-attention allocates an (n_heads × seq_len × seq_len)
        matrix per sample per forward pass.  Memory therefore scales as
        O(seq_len²), not O(seq_len).  At 8 192 tokens that is ≈ 2 GB per
        sample for Qwen3-0.6B, so the per-sample estimate must use quadratic
        scaling to produce a safe batch size.

        Falls back to 64 on CPU or when detection fails.
        """
        self._resolve_device()
        if self._resolved_device != "cuda":
            return 64  # CPU default

        try:
            import torch

            # Flush the allocator cache so mem_get_info reflects the real free
            # memory after model weights are fully resident in VRAM.
            torch.cuda.empty_cache()
            free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
            free_gb = free_bytes / 1e9

            seq_len = getattr(self._model, "max_seq_length", None) or 512

            dim = self.dimension
            if dim >= 1024:
                # bge-large / Qwen3-Embedding-*
                # Empirical baseline at 512 tokens: ~8 MB (attention matrix
                # + hidden-state activations for a ~0.6-1 B param model).
                # Attention scales as seq_len², so we square the ratio.
                base_per_sample_mb = 8.0
            else:
                # bge-small (~33 M params): lighter attention heads
                base_per_sample_mb = 2.0

            # *** Quadratic scaling ***: attention memory ∝ seq_len²
            seq_scale = max(1.0, (seq_len / 512) ** 2)
            per_sample_mb = base_per_sample_mb * seq_scale

            # Keep 20 % of free VRAM as a safety buffer for allocator overhead.
            usable_gb = free_gb * 0.80
            batch_size = max(1, int(usable_gb * 1024 / per_sample_mb))
            return min(batch_size, 128)  # cap for stability
        except Exception:
            return 8  # safe GPU fallback when detection fails

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed passages with automatic OOM recovery.

        If the GPU runs out of memory mid-batch the batch size is halved and
        the entire encode call is retried from the start.  This provides a
        safety net independent of the ``_compute_batch_size`` estimate.
        """
        self._ensure_loaded()
        batch_size = self._compute_batch_size()

        while True:
            try:
                return self._model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                )
            except RuntimeError as exc:
                oom = "out of memory" in str(exc).lower()
                if not oom or batch_size <= 1:
                    raise
                try:
                    import torch

                    torch.cuda.empty_cache()
                except ImportError:
                    pass
                batch_size = max(1, batch_size // 2)
                print(f"\nGPU OOM detected – retrying with batch_size={batch_size}")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query with the appropriate instruction prefix for the model."""
        self._ensure_loaded()
        if "qwen3" in self._model_name.lower():
            # Qwen3 Embedding uses an explicit instruction header on the query side.
            # Documents are encoded without any prefix.
            prefixed = f"Instruct: {QWEN3_QUERY_INSTRUCTION}\nQuery:{query}"
        else:
            prefixed = BGE_QUERY_INSTRUCTION + query
        return self._model.encode(
            prefixed,
            normalize_embeddings=True,
        )

    @property
    def dimension(self) -> int:
        self._resolve_device()
        return MODEL_CONFIGS.get(self._model_name, 384)
