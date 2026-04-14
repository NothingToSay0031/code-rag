from __future__ import annotations

import gc
import os
import sys

import numpy as np
from tqdm import tqdm

# Reduce CUDA allocator fragmentation.  Set before any torch import so the
# flag takes effect when the CUDA context is first created.
# expandable_segments is only supported on Linux; on Windows use
# max_split_size_mb to reduce fragmentation from large alloc/free cycles.
if sys.platform == "linux":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
elif sys.platform == "win32":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

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

    def _get_free_vram_gb(self) -> float:
        """Return current free VRAM in GB.  Returns 0.0 on non-CUDA or error."""
        if self._resolved_device != "cuda":
            return 0.0
        try:
            import torch

            torch.cuda.empty_cache()
            free_bytes, _ = torch.cuda.mem_get_info(0)
            return free_bytes / 1e9
        except Exception:
            return 0.0

    def _compute_batch_size(self) -> int:
        """Backward-compatible wrapper: safe batch size for max_seq_length.

        Prefer :meth:`_compute_batch_size_for_seq_len` when the actual token
        length of the current group is known.
        """
        self._resolve_device()
        if self._resolved_device != "cuda":
            return 64
        seq_len = getattr(self._model, "max_seq_length", None) or 512
        return self._compute_batch_size_for_seq_len(seq_len, self._get_free_vram_gb())

    def _compute_batch_size_for_seq_len(self, seq_len: int, free_gb: float) -> int:
        """Calculate a safe mini-batch size for a specific sequence length.

        Args:
            seq_len:  Maximum token length among texts in the current group.
            free_gb:  Free VRAM in GB (obtain once per encode call, reuse).

        Transformer self-attention memory scales as O(seq_len²).  We apply
        this quadratic factor relative to a 512-token baseline and keep 35%
        of free VRAM as a safety buffer for allocator overhead and
        fragmentation (important on Windows where expandable_segments is
        unsupported).
        """
        if self._resolved_device != "cuda" or free_gb <= 0:
            return 64 if self._resolved_device != "cuda" else 8

        dim = self.dimension
        # Empirical baseline at 512 tokens.
        #
        # Initial estimate was 8 MB, but empirical data on Qwen3-0.6B (Windows,
        # no FlashAttention) shows OOM at batch=13 with 10.4 GB free at seq=4096
        # (implying >800 MB/sample, formula said 512 MB) and OOM at batch=64
        # with seq≈1316 (implying >162 MB/sample, formula said 52 MB).
        # The consistent underestimate factor is ~3–4×, so we use 32 MB as the
        # baseline.  On Linux with FlashAttention this is overly conservative
        # (O(n) not O(n²) attention memory), but safe is better than OOM.
        base_per_sample_mb = 32.0 if dim >= 1024 else 2.0

        # Quadratic scaling: attention matrix ∝ seq_len²
        seq_scale = max(1.0, (max(seq_len, 1) / 512) ** 2)
        per_sample_mb = base_per_sample_mb * seq_scale

        # Keep 50% of free VRAM as buffer.  Windows CUDA allocator uses
        # max_split_size_mb:128 which causes fragmentation over long runs;
        # the extra headroom prevents late-batch OOM as the allocator drifts.
        usable_gb = free_gb * 0.50
        batch_size = max(1, int(usable_gb * 1024 / per_sample_mb))
        return min(batch_size, 128)

    def _count_tokens_fast(self, text: str) -> int:
        """Count BPE tokens using the cached tokenizer.

        Falls back to whitespace-split word count when the tokenizer is not
        yet loaded.  Called once per chunk before the batching loop so
        it should be fast — no model weights involved.
        """
        if self._tokenizer_obj is not None:
            return len(self._tokenizer_obj.encode(text, add_special_tokens=False))
        return max(1, len(text.split()))

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed passages with length-aware batching to prevent VRAM overflow.

        Standard ``model.encode()`` uses one fixed batch size for the entire
        list.  When long (4096-token) and short (50-token) chunks are mixed,
        every sample in a mini-batch gets padded to the longest sequence —
        wasting VRAM and computation.  Worse, a batch size computed for the
        worst-case length is too conservative for short sequences, leaving
        most of the GPU idle.

        This method:
        1. Pre-computes BPE token lengths for every chunk.
        2. Sorts chunks ascending by length (short → cheap → large batches).
        3. For each group, computes a fresh batch size based on that group's
           actual maximum length — short groups get batch_size ≈ 100+,
           long groups get batch_size ≈ 6–12.
        4. Calls :meth:`_encode_with_retry` per group (OOM halving is
           contained to the small group, not the full 5 000-chunk list).
        """
        self._ensure_loaded()

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # --- CPU fast-path: no memory pressure, single encode call ----------
        if self._resolved_device != "cuda":
            return self._model.encode(  # type: ignore[union-attr]
                texts,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,
            )

        # --- GPU path: length-aware batching --------------------------------
        lengths = [self._count_tokens_fast(t) for t in texts]

        # Sort ascending: short sequences → large batches first (fast wins),
        # long sequences → small but safe batches later.
        sorted_idx = sorted(range(len(texts)), key=lambda i: lengths[i])

        # Sample free VRAM once after the model is fully resident.
        # Re-using this avoids a CUDA context sync per mini-batch.
        # It is refreshed after any OOM event inside _encode_with_retry.
        free_gb = self._get_free_vram_gb()

        min_len = lengths[sorted_idx[0]]
        max_len = lengths[sorted_idx[-1]]
        bs_short = self._compute_batch_size_for_seq_len(min_len, free_gb)
        bs_long = self._compute_batch_size_for_seq_len(max_len, free_gb)
        print(
            f"  Token lengths: min={min_len}, max={max_len} | "
            f"batch_size: {bs_long} (long) → {bs_short} (short) | "
            f"free VRAM: {free_gb:.1f} GB"
        )

        all_embeddings: list[np.ndarray | None] = [None] * len(texts)
        groups_done = 0
        pos = 0
        pbar = tqdm(total=len(texts), desc="Embedding", unit="chunk")
        try:
            while pos < len(sorted_idx):
                groups_done += 1

                # Periodically flush the Python GC and CUDA allocator cache
                # to prevent cumulative drift from hundreds of encode() calls.
                if groups_done % 50 == 0:
                    gc.collect()
                    free_gb = self._get_free_vram_gb()

                group_min_len = lengths[sorted_idx[pos]]

                # Refresh VRAM before expensive long-sequence groups.
                if group_min_len > 1024:
                    updated = self._get_free_vram_gb()
                    if updated < free_gb:
                        free_gb = updated

                # ----------------------------------------------------------------
                # Find the optimal batch size via binary search.
                #
                # Goal: largest B in [1, 128] where the TAIL of the proposed
                # group sorted_idx[pos : pos+B] is safe for B samples.
                #
                # Why not "compute from min_len then shrink"?
                #   pos=7063: min_len=500 → batch_size=128
                #   group[7063:7191] tail_len=4096 → safe_bs=2 → batch_size=2
                #   group[7063:7065] tail_len=501  → safe_bs=128 ≥ 2 → break
                #   Result: process only 2 chunks of 500 tokens (step-2 crawl!)
                #
                # Binary search finds the true optimal (e.g. B=19 here) in
                # O(log 128) ≈ 7 probes instead of missing the right answer.
                # ----------------------------------------------------------------
                cap = min(len(sorted_idx) - pos, 128)
                lo, hi = 1, cap
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    tail_len = lengths[sorted_idx[pos + mid - 1]]
                    safe_bs = self._compute_batch_size_for_seq_len(tail_len, free_gb)
                    if safe_bs >= mid:
                        lo = mid  # mid samples fit within VRAM → try larger
                    else:
                        hi = mid - 1  # tail too long for mid samples → smaller
                batch_size = lo
                group_end = pos + batch_size

                group_orig_idx = sorted_idx[pos:group_end]
                group_texts = [texts[j] for j in group_orig_idx]

                try:
                    group_embs = self._encode_with_retry(group_texts)
                except RuntimeError:
                    # Hard failure: refresh VRAM so subsequent groups are
                    # sized more conservatively before re-raising.
                    free_gb = self._get_free_vram_gb()
                    raise

                for k, orig_idx in enumerate(group_orig_idx):
                    all_embeddings[orig_idx] = group_embs[k]

                pbar.update(len(group_orig_idx))
                pos = group_end
        finally:
            pbar.close()

        return np.vstack(all_embeddings)  # type: ignore[arg-type]

    def _encode_with_retry(self, texts: list[str]) -> np.ndarray:
        """Encode a length-homogeneous group with OOM halving retry.

        Unlike the previous implementation which restarted the entire
        5 000-chunk list after OOM, here we only retry the small group,
        so recovery is cheap and fast.
        """
        batch_size = len(texts)
        while True:
            try:
                return self._model.encode(  # type: ignore[union-attr]
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,  # outer tqdm tracks overall progress
                    normalize_embeddings=True,
                )
            except RuntimeError as exc:
                oom = "out of memory" in str(exc).lower()
                if not oom or batch_size <= 1:
                    raise
                # Release any tensors left behind by the failed encode.
                gc.collect()
                try:
                    import torch

                    torch.cuda.empty_cache()
                except ImportError:
                    pass
                batch_size = max(1, batch_size // 2)
                print(f"\nGPU OOM – retrying with batch_size={batch_size}")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query with the appropriate instruction prefix for the model."""
        self._ensure_loaded()
        assert self._model is not None  # _ensure_loaded always sets this
        if "qwen3" in self._model_name.lower():
            # Qwen3 Embedding uses an explicit instruction header on the query side.
            # Documents are encoded without any prefix.
            prefixed = f"Instruct: {QWEN3_QUERY_INSTRUCTION}\nQuery:{query}"
        else:
            prefixed = BGE_QUERY_INSTRUCTION + query
        return np.asarray(
            self._model.encode(
                prefixed,
                normalize_embeddings=True,
            )
        )

    @property
    def dimension(self) -> int:
        self._resolve_device()
        return MODEL_CONFIGS.get(self._model_name, 384)
