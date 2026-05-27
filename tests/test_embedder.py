"""Tests for embedder.py — model loading, embedding quality, and correctness.

Uses BAAI/bge-small-en-v1.5 (384-dim) as prescribed by the user.
"""

from __future__ import annotations

import numpy as np
import pytest

from code_rag.indexer.embedder import Embedder, MODEL_CONFIGS, BGE_QUERY_INSTRUCTION


MODEL = "BAAI/bge-small-en-v1.5"
EXPECTED_DIM = 384


# ---------------------------------------------------------------------------
# Model config / static properties (no loading)
# ---------------------------------------------------------------------------


class TestEmbedderConfig:
    def test_model_in_configs(self):
        """BGE-small must be in MODEL_CONFIGS with dimension 384."""
        assert MODEL in MODEL_CONFIGS
        assert MODEL_CONFIGS[MODEL] == EXPECTED_DIM

    def test_dimension_without_loading_model(self):
        """dimension property should work without loading model weights."""
        e = Embedder(model_name=MODEL, device="cpu")
        assert e.dimension == EXPECTED_DIM

    def test_effective_max_seq_length_without_loading(self):
        """effective_max_seq_length should work without loading model weights."""
        e = Embedder(model_name=MODEL, device="cpu")
        cap = e.effective_max_seq_length
        assert cap > 0
        assert cap <= 512  # BGE-small has 512 position embeddings

    def test_is_not_static(self):
        """BGE-small is not a static model2vec model."""
        e = Embedder(model_name=MODEL, device="cpu")
        assert e.is_static is False

    def test_tokenizer_without_full_model_load(self):
        """Tokenizer should be accessible without loading model weights."""
        e = Embedder(model_name=MODEL, device="cpu")
        tok = e.tokenizer
        assert tok is not None
        # Tokenizer should actually work
        ids = tok.encode("int32 GetWeight() const", add_special_tokens=False)
        assert len(ids) > 0


# ---------------------------------------------------------------------------
# Embedding correctness (requires model download — CPU only),
# skipped in CI without network or on first run if model not cached.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def embedder():
    """Module-scoped: load BGE-small once for all embedding tests."""
    e = Embedder(model_name=MODEL, device="cpu")
    e._ensure_loaded()
    return e


class TestEmbeddingCorrectness:
    """Verify embedding output is sensible — not just non-crashing."""

    def test_embed_passages_returns_correct_shape(self, embedder):
        texts = [
            "void UAnimationAsset::SetSkeleton(USkeleton* NewSkeleton);",
            "int32 GetWeight() const { return FMath::Clamp(TotalWeight, 0.f, 1.f); }",
            "/** Animation Extraction Context */",
        ]
        vecs = embedder.embed_passages(texts)
        assert vecs.shape == (3, EXPECTED_DIM)
        assert vecs.dtype == np.float32

    def test_embed_single_passage(self, embedder):
        vecs = embedder.embed_passages(["FVector GetFilterLastOutput() const;"])
        assert vecs.shape == (1, EXPECTED_DIM)

    def test_embed_query_returns_correct_shape(self, embedder):
        vec = embedder.embed_query("animation skeleton set function")
        assert vec.shape == (EXPECTED_DIM,)
        assert vec.dtype == np.float32

    def test_embed_empty_string(self, embedder):
        vecs = embedder.embed_passages([""])
        assert vecs.shape == (1, EXPECTED_DIM)
        assert not np.all(np.isnan(vecs))

    def test_no_nan_or_inf(self, embedder):
        texts = [
            "UAnimationAsset::SetSkeleton",
            "",
            "template<typename T> void Foo() {}",
            "#include <vector>",
        ]
        vecs = embedder.embed_passages(texts)
        assert not np.any(np.isnan(vecs)), "Embeddings contain NaN"
        assert not np.any(np.isinf(vecs)), "Embeddings contain Inf"

    def test_embeddings_are_not_all_identical(self, embedder):
        texts = [
            "void SetSkeleton(USkeleton* NewSkeleton);",
            "float GetBlendWeight() const;",
            "struct FBlendSampleData { int32 Index; };",
        ]
        vecs = embedder.embed_passages(texts)
        # Pairwise cosine similarities should be < 0.99 (not identical)
        from numpy.linalg import norm
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = np.dot(vecs[i], vecs[j]) / (norm(vecs[i]) * norm(vecs[j]))
                assert sim < 0.99, (
                    f"Texts {i} and {j} have near-identical embeddings (sim={sim:.4f})"
                )

    def test_l2_normalized(self, embedder):
        """BGE embeddings should be L2-normalized (unit vectors)."""
        texts = [
            "void TickAssetPlayer(FAnimTickRecord& Instance);",
            "/** Root motion extraction context */",
        ]
        vecs = embedder.embed_passages(texts)
        from numpy.linalg import norm
        for i, v in enumerate(vecs):
            n = norm(v)
            assert abs(n - 1.0) < 0.01, (
                f"Vector {i} norm={n:.6f}, expected ~1.0"
            )


class TestSemanticSimilarity:
    """Verify similar code snippets cluster closer than unrelated ones."""

    def test_similar_ue_snippets_closer_than_dissimilar(self, embedder):
        # UE animation-related snippets
        animation_texts = [
            "void UAnimationAsset::TickAssetPlayer(FAnimTickRecord& Instance, FAnimNotifyQueue& NotifyQueue, FAnimAssetTickContext& Context) const;",
            "virtual float GetPlayLength() const { return 0.f; };",
            "FMarkerTickRecord MarkerTickRecord;",
        ]
        # Unrelated infrastructure code
        infra_texts = [
            "int32 GetNumberOfThreadTasks(int32 Num, int32 MinBatchSize, EParallelForFlags Flags);",
            "void ParallelFor(int32 Num, TFunctionRef<void(int32)> Body, bool bForceSingleThread);",
            "std::atomic_int IncompleteBatches { 0 };",
        ]
        all_texts = animation_texts + infra_texts
        vecs = embedder.embed_passages(all_texts)

        from numpy.linalg import norm
        def cos(v1, v2):
            return float(np.dot(v1, v2) / (norm(v1) * norm(v2)))

        # Intra-group similarity
        intra_anim = [cos(vecs[i], vecs[j]) for i in range(3) for j in range(i + 1, 3)]
        intra_infra = [cos(vecs[3 + i], vecs[3 + j]) for i in range(3) for j in range(i + 1, 3)]

        # Cross-group similarity
        cross = [cos(vecs[i], vecs[3 + j]) for i in range(3) for j in range(3)]

        avg_intra = np.mean(intra_anim + intra_infra)
        avg_cross = np.mean(cross)

        assert avg_intra > avg_cross, (
            f"Intra-group similarity ({avg_intra:.4f}) should exceed "
            f"cross-group ({avg_cross:.4f})"
        )

    def test_exact_same_text_max_similarity(self, embedder):
        text = "void UBTCompositeNode::OnChildActivation(FBehaviorTreeSearchData& SearchData, int32 ChildIndex) const;"
        vecs = embedder.embed_passages([text, text])
        from numpy.linalg import norm
        sim = float(np.dot(vecs[0], vecs[1]) / (norm(vecs[0]) * norm(vecs[1])))
        assert sim > 0.999, f"Same text similarity={sim:.6f}, expected > 0.999"


class TestQueryEmbedding:
    """BGE models require a query instruction prefix for asymmetric search."""

    def test_query_uses_bge_instruction(self, embedder, monkeypatch):
        """Verify embed_query sends a different prompt than embed_passages."""
        query = "animation skeleton blending"
        # embed_query should add the BGE instruction; embed_passages should not
        q_vec = embedder.embed_query(query)
        p_vecs = embedder.embed_passages([query])
        # They should be different because query has the instruction prefix
        from numpy.linalg import norm
        sim = float(np.dot(q_vec, p_vecs[0]) / (norm(q_vec) * norm(p_vecs[0])))
        assert sim < 0.99, (
            f"Query and passage embeddings too similar ({sim:.4f}) — "
            f"query instruction may not be applied"
        )


class TestBatchProcessing:
    """Test that the embedder handles batches of varying sizes correctly."""

    def test_large_batch(self, embedder):
        """A batch of 100 texts should process without error."""
        texts = [
            f"void Function_{i}(int32 Arg) {{ return Arg + {i}; }}"
            for i in range(100)
        ]
        vecs = embedder.embed_passages(texts)
        assert vecs.shape == (100, EXPECTED_DIM)

    def test_mixed_length_texts(self, embedder):
        """Short and long texts mixed together should work."""
        texts = [
            "int x;",
            "void UAnimationAsset::TickAssetPlayer(FAnimTickRecord& Instance, FAnimNotifyQueue& NotifyQueue, FAnimAssetTickContext& Context) const { /* 100+ chars of implementation */ }",
            "float GetWeight() const;",
        ]
        vecs = embedder.embed_passages(texts)
        assert vecs.shape == (3, EXPECTED_DIM)
