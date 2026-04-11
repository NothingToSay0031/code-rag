from __future__ import annotations

from code_rag.retriever.semantic import SemanticRetriever
from code_rag.retriever.keyword import KeywordRetriever
from code_rag.models import SearchResult


# Reciprocal Rank Fusion smoothing constant (Cormack et al., SIGIR 2009).
# k=60 is the original paper's recommendation; not critical to tune.
_RRF_K = 60

# Default path substrings to exclude from search results (case-insensitive).
# Prevents test files from dominating results in repos with large test suites.
DEFAULT_EXCLUDE_PATHS: list[str] = ["Test"]

# When path-filtering is active we need to over-fetch so that after
# discarding excluded paths we still return the requested top_k results.
_FILTER_OVERFETCH = 10


def _resolve_exclude(exclude_paths: list[str] | None) -> list[str]:
    """Return the effective exclusion list.

    - ``None``  → use :data:`DEFAULT_EXCLUDE_PATHS` (filters test files by default)
    - ``[]``    → no path filtering
    - ``[...]`` → caller-supplied list
    """
    return DEFAULT_EXCLUDE_PATHS if exclude_paths is None else exclude_paths


def _path_matches_any(file_path: str, patterns: list[str]) -> bool:
    """Return True if *file_path* contains any pattern (case-insensitive)."""
    fp = file_path.replace("\\", "/").lower()
    return any(p.lower() in fp for p in patterns)


class HybridRetriever:
    def __init__(self, semantic: SemanticRetriever, keyword: KeywordRetriever):
        self._semantic = semantic
        self._keyword = keyword

    def search(
        self,
        query: str,
        top_k: int = 10,
        language: str | None = None,
        exclude_paths: list[str] | None = None,
    ) -> list[SearchResult]:
        """Hybrid (semantic + BM25) search with optional path exclusion.

        Args:
            query: Search query string.
            top_k: Maximum results to return.
            language: Optional language filter applied to both retrievers.
            exclude_paths: Path substrings to suppress (case-insensitive).
                ``None`` → use :data:`DEFAULT_EXCLUDE_PATHS` (excludes "Test").
                ``[]``   → no path filtering.
        """
        excl = _resolve_exclude(exclude_paths)
        # When filtering, over-fetch so we still return top_k after exclusion
        fetch_k = top_k * _FILTER_OVERFETCH if excl else top_k

        semantic_results = self._semantic.search(
            query, top_k=fetch_k * 2, language=language
        )
        keyword_results = self._keyword.search(query, top_k=fetch_k * 2)

        # If language filter specified, also filter keyword results
        if language:
            keyword_results = [
                r for r in keyword_results if r.chunk.language == language
            ]

        # RRF: score(d) = Σ 1/(k + rank_i)
        # Dedup key: (file_path, start_line, end_line)
        rrf_scores: dict[tuple, float] = {}
        result_map: dict[tuple, SearchResult] = {}

        # Semantic ranking contribution (1-based rank)
        for rank, r in enumerate(semantic_results, start=1):
            key = (r.chunk.file_path, r.chunk.start_line, r.chunk.end_line)
            rrf_scores[key] = 1.0 / (_RRF_K + rank)
            result_map[key] = r

        # BM25 ranking contribution (1-based rank, additive)
        for rank, r in enumerate(keyword_results, start=1):
            key = (r.chunk.file_path, r.chunk.start_line, r.chunk.end_line)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
            if key not in result_map:
                result_map[key] = r

        # Sort by RRF score descending, apply path filter, slice to top_k
        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

        output: list[SearchResult] = []
        for k in sorted_keys:
            if excl and _path_matches_any(result_map[k].chunk.file_path, excl):
                continue
            output.append(
                SearchResult(
                    chunk=result_map[k].chunk,
                    score=rrf_scores[k],
                    source="hybrid",
                )
            )
            if len(output) >= top_k:
                break
        return output

    def search_code(
        self,
        query: str,
        top_k: int = 10,
        language: str | None = None,
        exclude_paths: list[str] | None = None,
    ) -> list[SearchResult]:
        """Search for code chunks with optional path exclusion.

        Calls :meth:`search` with an amplified ``top_k`` so that after
        discarding excluded paths exactly ``top_k`` code chunks are returned.
        """
        excl = _resolve_exclude(exclude_paths)
        fetch_k = top_k * (_FILTER_OVERFETCH if excl else 2)
        # Disable filtering inside search(); we apply it here to avoid double-pass
        results = self.search(query, top_k=fetch_k, language=language, exclude_paths=[])
        code_results = [
            r
            for r in results
            if r.chunk.chunk_type == "code"
            and (language is None or r.chunk.language == language)
            and not (excl and _path_matches_any(r.chunk.file_path, excl))
        ]
        return code_results[:top_k]

    def search_docs(
        self,
        query: str,
        top_k: int = 10,
        exclude_paths: list[str] | None = None,
    ) -> list[SearchResult]:
        """Search for documentation chunks with optional path exclusion."""
        excl = _resolve_exclude(exclude_paths)
        fetch_k = top_k * (_FILTER_OVERFETCH if excl else 2)
        results = self.search(query, top_k=fetch_k, exclude_paths=[])
        doc_results = [
            r
            for r in results
            if r.chunk.chunk_type == "doc"
            and not (excl and _path_matches_any(r.chunk.file_path, excl))
        ]
        return doc_results[:top_k]
