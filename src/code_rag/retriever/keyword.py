from code_rag.storage.bm25_store import BM25Store
from code_rag.models import Chunk, SearchResult


class KeywordRetriever:
    def __init__(self, bm25_store: BM25Store):
        self._bm25_store = bm25_store

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Keyword search using BM25."""
        if not query or not query.strip():
            return []

        results = self._bm25_store.search(query, top_k=top_k)

        search_results = []
        for chunk, score in results:
            search_results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    source="keyword",
                )
            )
        return search_results
