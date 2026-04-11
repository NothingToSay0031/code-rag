from code_rag.indexer.embedder import Embedder
from code_rag.storage.vector_store import VectorStore
from code_rag.models import Chunk, SearchResult
import json


class SemanticRetriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self._vector_store = vector_store
        self._embedder = embedder

    def search(
        self,
        query: str,
        top_k: int = 10,
        language: str | None = None,
        chunk_type: str | None = None,
    ) -> list[SearchResult]:
        """Semantic search using dense vectors."""
        embedding = self._embedder.embed_query(query)

        # Build filter expression
        filters = []
        if language:
            filters.append(f'language == "{language}"')
        if chunk_type:
            filters.append(f'chunk_type == "{chunk_type}"')
        filter_expr = " and ".join(filters) if filters else None

        results = self._vector_store.search(
            embedding, top_k=top_k, filter_expr=filter_expr
        )

        search_results = []
        for hit in results:
            chunk = Chunk(
                text=hit.get("text", ""),
                file_path=hit.get("file_path", ""),
                start_line=hit.get("start_line", 0),
                end_line=hit.get("end_line", 0),
                chunk_type=hit.get("chunk_type", "code"),
                language=hit.get("language", None) or None,
                symbol_name=hit.get("symbol_name", None) or None,
                symbol_kind=hit.get("symbol_kind", None) or None,
                metadata=json.loads(hit.get("metadata_json", "{}"))
                if "metadata_json" in hit
                else {},
            )
            search_results.append(
                SearchResult(
                    chunk=chunk,
                    score=hit.get("score", 0.0),
                    source="semantic",
                )
            )
        return search_results

    def search_docs(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Convenience: search only doc-type chunks."""
        return self.search(query, top_k=top_k, chunk_type="doc")
