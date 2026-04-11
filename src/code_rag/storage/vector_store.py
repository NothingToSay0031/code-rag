from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from code_rag.models import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB vector store wrapper for code chunk embeddings.

    Uses ChromaDB PersistentClient for embedded, cross-platform vector storage.
    Data is persisted to a local directory automatically.

    If ChromaDB fails to initialise (e.g. database version mismatch) the store
    enters a *degraded* mode where all reads return empty results and writes
    are silently dropped.  Call :pyattr:`available` to check.
    """

    def __init__(self, db_path: Path, dimension: int):
        self._db_path = db_path
        self._dimension = dimension
        self._available = False
        self._client: Any = None
        self._collection: Any = None

        db_path.parent.mkdir(parents=True, exist_ok=True)
        chroma_dir = db_path.parent / "chroma_db"

        self._client, self._collection = self._open_chroma(chroma_dir)
        self._available = self._collection is not None

    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """``True`` when ChromaDB is operational; ``False`` in degraded mode."""
        return self._available

    # ------------------------------------------------------------------

    @staticmethod
    def _open_chroma(chroma_dir: Path) -> tuple[Any, Any]:
        """Try to open ChromaDB.  Returns ``(None, None)`` on failure."""
        try:
            client = chromadb.PersistentClient(path=str(chroma_dir))
            collection = client.get_or_create_collection(
                name="code_chunks",
                metadata={"hnsw:space": "cosine"},
            )
            return client, collection
        except Exception as exc:
            logger.error(
                "ChromaDB open failed (%s: %s). "
                "Vector search will be disabled for this index. "
                "Re-run 'code-rag init' to rebuild the index.",
                type(exc).__name__,
                exc,
            )
        return None, None

    def _next_base_id(self) -> int:
        """Derive the next safe integer ID from existing data."""
        if not self._available:
            return 0
        existing = self._collection.get(include=[])
        if not existing["ids"]:
            return 0
        return max(int(x) for x in existing["ids"]) + 1

    def insert(self, chunks: list[Chunk], embeddings: np.ndarray) -> list[int]:
        if not self._available:
            logger.warning(
                "VectorStore unavailable — skipping insert of %d chunks", len(chunks)
            )
            return []
        base_id = self._next_base_id()
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        embedding_list: list[list[float]] = []

        for i, chunk in enumerate(chunks):
            chunk_id = base_id + i
            ids.append(str(chunk_id))
            documents.append(chunk.text[:65535])
            metadatas.append(
                {
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "chunk_type": chunk.chunk_type,
                    "language": chunk.language or "",
                    "symbol_name": chunk.symbol_name or "",
                    "symbol_kind": chunk.symbol_kind or "",
                    "metadata_json": json.dumps(chunk.metadata),
                }
            )
            embedding_list.append(embeddings[i].tolist())

        # ChromaDB add() has a batch size limit; split if needed
        batch_size = 5000
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self._collection.add(
                ids=ids[start:end],
                embeddings=embedding_list[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        return list(range(base_id, base_id + len(chunks)))

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_expr: str | None = None,
    ) -> list[dict]:
        if not self._available:
            return []
        if self._collection.count() == 0:
            return []

        where = self._parse_filter(filter_expr) if filter_expr else None
        # Clamp top_k to collection size (ChromaDB requires n_results <= count)
        n = min(top_k, self._collection.count())

        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        output = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i] if results["distances"] else 0.0
            # cosine space: distance = 1 - cosine_similarity → score = 1 - distance
            score = 1.0 - distance
            entry = {
                "text": results["documents"][0][i],
                "file_path": meta.get("file_path", ""),
                "start_line": meta.get("start_line", 0),
                "end_line": meta.get("end_line", 0),
                "chunk_type": meta.get("chunk_type", "code"),
                "language": meta.get("language", ""),
                "symbol_name": meta.get("symbol_name", ""),
                "symbol_kind": meta.get("symbol_kind", ""),
                "metadata_json": meta.get("metadata_json", "{}"),
                "score": score,
                "id": int(doc_id),
            }
            output.append(entry)
        return output

    def delete_by_file(self, file_path: str):
        if not self._available:
            return
        # ChromaDB delete with where filter; no-op if nothing matches
        try:
            self._collection.delete(where={"file_path": file_path})
        except Exception:
            pass

    def get_by_file(self, file_path: str) -> list[dict]:
        if not self._available:
            return []
        try:
            results = self._collection.get(
                where={"file_path": file_path},
                include=["documents", "metadatas"],
            )
        except Exception:
            return []

        if not results["ids"]:
            return []

        output = []
        for i, doc_id in enumerate(results["ids"]):
            meta = results["metadatas"][i]
            output.append(
                {
                    "text": results["documents"][i],
                    "file_path": meta.get("file_path", ""),
                    "start_line": meta.get("start_line", 0),
                    "end_line": meta.get("end_line", 0),
                    "chunk_type": meta.get("chunk_type", "code"),
                    "language": meta.get("language", ""),
                    "symbol_name": meta.get("symbol_name", ""),
                    "symbol_kind": meta.get("symbol_kind", ""),
                }
            )
        return output

    def count(self) -> int:
        if not self._available:
            return 0
        return self._collection.count()

    def close(self):
        pass  # ChromaDB PersistentClient auto-persists; no explicit close needed

    @staticmethod
    def _parse_filter(filter_expr: str) -> dict | None:
        """Parse Milvus-style filter 'field == "value"' to ChromaDB where clause."""
        parts = [p.strip() for p in filter_expr.split(" and ")]
        conditions = []
        for part in parts:
            m = re.match(r'(\w+)\s*==\s*"([^"]*)"', part)
            if m:
                conditions.append({m.group(1): m.group(2)})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
