import os
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from code_rag.models import Chunk

# ---------------------------------------------------------------------------
# Code-aware tokenizer for BM25
# ---------------------------------------------------------------------------

# Regex patterns for splitting identifiers
_CAMEL_RE1 = re.compile(r"(.)([A-Z][a-z]+)")  # aB → a B
_CAMEL_RE2 = re.compile(r"([a-z0-9])([A-Z])")  # aB → a B (after RE1)
_CODE_DELIMITERS = re.compile(r"[^a-zA-Z0-9_]+")

# English stop words — NOT code keywords (BM25 IDF handles those naturally)
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
    }
)


def _split_identifier(token: str) -> list[str]:
    """Split a code identifier into sub-tokens.

    handleAuthCallback → [handleauthcallback, handle, auth, callback]
    get_file_symbols  → [get, file, symbols, get_file_symbols]
    HTTPResponse      → [httpresponse, http, response]
    """
    parts: list[str] = []
    lower = token.lower()

    # Split snake_case
    if "_" in token:
        for sub in token.split("_"):
            if sub:
                parts.append(sub.lower())

    # Split camelCase (two-pass: standard transitions then acronyms)
    camel = _CAMEL_RE1.sub(r"\1 \2", token)
    camel = _CAMEL_RE2.sub(r"\1 \2", camel)
    for sub in camel.split():
        sub_lower = sub.lower()
        if sub_lower and sub_lower not in parts:
            parts.append(sub_lower)

    # Always include the full original (lowered) for exact matching
    if lower not in parts:
        parts.insert(0, lower)

    return parts


def _tokenize_code(text: str) -> list[str]:
    """Code-aware tokenizer for BM25.

    - Splits on non-alphanumeric/underscore boundaries
    - Expands camelCase and snake_case into sub-tokens
    - Preserves originals for exact matching
    - Filters English stop words (NOT code keywords — IDF handles those)
    """
    if not text:
        return []

    raw_tokens = _CODE_DELIMITERS.split(text)
    tokens: list[str] = []
    for raw in raw_tokens:
        if not raw or len(raw) < 2:
            continue
        for sub in _split_identifier(raw):
            if len(sub) >= 2 and sub not in _STOP_WORDS:
                tokens.append(sub)
    return tokens


class BM25Store:
    def __init__(self, store_path: Path):
        self._store_path = store_path
        self._corpus: list[list[str]] = []  # tokenized docs
        self._doc_ids: list[int] = []  # chunk IDs
        self._chunks: list[Chunk] = []  # full chunk data for retrieval
        self._file_map: dict[str, list[int]] = {}  # file_path → corpus indices
        self._bm25: BM25Okapi | None = None
        self._dirty = True
        if store_path.exists():
            self._load()

    def _tokenize(self, text: str) -> list[str]:
        return _tokenize_code(text)

    def add(self, chunk_id: int, chunk: Chunk):
        tokens = self._tokenize(chunk.text)
        idx = len(self._corpus)
        self._corpus.append(tokens)
        self._doc_ids.append(chunk_id)
        self._chunks.append(chunk)
        self._file_map.setdefault(chunk.file_path, []).append(idx)
        self._dirty = True

    def add_batch(self, chunk_ids: list[int], chunks: list[Chunk]):
        for cid, chunk in zip(chunk_ids, chunks):
            tokens = self._tokenize(chunk.text)
            idx = len(self._corpus)
            self._corpus.append(tokens)
            self._doc_ids.append(cid)
            self._chunks.append(chunk)
            self._file_map.setdefault(chunk.file_path, []).append(idx)
        self._dirty = True

    def remove_by_file(self, file_path: str):
        indices = self._file_map.pop(file_path, [])
        if not indices:
            return
        remove_set = set(indices)
        new_corpus = []
        new_doc_ids = []
        new_chunks = []
        new_file_map: dict[str, list[int]] = {}
        for i, (tokens, doc_id, chunk) in enumerate(
            zip(self._corpus, self._doc_ids, self._chunks)
        ):
            if i in remove_set:
                continue
            new_idx = len(new_corpus)
            new_corpus.append(tokens)
            new_doc_ids.append(doc_id)
            new_chunks.append(chunk)
            new_file_map.setdefault(chunk.file_path, []).append(new_idx)
        self._corpus = new_corpus
        self._doc_ids = new_doc_ids
        self._chunks = new_chunks
        self._file_map = new_file_map
        self._dirty = True

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        if not self._corpus:
            return []
        if self._dirty or self._bm25 is None:
            self._bm25 = BM25Okapi(self._corpus)
            self._dirty = False
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        query_tokens = set(tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            # In very small corpora, rank_bm25 can assign zero IDF to matching
            # terms. Keep real lexical hits instead of dropping them as no-match.
            if scores[idx] > 0 or query_tokens.intersection(self._corpus[idx]):
                results.append((self._chunks[idx], float(scores[idx])))
        return results

    def save(self):
        """Persist BM25 data with atomic write (temp + rename)."""
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "corpus": self._corpus,
            "doc_ids": self._doc_ids,
            "chunks": self._chunks,
            "file_map": self._file_map,
        }
        tmp_path = self._store_path.with_suffix(".pkl.tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(data, f)
        # Atomic rename (overwrites existing on POSIX; Windows needs remove first)
        if os.name == "nt" and self._store_path.exists():
            self._store_path.unlink()
        tmp_path.rename(self._store_path)

    def _load(self):
        try:
            with open(self._store_path, "rb") as f:
                data = pickle.load(f)
            self._corpus = data["corpus"]
            self._doc_ids = data["doc_ids"]
            self._chunks = data["chunks"]
            self._file_map = data["file_map"]
            self._dirty = True
        except (EOFError, pickle.UnpicklingError, KeyError, Exception) as exc:
            print(
                f"WARNING: BM25 store corrupt ({type(exc).__name__}: {exc}). "
                f"Starting fresh — BM25 keyword search will be rebuilt."
            )
            self._corpus = []
            self._doc_ids = []
            self._chunks = []
            self._file_map = {}
            self._dirty = True

    def count(self) -> int:
        return len(self._corpus)
