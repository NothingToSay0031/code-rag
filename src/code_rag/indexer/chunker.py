from __future__ import annotations

import re
from typing import Any

from code_rag.models import Chunk
from code_rag.indexer.parser import parse_file, get_ast_children, ASTNode

# ---------------------------------------------------------------------------
# Tokenizer-aware token counting
# ---------------------------------------------------------------------------

_tokenizer: Any = None


def set_tokenizer(tokenizer) -> None:
    """Set the BPE/WordPiece tokenizer for accurate token counting.

    When set, :func:`count_tokens` and :func:`sliding_window_split` use the
    real tokenizer instead of word-based approximation.  Call this once at
    pipeline startup with the embedding model's tokenizer.

    Pass ``None`` to clear the tokenizer (e.g. for static models that have
    no HuggingFace tokenizer).
    """
    global _tokenizer
    _tokenizer = tokenizer
    if tokenizer is None:
        return
    # Remove the model_max_length constraint so encode() doesn't emit
    # "Token indices sequence length is longer than the specified maximum
    # sequence length" warnings when we're only counting tokens.
    if hasattr(tokenizer, "model_max_length"):
        tokenizer.model_max_length = int(1e9)


def count_tokens(text: str) -> int:
    """Count tokens in *text*.

    Uses the embedding model's tokenizer when available (set via
    :func:`set_tokenizer`), otherwise falls back to ``len(text.split())``.
    """
    if _tokenizer is not None:
        return len(_tokenizer.encode(text, add_special_tokens=False))
    return max(1, len(text.split())) if text else 0


def sliding_window_split(
    text: str, max_tokens: int, overlap_fraction: float = 0.1
) -> list[tuple[str, int]]:
    if max_tokens <= 0:
        return [(text, 0)] if text.strip() else []
    if _tokenizer is not None:
        return _sliding_window_tokenizer(text, max_tokens, overlap_fraction)
    return _sliding_window_words(text, max_tokens, overlap_fraction)


def _sliding_window_words(
    text: str, max_tokens: int, overlap_fraction: float
) -> list[tuple[str, int]]:
    """Word-based sliding window (fallback when no tokenizer is set)."""
    words = text.split()
    if not words:
        return [(text, 0)] if text.strip() else []

    # Find each word's character offset in the original text so we can
    # compute accurate per-window line numbers.
    word_positions: list[int] = []
    search_pos = 0
    for w in words:
        idx = text.find(w, search_pos)
        word_positions.append(idx if idx >= 0 else search_pos)
        if idx >= 0:
            search_pos = idx + len(w)

    window_size = max(1, max_tokens)
    step = max(1, window_size - int(window_size * overlap_fraction))
    windows: list[tuple[str, int]] = []
    for i in range(0, len(words), step):
        start_char = word_positions[i]
        end_idx = min(i + window_size, len(words))
        # Slice the original text to preserve all whitespace and line breaks.
        if end_idx < len(words):
            end_char = word_positions[end_idx]
        else:
            end_char = len(text)
        window = text[start_char:end_char].strip("\n")
        if window:
            windows.append((window, start_char))
        if i + window_size >= len(words):
            break
    return windows if windows else [(text, 0)]


def _sliding_window_tokenizer(
    text: str, max_tokens: int, overlap_fraction: float
) -> list[tuple[str, int]]:
    """Token-based sliding window using the real BPE/WordPiece tokenizer.

    Uses character offset mapping to extract original text spans rather than
    re-decoding token IDs.  Decoding re-tokenised IDs inserts spaces between
    subword pieces and corrupts non-ASCII characters (e.g. Chinese comments),
    so the stored ``Chunk.text`` would differ from the real source code.

    Falls back to ``tokenizer.decode()`` when offset mapping is unavailable
    (slow / non-fast tokenizers).

    Returns ``(window_text, char_offset)`` pairs so callers can compute
    per-window line numbers without a second find pass.
    """
    # Prefer offset-mapping path: slice original text instead of decoding.
    offsets: list[tuple[int, int]] | None = None
    try:
        encoding = _tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        token_ids: list[int] = encoding["input_ids"]
        offsets = encoding["offset_mapping"]
    except Exception:
        try:
            token_ids = _tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            return _sliding_window_words(text, max_tokens, overlap_fraction)

    if not token_ids:
        return [(text, 0)] if text.strip() else []

    window_size = max_tokens
    step = max(1, window_size - int(window_size * overlap_fraction))
    windows: list[tuple[str, int]] = []
    for i in range(0, len(token_ids), step):
        end_idx = min(i + window_size, len(token_ids))
        if offsets is not None:
            # Slice original text using character-level token boundaries.
            start_char = offsets[i][0]
            end_char = offsets[end_idx - 1][1]
            window_text = text[start_char:end_char]
        else:
            # Fallback: decode (may alter formatting/encoding for non-ASCII).
            window_text = _tokenizer.decode(
                token_ids[i:end_idx], skip_special_tokens=True
            )
            start_char = 0
        if window_text.strip():
            windows.append((window_text, start_char))
        if i + window_size >= len(token_ids):
            break
    return windows if windows else [(text, 0)]


# ---------------------------------------------------------------------------
# Helpers for computing per-window line offsets in sliding-window chunks
# ---------------------------------------------------------------------------

def _sliding_chunks(
    text: str,
    max_tokens: int,
    file_path: str,
    chunk_type: str,
    language: str | None,
    base_start_line: int,
    symbol_name: str | None = None,
    symbol_kind: str | None = None,
    extra_meta: dict | None = None,
    header_lines: int = 0,
) -> list[Chunk]:
    """Split *text* into sliding-window chunks with correct per-window line numbers."""
    windows = sliding_window_split(text, max_tokens)
    chunks = []
    for i, (w, char_offset) in enumerate(windows):
        meta: dict = {
            **(extra_meta or {}),
            "token_count": count_tokens(w),
        }
        if header_lines:
            meta["header_lines"] = header_lines
        if len(windows) > 1:
            meta["window"] = i
        window_start = base_start_line + text[:char_offset].count("\n")
        window_end = window_start + w.count("\n")
        chunks.append(
            Chunk(
                text=w,
                file_path=file_path,
                start_line=window_start,
                end_line=window_end,
                chunk_type=chunk_type,
                language=language,
                symbol_name=symbol_name,
                symbol_kind=symbol_kind,
                metadata=meta,
            )
        )
    return chunks


_C_COMMENT_RE = re.compile(r"//.*?$|/\*.*?\*/", re.DOTALL | re.MULTILINE)


def _meaningful_content(text: str, min_chars: int = 15) -> bool:
    """Return True if *text* has enough non-comment content to be worth indexing."""
    stripped = _C_COMMENT_RE.sub("", text)
    return len(stripped.strip()) >= min_chars


def _extract_declaration_line(node_text: str) -> str:
    """Strips leading blank lines to find the first content line of a node.

    Returns the first non-blank line of *node_text*, which is typically the
    declaration line (e.g. ``class Foo``, ``void bar()``).
    """
    stripped = node_text.lstrip("\n")
    if not stripped:
        return node_text[: node_text.find("\n")] if "\n" in node_text else node_text
    first_line = stripped[: stripped.find("\n")] if "\n" in stripped else stripped
    return first_line


# ---------------------------------------------------------------------------
# Public chunking API
# ---------------------------------------------------------------------------

def chunk_file(
    source: str,
    file_path: str,
    language: str | None,
    chunk_type: str,
    max_tokens: int = 512,
    source_bytes: bytes | None = None,
    ast_nodes: list | None = None,
) -> list[Chunk]:
    if not source.strip():
        return []
    if chunk_type == "code" and language:
        return chunk_code(source, file_path, language, max_tokens, source_bytes, ast_nodes)
    elif chunk_type == "doc":
        return chunk_docs(source, file_path, max_tokens)
    elif chunk_type == "config":
        return chunk_config(source, file_path, max_tokens)
    return []


def chunk_code(
    source: str,
    file_path: str,
    language: str,
    max_tokens: int,
    source_bytes: bytes | None = None,
    ast_nodes: list | None = None,
) -> list[Chunk]:
    if source_bytes is not None and ast_nodes is not None:
        nodes = ast_nodes
    else:
        source_bytes = source.encode("utf-8")  # type: ignore[assignment]
        try:
            tree = parse_file(source_bytes, language)
            nodes = get_ast_children(tree, source_bytes, language)
        except Exception:
            # Parser failed — fall back to sliding window so we respect max_tokens
            # rather than dumping the entire (potentially huge) file as one chunk.
            return _sliding_chunks(
                source, max_tokens, file_path, "code", language,
                base_start_line=1,
            )

    if not nodes:
        # No top-level AST nodes — apply sliding window instead of returning
        # the whole file as a single unbounded chunk.
        return _sliding_chunks(
            source, max_tokens, file_path, "code", language,
            base_start_line=1,
        )

    chunks = []
    prev_end = 0  # byte position of file start

    for node in nodes:
        # Gap chunk for content between previous node (or file start) and this node
        if node.start_byte > prev_end:
            gap_bytes = source_bytes[prev_end : node.start_byte]
            gap_text = gap_bytes.decode("utf-8", errors="replace")
            if _meaningful_content(gap_text):
                gap_start = 1 + source_bytes[:prev_end].count(b"\n")
                if count_tokens(gap_text) <= max_tokens:
                    chunks.append(
                        Chunk(
                            text=gap_text,
                            file_path=file_path,
                            start_line=gap_start,
                            end_line=gap_start + gap_text.count("\n"),
                            chunk_type="code",
                            language=language,
                            symbol_name=None,
                            symbol_kind=None,
                            metadata={"token_count": count_tokens(gap_text)},
                        )
                    )
                else:
                    chunks.extend(
                        _sliding_chunks(
                            gap_text, max_tokens, file_path, "code", language,
                            base_start_line=gap_start,
                        )
                    )

        chunks.extend(
            _chunk_ast_node(
                node, source_bytes, file_path, language, max_tokens, context_prefix=""
            )
        )
        prev_end = node.end_byte

    # Gap chunk for content after last node
    total_bytes = len(source_bytes)
    if total_bytes > prev_end:
        tail_bytes = source_bytes[prev_end:total_bytes]
        tail_text = tail_bytes.decode("utf-8", errors="replace")
        if _meaningful_content(tail_text):
            tail_start = 1 + source_bytes[:prev_end].count(b"\n")
            if count_tokens(tail_text) <= max_tokens:
                chunks.append(
                    Chunk(
                        text=tail_text,
                        file_path=file_path,
                        start_line=tail_start,
                        end_line=tail_start + tail_text.count("\n"),
                        chunk_type="code",
                        language=language,
                        symbol_name=None,
                        symbol_kind=None,
                        metadata={"token_count": count_tokens(tail_text)},
                    )
                )
            else:
                chunks.extend(
                    _sliding_chunks(
                        tail_text, max_tokens, file_path, "code", language,
                        base_start_line=tail_start,
                    )
                )

    return chunks


def _chunk_ast_node(
    node: ASTNode,
    source_bytes: bytes,
    file_path: str,
    language: str,
    max_tokens: int,
    context_prefix: str,
) -> list[Chunk]:
    node_text = source_bytes[node.start_byte : node.end_byte].decode(
        "utf-8", errors="replace"
    )
    # Warn if decode produced replacement characters (non-UTF-8 source).
    if "�" in node_text:
        pass  # already handled by errors="replace"; chunk text is best-effort

    full_text = context_prefix + node_text if context_prefix else node_text

    symbol_name = node.symbol.name if node.symbol else None
    symbol_kind = node.symbol.kind if node.symbol else None
    start_line = node.start_line  # already 1-based from parser
    end_line = node.end_line

    # Doc comment for this node
    doc = f"# {node.doc_comment}\n" if node.doc_comment else ""
    combined = doc + full_text

    # If it fits, return as single chunk
    if count_tokens(combined) <= max_tokens:
        header_lines = (doc + context_prefix).count("\n")
        metadata: dict = {"token_count": count_tokens(combined)}
        if header_lines:
            metadata["header_lines"] = header_lines
        return [
            Chunk(
                text=combined,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                chunk_type="code",
                language=language,
                symbol_name=symbol_name,
                symbol_kind=symbol_kind,
                metadata=metadata,
            )
        ]

    # If node has children, recurse with context prefix, filling gaps between
    # children to preserve non-symbol content (member variables, using decls,
    # friend decls, static_assert, etc.).
    if node.children:
        first_line = _extract_declaration_line(node_text)
        child_prefix = context_prefix + first_line + "\n"
        child_chunks = []
        prev_end = node.start_byte

        for child in node.children:
            # Gap chunk for non-symbol content between previous child and this one
            if child.start_byte > prev_end:
                gap_bytes = source_bytes[prev_end : child.start_byte]
                gap_text = gap_bytes.decode("utf-8", errors="replace")
                if _meaningful_content(gap_text):
                    gap_full = context_prefix + gap_text if context_prefix else gap_text
                    gap_start = start_line + source_bytes[
                        node.start_byte:prev_end
                    ].decode("utf-8", errors="replace").count("\n")
                    gap_header_lines = context_prefix.count("\n")
                    if count_tokens(gap_full) <= max_tokens:
                        gap_meta: dict = {"token_count": count_tokens(gap_full)}
                        if gap_header_lines:
                            gap_meta["header_lines"] = gap_header_lines
                        child_chunks.append(
                            Chunk(
                                text=gap_full,
                                file_path=file_path,
                                start_line=gap_start,
                                end_line=gap_start + gap_text.count("\n"),
                                chunk_type="code",
                                language=language,
                                symbol_name=symbol_name,
                                symbol_kind=symbol_kind,
                                metadata=gap_meta,
                            )
                        )
                    else:
                        child_chunks.extend(
                            _sliding_chunks(
                                gap_full, max_tokens, file_path, "code", language,
                                base_start_line=gap_start - gap_header_lines,
                                symbol_name=symbol_name,
                                symbol_kind=symbol_kind,
                                header_lines=gap_header_lines,
                            )
                        )

            child_chunks.extend(
                _chunk_ast_node(
                    child, source_bytes, file_path, language, max_tokens, child_prefix
                )
            )
            prev_end = child.end_byte

        # Gap chunk for content after the last child
        if node.end_byte > prev_end:
            tail_bytes = source_bytes[prev_end : node.end_byte]
            tail_text = tail_bytes.decode("utf-8", errors="replace")
            if _meaningful_content(tail_text):
                tail_full = context_prefix + tail_text if context_prefix else tail_text
                tail_start = start_line + source_bytes[
                    node.start_byte:prev_end
                ].decode("utf-8", errors="replace").count("\n")
                tail_header_lines = context_prefix.count("\n")
                if count_tokens(tail_full) <= max_tokens:
                    tail_meta: dict = {"token_count": count_tokens(tail_full)}
                    if tail_header_lines:
                        tail_meta["header_lines"] = tail_header_lines
                    child_chunks.append(
                        Chunk(
                            text=tail_full,
                            file_path=file_path,
                            start_line=tail_start,
                            end_line=tail_start + tail_text.count("\n"),
                            chunk_type="code",
                            language=language,
                            symbol_name=symbol_name,
                            symbol_kind=symbol_kind,
                            metadata=tail_meta,
                        )
                    )
                else:
                    child_chunks.extend(
                        _sliding_chunks(
                            tail_full, max_tokens, file_path, "code", language,
                            base_start_line=tail_start - tail_header_lines,
                            symbol_name=symbol_name,
                            symbol_kind=symbol_kind,
                            header_lines=tail_header_lines,
                        )
                    )

        if child_chunks:
            return child_chunks

    # Leaf node too large: sliding window.
    # Include doc comment in the text being split so it is not lost.
    split_text = doc + full_text
    header_lines = (doc + context_prefix).count("\n")
    return _sliding_chunks(
        split_text, max_tokens, file_path, "code", language,
        base_start_line=start_line,
        symbol_name=symbol_name,
        symbol_kind=symbol_kind,
        header_lines=header_lines,
    )


_HEADING_RE = re.compile(r"^#{1,6}\s")


def chunk_docs(source: str, file_path: str, max_tokens: int) -> list[Chunk]:
    sections = []
    current_lines: list[str] = []
    current_start = 1
    current_heading_level = 0

    for i, line in enumerate(source.split("\n"), 1):
        if _HEADING_RE.match(line) and current_lines:
            sections.append(
                (current_start, i - 1, "\n".join(current_lines), current_heading_level)
            )
            current_lines = [line]
            current_start = i
            # Count heading level (number of leading #)
            current_heading_level = len(line) - len(line.lstrip("#"))
        else:
            if _HEADING_RE.match(line) and not current_lines:
                current_heading_level = len(line) - len(line.lstrip("#"))
            current_lines.append(line)

    if current_lines:
        sections.append(
            (
                current_start,
                current_start + len(current_lines) - 1,
                "\n".join(current_lines),
                current_heading_level,
            )
        )

    chunks = []
    for start, end, text, heading_level in sections:
        if not text.strip():
            continue
        meta: dict = {}
        if heading_level:
            meta["heading_level"] = heading_level
        meta["token_count"] = count_tokens(text)
        if count_tokens(text) <= max_tokens:
            chunks.append(
                Chunk(
                    text=text,
                    file_path=file_path,
                    start_line=start,
                    end_line=end,
                    chunk_type="doc",
                    language=None,
                    symbol_name=None,
                    symbol_kind=None,
                    metadata=meta,
                )
            )
        else:
            windows = sliding_window_split(text, max_tokens)
            for j, (w, char_offset) in enumerate(windows):
                window_start = start + text[:char_offset].count("\n")
                chunks.append(
                    Chunk(
                        text=w,
                        file_path=file_path,
                        start_line=window_start,
                        end_line=window_start + w.count("\n"),
                        chunk_type="doc",
                        language=None,
                        symbol_name=None,
                        symbol_kind=None,
                        metadata={**meta, "window": j, "token_count": count_tokens(w)},
                    )
                )
    return chunks


def chunk_config(source: str, file_path: str, max_tokens: int = 512) -> list[Chunk]:
    import os

    ext = os.path.splitext(file_path)[1].lower()
    file_type_map = {
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
    }
    file_type = file_type_map.get(ext, os.path.basename(file_path).lower())
    meta_base: dict = {"file_type": file_type}
    total_lines = source.count("\n") + 1

    # Large config files (e.g. package-lock.json) can exceed the model's
    # context window.  Apply the same sliding-window split used elsewhere.
    if count_tokens(source) > max_tokens:
        return _sliding_chunks(
            source, max_tokens, file_path, "config", None,
            base_start_line=1,
            extra_meta=meta_base,
        )
    else:
        return [
            Chunk(
                text=source,
                file_path=file_path,
                start_line=1,
                end_line=total_lines,
                chunk_type="config",
                language=None,
                symbol_name=None,
                symbol_kind=None,
                metadata={**meta_base, "token_count": count_tokens(source)},
            )
        ]
