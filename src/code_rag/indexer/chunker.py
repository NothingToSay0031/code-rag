from __future__ import annotations

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
    """
    global _tokenizer
    _tokenizer = tokenizer
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
    return len(text.split())


def sliding_window_split(
    text: str, max_tokens: int, overlap_fraction: float = 0.1
) -> list[str]:
    if _tokenizer is not None:
        return _sliding_window_tokenizer(text, max_tokens, overlap_fraction)
    return _sliding_window_words(text, max_tokens, overlap_fraction)


def _sliding_window_words(
    text: str, max_tokens: int, overlap_fraction: float
) -> list[str]:
    """Word-based sliding window (fallback when no tokenizer is set)."""
    words = text.split()
    window_size = max_tokens
    step = max(1, window_size - int(window_size * overlap_fraction))
    windows = []
    for i in range(0, len(words), step):
        window = " ".join(words[i : i + window_size])
        if window:
            windows.append(window)
        if i + window_size >= len(words):
            break
    return windows


def _sliding_window_tokenizer(
    text: str, max_tokens: int, overlap_fraction: float
) -> list[str]:
    """Token-based sliding window using the real BPE/WordPiece tokenizer.

    Uses character offset mapping to extract original text spans rather than
    re-decoding token IDs.  Decoding re-tokenised IDs inserts spaces between
    subword pieces and corrupts non-ASCII characters (e.g. Chinese comments),
    so the stored ``Chunk.text`` would differ from the real source code.

    Falls back to ``tokenizer.decode()`` when offset mapping is unavailable
    (slow / non-fast tokenizers).
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
        # Slow tokenizer or model that doesn't support offset_mapping.
        token_ids = _tokenizer.encode(text, add_special_tokens=False)

    if not token_ids:
        return [text] if text.strip() else []

    window_size = max_tokens
    step = max(1, window_size - int(window_size * overlap_fraction))
    windows: list[str] = []
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
        if window_text.strip():
            windows.append(window_text)
        if i + window_size >= len(token_ids):
            break
    return windows if windows else [text]


def chunk_file(
    source: str,
    file_path: str,
    language: str | None,
    chunk_type: str,
    max_tokens: int = 512,
) -> list[Chunk]:
    if not source.strip():
        return []
    if chunk_type == "code" and language:
        return chunk_code(source, file_path, language, max_tokens)
    elif chunk_type == "doc":
        return chunk_docs(source, file_path, max_tokens)
    elif chunk_type == "config":
        return chunk_config(source, file_path, max_tokens)
    return []


def chunk_code(
    source: str, file_path: str, language: str, max_tokens: int
) -> list[Chunk]:
    source_bytes = source.encode("utf-8")
    try:
        tree = parse_file(source_bytes, language)
        nodes = get_ast_children(tree, source_bytes, language)
    except Exception:
        # Parser failed — fall back to sliding window so we respect max_tokens
        # rather than dumping the entire (potentially huge) file as one chunk.
        windows = sliding_window_split(source, max_tokens)
        return [
            Chunk(
                text=w,
                file_path=file_path,
                start_line=1,
                end_line=source.count("\n") + 1,
                chunk_type="code",
                language=language,
                symbol_name=None,
                symbol_kind=None,
                metadata={"window": i} if len(windows) > 1 else {},
            )
            for i, w in enumerate(windows)
        ]

    if not nodes:
        # No top-level AST nodes — apply sliding window instead of returning
        # the whole file as a single unbounded chunk.
        windows = sliding_window_split(source, max_tokens)
        return [
            Chunk(
                text=w,
                file_path=file_path,
                start_line=1,
                end_line=source.count("\n") + 1,
                chunk_type="code",
                language=language,
                symbol_name=None,
                symbol_kind=None,
                metadata={"window": i} if len(windows) > 1 else {},
            )
            for i, w in enumerate(windows)
        ]

    chunks = []
    for node in nodes:
        chunks.extend(
            _chunk_ast_node(
                node, source_bytes, file_path, language, max_tokens, context_prefix=""
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
    full_text = context_prefix + node_text if context_prefix else node_text

    symbol_name = node.symbol.name if node.symbol else None
    symbol_kind = node.symbol.kind if node.symbol else None
    start_line = node.start_line  # already 1-based from parser
    end_line = node.end_line

    # If it fits, return as single chunk
    doc = f"# {node.doc_comment}\n" if node.doc_comment else ""
    if count_tokens(doc + full_text) <= max_tokens:
        # Count extra lines prepended before the actual file content so that
        # _build_snippet can map code_lines[i] → file line (start_line + i)
        # correctly.  chunk.text = doc + context_prefix + node_text, so the
        # file content starts at line index (doc + context_prefix).count("\n").
        header_lines = (doc + context_prefix).count("\n")
        metadata: dict = {"header_lines": header_lines} if header_lines else {}
        return [
            Chunk(
                text=doc + full_text,
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

    # If node has children, recurse with context prefix
    if node.children:
        # Build context: first line of parent
        first_line = node_text.split("\n")[0]
        child_prefix = context_prefix + first_line + "\n"
        child_chunks = []
        for child in node.children:
            child_chunks.extend(
                _chunk_ast_node(
                    child, source_bytes, file_path, language, max_tokens, child_prefix
                )
            )
        if child_chunks:
            return child_chunks

    # Leaf node too large: sliding window
    windows = sliding_window_split(full_text, max_tokens)
    return [
        Chunk(
            text=w,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="code",
            language=language,
            symbol_name=symbol_name,
            symbol_kind=symbol_kind,
            metadata={"window": i},
        )
        for i, w in enumerate(windows)
    ]


def chunk_docs(source: str, file_path: str, max_tokens: int) -> list[Chunk]:
    sections = []
    current_lines: list[str] = []
    current_start = 1
    current_heading_level = 0

    for i, line in enumerate(source.split("\n"), 1):
        if line.startswith("#") and current_lines:
            sections.append(
                (current_start, i - 1, "\n".join(current_lines), current_heading_level)
            )
            current_lines = [line]
            current_start = i
            # Count heading level (number of leading #)
            current_heading_level = len(line) - len(line.lstrip("#"))
        else:
            if line.startswith("#") and not current_lines:
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
        meta = {"heading_level": heading_level} if heading_level else {}
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
            for j, w in enumerate(sliding_window_split(text, max_tokens)):
                chunks.append(
                    Chunk(
                        text=w,
                        file_path=file_path,
                        start_line=start,
                        end_line=end,
                        chunk_type="doc",
                        language=None,
                        symbol_name=None,
                        symbol_kind=None,
                        metadata={**meta, "window": j},
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
    meta_base = {"file_type": file_type}
    total_lines = source.count("\n") + 1

    # Large config files (e.g. package-lock.json) can exceed the model's
    # context window.  Apply the same sliding-window split used elsewhere.
    if count_tokens(source) > max_tokens:
        windows = sliding_window_split(source, max_tokens)
    else:
        windows = [source]

    return [
        Chunk(
            text=w,
            file_path=file_path,
            start_line=1,
            end_line=total_lines,
            chunk_type="config",
            language=None,
            symbol_name=None,
            symbol_kind=None,
            metadata={**meta_base, "window": i} if len(windows) > 1 else meta_base,
        )
        for i, w in enumerate(windows)
    ]
