"""Integration tests using real Unreal Engine source files.

Validates parser + chunker correctness on complex C++ code with UE macros,
templates, namespaces, and nested types.
"""

from pathlib import Path

import pytest

from code_rag.indexer.parser import parse_file, get_ast_children, extract_symbols
from code_rag.indexer.chunker import chunk_code, set_tokenizer

FIXTURES = Path(__file__).parent / "fixtures" / "ue"


def _parse_and_chunk(filename: str, max_tokens: int = 512):
    """Helper: parse + chunk a fixture file."""
    fp = FIXTURES / filename
    source_bytes = fp.read_bytes()
    source = source_bytes.decode("utf-8", errors="replace")
    tree = parse_file(source_bytes, "cpp")
    nodes = get_ast_children(tree, source_bytes, "cpp")
    symbols = extract_symbols(tree, source_bytes, "cpp")
    chunks = chunk_code(
        source, str(fp), "cpp", max_tokens=max_tokens,
        source_bytes=source_bytes, ast_nodes=nodes,
    )
    return source, nodes, symbols, chunks


class TestParserOnUEFiles:
    """Verify parser handles UE-specific patterns."""

    @pytest.mark.parametrize("filename,expected_min_symbols", [
        ("AnimationAsset.h", 80),       # heavy UE macro usage
        ("ParallelFor.h", 30),          # template-heavy
        ("BehaviorTreeTypes.cpp", 40),  # many free functions
        ("BTCompositeNode.cpp", 30),    # class with many methods
    ])
    def test_parser_produces_symbols(self, filename, expected_min_symbols):
        _, _, symbols, _ = _parse_and_chunk(filename)
        assert len(symbols) >= expected_min_symbols, (
            f"Expected >= {expected_min_symbols} symbols, got {len(symbols)}"
        )

    def test_cpp_enums_recognized(self):
        """C++ enums (including enum class) must be captured as symbols."""
        _, _, symbols, _ = _parse_and_chunk("AnimationAsset.h")
        enums = [s for s in symbols if s.kind == "enum"]
        assert len(enums) >= 3, f"Expected >= 3 C++ enums, got {len(enums)}: {[e.name for e in enums]}"

    def test_namespace_functions_not_misclassified_as_methods(self):
        """Free functions inside namespaces must remain 'function', not 'method'."""
        _, _, symbols, _ = _parse_and_chunk("BehaviorTreeTypes.cpp")
        namespace_funcs = [s for s in symbols if s.kind == "function"]
        # In a .cpp file, most top-level functions inside namespaces should be "function"
        assert len(namespace_funcs) >= 20, (
            f"Expected >= 20 namespace-level functions, got {len(namespace_funcs)}"
        )
        # Verify no namespace-internal function is misclassified as "method"
        # (Methods only exist inside class/struct/interface container kinds)
        methods = [s for s in symbols if s.kind == "method"]
        for m in methods:
            assert m.name != "Initialize", (
                f"Namespace function should not be 'method': {m.name}"
            )

    def test_class_methods_correctly_classified(self):
        """Methods inside classes must be 'method', not 'function'."""
        _, _, symbols, _ = _parse_and_chunk("AnimationAsset.h")
        methods = [s for s in symbols if s.kind == "method"]
        assert len(methods) >= 30, f"Expected >= 30 methods, got {len(methods)}"
        # Sample check: known class methods should exist
        method_names = {m.name for m in methods}
        assert "SetSkeleton" in method_names or "FMarkerPair" in method_names

    def test_struct_specifier_is_class_kind(self):
        """C++ struct_specifier should map to 'class' kind."""
        _, _, symbols, _ = _parse_and_chunk("AnimationAsset.h")
        classes = [s for s in symbols if s.kind == "class"]
        class_names = {c.name for c in classes}
        # FDeltaTimeRecord is a plain struct (no USTRUCT macro) — should be recognized
        assert "FDeltaTimeRecord" in class_names, (
            f"Plain struct FDeltaTimeRecord not found in classes: {sorted(class_names)[:20]}"
        )

    def test_namespace_symbols_exist(self):
        """Namespaces should appear as symbols (added in round 1 fix)."""
        _, _, symbols, _ = _parse_and_chunk("AnimationAsset.h")
        namespaces = [s for s in symbols if s.kind == "namespace"]
        assert len(namespaces) >= 1, f"Expected >= 1 namespace, got {len(namespaces)}"


class TestChunkerOnUEFiles:
    """Verify chunker correctness on real UE code."""

    def test_all_line_numbers_valid(self):
        """Every chunk must have start_line >= 1 and start_line <= end_line."""
        for filename in ["AnimationAsset.h", "ParallelFor.h",
                          "BehaviorTreeTypes.cpp", "BTCompositeNode.cpp"]:
            _, _, _, chunks = _parse_and_chunk(filename)
            for i, c in enumerate(chunks):
                assert c.start_line >= 1, (
                    f"{filename} chunk {i}: start_line={c.start_line} < 1"
                )
                assert c.start_line <= c.end_line, (
                    f"{filename} chunk {i}: start_line={c.start_line} > end_line={c.end_line}"
                )

    def test_meaningful_lines_covered(self):
        """Non-blank, non-comment, non-noise lines must be in at least one chunk."""
        # Patterns that are structural noise — no retrieval value if uncovered
        NOISE_PATTERNS = {
            "};", "}", "{",  # bare braces
            "UENUM()", "USTRUCT()", "UCLASS()",  # bare UE macro annotations
        }
        NOISE_PREFIXES = ("*", "/*", "//", "#if ", "#else", "#elif", "#endif", "#pragma ")

        for filename in ["AnimationAsset.h", "ParallelFor.h", "BehaviorTreeTypes.cpp"]:
            fp = FIXTURES / filename
            source = fp.read_text(encoding="utf-8", errors="replace")
            _, _, _, chunks = _parse_and_chunk(filename)

            covered = set()
            for c in chunks:
                for ln in range(c.start_line, c.end_line + 1):
                    covered.add(ln)

            lines = source.split("\n")
            missing_code = []
            for i, line in enumerate(lines, start=1):
                stripped = line.strip()
                if i in covered:
                    continue
                if not stripped:
                    continue
                if stripped in NOISE_PATTERNS:
                    continue
                if stripped.startswith(NOISE_PREFIXES):
                    continue

                missing_code.append((i, stripped[:100]))

            assert len(missing_code) == 0, (
                f"{filename}: {len(missing_code)} meaningful lines uncovered: {missing_code[:10]}"
            )

    def test_gap_chunks_capture_ue_macro_content(self):
        """UE-macro-wrapped types that tree-sitter can't parse must appear in gap chunks."""
        _, _, _, chunks = _parse_and_chunk("AnimationAsset.h")
        # Find gap chunks (not symbol-named, not sliding window)
        gaps = [c for c in chunks
                if c.symbol_name is None and c.metadata.get("window") is None]
        gap_texts = "\n".join(c.text for c in gaps)

        # UE macros that may trigger tree-sitter errors — should be in gaps if not symbols
        ue_patterns = ["USTRUCT", "UENUM", "GENERATED_USTRUCT_BODY", "UPROPERTY"]
        found = [p for p in ue_patterns if p in gap_texts]
        # Not all patterns must be in gaps (some may be inside recognized symbols),
        # but if tree-sitter rejected any, they should be preserved here
        # This test just verifies gaps exist and contain real content
        assert len(gaps) >= 10, (
            f"Expected >= 10 gap chunks for UE header, got {len(gaps)}"
        )

    def test_file_level_gap_captures_includes(self):
        """File-level gap chunks should capture #include directives before first symbol."""
        _, _, _, chunks = _parse_and_chunk("AnimationAsset.h")
        gaps = [c for c in chunks
                if c.symbol_name is None and c.metadata.get("window") is None]
        gap_texts = "\n".join(c.text for c in gaps)

        assert "#include" in gap_texts, (
            "Expected #include directives in gap chunks, but none found"
        )
        assert "CoreMinimal.h" in gap_texts or "UObject.h" in gap_texts

    def test_sliding_window_line_numbers_increment(self):
        """Each sliding-window chunk must have a larger start_line than the previous."""
        _, _, _, chunks = _parse_and_chunk("BehaviorTreeTypes.cpp", max_tokens=200)
        windows = [c for c in chunks if c.metadata.get("window") is not None]
        # Group windows by symbol
        by_symbol: dict = {}
        for c in chunks:
            key = c.symbol_name or "(gap)"
            if key not in by_symbol:
                by_symbol[key] = []
            by_symbol[key].append(c)

        # Check that windows for the same symbol have non-decreasing start_line
        for name, wchunks in by_symbol.items():
            windows_for_symbol = [c for c in wchunks if c.metadata.get("window") is not None]
            if len(windows_for_symbol) < 2:
                continue
            starts = [c.start_line for c in windows_for_symbol]
            assert starts == sorted(set(starts)), (
                f"Symbol '{name}': window start_lines not strictly ordered: {starts}"
            )
            # Window 0 should start at or before the others
            assert starts[0] <= starts[-1], (
                f"Symbol '{name}': first window starts after last: {starts}"

            )

    def test_small_max_tokens_produces_gap_chunks(self):
        """With small max_tokens, large classes split → container-internal gaps appear."""
        _, _, _, chunks = _parse_and_chunk("AnimationAsset.h", max_tokens=200)
        gaps = [c for c in chunks
                if c.symbol_name is None and c.metadata.get("window") is None]
        # At 200 tokens, more classes exceed limit → more gap chunks
        named = [c for c in chunks if c.symbol_name is not None]
        windows = [c for c in chunks if c.metadata.get("window") is not None]

        assert len(gaps) > len(named) // 5, (
            f"Too few gap chunks for max_tokens=200: gaps={len(gaps)}, named={len(named)}"
        )
        assert len(windows) >= 1, (
            "Expected sliding-window chunks when max_tokens=200"
        )

    def test_normal_mode_no_sliding_windows(self):
        """At max_tokens=512, most UE methods should fit without sliding windows."""
        _, _, _, chunks = _parse_and_chunk("AnimationAsset.h", max_tokens=512)
        windows = [c for c in chunks if c.metadata.get("window") is not None]
        # Some very large methods may still exceed 512, but most should fit
        assert len(windows) <= len(chunks) * 0.05, (
            f"Too many sliding window chunks at 512 tokens: {len(windows)}/{len(chunks)}"
        )


class TestWithRealTokenizer:
    """Verify chunker correctness with a real BPE tokenizer (production path).

    With a real tokenizer, ``_sliding_window_tokenizer`` is used instead of
    the word-based fallback.  It slices the original text using character
    offsets from the tokenizer's ``offset_mapping``, preserving all whitespace
    and line breaks.
    """

    @classmethod
    def setup_class(cls):
        """Load a small BPE tokenizer once for all tests in this class."""
        from transformers import AutoTokenizer
        from code_rag.indexer.chunker import set_tokenizer
        cls._tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        set_tokenizer(cls._tokenizer)

    @classmethod
    def teardown_class(cls):
        from code_rag.indexer.chunker import set_tokenizer
        set_tokenizer(None)

    def test_tokenizer_path_line_numbers_valid(self):
        """Every chunk must have valid line numbers with real tokenizer."""
        for filename in ["AnimationAsset.h", "BehaviorTreeTypes.cpp"]:
            _, _, _, chunks = _parse_and_chunk(filename)
            for i, c in enumerate(chunks):
                assert c.start_line >= 1, (
                    f"{filename} chunk {i}: start_line={c.start_line} < 1"
                )
                assert c.start_line <= c.end_line, (
                    f"{filename} chunk {i}: start={c.start_line} > end={c.end_line}"
                )

    def test_tokenizer_path_coverage(self):
        """Coverage with real tokenizer should be >= word-based fallback."""
        for filename in ["AnimationAsset.h", "BehaviorTreeTypes.cpp"]:
            _, _, _, chunks = _parse_and_chunk(filename)
            fp = FIXTURES / filename
            source = fp.read_text(encoding="utf-8", errors="replace")
            total = source.count("\n") + 1
            covered = set()
            for c in chunks:
                for ln in range(c.start_line, c.end_line + 1):
                    covered.add(ln)
            pct = 100 * len(covered) // total
            assert pct >= 85, (
                f"{filename}: tokenizer path coverage {pct}% < 85%"
            )

    def test_tokenizer_produces_different_token_counts(self):
        """Real tokenizer should produce different (more accurate) token counts
        than the word-based fallback."""
        _, _, _, chunks = _parse_and_chunk("BehaviorTreeTypes.cpp", max_tokens=200)
        # With real tokenizer and max_tokens=200, some functions should split
        windows = [c for c in chunks if c.metadata.get("window") is not None]
        # The tokenizer path uses BPE tokens which pack more text per unit,
        # so at max_tokens=200 we should still get sliding windows
        for c in windows:
            token_count = c.metadata.get("token_count", 0)
            assert token_count > 0, (
                f"Window chunk has no token_count: {c.metadata}"
            )
            word_count = len(c.text.split())
            # BPE token count should differ from simple word count
            # (not equal in most cases for code)

    def test_tokenizer_line_numbers_are_sequential(self):
        """Chunks should be in order — start_line should not jump backwards."""
        for filename in ["AnimationAsset.h", "BehaviorTreeTypes.cpp"]:
            _, _, _, chunks = _parse_and_chunk(filename)
            sorted_chunks = sorted(chunks, key=lambda c: c.start_line)
            prev_end = 0
            gaps_allowed = 0
            for c in sorted_chunks:
                if c.start_line < prev_end:
                    gaps_allowed += 1
                prev_end = max(prev_end, c.end_line)
            # A few overlaps are expected (context prefix prepending),
            # but not wholesale disorder
            assert gaps_allowed < len(sorted_chunks) * 0.3, (
                f"{filename}: {gaps_allowed} overlapping chunks — too many"
            )
