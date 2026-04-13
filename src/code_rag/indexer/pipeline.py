from __future__ import annotations

import gc
import time
from dataclasses import dataclass, replace
from pathlib import Path

from tqdm import tqdm

from code_rag.config import CodeRagConfig
from code_rag.indexer.chunker import chunk_file
from code_rag.indexer.chunker import set_tokenizer
from code_rag.indexer.discovery import (
    classify_file,
    detect_language,
    discover_files,
    get_file_fingerprint,
)
from code_rag.indexer.embedder import Embedder
from code_rag.indexer.parser import extract_symbols, parse_file
from code_rag.models import FileInfo
from code_rag.storage.bm25_store import BM25Store
from code_rag.storage.metadata import MetadataStore
from code_rag.storage.vector_store import VectorStore


@dataclass
class IndexStats:
    files_processed: int
    chunks_created: int
    files_skipped: int
    files_deleted: int
    elapsed_seconds: float


class IndexPipeline:
    def __init__(self, config: CodeRagConfig):
        self._config = config
        self._data_dir = config.repo_path / config.data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._vector_db_path = self._data_dir / "vectors.db"
        self._vector_db_path.touch(exist_ok=True)

        self._embedder = Embedder(
            config.model_name,
            config.resolve_device(),
            max_seq_length=config.max_chunk_tokens,
        )
        # Set the real tokenizer for accurate token counting in chunker
        set_tokenizer(self._embedder.tokenizer)
        self._vector_store = VectorStore(self._vector_db_path, self._embedder.dimension)
        self._bm25_store = BM25Store(self._data_dir / "bm25.pkl")
        self._metadata = MetadataStore(self._data_dir / "metadata.json")

    def run(
        self,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        checkpoint_every: int = 5_000,
    ) -> IndexStats:
        """Run the indexing pipeline with checkpoint/resume support.

        Args:
            checkpoint_every: Max chunks per embedding checkpoint batch.
                After each batch, metadata + bm25 + vectors are persisted so
                the process can resume from the last checkpoint on crash.
                Default is 5k chunks.
        """
        start = time.time()

        config = self._config
        if include_patterns is not None or exclude_patterns is not None:
            config = replace(
                self._config,
                include_patterns=(
                    self._config.include_patterns
                    if include_patterns is None
                    else include_patterns
                ),
                exclude_patterns=(
                    self._config.exclude_patterns
                    if exclude_patterns is None
                    else exclude_patterns
                ),
            )

        files = discover_files(config.repo_path, config)
        print(f"Discovered {len(files)} files")

        current_fingerprints: dict[str, str] = {}
        for file_path in tqdm(files, desc="Fingerprinting", unit="file"):
            abs_path = config.repo_path / file_path
            current_fingerprints[str(file_path)] = get_file_fingerprint(abs_path)

        new_files, modified_files, deleted_files = self._metadata.get_changed_files(
            current_fingerprints
        )

        for file_path in deleted_files:
            self._vector_store.delete_by_file(file_path)
            self._bm25_store.remove_by_file(file_path)
            self._metadata.remove_file(file_path)

        files_to_process = new_files + modified_files

        for file_path in modified_files:
            self._vector_store.delete_by_file(file_path)
            self._bm25_store.remove_by_file(file_path)
            self._metadata.remove_file(file_path)

        # -- Resume detection --------------------------------------------------
        # Files parsed in a previous run (metadata saved at checkpoint) but
        # never fully embedded (no chunk_ids).  They need re-parsing because
        # chunk data is only held in memory.
        resumed_files: list[str] = []
        already_scheduled = set(files_to_process)
        for fp in current_fingerprints:
            if fp in already_scheduled:
                continue
            info = self._metadata.get_file_info(fp)
            if info and info.chunk_count > 0 and not self._metadata.get_chunk_ids(fp):
                resumed_files.append(fp)

        if resumed_files:
            print(
                f"Resuming: {len(resumed_files)} files parsed previously "
                f"but not yet embedded"
            )
            files_to_process.extend(resumed_files)

        files_skipped = len(files) - len(files_to_process)
        resume_note = f", resuming: {len(resumed_files)}" if resumed_files else ""
        print(
            f"Files to process: {len(files_to_process)} "
            f"(new: {len(new_files)}, modified: {len(modified_files)}, "
            f"deleted: {len(deleted_files)}, skipped: {files_skipped}"
            f"{resume_note})"
        )

        # == Phase 1: Parse & chunk ============================================
        all_file_chunks: dict[str, list] = {}

        for file_path_str in tqdm(
            files_to_process, desc="Parsing & chunking", unit="file"
        ):
            abs_path = config.repo_path / file_path_str

            try:
                content = abs_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    content = abs_path.read_text(encoding="latin-1")
                except Exception:
                    self._metadata.remove_file(file_path_str)
                    continue
            except Exception:
                self._metadata.remove_file(file_path_str)
                continue

            language = detect_language(abs_path, config.custom_type_mappings)
            file_type = classify_file(abs_path, language)
            if file_type == "unknown":
                self._metadata.set_file_info(
                    file_path_str,
                    FileInfo(
                        path=file_path_str,
                        sha256=current_fingerprints[file_path_str],
                        language=None,
                        size=abs_path.stat().st_size,
                        chunk_count=0,
                    ),
                )
                continue

            symbols = []
            if file_type == "code" and language:
                try:
                    source_bytes = content.encode("utf-8")
                    tree = parse_file(source_bytes, language)
                    symbols = extract_symbols(tree, source_bytes, language)
                    for symbol in symbols:
                        symbol.file_path = file_path_str
                except Exception:
                    symbols = []

            chunks = chunk_file(
                content,
                file_path_str,
                language,
                file_type,
                config.max_chunk_tokens,
            )
            if not chunks:
                self._metadata.set_file_info(
                    file_path_str,
                    FileInfo(
                        path=file_path_str,
                        sha256=current_fingerprints[file_path_str],
                        language=language,
                        size=abs_path.stat().st_size,
                        chunk_count=0,
                    ),
                )
                continue

            all_file_chunks[file_path_str] = chunks

            self._metadata.set_file_info(
                file_path_str,
                FileInfo(
                    path=file_path_str,
                    sha256=current_fingerprints[file_path_str],
                    language=language,
                    size=abs_path.stat().st_size,
                    chunk_count=len(chunks),
                ),
            )
            self._metadata.set_symbols(file_path_str, symbols)

        # -- Checkpoint after parsing ------------------------------------------
        self._metadata.save()
        total_chunks = sum(len(c) for c in all_file_chunks.values())
        print(
            f"Parsing complete: {len(all_file_chunks)} files with chunks, "
            f"{total_chunks} chunks total. Metadata checkpoint saved."
        )

        # == Phase 2: Embed in batches with checkpoints ========================
        # Batch by chunk count (not file count) for uniform batch sizes.
        files_to_embed = [f for f, chunks in all_file_chunks.items() if chunks]
        chunks_created = 0

        if files_to_embed:
            print(
                f"Embedding {total_chunks} chunks "
                f"(checkpoint every ~{checkpoint_every} chunks)..."
            )

            # Build batches: accumulate files until chunk count hits threshold
            batches: list[list[str]] = []
            cur_batch: list[str] = []
            cur_chunk_count = 0
            for f in files_to_embed:
                n = len(all_file_chunks[f])
                cur_batch.append(f)
                cur_chunk_count += n
                if cur_chunk_count >= checkpoint_every:
                    batches.append(cur_batch)
                    cur_batch = []
                    cur_chunk_count = 0
            if cur_batch:
                batches.append(cur_batch)

            files_done = 0
            for batch_num, batch_files in enumerate(batches, 1):
                # Clean up any partial data from a previously failed batch
                # so re-running is idempotent.
                for f in batch_files:
                    self._vector_store.delete_by_file(f)
                    self._bm25_store.remove_by_file(f)

                batch_chunks = []
                batch_file_map: list[tuple[str, int]] = []
                for f in batch_files:
                    fc = all_file_chunks[f]
                    batch_file_map.append((f, len(fc)))
                    batch_chunks.extend(fc)

                if not batch_chunks:
                    files_done += len(batch_files)
                    continue

                print(
                    f"\n-- Batch {batch_num}/{len(batches)}: "
                    f"{len(batch_files)} files, "
                    f"{len(batch_chunks)} chunks --"
                )

                try:
                    embeddings = self._embedder.embed_passages(
                        [chunk.text for chunk in batch_chunks]
                    )
                except Exception as exc:
                    # CUDA errors (OOM, driver crash) — save what we have and
                    # re-raise so the process can be restarted to resume.
                    print(
                        f"\n!!! Embedding failed at batch {batch_num}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    print("Saving checkpoint before exit...")
                    self._bm25_store.save()
                    self._metadata.save()
                    print(
                        f"Checkpoint saved. "
                        f"{chunks_created} chunks embedded so far. "
                        f"Re-run to resume from batch {batch_num}."
                    )
                    raise

                chunk_ids = self._vector_store.insert(batch_chunks, embeddings)
                chunks_created += len(chunk_ids)
                self._bm25_store.add_batch(chunk_ids, batch_chunks)

                # Update chunk_ids in metadata
                idx = 0
                for f, n in batch_file_map:
                    file_chunk_ids = chunk_ids[idx : idx + n]
                    self._metadata.set_chunk_ids(f, file_chunk_ids)
                    idx += n

                # Persist checkpoint
                self._bm25_store.save()
                self._metadata.save()

                # Free GPU memory between batches to prevent fragmentation
                del embeddings
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

                files_done += len(batch_files)
                print(
                    f"  Checkpoint saved "
                    f"({files_done}/{len(files_to_embed)} files, "
                    f"{chunks_created} chunks embedded so far)"
                )

        # == Phase 3: Rebuild BM25 for any indexed files missing from BM25 =====
        # This happens when BM25 store was corrupt/lost but ChromaDB + metadata
        # survived.  We re-parse those files to rebuild keyword search.
        self._rebuild_bm25_if_needed(config)

        elapsed = time.time() - start
        print(
            f"Indexed {len(files_to_process)} files, "
            f"{chunks_created} chunks in {elapsed:.1f}s"
        )

        return IndexStats(
            files_processed=len(files_to_process),
            chunks_created=chunks_created,
            files_skipped=files_skipped,
            files_deleted=len(deleted_files),
            elapsed_seconds=elapsed,
        )

    @property
    def is_indexed(self) -> bool:
        return self._metadata.count_files() > 0

    def get_stats(self) -> dict:
        return {
            "files": self._metadata.count_files(),
            "symbols": self._metadata.count_symbols(),
            "chunks": self._vector_store.count(),
        }

    def _rebuild_bm25_if_needed(self, config: CodeRagConfig) -> None:
        """Rebuild BM25 entries for indexed files missing from BM25 store.

        When BM25 data is lost (corrupt pkl, manual deletion), ChromaDB and
        metadata may still be intact.  This re-parses those files to restore
        keyword search without re-embedding.
        """
        # Collect files that have chunk_ids (= fully embedded) but aren't in BM25
        bm25_files = set(self._bm25_store._file_map.keys())
        files_needing_bm25: list[str] = []
        for fp in self._metadata.get_all_files():
            chunk_ids = self._metadata.get_chunk_ids(fp)
            if chunk_ids and fp not in bm25_files:
                files_needing_bm25.append(fp)

        if not files_needing_bm25:
            return

        print(
            f"Rebuilding BM25 keyword index for {len(files_needing_bm25)} "
            f"files (data was lost/corrupt)..."
        )

        rebuilt = 0
        for file_path_str in tqdm(
            files_needing_bm25, desc="Rebuilding BM25", unit="file"
        ):
            abs_path = config.repo_path / file_path_str
            if not abs_path.exists():
                continue

            try:
                content = abs_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    content = abs_path.read_text(encoding="latin-1")
                except Exception:
                    continue
            except Exception:
                continue

            info = self._metadata.get_file_info(file_path_str)
            if not info or not info.language:
                continue

            language = info.language
            file_type = classify_file(abs_path, language)
            chunks = chunk_file(
                content,
                file_path_str,
                language,
                file_type,
                config.max_chunk_tokens,
            )
            if not chunks:
                continue

            chunk_ids = self._metadata.get_chunk_ids(file_path_str)
            if len(chunks) != len(chunk_ids):
                # Chunk count mismatch — skip to avoid data corruption
                continue

            self._bm25_store.add_batch(chunk_ids, chunks)
            rebuilt += 1

        if rebuilt > 0:
            self._bm25_store.save()
            print(f"BM25 rebuilt for {rebuilt} files.")
