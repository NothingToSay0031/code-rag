from __future__ import annotations

import gc
import os
import threading
import time
from concurrent.futures import as_completed, FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePath
from queue import Queue

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
from code_rag.indexer.parser import get_ast_children, parse_file
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


@dataclass
class _FileProcessResult:
    """Result from a worker thread that parsed and chunked one file."""

    file_path_str: str
    status: str  # "success", "unknown", "no_chunks", "error"
    chunks: list = field(default_factory=list)
    symbols: list = field(default_factory=list)
    file_info: FileInfo | None = None


def _relative_path_key(path: PurePath) -> str:
    """Return the canonical relative path key used by metadata and stores."""
    return path.as_posix()


def _parse_and_chunk_file(
    file_path_str: str,
    repo_path: Path,
    fingerprint: str,
    custom_type_mappings: dict[str, str] | None,
    max_chunk_tokens: int,
) -> _FileProcessResult:
    """Read, parse and chunk a single file.  Designed for use inside a
    ``ThreadPoolExecutor`` — tree-sitter (C) and file I/O release the GIL,
    so real parallelism is achieved in the thread pool."""
    abs_path = repo_path / file_path_str

    try:
        content = abs_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = abs_path.read_text(encoding="latin-1")
        except Exception:
            return _FileProcessResult(file_path_str, "error")
    except Exception:
        return _FileProcessResult(file_path_str, "error")

    language = detect_language(abs_path, custom_type_mappings)
    file_type = classify_file(abs_path, language)

    if file_type == "unknown":
        return _FileProcessResult(
            file_path_str,
            "unknown",
            file_info=FileInfo(
                path=file_path_str,
                sha256=fingerprint,
                language=None,
                size=abs_path.stat().st_size,
                chunk_count=0,
            ),
        )

    symbols = []
    _ast_nodes: list = []
    _source_bytes: bytes | None = None
    if file_type == "code" and language:
        _source_bytes = content.encode("utf-8")
        # Files over 1MB can take 30-120+ seconds in tree-sitter and are
        # typically auto-generated or shader code where AST structure is less
        # useful.  Fall back to sliding-window chunking instead.
        if len(_source_bytes) <= 1_000_000:
            try:
                tree = parse_file(_source_bytes, language)
                _ast_nodes = get_ast_children(tree, _source_bytes, language)
                # Extract symbols from AST nodes (single walk)
                _work = list(_ast_nodes)
                while _work:
                    node = _work.pop()
                    if node.symbol is not None:
                        node.symbol.file_path = file_path_str
                        symbols.append(node.symbol)
                    _work.extend(node.children)
            except Exception:
                symbols = []
                _ast_nodes = []

    chunks = chunk_file(
        content, file_path_str, language, file_type, max_chunk_tokens,
        source_bytes=_source_bytes,
        ast_nodes=_ast_nodes if _ast_nodes else None,
    )

    if not chunks:
        return _FileProcessResult(
            file_path_str,
            "no_chunks",
            file_info=FileInfo(
                path=file_path_str,
                sha256=fingerprint,
                language=language,
                size=abs_path.stat().st_size,
                chunk_count=0,
            ),
        )

    return _FileProcessResult(
        file_path_str,
        "success",
        chunks=chunks,
        symbols=symbols,
        file_info=FileInfo(
            path=file_path_str,
            sha256=fingerprint,
            language=language,
            size=abs_path.stat().st_size,
            chunk_count=len(chunks),
        ),
    )


def _read_and_chunk_for_bm25(
    file_path_str: str,
    repo_path: Path,
    language: str,
    max_chunk_tokens: int,
) -> tuple[str, list | None]:
    """Read a file and chunk it for BM25 rebuild.  Returns ``(path, chunks)``
    or ``(path, None)`` when the file cannot be read/chunked."""
    abs_path = repo_path / file_path_str
    if not abs_path.exists():
        return file_path_str, None

    try:
        content = abs_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = abs_path.read_text(encoding="latin-1")
        except Exception:
            return file_path_str, None
    except Exception:
        return file_path_str, None

    file_type = classify_file(abs_path, language)
    chunks = chunk_file(
        content, file_path_str, language, file_type, max_chunk_tokens
    )
    return file_path_str, chunks if chunks else None


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

        # Persist model choice so the server can recreate the exact embedder
        # without relying on CLI args or environment at serve time.
        config.save()

    def run(
        self,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        checkpoint_every: int = 5_000,
    ) -> IndexStats:
        """Run the indexing pipeline with checkpoint/resume support.

        Args:
            checkpoint_every: Max chunks per embedding checkpoint batch.
                Default is 5k chunks.  For static models (model2vec) that
                have no VRAM pressure, the default is automatically raised
                to 50k to reduce I/O overhead.
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
        max_workers = min(8, (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_path: dict = {}
            for file_path in files:
                abs_path = config.repo_path / file_path
                future_to_path[
                    pool.submit(get_file_fingerprint, abs_path)
                ] = _relative_path_key(file_path)

            for future in tqdm(
                as_completed(future_to_path), desc="Fingerprinting", unit="file",
                total=len(files),
            ):
                key = future_to_path[future]
                current_fingerprints[key] = future.result()

        print("Computing delta (new / modified / deleted files)...")
        new_files, modified_files, deleted_files = self._metadata.get_changed_files(
            current_fingerprints
        )
        print(
            f"  Delta: {len(new_files)} new, {len(modified_files)} modified, "
            f"{len(deleted_files)} deleted"
        )

        if deleted_files:
            print(f"  Cleaning up {len(deleted_files)} deleted files...")
            deleted_chunk_ids: dict[str, list[int]] = {}
            for fp in deleted_files:
                cids = self._metadata.get_chunk_ids(fp)
                if cids:
                    deleted_chunk_ids[fp] = cids
            self._vector_store.delete_by_files(deleted_files, deleted_chunk_ids)
            self._bm25_store.remove_by_files(deleted_files)
            for file_path in deleted_files:
                self._metadata.remove_file(file_path)

        files_to_process = new_files + modified_files

        if modified_files:
            print(f"  Cleaning up {len(modified_files)} modified files...")
            modified_chunk_ids: dict[str, list[int]] = {}
            for fp in modified_files:
                cids = self._metadata.get_chunk_ids(fp)
                if cids:
                    modified_chunk_ids[fp] = cids
            self._vector_store.delete_by_files(modified_files, modified_chunk_ids)
            self._bm25_store.remove_by_files(modified_files)
            for file_path in modified_files:
                self._metadata.remove_file(file_path)

        # -- Resume detection --------------------------------------------------
        # Files parsed in a previous run (metadata saved at checkpoint) but
        # never fully embedded (no chunk_ids).  They need re-parsing because
        # chunk data is only held in memory.
        print("Checking for partially-processed files from previous run...")
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

        # Using daemon threads so that if one gets stuck in tree-sitter C code
        # the process can still exit.  Results flow back through a thread-safe
        # queue and the main thread collects them with a per-file timeout.

        # Sort by file size (smallest first) so threads process quick files
        # first and large files are naturally spread across threads at the end.
        # This keeps the progress bar moving and reduces the chance that all
        # threads simultaneously hit a multi-megabyte file.
        _sized: list[tuple[str, int]] = []
        for fp in files_to_process:
            try:
                sz = (config.repo_path / fp).stat().st_size
            except OSError:
                sz = 0
            _sized.append((fp, sz))
        _sized.sort(key=lambda x: x[1])
        work_items: list[str] = [fp for fp, _ in _sized]
        work_lock = threading.Lock()
        work_idx = 0
        result_queue: Queue[_FileProcessResult | None] = Queue()

        def worker() -> None:
            nonlocal work_idx
            while True:
                with work_lock:
                    if work_idx >= len(work_items):
                        return
                    file_path_str = work_items[work_idx]
                    work_idx += 1

                result = _parse_and_chunk_file(
                    file_path_str,
                    config.repo_path,
                    current_fingerprints[file_path_str],
                    config.custom_type_mappings,
                    config.max_chunk_tokens,
                )
                result_queue.put(result)

        threads = [
            threading.Thread(target=worker, daemon=True)
            for _ in range(max_workers)
        ]
        for t in threads:
            t.start()

        PARALLEL_TIMEOUT = 120
        SHORT_POLL = 5  # check for stall every N seconds
        collected = 0
        total = len(files_to_process)
        last_progress = time.time()
        pbar = tqdm(total=total, desc="Parsing & chunking", unit="file")

        while collected < total:
            try:
                result = result_queue.get(timeout=SHORT_POLL)
            except Exception:  # queue.Empty
                if time.time() - last_progress > PARALLEL_TIMEOUT:
                    break
                continue

            last_progress = time.time()

            if result is None:
                continue

            collected += 1
            pbar.update(1)

            if result.status == "error":
                self._metadata.remove_file(result.file_path_str)
            elif result.status in ("unknown", "no_chunks"):
                self._metadata.set_file_info(
                    result.file_path_str, result.file_info
                )
            elif result.status == "success":
                all_file_chunks[result.file_path_str] = result.chunks
                self._metadata.set_file_info(
                    result.file_path_str, result.file_info
                )
                self._metadata.set_symbols(
                    result.file_path_str, result.symbols
                )

        # Collect files that didn't complete in the parallel phase
        with work_lock:
            stuck = [work_items[i] for i in range(work_idx, len(work_items))]

        pbar.close()

        # -- Retry stuck files with per-file timeout -------------------------------
        # Tree-sitter can hang on certain files (large generated C, shaders).
        # Use ThreadPoolExecutor so slow files don't serialize the retry.
        if stuck:
            PER_FILE_TIMEOUT = 30
            skipped_timeout = 0
            print(
                f"\n{len(stuck)} file(s) timed out in parallel parse. "
                f"Retrying with {max_workers} threads (per-file timeout={PER_FILE_TIMEOUT}s)..."
            )
            pbar2 = tqdm(total=len(stuck), desc="Parsing (retry)", unit="file")

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_to_path = {}
                for fp in stuck:
                    future = pool.submit(
                        _parse_and_chunk_file,
                        fp,
                        config.repo_path,
                        current_fingerprints[fp],
                        config.custom_type_mappings,
                        config.max_chunk_tokens,
                    )
                    future_to_path[future] = fp

                pending = set(future_to_path.keys())
                while pending:
                    done, pending = wait(
                        pending,
                        timeout=PER_FILE_TIMEOUT,
                        return_when=FIRST_COMPLETED,
                    )

                    if not done:
                        skipped_timeout += len(pending)
                        for f in pending:
                            f.cancel()
                        pbar2.update(len(pending))
                        break

                    for future in done:
                        try:
                            result = future.result()
                        except Exception:
                            skipped_timeout += 1
                            pbar2.update(1)
                            continue

                        if result.status == "error":
                            self._metadata.remove_file(result.file_path_str)
                        elif result.status in ("unknown", "no_chunks"):
                            self._metadata.set_file_info(
                                result.file_path_str, result.file_info
                            )
                        elif result.status == "success":
                            all_file_chunks[result.file_path_str] = result.chunks
                            self._metadata.set_file_info(
                                result.file_path_str, result.file_info
                            )
                            self._metadata.set_symbols(
                                result.file_path_str, result.symbols
                            )

                        pbar2.update(1)

            pbar2.close()
            if skipped_timeout:
                print(
                    f"  Skipped {skipped_timeout} file(s) due to "
                    f"{PER_FILE_TIMEOUT}s parse timeout"
                )

        # -- Checkpoint after parsing ------------------------------------------
        self._metadata.save()
        total_chunks = sum(len(c) for c in all_file_chunks.values())
        print(
            f"Parsing complete: {len(all_file_chunks)} files with chunks, "
            f"{total_chunks} chunks total. Metadata checkpoint saved."
        )

        # == Phase 2: Embed in batches with checkpoints ========================
        # Static models (model2vec) have no VRAM pressure — checkpoint I/O is
        # pure overhead.  Build a single batch so we only persist at the end.
        if self._embedder._is_static():
            checkpoint_every = max(checkpoint_every, total_chunks + 1)

        # Batch by chunk count (not file count) for uniform batch sizes.
        # Sort files by average chunk length (ascending) so that short-chunk
        # files (headers, shaders) fill early batches and benefit from large
        # batch sizes, while long-chunk files are grouped together at the end.
        # Total compute is unchanged; this reduces batch-to-batch variance.
        files_to_embed = sorted(
            (f for f, chunks in all_file_chunks.items() if chunks),
            key=lambda f: (
                sum(len(c.text) for c in all_file_chunks[f]) / len(all_file_chunks[f])
            ),
        )
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
                    # On transient CUDA errors (e.g. cudaErrorUnknown,
                    # AcceleratorError) the embed_passages retry logic may
                    # have already exhausted its per-group retries.  Before
                    # giving up on the entire batch, try one more time with
                    # a full GPU reset — these errors are often transient
                    # (WDDM TDR, driver hiccup) and a fresh attempt succeeds.
                    from code_rag.indexer.embedder import Embedder

                    if Embedder._is_cuda_error(exc):
                        print(
                            f"\n!!! CUDA error at batch {batch_num}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        print("Attempting GPU reset + retry for entire batch...")
                        gc.collect()
                        try:
                            import torch

                            if torch.cuda.is_available():
                                try:
                                    torch.cuda.empty_cache()
                                except Exception:
                                    pass
                                try:
                                    torch.cuda.synchronize()
                                except RuntimeError:
                                    pass
                        except ImportError:
                            pass
                        try:
                            embeddings = self._embedder.embed_passages(
                                [chunk.text for chunk in batch_chunks]
                            )
                        except Exception as retry_exc:
                            # Retry also failed — checkpoint and exit.
                            print(
                                f"\n!!! Retry also failed: "
                                f"{type(retry_exc).__name__}: {retry_exc}"
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
                    else:
                        # Non-CUDA error (e.g. logic bug) — checkpoint and
                        # re-raise immediately; retrying won't help.
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
        max_workers = min(8, (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_path: dict = {}
            for file_path_str in files_needing_bm25:
                info = self._metadata.get_file_info(file_path_str)
                if not info or not info.language:
                    continue
                future_to_path[
                    pool.submit(
                        _read_and_chunk_for_bm25,
                        file_path_str,
                        config.repo_path,
                        info.language,
                        config.max_chunk_tokens,
                    )
                ] = file_path_str

            for future in tqdm(
                as_completed(future_to_path), desc="Rebuilding BM25", unit="file",
                total=len(future_to_path),
            ):
                file_path_str, chunks = future.result()
                if chunks is None:
                    continue

                chunk_ids = self._metadata.get_chunk_ids(file_path_str)
                if len(chunks) != len(chunk_ids):
                    continue

                self._bm25_store.add_batch(chunk_ids, chunks)
                rebuilt += 1

        if rebuilt > 0:
            self._bm25_store.save()
            print(f"BM25 rebuilt for {rebuilt} files.")
