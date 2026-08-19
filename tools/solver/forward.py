"""Forward enumeration pass of the perfect-play solver (offline).

Enumerates every reachable *non-terminal* canonical position of a fixed-seed game,
layer by layer, from the opening down toward the endgame, writing each layer to disk
for the backward value pass. See ``SPEC.md`` beside this file for the design, and
``NOTES.md`` for the measurements and the reasoning behind the constants.

A layer is expanded in parallel and then **range-partitioned into buckets by sampled
splitters** (a sample sort), each deduplicated independently. Nothing is ever globally
sorted or merged, so peak memory is one bucket rather than one layer — which is what
makes the low-symmetry seeds solvable at all.

    uv run python tools/solver/forward.py [SEED] [OUT_DIR] [CHUNK] [WORKERS]
"""

from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from snakes_and_mice.core import Cell
from snakes_and_mice.players.symmetry import CELL_COUNT
from enumeration import FULL, canon, child_keys

def _tunable(name: str, default: int) -> int:
    """An override from the environment, for exercising the scaling paths on C3."""
    return int(os.environ.get(name, default))


SHARD_ROWS: int = _tunable("SOLVER_SHARD_ROWS", 4_000_000)
"""Target parent rows per shard — sets parallel granularity, *not* memory."""

FLUSH_ROWS: int = _tunable("SOLVER_FLUSH_ROWS", 16_000_000)
"""Accumulated child keys after which a worker writes a file and starts fresh.

This, not ``SHARD_ROWS``, is what bounds worker memory, and getting that backwards
cost a run: parent rows say nothing about how many children survive dedup, and the
ratio swings with the side to move (see the duplication note below) — ~1x when the
snake moves, 8-15x when the mouse does. Sizing shards by input rows and calibrating
on a *snake* layer put 16 workers at ~3.5 GB each on the first mouse layer of A1, and
the machine ran out of application memory. Capping the *output* keeps a worker at
roughly ``FLUSH_ROWS`` x 8 bytes x ~3 (accumulate, concatenate, sort) regardless of
layer or seed: ~0.4 GB here, ~7 GB across 16 workers."""

TARGET_BUCKET_ROWS: int = _tunable("SOLVER_BUCKET_ROWS", 16_000_000)
"""Target shard rows per bucket — bounds each worker's peak in the dedup phase."""

MIN_BUCKETS: int = _tunable("SOLVER_MIN_BUCKETS", 128)
MAX_BUCKETS: int = _tunable("SOLVER_MAX_BUCKETS", 4096)

SAMPLES_PER_SHARD: int = 1024
"""Splitter samples drawn from each shard."""

_KEY_MAX: np.uint64 = np.uint64(np.iinfo(np.uint64).max)

ExpandJob = tuple[list[tuple[str, int, int]], bool, int, str]
Shard = tuple[str, int, int]
"""A shard file and the first/last key it holds, for skipping it cheaply."""
DedupJob = tuple[int, np.uint64, np.uint64, list[str], str]


def _expand_shard(job: ExpandJob) -> list[Shard]:
    """Phase 1: expand a set of row-ranges into sorted unique child keys on disk.

    ``sources`` is a list of ``(layer_file, start, stop)`` ranges — a shard is a
    contiguous run of the layer, which may span several bucket files. Emits **one or
    more** files, flushing whenever the accumulated keys reach ``FLUSH_ROWS`` so peak
    memory never depends on how much the layer duplicates. Returns each file with its
    first and last key, so the dedup phase can skip files outside its range.
    """
    sources, mouse_to_move, chunk, out_prefix = job
    parts: list[np.ndarray] = []
    held: int = 0
    written: list[Shard] = []

    def flush() -> None:
        nonlocal parts, held
        if not parts:
            return
        merged: np.ndarray = np.unique(np.concatenate(parts))
        parts = []
        held = 0
        path: str = f"{out_prefix}_{len(written):04d}.npy"
        np.save(path, merged)
        written.append((path, int(merged[0]), int(merged[-1])))

    for path, start, stop in sources:
        layer: np.ndarray = np.load(path, mmap_mode="r")
        for base in range(start, stop, chunk):
            end: int = min(base + chunk, stop)
            block: np.ndarray = np.asarray(layer[base:end])
            m: np.ndarray = ((block >> np.uint64(CELL_COUNT)) & FULL).astype(np.uint32)
            s: np.ndarray = (block & np.uint64(FULL)).astype(np.uint32)
            keys: np.ndarray = child_keys(m, s, mouse_to_move)
            if keys.size:
                parts.append(keys)
                held += int(keys.size)
                if held >= FLUSH_ROWS:
                    flush()
    flush()
    return written


def _dedup_range(job: DedupJob) -> tuple[int, int]:
    """Phase 2: dedup one key range ``[lo, hi)`` across every shard.

    Each shard is sorted, so the range is found with two ``searchsorted`` probes and
    read as one contiguous slice — the rest of the shard is never paged in. Returns
    ``(bucket, size)``.
    """
    bucket, lo, hi, shards, out_path = job
    parts: list[np.ndarray] = []
    for shard in shards:
        keys: np.ndarray = np.load(shard, mmap_mode="r")
        if keys.shape[0] == 0:
            continue
        start: int = int(np.searchsorted(keys, lo, side="left"))
        stop: int = int(np.searchsorted(keys, hi, side="left"))
        if stop > start:
            parts.append(np.asarray(keys[start:stop]))
    result: np.ndarray = (
        np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.uint64)
    )
    np.save(out_path, result)
    return bucket, int(result.shape[0])


def _splitters(shards: list[Shard], buckets: int) -> np.ndarray:
    """Ascending key splitters cutting the shards' rows into ``buckets`` even parts.

    Samples each (sorted) shard on a stride and takes quantiles of the pooled
    sample. Ties are collapsed, so the result may be shorter than ``buckets - 1``
    and the caller must take its length as authoritative.
    """
    samples: list[np.ndarray] = []
    for path, _first, _last in shards:
        keys: np.ndarray = np.load(path, mmap_mode="r")
        n: int = keys.shape[0]
        if n == 0:
            continue
        stride: int = max(1, n // SAMPLES_PER_SHARD)
        samples.append(np.asarray(keys[::stride]))
    if not samples:
        return np.empty(0, dtype=np.uint64)
    pool: np.ndarray = np.sort(np.concatenate(samples))
    cuts: np.ndarray = (np.arange(1, buckets, dtype=np.int64) * pool.size) // buckets
    return np.unique(pool[cuts])


def _bounds(splitters: np.ndarray) -> list[tuple[np.uint64, np.uint64]]:
    """The ``[lo, hi)`` key range of each bucket, in ascending order."""
    edges: np.ndarray = np.concatenate(
        [
            np.zeros(1, dtype=np.uint64),
            splitters.astype(np.uint64),
            np.array([_KEY_MAX], dtype=np.uint64),
        ]
    )
    return [(edges[i], edges[i + 1]) for i in range(edges.size - 1)]


def bucket_path(out_dir: Path, empties: int, bucket: int) -> Path:
    """Where one bucket file of a layer lives.

    Four digits, not three: a layer can now exceed 1000 buckets, and three-digit names
    stop sorting in bucket order at that point (``b1000`` < ``b999`` as text).
    """
    return out_dir / f"layer_{empties:02d}_b{bucket:04d}.npy"


def _layer_files(out_dir: Path, empties: int, sizes: list[int]) -> list[tuple[str, int]]:
    """The ``(path, size)`` of each non-empty bucket file, in key order."""
    return [
        (str(bucket_path(out_dir, empties, b)), size)
        for b, size in enumerate(sizes)
        if size > 0
    ]


def _split_into_shards(
    files: list[tuple[str, int]], shards: int
) -> list[list[tuple[str, int, int]]]:
    """Cut the layer's rows into ``shards`` equal contiguous runs, in key order.

    A run may span more than one bucket file, so each shard is a list of ranges.
    """
    total: int = sum(size for _, size in files)
    if total == 0:
        return []
    per: int = max(1, (total + shards - 1) // shards)
    result: list[list[tuple[str, int, int]]] = []
    current: list[tuple[str, int, int]] = []
    room: int = per
    for path, size in files:
        pos: int = 0
        while pos < size:
            take: int = min(room, size - pos)
            current.append((path, pos, pos + take))
            pos += take
            room -= take
            if room == 0:
                result.append(current)
                current = []
                room = per
    if current:
        result.append(current)
    return result


def main() -> None:
    seed_label: str = sys.argv[1] if len(sys.argv) > 1 else "C3"
    out_dir: Path = Path(sys.argv[2] if len(sys.argv) > 2 else "solve-data") / seed_label
    chunk: int = int(sys.argv[3]) if len(sys.argv) > 3 else 50_000
    workers: int = int(sys.argv[4]) if len(sys.argv) > 4 else (os.cpu_count() or 1)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_cell: Cell = Cell.from_label(seed_label)
    seed_bit: int = seed_cell.row * 5 + seed_cell.col
    root: np.ndarray = canon(
        np.array([0], dtype=np.uint32), np.array([1 << seed_bit], dtype=np.uint32)
    )
    np.save(bucket_path(out_dir, 24, 0), root)
    sizes: list[int] = [1]

    mouse_to_move: bool = True  # the mouse moves first, at 24 empties

    print(
        f"seed {seed_label}  chunk {chunk}  workers {workers}  "
        f"shard_rows {SHARD_ROWS}  bucket_rows {TARGET_BUCKET_ROWS}",
        flush=True,
    )
    print(
        "empties   distinct        cum     peak    secs   expand   dedup"
        "     shardrows  shards  buckets  maxbucket",
        flush=True,
    )
    cum: int = 0
    peak: int = 0
    try:
        with Pool(processes=workers) as pool:
            for empties in range(24, -1, -2):
                t: float = time.time()
                files: list[tuple[str, int]] = _layer_files(out_dir, empties, sizes)
                n: int = sum(size for _, size in files)
                cum += n
                peak = max(peak, n)

                # Enough shards to keep each worker's accumulated children bounded,
                # but never fewer than would keep the pool busy.
                shard_count: int = max(
                    1,
                    min(
                        n,
                        max(workers * 4, (n + SHARD_ROWS - 1) // SHARD_ROWS),
                    ),
                )
                expand_jobs: list[ExpandJob] = [
                    (
                        source,
                        mouse_to_move,
                        chunk,
                        str(out_dir / f"shard_{empties:02d}_{i:04d}"),
                    )
                    for i, source in enumerate(_split_into_shards(files, shard_count))
                ]
                shards: list[Shard] = [
                    written
                    for group in pool.map(_expand_shard, expand_jobs)
                    for written in group
                ]
                t_expand: float = time.time() - t

                t_dedup: float = time.time()
                shard_rows: int = sum(
                    int(np.load(path, mmap_mode="r").shape[0])
                    for path, _first, _last in shards
                )
                child: int = empties - 2
                # Scale buckets with the work so one bucket stays small regardless
                # of seed class; on C3's layers this lands on MIN_BUCKETS anyway.
                buckets: int = min(
                    MAX_BUCKETS,
                    max(
                        MIN_BUCKETS,
                        (shard_rows + TARGET_BUCKET_ROWS - 1) // TARGET_BUCKET_ROWS,
                    ),
                )
                bounds: list[tuple[np.uint64, np.uint64]] = _bounds(
                    _splitters(shards, buckets)
                )
                # A shard file whose key range misses this bucket has nothing to
                # contribute; skipping it here avoids opening it at all, which matters
                # once a layer runs to hundreds of files and hundreds of buckets.
                dedup_jobs: list[DedupJob] = [
                    (
                        b,
                        lo,
                        hi,
                        [
                            path
                            for path, first, last in shards
                            if np.uint64(first) < hi and np.uint64(last) >= lo
                        ],
                        str(bucket_path(out_dir, child, b)),
                    )
                    for b, (lo, hi) in enumerate(bounds)
                ]
                next_sizes: list[int] = [0] * len(bounds)
                for b, size in pool.map(_dedup_range, dedup_jobs):
                    next_sizes[b] = size
                dedup_secs: float = time.time() - t_dedup

                for path, _first, _last in shards:
                    os.remove(path)

                print(
                    f"{empties:5d}  {n:10d} {cum:11d} {peak:8d} {time.time()-t:7.1f}"
                    f" {t_expand:8.1f} {dedup_secs:7.1f} {shard_rows:13d}"
                    f" {len(shards):7d} {len(bounds):8d}"
                    f" {max(next_sizes, default=0):10d}",
                    flush=True,
                )
                sizes = next_sizes
                mouse_to_move = not mouse_to_move
                if sum(sizes) == 0:
                    break

    finally:
        # A run that is interrupted mid-layer would otherwise strand its
        # shards on disk — one abandoned layer-10 pass left 8 GB behind.
        for leftover in out_dir.glob("shard_*.npy"):
            leftover.unlink(missing_ok=True)

    print(f"DONE  total distinct {cum}  peak layer {peak}", flush=True)


if __name__ == "__main__":
    main()
