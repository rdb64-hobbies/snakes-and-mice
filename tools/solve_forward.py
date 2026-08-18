"""Forward enumeration pass of the perfect-play retrograde solver (offline).

Enumerates every reachable *non-terminal* canonical position of a game with a
fixed snake seed, layer by layer (a layer = all positions with the same number of
empty cells), from the opening down toward the endgame. This is both the first
half of the retrograde solve and the feasibility probe: it reports the size of
each layer (so we learn the peak-layer memory cost and the total) and writes each
layer's sorted canonical keys to disk for the backward value pass to reuse.

Everything is vectorized with numpy: within a layer every position has the same
fan-out, so children are generated a whole layer at a time (chunked to bound
memory) and canonicalized with precomputed 5-bit-chunk permutation tables. The
expansion of a layer is embarrassingly parallel, so it is sharded across a process
pool: each worker mmaps the (already-written) layer file, expands a row-range into
its unique child keys, and the parent merges the shards. Run:

    uv run python tools/solve_forward.py [SEED_LABEL] [OUT_DIR] [CHUNK] [WORKERS]
"""

from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from snakes_and_mice.core import Cell
from snakes_and_mice.players.perfect import _LINE_MASKS
from snakes_and_mice.players.symmetry import ALL_PERMS, CELL_COUNT

FULL: int = (1 << CELL_COUNT) - 1


def _build_tables() -> np.ndarray:
    """The permutation lookup tables as a (32, 5, 32) uint32 array.

    ``tables[p, k, v]`` is the image, under symmetry ``p``, of chunk ``k`` (bits
    ``5k..5k+4``) holding the 5-bit value ``v``.
    """
    tables: np.ndarray = np.zeros((len(ALL_PERMS), 5, 32), dtype=np.uint32)
    for p, perm in enumerate(ALL_PERMS):
        for k in range(5):
            base: int = k * 5
            for v in range(32):
                image: int = 0
                for bit in range(5):
                    if (v >> bit) & 1:
                        image |= 1 << perm[base + bit]
                tables[p, k, v] = image
    return tables


_TABLES: np.ndarray = _build_tables()


def _permute(masks: np.ndarray, p: int) -> np.ndarray:
    """Apply symmetry ``p`` to a uint32 array of 25-bit masks (vectorized)."""
    t: np.ndarray = _TABLES[p]
    return (
        t[0][masks & 31]
        | t[1][(masks >> 5) & 31]
        | t[2][(masks >> 10) & 31]
        | t[3][(masks >> 15) & 31]
        | t[4][(masks >> 20) & 31]
    )


def canon(mouse: np.ndarray, snake: np.ndarray) -> np.ndarray:
    """Canonical packed keys (uint64) for arrays of (mouse, snake) masks."""
    best: np.ndarray | None = None
    for p in range(len(ALL_PERMS)):
        pm: np.ndarray = _permute(mouse, p).astype(np.uint64)
        ps: np.ndarray = _permute(snake, p).astype(np.uint64)
        packed: np.ndarray = (pm << np.uint64(CELL_COUNT)) | ps
        best = packed if best is None else np.minimum(best, packed)
    assert best is not None
    return best


def wins(mine: np.ndarray) -> np.ndarray:
    """Boolean array: whether each mask completes some winning line."""
    result: np.ndarray = np.zeros(mine.shape, dtype=bool)
    for line in _LINE_MASKS:
        result |= (mine & line) == line
    return result


def cats(mouse: np.ndarray, snake: np.ndarray) -> np.ndarray:
    """Boolean array: whether every line is dead (both colors present)."""
    result: np.ndarray = np.ones(mouse.shape, dtype=bool)
    for line in _LINE_MASKS:
        result &= ((mouse & line) != 0) & ((snake & line) != 0)
    return result


_PAIRS: tuple[int, ...] = tuple(
    (1 << a) | (1 << b) for a in range(CELL_COUNT) for b in range(a + 1, CELL_COUNT)
)


def _chunk_child_keys(
    m: np.ndarray, s: np.ndarray, mouse_to_move: bool
) -> np.ndarray:
    """Unique canonical keys of the *non-terminal* children of a batch of positions.

    Generates every child across all 300 cell-pairs at once, drops the terminal
    ones (a win or a cat's game), then canonicalizes and dedups the whole batch —
    keeping the per-pair Python overhead well below the vectorized work.
    """
    occ: np.ndarray = m | s
    nm_parts: list[np.ndarray] = []
    ns_parts: list[np.ndarray] = []
    for bitab in _PAIRS:
        elig: np.ndarray = (occ & bitab) == 0
        if not elig.any():
            continue
        if mouse_to_move:
            nm_parts.append(m[elig] | bitab)
            ns_parts.append(s[elig])
        else:
            nm_parts.append(m[elig])
            ns_parts.append(s[elig] | bitab)
    if not nm_parts:
        return np.empty(0, dtype=np.uint64)
    nm: np.ndarray = np.concatenate(nm_parts)
    ns: np.ndarray = np.concatenate(ns_parts)
    mine: np.ndarray = nm if mouse_to_move else ns
    keep: np.ndarray = ~(wins(mine) | cats(nm, ns))
    if not keep.any():
        return np.empty(0, dtype=np.uint64)
    return np.unique(canon(nm[keep], ns[keep]))


def _expand_shard(job: tuple[str, int, int, bool, int, str]) -> str | None:
    """Expand one row-range of a layer file, writing its unique child keys to disk.

    Runs in a worker process: it mmaps the layer (so the big array is never
    pickled *in*), walks its ``[start, stop)`` range in ``chunk``-sized pieces,
    dedups the non-terminal children, and writes them to ``out_path`` — returning
    just that path (or ``None`` if empty) so the big array is never pickled *out*
    either. The parent merges the shard files.
    """
    path, start, stop, mouse_to_move, chunk, out_path = job
    layer: np.ndarray = np.load(path, mmap_mode="r")
    parts: list[np.ndarray] = []
    for base in range(start, stop, chunk):
        end: int = min(base + chunk, stop)
        block: np.ndarray = np.asarray(layer[base:end])
        m: np.ndarray = ((block >> np.uint64(CELL_COUNT)) & FULL).astype(np.uint32)
        s: np.ndarray = (block & np.uint64(FULL)).astype(np.uint32)
        keys: np.ndarray = _chunk_child_keys(m, s, mouse_to_move)
        if keys.size:
            parts.append(keys)
    if not parts:
        return None
    np.save(out_path, np.unique(np.concatenate(parts)))
    return out_path


def main() -> None:
    seed_label: str = sys.argv[1] if len(sys.argv) > 1 else "C3"
    out_dir: Path = Path(sys.argv[2] if len(sys.argv) > 2 else "solve-data") / seed_label
    chunk: int = int(sys.argv[3]) if len(sys.argv) > 3 else 400_000
    workers: int = int(sys.argv[4]) if len(sys.argv) > 4 else (os.cpu_count() or 1)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_cell: Cell = Cell.from_label(seed_label)
    seed_bit: int = seed_cell.row * 5 + seed_cell.col
    layer: np.ndarray = canon(
        np.array([0], dtype=np.uint32), np.array([1 << seed_bit], dtype=np.uint32)
    )
    mouse_to_move: bool = True  # the mouse moves first, at 24 empties

    print(f"seed {seed_label}  chunk {chunk}  workers {workers}", flush=True)
    print("empties   distinct        cum     peak    secs", flush=True)
    cum: int = 0
    peak: int = 0
    with Pool(processes=workers) as pool:
        for empties in range(24, -1, -2):
            t: float = time.time()
            n: int = layer.shape[0]
            path: str = str(out_dir / f"layer_{empties:02d}.npy")
            np.save(path, layer)
            cum += n
            peak = max(peak, n)

            # Shard the layer into more pieces than workers so a straggler shard
            # can't stall the pool; each worker mmaps the file it needs and writes
            # its result to disk, so only paths cross the process boundary.
            tasks: int = max(1, workers * 4)
            step: int = max(chunk, (n + tasks - 1) // tasks)
            jobs: list[tuple[str, int, int, bool, int, str]] = [
                (
                    path,
                    base,
                    min(base + step, n),
                    mouse_to_move,
                    chunk,
                    str(out_dir / f"shard_{empties:02d}_{base}.npy"),
                )
                for base in range(0, n, step)
            ]
            paths: list[str] = [p for p in pool.map(_expand_shard, jobs) if p]
            acc: np.ndarray = (
                np.unique(np.concatenate([np.load(p) for p in paths]))
                if paths
                else np.empty(0, dtype=np.uint64)
            )
            for p in paths:
                os.remove(p)

            print(
                f"{empties:5d}  {n:10d} {cum:11d} {peak:8d} {time.time()-t:7.1f}",
                flush=True,
            )
            layer = acc
            mouse_to_move = not mouse_to_move
            if layer.shape[0] == 0:
                break

    print(f"DONE  total distinct {cum}  peak layer {peak}", flush=True)


if __name__ == "__main__":
    main()
