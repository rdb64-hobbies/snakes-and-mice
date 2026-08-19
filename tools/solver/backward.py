"""Backward value pass of the perfect-play retrograde solver (offline).

Consumes the layers written by :mod:`forward` and assigns every
enumerated position its exact game value, working *up* from the endgame: a layer's
values depend only on the layer below it, which is already final.

The value is negamax, from the perspective of the side to move. For a position P at
``empties`` empty cells, every legal move (a pair of empty cells) yields a child C
two cells fuller, and::

    value(P) = max over moves of  +WIN - depth(P)   if the move completes a line
                                  0                 if C is a cat's game
                                  -value(C)         otherwise

``depth`` is read off the position (``(occupied - 1) // 2``), not off the search, so
values are globally consistent no matter which layer computed them: a win scores
higher the sooner it lands, and because a loss is a negated win, a lost position
prefers the *longest* defence. Terminal children were dropped by the forward pass, so
they are re-detected here rather than looked up; every non-terminal child is
guaranteed to be present in the layer below. (SPEC allows a one-piece move only when
it ends the game, so every *non-terminal* child comes from a two-piece move and the
enumeration below is complete; a one- and a two-piece win score alike anyway.)

The win score is ``WIN - depth(P)``, the depth of the position the winning move is
made *from*, matching :meth:`PerfectPlayer._negamax`, which returns ``_WIN - depth``
as soon as the side to move *has* a winning move. Scoring the child's depth instead
would shift every value by one and silently break a runtime player that mixes table
lookups with live search.

Children are generated **rectangularly**: every position in a layer has exactly
``C(empties, 2)`` moves, so a chunk of ``n`` parents expands to an ``(n, C(empties, 2))``
grid and the negamax maximum is one ``max(axis=1)``. Child values are found with a
single ``np.searchsorted`` into the layer below, which is globally sorted (the forward
pass's buckets are key ranges in order), memory-mapped so all workers share one copy.

Writes ``keys_NN.npy`` (the layer, flattened from its buckets) and ``vals_NN.npy``
(int16 values, positionally aligned) per layer.

    uv run python tools/solver/backward.py [SEED] [OUT_DIR] [CHUNK] [WORKERS] [KEEP]
"""

from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from snakes_and_mice.players.symmetry import CELL_COUNT
from enumeration import FULL, canon, cats, wins

WIN: int = 1000
"""Score of a win at depth 0 — larger than any reachable depth, so wins dominate."""

_BIT: np.ndarray = np.left_shift(
    np.uint32(1), np.arange(CELL_COUNT, dtype=np.uint32)
)
"""``_BIT[i]`` is the mask of cell ``i``, for turning cell indices into masks."""


def _empty_cells(m: np.ndarray, s: np.ndarray, empties: int) -> np.ndarray:
    """The ``(n, empties)`` ascending indices of each position's empty cells."""
    free: np.ndarray = (~(m | s)) & np.uint32(FULL)
    bits: np.ndarray = (free[:, None] >> np.arange(CELL_COUNT, dtype=np.uint32)) & 1
    # Empty cells sort first (False < True) and stable keeps them in index order.
    return np.argsort(bits == 0, axis=1, kind="stable")[:, :empties]


def _pair_indices(empties: int) -> tuple[np.ndarray, np.ndarray]:
    """The ``C(empties, 2)`` unordered pairs of empty-cell slots, as two index arrays."""
    first, second = np.triu_indices(empties, k=1)
    return first, second


def _child_masks(
    m: np.ndarray, s: np.ndarray, empties: int, mouse_to_move: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every child of a batch of positions, as ``(mouse, snake, mover)`` grids.

    Each is ``(n, C(empties, 2))``: row ``i`` holds all children of parent ``i``, so
    the negamax maximum over a parent's moves is a reduction along axis 1.
    """
    cells: np.ndarray = _empty_cells(m, s, empties)
    first, second = _pair_indices(empties)
    pair: np.ndarray = _BIT[cells[:, first]] | _BIT[cells[:, second]]
    if mouse_to_move:
        child_mouse: np.ndarray = m[:, None] | pair
        child_snake: np.ndarray = np.broadcast_to(s[:, None], child_mouse.shape)
        return child_mouse, child_snake, child_mouse
    child_snake = s[:, None] | pair
    child_mouse = np.broadcast_to(m[:, None], child_snake.shape)
    return child_mouse, child_snake, child_snake


ValueJob = tuple[str, int, int, int, bool, int, str, str, str]


def _value_shard(job: ValueJob) -> str:
    """Value one row-range of a layer, writing the values to disk.

    ``child_keys_path``/``child_vals_path`` are the layer below (empty strings when
    there is none, i.e. every child is terminal).
    """
    (
        keys_path,
        start,
        stop,
        empties,
        mouse_to_move,
        chunk,
        child_keys_path,
        child_vals_path,
        out_path,
    ) = job
    layer: np.ndarray = np.load(keys_path, mmap_mode="r")
    child_keys: np.ndarray | None = None
    child_vals: np.ndarray | None = None
    if child_keys_path:
        child_keys = np.load(child_keys_path, mmap_mode="r")
        child_vals = np.load(child_vals_path, mmap_mode="r")

    # Every position in this layer shares one depth, so a win found from any of
    # them scores the same. This is the *parent's* depth — see the module docstring.
    win_value: np.int16 = np.int16(WIN - (24 - empties) // 2)

    out: np.ndarray = np.empty(stop - start, dtype=np.int16)
    for base in range(start, stop, chunk):
        end: int = min(base + chunk, stop)
        block: np.ndarray = np.asarray(layer[base:end])
        m: np.ndarray = ((block >> np.uint64(CELL_COUNT)) & FULL).astype(np.uint32)
        s: np.ndarray = (block & np.uint64(FULL)).astype(np.uint32)

        child_mouse, child_snake, mover = _child_masks(m, s, empties, mouse_to_move)
        won: np.ndarray = wins(mover)
        drawn: np.ndarray = cats(child_mouse, child_snake) & ~won
        values: np.ndarray = np.zeros(child_mouse.shape, dtype=np.int16)
        values[won] = win_value

        live: np.ndarray = ~(won | drawn)
        if live.any():
            if child_keys is None or child_vals is None:
                raise AssertionError(
                    f"layer {empties} has non-terminal children but no layer below"
                )
            keys: np.ndarray = canon(child_mouse[live], child_snake[live])
            idx: np.ndarray = np.searchsorted(child_keys, keys)
            # Every non-terminal child must be in the layer below — the forward pass
            # built that layer as exactly this set. A miss means the layers disagree.
            if idx.max(initial=0) >= child_keys.shape[0] or not np.array_equal(
                np.asarray(child_keys)[idx], keys
            ):
                raise AssertionError(f"layer {empties}: child missing from layer below")
            values[live] = -np.asarray(child_vals)[idx]

        out[base - start : end - start] = values.max(axis=1)

    np.save(out_path, out)
    return out_path


def _bucket_files(out_dir: Path, empties: int) -> list[Path]:
    """A layer's bucket files in bucket order.

    Sorted *numerically* on the index, not lexicographically: bucket counts now scale
    with the layer, and a run with more than 1000 buckets writes four-digit names
    while older runs wrote three, so text order is not bucket order across both.
    """
    return sorted(
        out_dir.glob(f"layer_{empties:02d}_b*.npy"),
        key=lambda path: int(path.stem.rsplit("_b", 1)[1]),
    )


def _flatten_layer(out_dir: Path, empties: int) -> tuple[str, int]:
    """Concatenate a layer's bucket files into one sorted ``keys_NN.npy``.

    The forward pass's buckets are key *ranges* in ascending order, so writing them in
    bucket order is already globally sorted — no merge needed. Copied bucket by bucket
    through a memmap rather than concatenated in memory: an A-class peak layer is
    ~17 GB, and building it as one array (plus an ``np.diff`` to check the ordering)
    would take twice that in the parent.
    """
    flat: Path = out_dir / f"keys_{empties:02d}.npy"
    if flat.exists():
        return str(flat), int(np.load(flat, mmap_mode="r").shape[0])

    files: list[Path] = _bucket_files(out_dir, empties)
    sizes: list[int] = [int(np.load(f, mmap_mode="r").shape[0]) for f in files]
    total: int = sum(sizes)
    out: np.memmap[Any, np.dtype[np.uint64]] = np.lib.format.open_memmap(
        flat, mode="w+", dtype=np.uint64, shape=(total,)
    )
    at: int = 0
    previous: int = -1
    for path, size in zip(files, sizes):
        if size == 0:
            continue
        keys: np.ndarray = np.load(path)
        if size > 1 and not bool((np.diff(keys) > 0).all()):
            raise AssertionError(f"{path.name} is not sorted")
        if int(keys[0]) <= previous:
            raise AssertionError(f"{path.name} overlaps the previous bucket")
        previous = int(keys[-1])
        out[at : at + size] = keys
        at += size
    out.flush()
    del out
    return str(flat), total


def main() -> None:
    seed_label: str = sys.argv[1] if len(sys.argv) > 1 else "C3"
    out_dir: Path = Path(sys.argv[2] if len(sys.argv) > 2 else "solve-data") / seed_label
    chunk: int = int(sys.argv[3]) if len(sys.argv) > 3 else 50_000
    workers: int = int(sys.argv[4]) if len(sys.argv) > 4 else (os.cpu_count() or 1)
    # Space-saving mode: with a positive threshold, a layer's buckets are dropped once
    # flattened, and a layer's keys/values are dropped once the layer above it is
    # valued and the layer is below the threshold. Deep layers exist only to value the
    # shallow ones; the runtime player live-solves below ~16 empties. 0 keeps all.
    keep_empties: int = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    print(
        f"seed {seed_label}  chunk {chunk}  workers {workers}  keep>={keep_empties}",
        flush=True,
    )
    print("empties   positions    secs      win     draw     loss   value(root)", flush=True)

    child_keys_path: str = ""
    child_vals_path: str = ""
    try:
        with Pool(processes=workers) as pool:
            for empties in range(2, 25, 2):
                t: float = time.time()
                keys_path, n = _flatten_layer(out_dir, empties)
                if n == 0:
                    continue
                mouse_to_move: bool = ((24 - empties) // 2) % 2 == 0

                shard_count: int = max(1, min(workers * 4, (n + chunk - 1) // chunk))
                step: int = (n + shard_count - 1) // shard_count
                jobs: list[ValueJob] = [
                    (
                        keys_path,
                        base,
                        min(base + step, n),
                        empties,
                        mouse_to_move,
                        chunk,
                        child_keys_path,
                        child_vals_path,
                        str(out_dir / f"vshard_{empties:02d}_{i:04d}.npy"),
                    )
                    for i, base in enumerate(range(0, n, step))
                ]
                shards: list[str] = pool.map(_value_shard, jobs)
                values: np.ndarray = np.concatenate([np.load(p) for p in shards])
                for path in shards:
                    os.remove(path)

                vals_path: Path = out_dir / f"vals_{empties:02d}.npy"
                np.save(vals_path, values)

                won: int = int((values > 0).sum())
                drawn: int = int((values == 0).sum())
                lost: int = int((values < 0).sum())
                root: str = str(values[0]) if n == 1 else ""
                print(
                    f"{empties:5d}  {n:10d} {time.time()-t:7.1f} "
                    f"{won:8d} {drawn:8d} {lost:8d}   {root}",
                    flush=True,
                )
                if keep_empties:
                    for spent in _bucket_files(out_dir, empties):
                        spent.unlink(missing_ok=True)
                    # Layer `empties - 2` has now done its only job.
                    if child_keys_path and empties - 2 < keep_empties:
                        Path(child_keys_path).unlink(missing_ok=True)
                        Path(child_vals_path).unlink(missing_ok=True)
                child_keys_path = keys_path
                child_vals_path = str(vals_path)
    finally:
        for leftover in out_dir.glob("vshard_*.npy"):
            leftover.unlink(missing_ok=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
