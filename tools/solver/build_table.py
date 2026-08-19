"""Pack a solved seed's upper plies into the compact table the player loads.

The solver writes `keys_NN.npy` / `vals_NN.npy` per layer, which are numpy files.
numpy is a dev-group dependency and the player does not use it, so this rewrites the
layers the player reads into a flat binary that loads with `array.frombytes`.

Only layers **>= 16 empties** are included, and note they sit *two below* the layers
at which the player moves: choosing at `E` empty cells means valuing children at
`E-2`. Storing 22/20/18/16 covers the player's choices at 24/22/20/18; it
live-searches from 16 down. `vals_24` is never consulted and is not written.

Layout (all little-endian)::

    magic     8s   b"SNM-PERF"
    version   B    1
    seed_bit  B    representative seed cell index, 0..24
    layers    B    number of layer sections
    reserved  B
    then `layers` x (empties B, reserved B, count I)
    then, per layer in that order: count x uint64 keys, count x int16 values

Keys are the canonical keys of :func:`symmetry.canonical_key`, ascending, so the
player finds one with a plain binary search. Values are negamax scores for the side
to move at that layer.

    uv run python tools/solver/build_table.py SOLVE_DIR OUT_DIR [SEED...]
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

MAGIC: bytes = b"SNM-PERF"
VERSION: int = 1
TABLE_LAYERS: tuple[int, ...] = (22, 20, 18, 16)
"""Layers the player consults, deepest last — see the module docstring."""


def _seed_bit(label: str) -> int:
    """The 0..24 cell index of a seed label like ``A1``."""
    return (ord(label[0]) - ord("A")) * 5 + (int(label[1:]) - 1)


def build(solve_dir: Path, out_path: Path, seed: str) -> None:
    keys_by_layer: dict[int, np.ndarray] = {}
    vals_by_layer: dict[int, np.ndarray] = {}
    for empties in TABLE_LAYERS:
        keys_path: Path = solve_dir / f"keys_{empties:02d}.npy"
        vals_path: Path = solve_dir / f"vals_{empties:02d}.npy"
        if not keys_path.exists():
            raise FileNotFoundError(f"{seed}: missing {keys_path}")
        keys: np.ndarray = np.load(keys_path)
        vals: np.ndarray = np.load(vals_path)
        if keys.shape != vals.shape:
            raise AssertionError(f"{seed} layer {empties}: keys/values length mismatch")
        if keys.size > 1 and not bool((np.diff(keys) > 0).all()):
            raise AssertionError(f"{seed} layer {empties}: keys not ascending")
        keys_by_layer[empties] = keys
        vals_by_layer[empties] = vals

    with out_path.open("wb") as handle:
        handle.write(
            struct.pack(
                "<8sBBBB", MAGIC, VERSION, _seed_bit(seed), len(TABLE_LAYERS), 0
            )
        )
        for empties in TABLE_LAYERS:
            handle.write(
                struct.pack("<BBI", empties, 0, int(keys_by_layer[empties].size))
            )
        for empties in TABLE_LAYERS:
            handle.write(keys_by_layer[empties].astype("<u8").tobytes())
            handle.write(vals_by_layer[empties].astype("<i2").tobytes())

    total: int = sum(int(k.size) for k in keys_by_layer.values())
    print(
        f"{seed}: {total:>10,} positions  {out_path.stat().st_size/2**20:8.2f} MB"
        f"  -> {out_path}"
    )


def main() -> None:
    solve_root: Path = Path(sys.argv[1])
    out_dir: Path = Path(sys.argv[2])
    seeds: list[str] = sys.argv[3:] or ["C3", "A1", "A2", "A3"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        build(solve_root / seed, out_dir / f"{seed}.table", seed)


if __name__ == "__main__":
    main()
