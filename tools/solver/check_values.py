"""Check solver values against ``PerfectPlayer``'s live search.

The table is only useful if it agrees exactly with the search the runtime player
falls back on below the lookup threshold. This samples positions from a solved
layer, decodes them, and re-derives each value with
:meth:`PerfectPlayer._negamax` — the same code path a live game would take.

Sampling a *deep* layer keeps the live search cheap; the values there are the ones
every shallower layer was built from, so an error anywhere below would show up here.

    uv run python tools/solver/check_values.py OUT_DIR EMPTIES [SAMPLES]
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

from snakes_and_mice.core import Side
from snakes_and_mice.players.perfect import _WIN, PerfectPlayer
from snakes_and_mice.players.symmetry import CELL_COUNT

FULL: int = (1 << CELL_COUNT) - 1


def main() -> None:
    out_dir: Path = Path(sys.argv[1])
    empties: int = int(sys.argv[2])
    samples: int = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    keys: np.ndarray = np.load(out_dir / f"keys_{empties:02d}.npy", mmap_mode="r")
    values: np.ndarray = np.load(out_dir / f"vals_{empties:02d}.npy", mmap_mode="r")
    n: int = int(keys.shape[0])
    print(f"layer {empties}: {n} positions, checking {samples}")

    rng: random.Random = random.Random(20260818)
    picks: list[int] = [rng.randrange(n) for _ in range(samples)]

    mouse_to_move: bool = ((24 - empties) // 2) % 2 == 0
    side: Side = Side.MOUSE if mouse_to_move else Side.SNAKE
    depth: int = (24 - empties) // 2

    mismatches: int = 0
    for i in picks:
        key: int = int(keys[i])
        mouse: int = (key >> CELL_COUNT) & FULL
        snake: int = key & FULL
        assert 25 - (mouse.bit_count() + snake.bit_count()) == empties

        player: PerfectPlayer = PerfectPlayer()
        live: int = player._negamax(mouse, snake, side, depth, -_WIN - 1, _WIN + 1)
        table: int = int(values[i])
        if live != table:
            mismatches += 1
            if mismatches <= 5:
                print(f"  MISMATCH idx={i} mouse={mouse:#x} snake={snake:#x} "
                      f"table={table} live={live}")

    print(f"mismatches: {mismatches}/{samples}")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
