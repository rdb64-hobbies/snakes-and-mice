"""Time ``PerfectPlayer.choose_move`` on real positions from a solved layer.

This is the measurement that sets the lookup table's depth threshold: the table has
to cover every layer the live search cannot finish in reasonable time, and each step
deeper costs ~5x the table size. It times the **real** entry point, not ``_negamax``
— ``choose_move`` evaluates every root move with a full window (no root pruning) so
it can collect all moves that tie for best, which is strictly more work than a single
alpha-beta call from the root.

The table's value is also checked against the search's own root value, so a slow
result is at least a correct one.

    uv run python tools/solver/bench_live_search.py OUT_DIR EMPTIES [SAMPLES] [SEED_LABEL] [drawn]
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np

from snakes_and_mice.core import BOARD_SIZE, Cell, Side
from snakes_and_mice.players.perfect import PerfectPlayer
from snakes_and_mice.players.symmetry import CELL_COUNT

FULL: int = (1 << CELL_COUNT) - 1


def _cells(mask: int) -> list[Cell]:
    return [
        Cell(i // BOARD_SIZE, i % BOARD_SIZE)
        for i in range(CELL_COUNT)
        if (mask >> i) & 1
    ]


def main() -> None:
    out_dir: Path = Path(sys.argv[1])
    empties: int = int(sys.argv[2])
    samples: int = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    label: str = sys.argv[4] if len(sys.argv) > 4 else out_dir.name
    # Positions with an immediate win short-circuit before any search, so random
    # sampling mostly measures nothing. "drawn" restricts to value 0 — the positions
    # that force a full-width search, i.e. the case the threshold has to survive.
    hard_only: bool = "drawn" in sys.argv[5:]

    keys: np.ndarray = np.load(out_dir / f"keys_{empties:02d}.npy", mmap_mode="r")
    values: np.ndarray = np.load(out_dir / f"vals_{empties:02d}.npy", mmap_mode="r")
    n: int = int(keys.shape[0])

    mouse_to_move: bool = ((24 - empties) // 2) % 2 == 0
    side: Side = Side.MOUSE if mouse_to_move else Side.SNAKE

    rng: random.Random = random.Random(20260819)
    picks: list[int] = []
    while len(picks) < samples:
        candidate: int = rng.randrange(n)
        if not hard_only or int(values[candidate]) == 0:
            picks.append(candidate)
    times: list[float] = []

    print(f"{label} layer {empties}: {n:,} positions, {side.name} to move, {samples} samples")
    for i in picks:
        key: int = int(keys[i])
        mouse: int = (key >> CELL_COUNT) & FULL
        snake: int = key & FULL

        player: PerfectPlayer = PerfectPlayer(rng=random.Random(1))
        # The canonical form places the seed somewhere in its orbit, so take the seed
        # from the position itself rather than assuming the run's label.
        snake_cells: list[Cell] = _cells(snake)
        player.start_game(side, snake_cells[0])
        player._board._cells = {c: Side.SNAKE for c in snake_cells}
        player._board._cells.update({c: Side.MOUSE for c in _cells(mouse)})
        player._tt = {}

        start: float = time.time()
        choice = player.choose_move()
        elapsed: float = time.time() - start
        times.append(elapsed)
        print(
            f"  idx={i:<10d} table={int(values[i]):>5d}  "
            f"tt={len(player._tt):>9,}  {elapsed:8.2f} s  move={choice.move}"
        )

    times.sort()
    print(
        f"  median {times[len(times)//2]:.2f} s | max {times[-1]:.2f} s | "
        f"total {sum(times):.1f} s"
    )


if __name__ == "__main__":
    main()
