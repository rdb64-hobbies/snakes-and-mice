"""A dataset of positions labelled with their exact solved value.

The critic is regressed against `evaluate()` rather than learned from self-play
alone (SPEC-rl-player.md, "The algorithm and network"): the game is solved, so
training budget spent rediscovering position values by trial and error is wasted.
This module produces the labels once and caches them, so both the pretraining
phase and the auxiliary regularizer that runs alongside PPO read the same file.

Cost, measured on this repo's tables (which cover 22/20/18/16 empty cells): a
lookup is free above 16 empties and the live search below it is ~45 ms at 14
empties, ~10 ms at 12, and under a millisecond from 10 down. So labelling tens of
thousands of positions is minutes, not hours — but it is minutes *every run*,
which is why it lands in a file.

Positions are sampled from **every** seed, not from the current policy: this
dataset is about the game, not about the agent, and is built before the agent
exists.

**Half the playouts are deliberately one-sided**, because symmetric ones do not
teach the critic what losing looks like. Measured on 400,000 symmetrically
sampled positions: 37.6% won for the mover, 62.0% drawn, and **0.4% lost**. The
asymmetry is structural rather than bad luck — "won for the mover" only means *I
have a win available now*, which a careless opponent concedes constantly, while
"lost for the mover" needs the **opponent** to hold a threat set no single move
answers, which is exactly what `SPEC-mistake-model.md` says ordinary play
produces at no useful rate. A value head fit to that data was accurate on won
(MSE 0.013) and drawn (0.005) positions and useless on lost ones (0.674, against
a target of −1 — it calls a lost position a draw). Neither a wider network (tried
up to 1.1 M parameters) nor more of the same labels (tried 9x) moved it.

So :func:`sample_positions` runs some playouts with one side building lines hard
and the other placing at random, and keeps only the **random** side's turns (see
:data:`ATTACK_SHARE`). Those are the positions where someone is actually in
trouble.

    uv run python tools/rl/solved.py [--out FILE] [--positions N]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from snakes_and_mice.board import LINES, Board
from snakes_and_mice.core import BOARD_SIZE, Cell, Side
from snakes_and_mice.mistake_model import SCORE_SPLIT, threat_score
from snakes_and_mice.players.perfect import evaluate
from snakes_and_mice.players.rl_net import board_masks, own_masks
from snakes_and_mice.players.table import PerfectTable, load_for_seed

DEFAULT_PATH: Path = Path("rl-models/solved-values.jsonl")
DEFAULT_POSITIONS: int = 400_000
"""How many positions to label by default.

400,000 rather than the 40,000 of earlier runs. Overall test error only improves
25% over that range — the curve flattens early, so volume alone was never the
lever — but it is what makes the danger filter above pay off, yielding ~10,000
distinct losing positions against ~1,600. Trained on them and scored against a
representative test set, the value head's error on losing positions falls from
0.674 to 0.253, with won and drawn positions unchanged.

A cache built by an older generator holds 40,000 rows and so is below this count,
which makes :func:`load_or_build` rebuild it rather than silently reuse a file
with the wrong distribution.
"""

ALL_SEEDS: tuple[Cell, ...] = tuple(
    Cell(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
)

ATTACK_SHARE: float = 0.5
"""Fraction of playouts run one-sided, to harvest positions that are *lost*.

Half, so the dataset keeps a broad sample of ordinary play alongside the
enriched one: a critic that only ever saw danger would be as badly calibrated as
one that never did. See this module's docstring for the measurements behind it.
"""

ATTACKER_GREEDY: float = 1.0
"""Line-building rate for the strong side of a one-sided playout."""

VICTIM_GREEDY: float = 0.0
"""Line-building rate for the weak side — pure random placement."""

BROAD_KEEP: float = 0.15
"""Chance of labelling a candidate the danger filter did *not* flag.

Labelling, not sampling, is what costs: a playout is nearly free while
:func:`~snakes_and_mice.players.perfect.evaluate` runs at ~200 positions a
second. So candidates are filtered before the expensive call, by the mistake
model's own :func:`~snakes_and_mice.mistake_model.threat_score` — measured over
12,001 positions to be an almost exact detector of a losing one:

| ``threat_score`` | share of positions | lost |
|---|---|---|
| 0 | 81.6% | 0.0% |
| 1 | 5.7% | 0.1% |
| 2 (:data:`SCORE_SPLIT`) | 12.7% | **7.1%** |

Every ``SCORE_SPLIT`` candidate is labelled; the rest are kept at this rate.
0.15 against a 12.7% flagged share puts the two halves at roughly parity, which
keeps a broad sample of ordinary play beside the enriched one — a critic that
only ever saw danger would be as badly calibrated as one that never did.
"""


@dataclass(frozen=True)
class Labelled:
    """One position in the mover's own frame, with its exact value.

    ``mine`` / ``theirs`` are the occupancy masks as the network sees them, and
    ``target`` is in ``[-1, 1]`` — the sign of the negamax score, which is the
    undiscounted return of the game from here under optimal play (the trainer runs
    at ``gamma = 1``, so sign and return agree exactly).
    """

    mine: int
    theirs: int
    target: float


def solved_target(board: Board, side: Side, table: PerfectTable | None) -> float:
    """``+1`` if ``side`` wins this position, ``-1`` if it loses, ``0`` if drawn.

    The exact score folds in distance-to-win (``_WIN - depth``), which the value
    head has no use for: at ``gamma = 1`` a win is worth 1 whenever it arrives, and
    asking a tanh head to also encode *when* would spend capacity on something the
    return never rewards.
    """
    score: int = evaluate(board, side, table)
    return float((score > 0) - (score < 0))


def sample_positions(
    count: int,
    rng: random.Random,
    greedy_rate: float = 0.5,
    attack_share: float = ATTACK_SHARE,
) -> list[Labelled]:
    """Label ``count`` distinct non-terminal positions with their exact value.

    ``greedy_rate`` is the chance that a step of a *symmetric* playout prefers
    cells on lines it already occupies rather than placing uniformly; purely
    random play stays in the drawn, featureless part of the tree.

    ``attack_share`` of playouts are instead **one-sided**: one side builds lines
    every move, the other places at random, and only the random side's turns are
    kept.

    Candidates then pass a danger filter before being labelled — all of those the
    mistake model flags, :data:`BROAD_KEEP` of the rest — which is what actually
    puts losing positions in the dataset. The one-sided playouts alone only lift
    them from 0.37% to 0.93%; the filter is the part that works.
    """
    tables: dict[str, PerfectTable | None] = {}
    seen: set[tuple[int, int, bool]] = set()
    labelled: list[Labelled] = []
    while len(labelled) < count:
        seed: Cell = rng.choice(ALL_SEEDS)
        if seed.label not in tables:
            tables[seed.label] = load_for_seed(seed)
        table: PerfectTable | None = tables[seed.label]
        for board, side in _sample_playout(seed, rng, greedy_rate, attack_share):
            mouse, snake = board_masks(board)
            key: tuple[int, int, bool] = (mouse, snake, side is Side.MOUSE)
            if key in seen:
                continue
            mine, theirs = own_masks(board, side)
            # The filter asks what the OPPONENT threatens with us to move, which
            # is `threat_score`'s own frame (attacker, defender-to-move).
            flagged: bool = threat_score(theirs, mine) >= SCORE_SPLIT
            if not flagged and rng.random() >= BROAD_KEEP:
                continue  # skipped before paying for a label, not after
            seen.add(key)
            labelled.append(
                Labelled(mine, theirs, solved_target(board, side, table))
            )
            if len(labelled) >= count:
                break
    return labelled


def _sample_playout(
    seed: Cell, rng: random.Random, greedy_rate: float, attack_share: float
) -> list[tuple[Board, Side]]:
    """One playout, either symmetric or one-sided, as ``attack_share`` decides.

    The one-sided case picks its attacker at random so neither Mouse nor Snake
    is systematically the one in trouble — the encoding is side-agnostic, but the
    seeded snake makes the two seats genuinely different positions to defend.
    """
    if rng.random() >= attack_share:
        rates: Mapping[Side, float] = {
            Side.MOUSE: greedy_rate, Side.SNAKE: greedy_rate
        }
        return _playout(seed, rng, rates)
    attacker: Side = Side.MOUSE if rng.random() < 0.5 else Side.SNAKE
    rates = {attacker: ATTACKER_GREEDY, attacker.other: VICTIM_GREEDY}
    return _playout(seed, rng, rates, record=attacker.other)


def _playout(
    seed: Cell,
    rng: random.Random,
    greedy: Mapping[Side, float],
    record: Side | None = None,
) -> list[tuple[Board, Side]]:
    """Non-terminal positions along one game, with the side to move.

    ``greedy[side]`` is that side's chance of preferring cells on its own lines.
    ``record`` keeps only that side's turns; ``None`` keeps every turn.

    The game is played out with `Board` itself, so a position in the dataset is
    reachable under the real rules by construction.
    """
    board: Board = Board(seed)
    side: Side = Side.MOUSE
    states: list[tuple[Board, Side]] = []
    while True:
        empties: list[Cell] = board.empty_cells()
        if len(empties) < 2:
            return states
        if record is None or side is record:
            states.append((board.copy(), side))
        is_greedy: bool = rng.random() < greedy[side]
        for cell in _two_cells(board, side, empties, rng, is_greedy):
            board.place(cell, side)
            if board.winner() is not None or board.is_cats_game():
                return states
        side = side.other


def _two_cells(
    board: Board, side: Side, empties: list[Cell], rng: random.Random, greedy: bool
) -> list[Cell]:
    """Two distinct empty cells: uniform, or weighted toward our own lines.

    The greedy draw is without replacement — the first pick is removed before the
    second — because a move must place two *distinct* cells (SPEC.md §2.5).
    """
    if not greedy:
        return rng.sample(empties, 2)
    pool: list[Cell] = list(empties)
    weights: list[float] = [1.0 + 4.0 * _own_lines(board, side, c) for c in pool]
    picked: list[Cell] = []
    for _ in range(2):
        at: int = rng.choices(range(len(pool)), weights=weights)[0]
        picked.append(pool.pop(at))
        weights.pop(at)
    return picked


def _own_lines(board: Board, side: Side, cell: Cell) -> int:
    """How many of ``cell``'s lines ``side`` already has a piece on and the
    opponent none — a crude "this cell builds something" weight, enough to steer a
    playout off uniform and into the part of the tree where positions are won and
    lost rather than dead."""
    total: int = 0
    for line in LINES:
        if cell not in line:
            continue
        occupants: list[Side | None] = [board.occupant(c) for c in line]
        if side.other not in occupants and side in occupants:
            total += 1
    return total


def write(path: Path, rows: list[Labelled]) -> None:
    """Write ``rows`` as JSON Lines, one position each."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {"mine": row.mine, "theirs": row.theirs, "target": row.target}
                )
                + "\n"
            )


def read(path: Path) -> list[Labelled]:
    """Read a dataset written by :func:`write`."""
    rows: list[Labelled] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            blob: dict[str, float] = json.loads(line)
            rows.append(
                Labelled(int(blob["mine"]), int(blob["theirs"]), float(blob["target"]))
            )
    return rows


def load_or_build(
    path: Path, count: int, rng: random.Random, *, quiet: bool = False
) -> list[Labelled]:
    """The cached dataset at ``path``, built and written there if it is absent."""
    if path.exists():
        rows: list[Labelled] = read(path)
        if len(rows) >= count:
            return rows[:count]
        if not quiet:
            print(
                f"{path} holds {len(rows)} positions, {count} wanted — rebuilding",
                file=sys.stderr,
            )
    started: float = time.time()
    rows = sample_positions(count, rng)
    write(path, rows)
    if not quiet:
        wins: int = sum(1 for r in rows if r.target > 0)
        losses: int = sum(1 for r in rows if r.target < 0)
        print(
            f"labelled {len(rows)} positions in {time.time() - started:.0f}s "
            f"({wins} won, {losses} lost, {len(rows) - wins - losses} drawn) "
            f"-> {path}",
            file=sys.stderr,
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="solved.py",
        description="Build the solved-value dataset the RL critic is regressed against.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--positions", type=int, default=DEFAULT_POSITIONS)
    parser.add_argument("--rng-seed", type=int, default=0)
    args: argparse.Namespace = parser.parse_args(argv)
    load_or_build(args.out, args.positions, random.Random(args.rng_seed))


if __name__ == "__main__":
    main()
