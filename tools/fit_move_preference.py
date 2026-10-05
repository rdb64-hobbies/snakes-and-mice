"""Fit what move a fallible player prefers, from the mistakes it has made.

Step 4 of `SPEC-rl-player.md`'s loop asks for a mistake model that *generalizes*.
The hand-coded one (`SPEC-mistake-model.md`) is validated but partial: measured
over 588 real `draw->loss` blunders by `qwen-3-8-rtx`, 33.8% happened at
`SCORE_SPLIT` against a 2.0% base rate — a real effect — while **62.4% happened
at score 0**, with no threat pressure at all. Two thirds of the exploitable
errors are invisible to it.

This fits the missing part from data already on disk, with no further LLM time.

## Why this is a ranking problem, not a classification one

`mistakes.jsonl` records only mistakes, so there are no "played this position
correctly" examples to classify against. But each record names the move that was
*chosen* from a fully specified board, and every other legal move from that board
is one that was *rejected*. So the learnable thing is a **preference over moves
within one position**, trained by softmax cross-entropy with the chosen move as
the label — a conditional logit.

Two consequences follow. Only **move-level** features can contribute, because a
position-level feature is constant across the candidates and cancels in the
softmax; that is a useful discipline rather than a limitation. And the result is
a model of *which move this player picks*, which is what steering toward its
errors actually needs — not merely which positions look dangerous.

## What it cannot tell you

The data is conditioned on blundering: every position here is one where the
player went wrong. A model fitted to them predicts move choice *given* that
setting, and will overstate how error-prone the player is in general. Fixing that
needs `mistakes.py` to log every graded move rather than only the bad ones, which
needs fresh games. Stated plainly because it bounds what any number below means.

    uv run python tools/fit_move_preference.py [--player NAME] [--mistakes FILE]
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Final

import torch
from torch import Tensor, nn, optim

from snakes_and_mice.board import LINES
from snakes_and_mice.core import BOARD_SIZE, Cell, Side
from snakes_and_mice.mistake_model import SCORE_SPLIT, threat_score

DEFAULT_MISTAKES: Final[Path] = Path("mistakes.jsonl")
DEFAULT_PLAYER: Final[str] = "qwen-3-8-rtx"

CELL_COUNT: Final[int] = BOARD_SIZE * BOARD_SIZE
LINE_MASKS: Final[tuple[int, ...]] = tuple(
    sum(1 << (c.row * BOARD_SIZE + c.col) for c in line) for line in LINES
)
DIAGONALS: Final[tuple[int, ...]] = LINE_MASKS[-2:]
CENTER: Final[int] = 1 << (2 * BOARD_SIZE + 2)
CORNERS: Final[int] = sum(
    1 << (r * BOARD_SIZE + c) for r in (0, 4) for c in (0, 4)
)

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "aligned",            # both cells share a row or a column
    "adjacent",           # the cells touch, including diagonally
    "advances_own",       # lines the move extends that the opponent is absent from
    "blocks_threat",      # opponent threat lines the move kills
    "answers_all",        # kills every opponent threat line
    "creates_split",      # leaves the opponent facing SCORE_SPLIT
    "creates_threat",     # leaves at least one live threat of ours
    "wins_now",           # completes a line
    "on_diagonal",        # cells lying on a diagonal
    "center",             # plays the center cell
    "corner",             # cells in the corners
)


@dataclass(frozen=True)
class Position:
    """One recorded position: the board, and the move the player chose.

    ``mine`` / ``theirs`` are masks from the *chooser's* side, and ``chosen`` is
    its move as a frozenset of cell indices so ordering cannot matter.
    """

    mine: int
    theirs: int
    chosen: frozenset[int]
    score: int


def _mask(labels: list[str]) -> int:
    return sum(1 << (Cell.from_label(s).row * BOARD_SIZE + Cell.from_label(s).col)
               for s in labels)


def load_positions(path: Path, player: str) -> list[Position]:
    """Every ``draw->loss`` blunder by ``player``, as a position plus its move.

    Only ``draw->loss`` is used. A ``win->draw`` record is a missed conversion,
    a different failure with a different cause — and measured at 100% score 0,
    so the two should not be pooled.
    """
    out: list[Position] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["player"] != player:
                continue
            if not (row["value_before"] == 0 and row["value_after"] < 0):
                continue
            mouse: int = _mask(row["board_before"]["mouse"])
            snake: int = _mask(row["board_before"]["snake"])
            side: Side = Side.MOUSE if row["side"] == "MOUSE" else Side.SNAKE
            mine, theirs = (mouse, snake) if side is Side.MOUSE else (snake, mouse)
            chosen = frozenset(
                Cell.from_label(s).row * BOARD_SIZE + Cell.from_label(s).col
                for s in row["move"]
            )
            played: int = sum(1 << i for i in chosen)
            if played & (mine | theirs):
                continue  # inconsistent record; the move was already occupied
            out.append(
                Position(mine, theirs, chosen, threat_score(theirs, mine))
            )
    return out


def _threat_lines(attacker: int, defender: int) -> list[int]:
    """Opponent lines holding 3+ of their pieces and none of ours."""
    return [
        line for line in LINE_MASKS
        if not line & defender and 1 <= (line & ~attacker).bit_count() <= 2
    ]


def features(mine: int, theirs: int, move: frozenset[int]) -> list[float]:
    """The feature vector for playing ``move`` from ``(mine, theirs)``.

    Every feature is a property of the *move*, since a position-level one would
    be identical across candidates and cancel in the softmax.
    """
    cells: list[int] = sorted(move)
    added: int = sum(1 << i for i in cells)
    after: int = mine | added
    threats: list[int] = _threat_lines(theirs, mine)

    rows = {i // BOARD_SIZE for i in cells}
    cols = {i % BOARD_SIZE for i in cells}
    aligned = float(len(cells) == 2 and (len(rows) == 1 or len(cols) == 1))
    if len(cells) == 2:
        dr = abs(cells[0] // BOARD_SIZE - cells[1] // BOARD_SIZE)
        dc = abs(cells[0] % BOARD_SIZE - cells[1] % BOARD_SIZE)
        adjacent = float(max(dr, dc) == 1)
    else:
        adjacent = 0.0

    advances = sum(
        1 for line in LINE_MASKS
        if line & added and not line & theirs and line & mine
    )
    blocked = sum(1 for line in threats if line & added)
    after_score = threat_score(after, theirs)
    return [
        aligned,
        adjacent,
        float(advances),
        float(blocked),
        float(bool(threats) and blocked == len(threats)),
        float(after_score >= SCORE_SPLIT),
        float(bool(_threat_lines(after, theirs))),
        float(any(after & line == line for line in LINE_MASKS)),
        float(bin(added & (DIAGONALS[0] | DIAGONALS[1])).count("1")),
        float(bool(added & CENTER)),
        float(bin(added & CORNERS).count("1")),
    ]


def candidates(position: Position) -> list[frozenset[int]]:
    """Every legal two-cell move from ``position``, the chosen one included."""
    free: int = ((1 << CELL_COUNT) - 1) & ~(position.mine | position.theirs)
    empties: list[int] = [i for i in range(CELL_COUNT) if (free >> i) & 1]
    return [frozenset(pair) for pair in combinations(empties, 2)]


@dataclass
class Example:
    """One position as a ranking problem: features per candidate, and the label."""

    matrix: Tensor
    label: int
    count: int


def build(positions: list[Position]) -> list[Example]:
    examples: list[Example] = []
    for position in positions:
        moves = candidates(position)
        if position.chosen not in moves:
            continue
        rows = [features(position.mine, position.theirs, m) for m in moves]
        examples.append(
            Example(
                torch.tensor(rows, dtype=torch.float32),
                moves.index(position.chosen),
                len(moves),
            )
        )
    return examples


def fit(train: list[Example], steps: int = 400) -> Tensor:
    """Conditional-logit weights: the chosen move scored above the rejected ones."""
    weights = torch.zeros(len(FEATURE_NAMES), requires_grad=True)
    opt = optim.Adam([weights], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.stack([
            nn.functional.cross_entropy(
                (e.matrix @ weights).unsqueeze(0), torch.tensor([e.label])
            )
            for e in train
        ]).mean()
        loss.backward()
        opt.step()
    return weights.detach()


def score(examples: list[Example], weights: Tensor) -> tuple[float, float, float]:
    """Top-1 accuracy, mean rank percentile of the chosen move, and chance."""
    hits = 0
    percentile = 0.0
    chance = 0.0
    for e in examples:
        s = e.matrix @ weights
        order = torch.argsort(s, descending=True).tolist()
        rank = order.index(e.label)
        hits += rank == 0
        percentile += 1.0 - rank / max(e.count - 1, 1)
        chance += 1.0 / e.count
    n = len(examples)
    return hits / n, percentile / n, chance / n


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="fit_move_preference.py",
        description="Fit which move a fallible player prefers, from its mistakes.",
    )
    parser.add_argument("--mistakes", type=Path, default=DEFAULT_MISTAKES)
    parser.add_argument("--player", default=DEFAULT_PLAYER)
    parser.add_argument("--holdout", type=float, default=0.3)
    parser.add_argument("--rng-seed", type=int, default=0)
    args: argparse.Namespace = parser.parse_args(argv)

    positions = load_positions(args.mistakes, args.player)
    print(f"{len(positions)} draw->loss positions for {args.player}")
    split = [p for p in positions if p.score >= SCORE_SPLIT]
    print(f"  {len(split)} at SCORE_SPLIT, {len(positions) - len(split)} below it")

    examples = build(positions)
    random.Random(args.rng_seed).shuffle(examples)
    cut = int((1 - args.holdout) * len(examples))
    train, test = examples[:cut], examples[cut:]
    print(f"{len(train)} train / {len(test)} test positions, "
          f"{sum(e.count for e in examples) / len(examples):.0f} candidate moves each")

    weights = fit(train)
    print("\nlearned preference (positive = makes the move more likely to be chosen)")
    for name, w in sorted(
        zip(FEATURE_NAMES, weights.tolist()), key=lambda kv: -abs(kv[1])
    ):
        print(f"  {name:<16}{w:+.3f}")

    for label, subset in (("train", train), ("test", test)):
        top1, pct, chance = score(subset, weights)
        print(f"\n{label}: top-1 {100*top1:.1f}% (chance {100*chance:.1f}%), "
              f"chosen move's mean rank percentile {100*pct:.1f}%")


if __name__ == "__main__":
    main()
