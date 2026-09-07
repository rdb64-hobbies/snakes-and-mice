"""Flagging mistakes: legal moves that give away game-theoretic value (§5).

A **mistake** is not a fault (§3). It is an ordinary, accepted, entirely legal
move that simply is not the best one available — and because the game is solved
(§10), that is a fact rather than an opinion: every reachable position has an
exact value under perfect play, so a move whose result is worth less than the
position the mover started from gave something away.

Detection reads the engine's authoritative board straight off the observer hooks
— the pre-move board from ``on_move_start``, the post-move one from
``on_move_end`` — so nothing here reconstructs a position or parses a log.

This module owns the *detection*; :mod:`console` owns board rendering, which the
callout reuses. Which sides are graded is the caller's decision, not this
observer's: it is handed a set of :class:`Side` and knows nothing about what kind
of player sits on either one (§5).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .board import Board
from .console import render_board
from .core import BOARD_SIZE, Cell, Move, Side, TurnOutcome
from .observer import ObservationLevel, Observer
from .players.perfect import evaluate
from .players.table import PerfectTable, load_for_seed
from .result import MatchResult

_ALL_CELLS: tuple[Cell, ...] = tuple(
    Cell(row, col) for row in range(BOARD_SIZE) for col in range(BOARD_SIZE)
)

DEFAULT_MISTAKES_PATH: Path = Path("mistakes.jsonl")
"""Where recorded mistakes accumulate unless a caller names another file."""


def _sign(value: int) -> int:
    """-1, 0 or +1: which side of the win/draw/loss split a value falls on."""
    return (value > 0) - (value < 0)


@dataclass(frozen=True)
class Mistake:
    """One legal, accepted move that left its mover worse off than it had to be.

    ``value_before`` is the position's exact value for the mover before it moved;
    ``value_after`` the value its actual move left behind, from that same
    mover's perspective. ``value_after < value_before`` always holds — the
    "before" value *is* the best move's value, so no move can beat it.
    """

    game: int  # 1-based, within the match
    turn: int  # 1-based, within the game
    side: Side
    player: str
    seed: Cell
    move: Move
    board_before: Board
    value_before: int
    value_after: int

    @property
    def hard(self) -> bool:
        """Whether this crossed a win/draw/loss boundary rather than a depth.

        Only hard mistakes are called out live (§5): a soft one settles for a
        slower win or a faster loss without changing who is winning, which is
        real but not what a watcher wants interrupting them.
        """
        return _sign(self.value_after) != _sign(self.value_before)

    def as_record(self) -> dict[str, object]:
        """A self-contained JSON record — enough to rebuild the position later.

        The board is written out as cell labels rather than as the move history
        that produced it, so a reader needs nothing but this one line (the same
        stance §6 takes for the results file).
        """
        mouse: list[str] = []
        snake: list[str] = []
        for cell in _ALL_CELLS:
            occupant: Side | None = self.board_before.occupant(cell)
            if occupant is Side.MOUSE:
                mouse.append(cell.label)
            elif occupant is Side.SNAKE:
                snake.append(cell.label)
        return {
            "game": self.game,
            "turn": self.turn,
            "side": self.side.name,
            "player": self.player,
            "seed": self.seed.label,
            "move": [cell.label for cell in self.move.cells],
            "board_before": {"mouse": mouse, "snake": snake},
            "value_before": self.value_before,
            "value_after": self.value_after,
            "hard": self.hard,
        }


def _transition(before: int, after: int) -> str:
    """How a hard mistake changed the result, in words."""
    was: str = {1: "win", 0: "draw", -1: "loss"}[_sign(before)]
    now: str = {1: "win", 0: "draw", -1: "loss"}[_sign(after)]
    return f"turning a {was} into a {now}"


def describe_mistake(mistake: Mistake) -> str:
    """The live callout for a flagged mistake: what happened, and from where."""
    return (
        f"\n⚠  Mistake — {mistake.player} ({mistake.side.value}) played "
        f"{mistake.move}, {_transition(mistake.value_before, mistake.value_after)} "
        f"({mistake.value_before} → {mistake.value_after}). "
        f"It moved from:\n\n{render_board(mistake.board_before)}"
    )


class MistakeObserver(Observer):
    """Grades every move by ``graded_sides`` against the solved game (§5).

    ``report`` prints hard mistakes as they happen; ``path`` records every
    mistake, hard or soft, as one JSON line each — appended as it is found, so
    an interrupted run still leaves every mistake it had seen. Either one alone
    is enough reason to grade.

    Grading needs the seed's opening table (§10). Without it, valuing the
    opening means searching the widest ply in the game, which is not slow but
    effectively unbounded — so a game whose table is missing is left ungraded
    (with the warning ``load_for_seed`` already prints) rather than stalling the
    match for the sake of a diagnostic.
    """

    def __init__(
        self,
        graded_sides: frozenset[Side],
        *,
        report: bool = True,
        path: Path | None = None,
    ) -> None:
        # Level is irrelevant here — the engine fires every hook regardless, and
        # flagging is orthogonal to how much ordinary play is being watched (§5).
        super().__init__(ObservationLevel.MOVE)
        self.graded_sides: frozenset[Side] = graded_sides
        self.mistakes: list[Mistake] = []
        self.graded_moves: dict[Side, int] = {side: 0 for side in graded_sides}
        self._report: bool = report
        self._path: Path | None = path
        self._names: dict[Side, str] = {}
        self._table: PerfectTable | None = None
        self._seed: Cell | None = None
        self._before: Board | None = None
        self._game: int = 0
        self._turn: int = 0

    def on_game_start(self, names: dict[Side, str], board: Board) -> None:
        self._names = names
        self._game += 1
        self._turn = 0
        self._before = None
        self._seed = board.seed
        # Cached by `load_for_seed`, so a match reloads nothing after game one.
        self._table = load_for_seed(board.seed)

    def on_move_start(self, side: Side, board: Board) -> None:
        self._turn += 1
        if side in self.graded_sides:
            # The engine mutates one board in place all game, so the pre-move
            # position has to be copied now — by on_move_end it has the move on it.
            self._before = board.copy()

    def on_move_end(
        self, side: Side, move: Move, board: Board, outcome: TurnOutcome
    ) -> None:
        before_board: Board | None = self._before
        self._before = None
        if side not in self.graded_sides or before_board is None:
            return
        if self._table is None:
            return  # ungraded: see the class docstring
        if outcome is TurnOutcome.WIN:
            return  # nothing beats winning outright, so this cannot be a mistake

        self.graded_moves[side] += 1
        before: int = evaluate(before_board, side, self._table)
        # A cat's game is worth 0 to both sides; otherwise the opponent is now to
        # move, so their value negates into this mover's.
        after: int = (
            0
            if outcome is TurnOutcome.CATS_GAME
            else -evaluate(board, side.other, self._table)
        )
        if after >= before:
            return

        assert self._seed is not None
        mistake = Mistake(
            game=self._game,
            turn=self._turn,
            side=side,
            player=self._names.get(side, side.value),
            seed=self._seed,
            move=move,
            board_before=before_board,
            value_before=before,
            value_after=after,
        )
        self.mistakes.append(mistake)
        if self._path is not None:
            self._append(mistake)
        if self._report and mistake.hard:
            print(describe_mistake(mistake))

    def on_match_end(self, result: MatchResult) -> None:
        if not self.graded_sides:
            return
        print()
        for side in sorted(self.graded_sides, key=lambda s: s.name):
            found = [m for m in self.mistakes if m.side is side]
            hard = sum(1 for m in found if m.hard)
            graded = self.graded_moves[side]
            print(
                f"Mistakes — {self._names.get(side, side.value)} ({side.value}): "
                f"{len(found)} of {graded} graded moves ({hard} crossing a "
                f"win/draw/loss line)"
            )
        if self._path is not None and self.mistakes:
            print(f"Recorded to {self._path}")

    def _append(self, mistake: Mistake) -> None:
        """Append one mistake as a JSON line, flushed immediately.

        Opened per mistake rather than held open: mistakes are rare, and a
        run that dies mid-match should still leave a complete file behind.
        """
        assert self._path is not None
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(mistake.as_record()) + "\n")


__all__ = [
    "DEFAULT_MISTAKES_PATH",
    "Mistake",
    "MistakeObserver",
    "describe_mistake",
]
