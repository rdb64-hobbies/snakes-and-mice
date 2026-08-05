"""A player that plays uniformly at random among legal moves.

A trivial baseline (and a useful sparring partner for the stronger players): it
tracks its own board via :meth:`observe_move` and, each turn, places two pieces
on two randomly chosen empty cells — or a single piece on the last empty cell,
which necessarily ends the game. It never claims an outcome; the engine judges
the result.

The randomness comes from an injectable :class:`random.Random`, so a seeded
instance yields fully reproducible games (handy for tests and replays).
"""

from __future__ import annotations

import random

from ..board import Board
from ..core import Cell, Move, MoveChoice, Side
from .base import Player


class RandomPlayer(Player):
    """Chooses a legal move uniformly at random each turn."""

    def __init__(
        self, name: str | None = None, rng: random.Random | None = None
    ) -> None:
        super().__init__(name)
        self._rng: random.Random = rng if rng is not None else random.Random()
        self._board: Board = Board()
        self._side: Side | None = None

    def start_game(self, side: Side) -> None:
        self._side = side
        self._board = Board()

    def observe_move(self, side: Side, move: Move) -> None:
        for cell in move.cells:
            if self._board.is_empty(cell):
                self._board.place(cell, side)

    def choose_move(self) -> MoveChoice:
        empties: list[Cell] = self._board.empty_cells()
        # Two pieces normally; a single piece only when one cell remains — which,
        # by filling the board, always ends the game and is therefore legal.
        count: int = min(2, len(empties))
        if count == 0:
            raise RuntimeError(
                f"{self.name}: asked to move with no empty cells left"
            )
        cells: list[Cell] = self._rng.sample(empties, count)
        return MoveChoice(Move.of(*cells))
