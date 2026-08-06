"""An interactive human player that reads moves from a console.

It prompts for a move, parses the typed labels (e.g. ``C3 D4``), and — so a
human is never knocked out of a game by a simple slip — **re-prompts** on any
mistake it can detect locally: an unparseable label, the wrong number of cells,
an off-board or repeated cell, a target that is already occupied, or a lone
piece that would not end the game. End-of-input (Ctrl-D) is treated as conceding
the turn (a ``PLAYER_FAULT``).

The engine remains the single source of truth; this local checking only spares
the human avoidable faults. Input and output are injectable, so the player can
be driven deterministically in tests.
"""

from __future__ import annotations

from collections.abc import Callable

from ..board import Board
from ..core import Cell, Move, MoveChoice, Side
from ..faults import IllegalMove, MoveUnavailable, PlayerFaultReason
from .base import Player

_REASON_HELP: dict[PlayerFaultReason, str] = {
    PlayerFaultReason.OFF_BOARD: "that cell is off the board (row A–E, column 1–5).",
    PlayerFaultReason.DUPLICATE_CELLS: "the two cells must be different.",
    PlayerFaultReason.WRONG_PIECE_COUNT: "enter one or two cells, e.g. C3 D4.",
}


class HumanPlayer(Player):
    """Reads a move interactively, re-prompting until it is locally valid."""

    def __init__(
        self,
        name: str | None = None,
        read_line: Callable[[str], str] | None = None,
        write: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(name)
        self._read_line: Callable[[str], str] = read_line or input
        self._write: Callable[[str], None] = write or (lambda line: print(line))
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
        assert self._side is not None, "choose_move called before start_game"
        prompt: str = (
            f"{self.name} ({self._side.value}), enter your move (e.g. C3 D4): "
        )
        while True:
            try:
                line: str = self._read_line(prompt)
            except EOFError:
                raise MoveUnavailable(
                    PlayerFaultReason.UNPARSEABLE_OUTPUT, "no input (end of file)"
                )
            move: Move | None = self._parse(line)
            if move is not None:
                return MoveChoice(move)

    def _parse(self, line: str) -> Move | None:
        """Parse and locally validate one input line. Returns the move, or
        ``None`` after reporting why the input was rejected (so the caller
        re-prompts)."""
        labels: list[str] = line.replace(",", " ").split()
        try:
            move: Move = Move.from_labels(*labels)
        except ValueError:
            self._reject("could not read that — use labels like C3 (row A–E, column 1–5).")
            return None
        except IllegalMove as exc:
            self._reject(_REASON_HELP[exc.reason])
            return None

        for cell in move.cells:
            if not self._board.is_empty(cell):
                self._reject(f"{cell} is already occupied.")
                return None

        if len(move.cells) == 1 and not self._ends_game(move.cells[0]):
            self._reject(
                "a single piece is only legal if it ends the game — enter two cells."
            )
            return None

        return move

    def _ends_game(self, cell: Cell) -> bool:
        """Whether playing ``cell`` alone would win or complete a cat's game."""
        assert self._side is not None
        scratch: Board = self._board.copy()
        scratch.place(cell, self._side)
        return scratch.winner() is self._side or scratch.is_cats_game()

    def _reject(self, message: str) -> None:
        self._write(f"  ! {message}")
