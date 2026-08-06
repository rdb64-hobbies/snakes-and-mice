"""Core value types for the game.

These are validated, immutable value types: constructing an off-board
:class:`Cell` or a :class:`Move` that is not two distinct cells raises
:class:`~snakes_and_mice.faults.IllegalMove`. Illegal *structure* is therefore
unrepresentable — once you hold a ``Cell`` or ``Move``, it is well-formed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .faults import IllegalMove, PlayerFaultReason

BOARD_SIZE = 5
"""The board is ``BOARD_SIZE`` × ``BOARD_SIZE`` cells."""


class Side(Enum):
    """A player's side. Mouse moves first; the snake seeds the board at B3."""

    MOUSE = "mouse"
    SNAKE = "snake"

    @property
    def other(self) -> Side:
        """The opposing side."""
        return Side.SNAKE if self is Side.MOUSE else Side.MOUSE


@dataclass(frozen=True)
class Cell:
    """A board coordinate, validated to be on the board at construction.

    ``row`` and ``col`` are zero-based: row 0 is the top row (``A``), col 0 is
    the leftmost column (``1``). The human-facing label (e.g. ``C3``) is available via
    :attr:`label` / :func:`str` and parsed by :meth:`from_label`.
    """

    row: int
    col: int

    def __post_init__(self) -> None:
        if not (0 <= self.row < BOARD_SIZE and 0 <= self.col < BOARD_SIZE):
            raise IllegalMove(
                PlayerFaultReason.OFF_BOARD,
                f"cell off board: row={self.row}, col={self.col}",
            )

    @property
    def label(self) -> str:
        """The human-facing label, e.g. ``C3``."""
        return f"{chr(ord('A') + self.row)}{self.col + 1}"

    def __str__(self) -> str:
        return self.label

    @classmethod
    def from_label(cls, label: str) -> Cell:
        """Parse a label like ``C3`` into a :class:`Cell`.

        Raises :class:`ValueError` if the text is not a well-formed label, and
        :class:`~snakes_and_mice.faults.IllegalMove` (``OFF_BOARD``) if the
        parsed coordinate lies off the board.
        """
        text: str = label.strip().upper()
        if len(text) != 2 or not ("A" <= text[0] <= "Z") or not text[1].isdigit():
            raise ValueError(f"invalid cell label: {label!r}")
        return cls(ord(text[0]) - ord("A"), int(text[1]) - 1)


@dataclass(frozen=True)
class Move:
    """A turn: one or two cells, in the order the player plays them.

    Order matters: the engine places the cells in turn and checks for a win after
    each, so a line completed by the first piece wins before the second is placed.

    A move normally has two distinct cells. A **single**-cell move is structurally
    valid but legal only when that one piece ends the game (a win or cat's game);
    the engine enforces that at apply-time. A count of zero or three-plus, or two
    identical cells, is rejected here at construction.
    """

    cells: tuple[Cell, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.cells) <= 2:
            raise IllegalMove(
                PlayerFaultReason.WRONG_PIECE_COUNT,
                f"a move must place one or two pieces, got {len(self.cells)}",
            )
        if len(self.cells) == 2 and self.cells[0] == self.cells[1]:
            raise IllegalMove(
                PlayerFaultReason.DUPLICATE_CELLS,
                f"move repeats a cell: {self.cells[0]}",
            )

    @classmethod
    def of(cls, *cells: Cell) -> Move:
        """Build a move from one or two cells."""
        return cls(tuple(cells))

    @classmethod
    def from_labels(cls, *labels: str) -> Move:
        """Build a move from one or two labels, e.g. ``Move.from_labels("A1", "B2")``."""
        return cls(tuple(Cell.from_label(label) for label in labels))

    def __str__(self) -> str:
        return " ".join(str(cell) for cell in self.cells)


class TurnOutcome(Enum):
    """A mover's self-assessment of the outcome of its own move."""

    WIN = "win"
    CATS_GAME = "cats_game"
    IN_PLAY = "in_play"


@dataclass(frozen=True)
class MoveChoice:
    """What a player returns from ``choose_move``: a move and an optional claim.

    ``claimed_outcome`` is the player's assessment of the state *after* its move.
    When supplied, the engine checks it against ground truth; any mismatch is a
    ``WRONG_OUTCOME_CLAIM`` fault. ``None`` means the player makes no claim.
    """

    move: Move
    claimed_outcome: TurnOutcome | None = None
