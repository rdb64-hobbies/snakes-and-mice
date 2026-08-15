"""The board: occupancy bookkeeping and line/win/cat's-game detection.

``Board`` is the engine's authoritative state, but it is deliberately reusable:
a player MAY compose one to track its own view of the game (see the ``Player``
abstraction). The board seeds itself with a snake at ``B3`` by default, or at
whatever cell a caller passes — the game loop uses this to vary the opening.
"""

from __future__ import annotations

from .core import BOARD_SIZE, Cell, Side
from .faults import IllegalMove, PlayerFaultReason


def _all_lines() -> tuple[tuple[Cell, ...], ...]:
    """The 12 winning lines: 5 rows, 5 columns, 2 diagonals."""
    lines: list[tuple[Cell, ...]] = []
    for r in range(BOARD_SIZE):
        lines.append(tuple(Cell(r, c) for c in range(BOARD_SIZE)))
    for c in range(BOARD_SIZE):
        lines.append(tuple(Cell(r, c) for r in range(BOARD_SIZE)))
    lines.append(tuple(Cell(i, i) for i in range(BOARD_SIZE)))
    lines.append(tuple(Cell(i, BOARD_SIZE - 1 - i) for i in range(BOARD_SIZE)))
    return tuple(lines)


LINES: tuple[tuple[Cell, ...], ...] = _all_lines()
"""The 12 lines a player must fully occupy to win."""

_SNAKE_START = Cell.from_label("B3")
"""The default snake seed cell — a starting position, not a move. Private: the
default lives only here, so other modules pass ``seed=None`` to mean "default"
and read the resolved cell back from :attr:`Board.seed`."""


class Board:
    """A 5×5 board tracking which side, if any, occupies each cell."""

    def __init__(self, seed: Cell | None = None) -> None:
        # ``None`` means the default seed; resolve it here so the board always
        # knows its concrete seed cell (exposed via the seed property).
        self._seed: Cell = seed if seed is not None else _SNAKE_START
        self._cells: dict[Cell, Side] = {self._seed: Side.SNAKE}

    @property
    def seed(self) -> Cell:
        """The cell the snake was seeded on at the start of this game."""
        return self._seed

    def copy(self) -> Board:
        """An independent copy of this board, safe to mutate without affecting
        the original — useful for a player exploring hypothetical moves."""
        clone: Board = Board(self._seed)
        clone._cells = dict(self._cells)
        return clone

    def occupant(self, cell: Cell) -> Side | None:
        """The side occupying ``cell``, or ``None`` if it is empty."""
        return self._cells.get(cell)

    def is_empty(self, cell: Cell) -> bool:
        """Whether ``cell`` has no piece on it."""
        return cell not in self._cells

    def place(self, cell: Cell, side: Side) -> None:
        """Place ``side``'s piece on ``cell``.

        Raises :class:`~snakes_and_mice.faults.IllegalMove` (``CELL_NOT_EMPTY``)
        if the cell is already occupied.
        """
        if cell in self._cells:
            raise IllegalMove(
                PlayerFaultReason.CELL_NOT_EMPTY, f"cell not empty: {cell}"
            )
        self._cells[cell] = side

    def is_full(self) -> bool:
        """Whether every cell is occupied."""
        return len(self._cells) == BOARD_SIZE * BOARD_SIZE

    def empty_cells(self) -> list[Cell]:
        """Every unoccupied cell, in row-major order (row ``A`` first)."""
        return [
            cell
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE)
            if self.is_empty(cell := Cell(r, c))
        ]

    def winner(self) -> Side | None:
        """The side occupying all cells of some line, or ``None``."""
        for line in LINES:
            first: Side | None = self._cells.get(line[0])
            if first is not None and all(
                self._cells.get(cell) is first for cell in line[1:]
            ):
                return first
        return None

    @staticmethod
    def _is_line_dead(occupants: list[Side | None]) -> bool:
        """A line is dead once it holds at least one piece of each side."""
        return Side.MOUSE in occupants and Side.SNAKE in occupants

    def is_cats_game(self) -> bool:
        """Whether every line is dead — no line can still be completed."""
        return all(
            self._is_line_dead([self._cells.get(cell) for cell in line])
            for line in LINES
        )
