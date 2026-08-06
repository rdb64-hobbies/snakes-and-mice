"""Tests for the Board: occupancy, win detection, and cat's-game detection."""

from __future__ import annotations

import pytest

from snakes_and_mice import (
    BOARD_SIZE,
    Board,
    Cell,
    IllegalMove,
    PlayerFaultReason,
    Side,
)

# A full board with every line mixed (no winner) — used for the draw test.
# Rows A..E, S = snake, M = mouse. B3 is the snake's seed.
_DRAW_GRID = [
    "SMSMS",  # A
    "SMSMS",  # B  (B3 = S)
    "MSMSM",  # C
    "SMSMS",  # D
    "MSMSM",  # E
]


def _fill_draw_board() -> Board:
    board = Board()
    for row, glyphs in enumerate(_DRAW_GRID):
        for col, glyph in enumerate(glyphs):
            cell = Cell(row, col)
            if board.is_empty(cell):  # B3 is already seeded
                board.place(cell, Side.SNAKE if glyph == "S" else Side.MOUSE)
    return board


def test_initial_board_has_snake_at_b3_only() -> None:
    board = Board()
    assert board.occupant(Cell.from_label("B3")) is Side.SNAKE
    occupied = [
        Cell(r, c)
        for r in range(5)
        for c in range(5)
        if not board.is_empty(Cell(r, c))
    ]
    assert occupied == [Cell.from_label("B3")]


def test_place_and_occupant() -> None:
    board = Board()
    board.place(Cell.from_label("A1"), Side.MOUSE)
    assert board.occupant(Cell.from_label("A1")) is Side.MOUSE
    assert board.is_empty(Cell.from_label("A2"))


def test_place_on_occupied_raises() -> None:
    board = Board()
    with pytest.raises(IllegalMove) as info:
        board.place(Cell.from_label("B3"), Side.MOUSE)
    assert info.value.reason is PlayerFaultReason.CELL_NOT_EMPTY


def test_no_winner_initially() -> None:
    assert Board().winner() is None


def test_empty_cells_excludes_seed_and_occupied() -> None:
    board = Board()
    empties = board.empty_cells()
    # Every cell but the seeded snake at B3.
    assert Cell.from_label("B3") not in empties
    assert len(empties) == BOARD_SIZE * BOARD_SIZE - 1
    # Row-major order: the first empty cell is A1.
    assert empties[0] == Cell.from_label("A1")

    board.place(Cell.from_label("A1"), Side.MOUSE)
    assert Cell.from_label("A1") not in board.empty_cells()
    assert len(board.empty_cells()) == BOARD_SIZE * BOARD_SIZE - 2


def test_copy_is_independent() -> None:
    board = Board()
    board.place(Cell.from_label("A1"), Side.MOUSE)
    clone = board.copy()
    # The clone starts equal to the original...
    assert clone.occupant(Cell.from_label("A1")) is Side.MOUSE
    assert clone.occupant(Cell.from_label("B3")) is Side.SNAKE
    # ...but mutating one does not affect the other.
    clone.place(Cell.from_label("C3"), Side.MOUSE)
    assert board.is_empty(Cell.from_label("C3"))
    assert not clone.is_empty(Cell.from_label("C3"))


def test_winner_by_row() -> None:
    board = Board()
    for col in range(5):
        board.place(Cell(0, col), Side.MOUSE)  # row A
    assert board.winner() is Side.MOUSE


def test_winner_by_column() -> None:
    board = Board()
    for row in range(5):
        board.place(Cell(row, 0), Side.MOUSE)  # column 1
    assert board.winner() is Side.MOUSE


def test_winner_by_main_diagonal() -> None:
    board = Board()
    for i in range(5):
        if board.is_empty(Cell(i, i)):  # skip B3? B3 is (1,2), not on this diagonal
            board.place(Cell(i, i), Side.SNAKE)
    assert board.winner() is Side.SNAKE


def test_winner_by_anti_diagonal() -> None:
    board = Board()
    for i in range(5):
        board.place(Cell(i, 4 - i), Side.MOUSE)
    assert board.winner() is Side.MOUSE


def test_not_cats_game_initially() -> None:
    assert Board().is_cats_game() is False


def test_full_mixed_board_is_cats_game() -> None:
    board = _fill_draw_board()
    assert board.is_full()
    assert board.winner() is None
    assert board.is_cats_game()
