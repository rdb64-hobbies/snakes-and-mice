"""Tests for the core value types: Side, Cell, Move."""

from __future__ import annotations

import pytest

from snakes_and_mice import Cell, IllegalMove, Move, PlayerFaultReason, Side


def test_side_other() -> None:
    assert Side.MOUSE.other is Side.SNAKE
    assert Side.SNAKE.other is Side.MOUSE


def test_cell_label_round_trip() -> None:
    for row in range(5):
        for col in range(5):
            cell = Cell(row, col)
            assert Cell.from_label(cell.label) == cell


def test_cell_center_is_c3() -> None:
    assert Cell.from_label("C3") == Cell(2, 2)
    assert str(Cell(2, 2)) == "C3"


def test_cell_from_label_is_case_insensitive() -> None:
    assert Cell.from_label("c3") == Cell(2, 2)
    assert Cell.from_label(" a1 ") == Cell(0, 0)


@pytest.mark.parametrize("row, col", [(-1, 0), (0, -1), (5, 0), (0, 5)])
def test_cell_off_board_raises(row: int, col: int) -> None:
    with pytest.raises(IllegalMove) as info:
        Cell(row, col)
    assert info.value.reason is PlayerFaultReason.OFF_BOARD


@pytest.mark.parametrize("label", ["F1", "A6", "Z9"])
def test_cell_from_label_off_board_raises(label: str) -> None:
    with pytest.raises(IllegalMove) as info:
        Cell.from_label(label)
    assert info.value.reason is PlayerFaultReason.OFF_BOARD


@pytest.mark.parametrize("label", ["", "A", "A12", "1A", "hi"])
def test_cell_from_label_malformed_raises_value_error(label: str) -> None:
    with pytest.raises(ValueError):
        Cell.from_label(label)


def test_move_of_two_distinct_cells() -> None:
    move = Move.from_labels("A1", "B2")
    assert move.cells == (Cell(0, 0), Cell(1, 1))


def test_move_duplicate_cells_raises() -> None:
    with pytest.raises(IllegalMove) as info:
        Move.of(Cell(0, 0), Cell(0, 0))
    assert info.value.reason is PlayerFaultReason.DUPLICATE_CELLS


def test_single_cell_move_is_constructible() -> None:
    move = Move.of(Cell(0, 0))
    assert move.cells == (Cell(0, 0),)
    assert Move.from_labels("A1").cells == (Cell(0, 0),)
    assert str(move) == "A1"


@pytest.mark.parametrize("count", [0, 3, 4])
def test_move_wrong_piece_count_raises(count: int) -> None:
    cells = [Cell(0, c) for c in range(count)]
    with pytest.raises(IllegalMove) as info:
        Move.of(*cells)
    assert info.value.reason is PlayerFaultReason.WRONG_PIECE_COUNT


def test_move_preserves_order() -> None:
    move = Move.from_labels("A5", "A1")
    assert move.cells[0] == Cell(0, 4)
    assert move.cells[1] == Cell(0, 0)
