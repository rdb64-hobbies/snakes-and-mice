"""Tests for the PerfectPlayer and its offline lookup table (§10).

The solved tables are large and are not part of a checkout, so nothing here depends
on them being installed: table behaviour is tested against a synthetic table built in
a temp directory, and the table-free path is exercised by pointing the loader at an
empty one. Positions are chosen near the endgame so the live search stays fast.
"""

from __future__ import annotations

import itertools
import random
import struct
from pathlib import Path

import pytest

from snakes_and_mice import Cell, Move, Side
from snakes_and_mice.players.perfect import _WIN, PerfectPlayer
from snakes_and_mice.players.symmetry import CELL_COUNT, canonical_key
from snakes_and_mice.players.table import (
    MAGIC,
    VERSION,
    PerfectTable,
    load_for_seed,
    seed_representative,
)

ORBIT_REPRESENTATIVES: dict[str, set[str]] = {
    "A1": {"A1", "A5", "B2", "B4", "D2", "D4", "E1", "E5"},
    "A2": {"A2", "A4", "B1", "B5", "D1", "D5", "E2", "E4"},
    "A3": {"A3", "B3", "C1", "C2", "C4", "C5", "D3", "E3"},
    "C3": {"C3"},
}


def _write_table(path: Path, layers: dict[int, list[tuple[int, int]]]) -> None:
    """Write a table file holding the given (key, value) pairs per layer."""
    ordered: list[int] = sorted(layers, reverse=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<8sBBBB", MAGIC, VERSION, 12, len(ordered), 0))
        for empties in ordered:
            handle.write(struct.pack("<BBI", empties, 0, len(layers[empties])))
        for empties in ordered:
            entries = sorted(layers[empties])
            for key, _value in entries:
                handle.write(struct.pack("<Q", key))
            for _key, value in entries:
                handle.write(struct.pack("<h", value))


def test_every_seed_maps_to_its_orbit_representative() -> None:
    # All 25 seeds must resolve to one of the four solved representatives, so four
    # tables cover every opening.
    for representative, orbit in ORBIT_REPRESENTATIVES.items():
        for label in orbit:
            assert seed_representative(Cell.from_label(label)).label == representative


def test_all_twenty_five_seeds_are_covered() -> None:
    covered = {
        seed_representative(Cell(row, col)).label
        for row in range(5)
        for col in range(5)
    }
    assert covered == set(ORBIT_REPRESENTATIVES)


def test_table_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "C3.table"
    _write_table(path, {16: [(5, -3), (9, 0), (2, 996)], 18: [(7, 0)]})
    table = PerfectTable.load(path)

    assert table.empties_covered == frozenset({16, 18})
    assert table.covers(16) and not table.covers(14)
    assert table.value(16, 2) == 996
    assert table.value(16, 5) == -3
    assert table.value(16, 9) == 0
    assert table.value(18, 7) == 0
    # A key that is not present is a miss, not an approximate hit.
    assert table.value(16, 6) is None
    assert table.value(14, 5) is None


def test_rejects_a_file_that_is_not_a_table(tmp_path: Path) -> None:
    path = tmp_path / "C3.table"
    path.write_bytes(b"not a table at all, really")
    with pytest.raises(ValueError, match="not a perfect-play table"):
        PerfectTable.load(path)


def test_missing_table_is_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A checkout without the solver's output must still produce a working player...
    monkeypatch.setenv("SNAKES_AND_MICE_TABLE_DIR", str(tmp_path))
    assert load_for_seed(Cell.from_label("C3")) is None

    # ...but must say so. Falling back silently looks identical to a hang, because
    # searching the opening can take hours per move.
    warned = capsys.readouterr().err
    assert "no perfect-play table" in warned
    assert str(tmp_path) in warned
    assert "C3.table" in warned
    assert "SNAKES_AND_MICE_TABLE_DIR" in warned


def test_unreadable_table_warns_and_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A damaged table must not stop play, and must not pass unnoticed either.
    (tmp_path / "C3.table").write_bytes(b"definitely not a table")
    monkeypatch.setenv("SNAKES_AND_MICE_TABLE_DIR", str(tmp_path))
    assert load_for_seed(Cell.from_label("C3")) is None

    warned = capsys.readouterr().err
    assert "could not read perfect-play table" in warned


def test_a_healthy_table_loads_quietly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_table(tmp_path / "C3.table", {16: [(1, 0)]})
    monkeypatch.setenv("SNAKES_AND_MICE_TABLE_DIR", str(tmp_path))
    assert load_for_seed(Cell.from_label("C3")) is not None
    assert capsys.readouterr().err == ""


def _endgame_player(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> PerfectPlayer:
    """A player with no table available, so it must search."""
    monkeypatch.setenv("SNAKES_AND_MICE_TABLE_DIR", str(tmp_path / "absent"))
    player = PerfectPlayer(rng=random.Random(1))
    player.start_game(Side.MOUSE, Cell.from_label("C3"))
    return player


def test_plays_a_winning_move_when_one_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    player = _endgame_player(monkeypatch, tmp_path)
    # Mouse holds four of row A; A5 completes it, and a one-piece move is the honest
    # representation of a win that needs only one piece.
    for label in ("A1", "A2", "A3", "A4"):
        player._board.place(Cell.from_label(label), Side.MOUSE)
    for label in ("B1", "B2", "B3"):
        player._board.place(Cell.from_label(label), Side.SNAKE)

    choice = player.choose_move()
    assert choice.move == Move.of(Cell.from_label("A5"))


def test_table_and_search_agree_on_the_chosen_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Build a table from the search itself for one position's children, then check the
    # table-driven choice scores the same as the searched choice. This is the property
    # that lets the player mix lookup and search within a single game.
    searcher = _endgame_player(monkeypatch, tmp_path)
    placed = {
        Side.MOUSE: ("A1", "A2", "B4", "E1", "E2", "D5"),
        Side.SNAKE: ("B1", "B2", "C1", "D1", "E4", "E5"),
    }
    for side, labels in placed.items():
        for label in labels:
            searcher._board.place(Cell.from_label(label), side)

    mouse, snake = searcher._masks()
    empties = searcher._empty_indices(mouse | snake)
    depth = (mouse.bit_count() + snake.bit_count() - 1) // 2

    entries: list[tuple[int, int]] = []
    best_searched = -_WIN - 1
    for a, b in itertools.combinations(empties, 2):
        added = (1 << a) | (1 << b)
        child_mouse, child_snake = mouse | added, snake
        if searcher._is_cats(child_mouse, child_snake):
            value = 0
        else:
            value = -searcher._negamax(
                child_mouse, child_snake, Side.SNAKE, depth + 1, -_WIN - 1, _WIN + 1
            )
            entries.append((canonical_key(child_mouse, child_snake), -value))
        best_searched = max(best_searched, value)

    table_dir = tmp_path / "with-table"
    table_dir.mkdir()
    _write_table(table_dir / "C3.table", {len(empties) - 2: entries})
    monkeypatch.setenv("SNAKES_AND_MICE_TABLE_DIR", str(table_dir))

    looker = PerfectPlayer(rng=random.Random(1))
    looker.start_game(Side.MOUSE, Cell.from_label("C3"))
    for side, labels in placed.items():
        for label in labels:
            looker._board.place(Cell.from_label(label), side)

    assert looker._table is not None
    chosen = looker._choose_from_table(mouse, snake, empties)
    assert chosen is not None, "the table covers every child, so lookup must succeed"

    # The chosen move must actually achieve the searched-for best value.
    cells = [c.row * 5 + c.col for c in chosen.move.cells]
    added = sum(1 << i for i in cells)
    achieved = -searcher._negamax(
        mouse | added, snake, Side.SNAKE, depth + 1, -_WIN - 1, _WIN + 1
    )
    assert achieved == best_searched


def test_incomplete_table_falls_back_to_search(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A table missing even one child must not produce a fast wrong answer.
    table_dir = tmp_path / "partial"
    table_dir.mkdir()
    _write_table(table_dir / "C3.table", {22: [(1, 0)]})
    monkeypatch.setenv("SNAKES_AND_MICE_TABLE_DIR", str(table_dir))

    player = PerfectPlayer(rng=random.Random(1))
    player.start_game(Side.MOUSE, Cell.from_label("C3"))
    mouse, snake = player._masks()
    empties = player._empty_indices(mouse | snake)
    assert player._table is not None
    assert player._choose_from_table(mouse, snake, empties) is None
