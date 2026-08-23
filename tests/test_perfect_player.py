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
from snakes_and_mice.players.perfect import (
    _ALL_DRAWN_ABOVE_EMPTIES,
    _CELLS_BY_INDEX,
    _LIVENESS_MAX_EMPTIES,
    _WIN,
    _Candidate,
    PerfectPlayer,
)
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


# --- Choosing among equally optimal moves (§10) -----------------------------------
#
# The tie-break ranks a pool whose members already share the exact minimax value, so
# the property that matters is that ranking never changes *which value* is achieved —
# only which of the equally optimal moves is played. The other property under test is
# the opposite one: that the keys stand aside wherever they cannot discriminate, so
# the opening stays uniformly random and games do not repeat.


def _random_position(
    player: PerfectPlayer, empties_target: int, rng: random.Random, tries: int = 200
) -> tuple[int, int, int] | None:
    """A random *live* position with Mouse to move, as (mouse, snake, depth).

    Random fills are often already decided — a cat's game especially — so keep
    drawing until one is a genuine choice: nobody has a completed line, the game is
    not over, and Mouse has no immediate win (a win is taken outright, never
    tie-broken). ``None`` means no live position turned up, which the caller should
    treat as a failed test rather than a skip.
    """
    for _ in range(tries):
        cells = list(range(CELL_COUNT))
        rng.shuffle(cells)
        mouse, snake = 0, 1 << cells[0]
        taken = 1
        while CELL_COUNT - taken > empties_target:
            mouse |= 1 << cells[taken]
            snake |= 1 << cells[taken + 1]
            taken += 2
        if player._completes(mouse) or player._completes(snake):
            continue
        if player._is_cats(mouse, snake) or player._wins_now(mouse, snake):
            continue
        return mouse, snake, (mouse.bit_count() + snake.bit_count() - 1) // 2
    return None


def _pool_and_best(
    player: PerfectPlayer, mouse: int, snake: int, depth: int
) -> tuple[list[_Candidate], int]:
    """Every move that ties for the best value, searched exactly."""
    empties = player._empty_indices(mouse | snake)
    best = -_WIN - 1
    pool: list[_Candidate] = []
    for a, b in itertools.combinations(empties, 2):
        added = (1 << a) | (1 << b)
        child_mouse, child_snake = mouse | added, snake
        if player._is_cats(child_mouse, child_snake):
            value = 0
        else:
            value = -player._negamax(
                child_mouse, child_snake, Side.SNAKE, depth + 1, -_WIN - 1, _WIN + 1
            )
        candidate = _Candidate(
            Move.of(_CELLS_BY_INDEX[a], _CELLS_BY_INDEX[b]), child_mouse, child_snake
        )
        if value > best:
            best, pool = value, [candidate]
        elif value == best:
            pool.append(candidate)
    return pool, best


def test_tie_break_never_changes_the_value_achieved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The whole design rests on this: narrowing a pool of equally optimal moves is
    # free. However the keys rank it, the move played must still achieve the best
    # value — so the player is exactly as perfect as it was before ranking existed.
    monkeypatch.setenv("SNAKES_AND_MICE_TABLE_DIR", str(tmp_path / "absent"))
    rng = random.Random(7)
    checked = 0
    for trial in range(40):
        player = PerfectPlayer(rng=random.Random(trial))
        player.start_game(Side.MOUSE, Cell.from_label("C3"))
        found = _random_position(player, 10, rng)
        if found is None:
            continue
        mouse, snake, depth = found
        pool, best = _pool_and_best(player, mouse, snake, depth)
        empties = len(player._empty_indices(mouse | snake))
        picked = player._pick(list(pool), best, empties, depth)
        # Re-derive the picked move's value rather than trusting the pool bookkeeping.
        cells = [c.row * 5 + c.col for c in picked.cells]
        added = sum(1 << i for i in cells)
        value = (
            0
            if player._is_cats(mouse | added, snake)
            else -player._negamax(
                mouse | added, snake, Side.SNAKE, depth + 1, -_WIN - 1, _WIN + 1
            )
        )
        assert value == best, f"played a move worth {value}, not {best}"
        checked += 1
    assert checked >= 20, "too few positions exercised to mean anything"


def test_the_opening_is_left_uniformly_random(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The point of the gates: at the top of the tree every reply is drawn, so no key
    # can discriminate and none may narrow. A deterministic opening would make two
    # deterministic players replay one identical game per seed (§5).
    monkeypatch.setenv("SNAKES_AND_MICE_TABLE_DIR", str(tmp_path / "absent"))
    player = PerfectPlayer(rng=random.Random(1))
    player.start_game(Side.MOUSE, Cell.from_label("C3"))
    mouse, snake = player._masks()
    empties = player._empty_indices(mouse | snake)
    pool = [
        _Candidate(
            Move.of(_CELLS_BY_INDEX[a], _CELLS_BY_INDEX[b]),
            mouse | (1 << a) | (1 << b),
            snake,
        )
        for a, b in itertools.combinations(empties, 2)
    ]
    assert len(pool) == 276

    # Trap counting is vacuous this shallow — every grandchild is drawn — and must
    # not even be attempted, since at 24 empties it is also the widest node in the
    # game. Liveness is gated off above the opening for the same reason.
    assert player._most_trapping(list(pool), 0, len(empties), 0) is not None
    assert len(player._most_trapping(list(pool), 0, len(empties), 0)) == len(pool)
    assert len(empties) - 4 >= _ALL_DRAWN_ABOVE_EMPTIES
    assert len(empties) > _LIVENESS_MAX_EMPTIES

    # ...so the pick really is spread over the whole pool.
    seen = {player._pick(list(pool), 0, len(empties), 0) for _ in range(400)}
    assert len(seen) > 150, f"opening collapsed to {len(seen)} distinct moves"


def test_chosen_move_maximizes_the_trap_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # When the count does discriminate, the played move must be one of the moves that
    # gives the opponent the most ways to go wrong — that is the entire mechanism.
    # The liveness key cannot disturb this: it narrows *within* whatever the trap
    # count leaves, so the pick stays inside the trap-maximal set.
    monkeypatch.setenv("SNAKES_AND_MICE_TABLE_DIR", str(tmp_path / "absent"))
    rng = random.Random(11)
    discriminating = 0
    for trial in range(60):
        player = PerfectPlayer(rng=random.Random(trial))
        player.start_game(Side.MOUSE, Cell.from_label("C3"))
        found = _random_position(player, 10, rng)
        if found is None:
            continue
        mouse, snake, depth = found
        empties = len(player._empty_indices(mouse | snake))
        pool, best = _pool_and_best(player, mouse, snake, depth)
        if best != 0 or len(pool) < 2:
            continue
        counts = {
            c.move: player._trap_count(c.mouse, c.snake, 0, depth, False)
            for c in pool
        }
        if len(set(counts.values())) < 2:
            continue
        discriminating += 1
        top = max(c for c in counts.values() if c is not None)
        for _ in range(20):
            assert counts[player._pick(list(pool), best, empties, depth)] == top
    assert discriminating >= 5, "no discriminating position found; test proves nothing"


def test_table_and_search_agree_on_trap_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The two tiers must agree about traps as well as values, since a game mixes them.
    # Build the table out of the search's own grandchild values, then require the
    # table-backed count to match the searched one exactly.
    monkeypatch.setenv("SNAKES_AND_MICE_TABLE_DIR", str(tmp_path / "absent"))
    searcher = PerfectPlayer(rng=random.Random(1))
    searcher.start_game(Side.MOUSE, Cell.from_label("C3"))
    found = _random_position(searcher, 12, random.Random(5))
    assert found is not None
    mouse, snake, depth = found
    pool, best = _pool_and_best(searcher, mouse, snake, depth)

    entries: dict[int, int] = {}
    for candidate in pool:
        child_mouse, child_snake = candidate.mouse, candidate.snake
        for a, b in itertools.combinations(
            searcher._empty_indices(child_mouse | child_snake), 2
        ):
            added = (1 << a) | (1 << b)
            gm, gs = child_mouse, child_snake | added
            if searcher._completes(child_snake | added) or searcher._is_cats(gm, gs):
                continue
            entries[canonical_key(gm, gs)] = searcher._negamax(
                gm, gs, Side.MOUSE, depth + 2, -_WIN - 1, _WIN + 1
            )

    table_dir = tmp_path / "grandchildren"
    table_dir.mkdir()
    _write_table(table_dir / "C3.table", {8: sorted(entries.items())})
    monkeypatch.setenv("SNAKES_AND_MICE_TABLE_DIR", str(table_dir))
    looker = PerfectPlayer(rng=random.Random(1))
    looker.start_game(Side.MOUSE, Cell.from_label("C3"))
    assert looker._table is not None and looker._table.covers(8)

    for candidate in pool:
        child_mouse, child_snake = candidate.mouse, candidate.snake
        by_search = searcher._trap_count(child_mouse, child_snake, 0, depth, False)
        by_table = looker._trap_count(child_mouse, child_snake, 0, depth, True)
        assert by_table == by_search


def test_incomplete_table_refuses_to_rank_on_partial_trap_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A table that cannot value every reply must leave the pool alone rather than
    # rank on whatever it happens to hold — the same stance `_choose_from_table` takes.
    table_dir = tmp_path / "partial"
    table_dir.mkdir()
    _write_table(table_dir / "C3.table", {8: [(1, 0)]})
    monkeypatch.setenv("SNAKES_AND_MICE_TABLE_DIR", str(table_dir))
    player = PerfectPlayer(rng=random.Random(1))
    player.start_game(Side.MOUSE, Cell.from_label("C3"))
    found = _random_position(player, 12, random.Random(5))
    assert found is not None
    mouse, snake, depth = found
    pool, best = _pool_and_best(player, mouse, snake, depth)
    assert len(player._most_trapping(list(pool), best, 12, depth)) == len(pool)


def test_liveness_rewards_concentration_over_breadth() -> None:
    # Three of ours in one live line is a win-next-turn threat, because a move places
    # two pieces. The same three spread across three lines threatens nothing, so the
    # key must not score them alike.
    def mask(*labels: str) -> int:
        return sum(1 << (Cell.from_label(x).row * 5 + Cell.from_label(x).col) for x in labels)

    # (A1, B2, C3 would *not* do as the spread case — those three are the start of
    # the main diagonal, so they are concentrated too.)
    concentrated = PerfectPlayer._liveness(mask("A1", "A2", "A3"), 0)
    spread = PerfectPlayer._liveness(mask("A1", "B3", "D5"), 0)
    assert concentrated > spread

    # And a line the opponent has touched is spent: it can never be completed.
    assert PerfectPlayer._liveness(mask("A1", "A2", "A3"), mask("A4")) < concentrated
