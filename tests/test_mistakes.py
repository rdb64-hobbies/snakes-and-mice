"""Tests for mistake flagging (§5).

The observer is driven by calling its hooks exactly as the engine does, against
hand-built boards and a tiny synthetic table, so nothing here needs a real match,
a model, or the installed solver output. What is under test is the wiring — the
right board captured at the right moment, the right sides graded, the
before/after diff and its hard/soft split, and what reaches the console versus
the file. Whether ``evaluate`` itself is correct is ``test_perfect_player``'s job.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from snakes_and_mice import Cell, Move, Side, TurnOutcome
from snakes_and_mice.board import Board
from snakes_and_mice.cli_common import (
    BUILTIN_KINDS,
    NON_GRADABLE_KINDS,
    gradable_sides,
    make_observer,
)
from snakes_and_mice.console import ConsoleObserver
from snakes_and_mice.mistakes import Mistake, MistakeObserver
from snakes_and_mice.observer import BroadcastObserver
from snakes_and_mice.players.symmetry import canonical_key
from table_helpers import write_table

NAMES: dict[Side, str] = {Side.MOUSE: "qwen", Side.SNAKE: "Perseus"}

# The position every test moves from: Mouse holds four of row A (A5 would win
# outright), Snake holds three of row B, and C3 is the seed. 8 pieces, so 17
# empty cells; playing two more leaves 15.
BEFORE_EMPTIES: int = 17
AFTER_EMPTIES: int = 15
MOUSE_CELLS: tuple[str, ...] = ("A1", "A2", "A3", "A4")
SNAKE_CELLS: tuple[str, ...] = ("B1", "B2", "B3")
PLAYED: tuple[str, ...] = ("C1", "D1")  # legal, but not the win sitting on A5


def _bit(label: str) -> int:
    cell = Cell.from_label(label)
    return 1 << (cell.row * 5 + cell.col)


def _mask(labels: tuple[str, ...]) -> int:
    return sum(_bit(label) for label in labels)


def _position() -> Board:
    board = Board(Cell.from_label("C3"))
    for label in MOUSE_CELLS:
        board.place(Cell.from_label(label), Side.MOUSE)
    for label in SNAKE_CELLS:
        board.place(Cell.from_label(label), Side.SNAKE)
    return board


def _install_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, before: int, after: int
) -> None:
    """A table valuing just this position and the one the played move reaches.

    ``before`` is the value for Mouse (to move) beforehand; ``after`` the value
    for Snake (to move) once Mouse has played, which the observer negates back
    into Mouse's terms.
    """
    mouse = _mask(MOUSE_CELLS)
    snake = _mask(SNAKE_CELLS) | _bit("C3")
    directory = tmp_path / "table"
    directory.mkdir(exist_ok=True)
    write_table(
        directory / "C3.table",
        {
            BEFORE_EMPTIES: [(canonical_key(mouse, snake), before)],
            AFTER_EMPTIES: [(canonical_key(mouse | _mask(PLAYED), snake), after)],
        },
    )
    monkeypatch.setenv("SNAKES_AND_MICE_TABLE_DIR", str(directory))


def _play(
    observer: MistakeObserver, outcome: TurnOutcome = TurnOutcome.IN_PLAY
) -> None:
    """Drive one Mouse turn through the observer the way the engine would."""
    board = _position()
    observer.on_game_start(NAMES, board)
    observer.on_move_start(Side.MOUSE, board)
    for label in PLAYED:
        board.place(Cell.from_label(label), Side.MOUSE)
    move = Move.of(*(Cell.from_label(label) for label in PLAYED))
    observer.on_move_end(Side.MOUSE, move, board, outcome)


def test_flags_a_move_that_throws_a_won_position_away(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_table(monkeypatch, tmp_path, before=997, after=900)
    observer = MistakeObserver(frozenset({Side.MOUSE}))
    _play(observer)

    assert len(observer.mistakes) == 1
    mistake: Mistake = observer.mistakes[0]
    assert mistake.side is Side.MOUSE
    assert mistake.player == "qwen"
    assert mistake.value_before == 997
    # Snake wins by 900 there, which is -900 to Mouse.
    assert mistake.value_after == -900
    assert mistake.hard  # a win turned into a loss crosses two boundaries
    assert mistake.game == 1 and mistake.turn == 1
    assert mistake.seed == Cell.from_label("C3")
    # The board carried is the one it moved *from*, not the one it produced.
    assert mistake.board_before.is_empty(Cell.from_label(PLAYED[0]))

    printed = capsys.readouterr().out
    assert "Mistake" in printed and "qwen" in printed
    assert "turning a win into a loss" in printed
    assert "🐭" in printed  # the position it moved from is rendered


def test_ignores_sides_it_was_not_asked_to_grade(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_table(monkeypatch, tmp_path, before=997, after=900)
    observer = MistakeObserver(frozenset({Side.SNAKE}))
    _play(observer)  # Mouse moves; only Snake is graded

    assert observer.mistakes == []
    assert observer.graded_moves == {Side.SNAKE: 0}


def test_never_flags_a_win(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Nothing beats winning outright, so a winning move cannot have given
    # anything away -- whatever the table would say about the position.
    _install_table(monkeypatch, tmp_path, before=997, after=900)
    observer = MistakeObserver(frozenset({Side.MOUSE}))
    _play(observer, outcome=TurnOutcome.WIN)

    assert observer.mistakes == []
    assert observer.graded_moves[Side.MOUSE] == 0


def test_a_cats_game_is_scored_as_a_draw_not_looked_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A drawn ending is worth 0 to both sides, so throwing a win away into one
    # is still a mistake -- and needs no value for the final position at all.
    _install_table(monkeypatch, tmp_path, before=997, after=12345)
    observer = MistakeObserver(frozenset({Side.MOUSE}))
    _play(observer, outcome=TurnOutcome.CATS_GAME)

    assert len(observer.mistakes) == 1
    assert observer.mistakes[0].value_after == 0
    assert observer.mistakes[0].hard


def test_soft_mistakes_are_recorded_but_not_called_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Still winning, just more slowly: real, worth recording, but not worth
    # interrupting a watcher over (§5).
    _install_table(monkeypatch, tmp_path, before=900, after=-800)
    observer = MistakeObserver(frozenset({Side.MOUSE}))
    _play(observer)

    assert len(observer.mistakes) == 1
    mistake = observer.mistakes[0]
    assert mistake.value_before == 900 and mistake.value_after == 800
    assert not mistake.hard
    assert "Mistake" not in capsys.readouterr().out


def test_an_optimal_move_is_not_a_mistake(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The "before" value *is* the best move's value, so a move that achieves it
    # gave up nothing. -997 for Snake afterwards is +997 for Mouse: unchanged.
    _install_table(monkeypatch, tmp_path, before=997, after=-997)
    observer = MistakeObserver(frozenset({Side.MOUSE}))
    _play(observer)

    assert observer.mistakes == []
    assert observer.graded_moves[Side.MOUSE] == 1  # graded, and found clean


def test_records_each_mistake_as_a_self_contained_json_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_table(monkeypatch, tmp_path, before=997, after=900)
    path = tmp_path / "mistakes.jsonl"
    observer = MistakeObserver(frozenset({Side.MOUSE}), report=False, path=path)
    _play(observer)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["side"] == "MOUSE" and record["player"] == "qwen"
    assert record["move"] == list(PLAYED)
    assert record["value_before"] == 997 and record["value_after"] == -900
    assert record["hard"] is True
    assert record["seed"] == "C3"
    # The position is carried as occupancy, so a reader needs nothing but this
    # line to rebuild it -- no move history, no seed replay.
    assert sorted(record["board_before"]["mouse"]) == sorted(MOUSE_CELLS)
    assert sorted(record["board_before"]["snake"]) == sorted(
        (*SNAKE_CELLS, "C3")
    )


def test_appends_across_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The file accumulates, like the tournament results file (§6): collecting
    # mistakes is done over many runs, not one.
    _install_table(monkeypatch, tmp_path, before=997, after=900)
    path = tmp_path / "mistakes.jsonl"
    for _ in range(2):
        observer = MistakeObserver(frozenset({Side.MOUSE}), report=False, path=path)
        _play(observer)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_a_game_with_no_table_is_left_ungraded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Valuing the opening without a table means searching the widest ply in the
    # game. A diagnostic must not stall the match it is annotating, so the game
    # goes ungraded instead.
    monkeypatch.setenv("SNAKES_AND_MICE_TABLE_DIR", str(tmp_path / "absent"))
    observer = MistakeObserver(frozenset({Side.MOUSE}))
    _play(observer)

    assert observer.mistakes == []
    assert observer.graded_moves[Side.MOUSE] == 0


# --- Composition and CLI wiring ---------------------------------------------


def test_gradable_sides_excludes_only_the_non_gradable_kinds() -> None:
    assert gradable_sides({Side.MOUSE: "qwen", Side.SNAKE: "perfect"}) == frozenset(
        {Side.MOUSE}
    )
    assert gradable_sides({Side.MOUSE: "random", Side.SNAKE: "human"}) == frozenset()
    assert gradable_sides({Side.MOUSE: "qwen", Side.SNAKE: "opus"}) == frozenset(
        {Side.MOUSE, Side.SNAKE}
    )


def test_gradability_is_not_the_same_question_as_how_to_build_a_player() -> None:
    # The two sets are equal today and are still kept apart, because they answer
    # different questions and are expected to diverge: a built-in player that is
    # neither optimal nor aimless -- the RL player (§3) -- would belong to
    # BUILTIN_KINDS while remaining very much worth grading. Anything not named
    # as non-gradable grades, whether or not it is a roster name.
    assert NON_GRADABLE_KINDS == BUILTIN_KINDS  # for now
    assert gradable_sides({Side.MOUSE: "rl"}) == frozenset({Side.MOUSE})


def test_make_observer_composes_watching_with_flagging() -> None:
    both = make_observer(
        "move", mistake_sides=frozenset({Side.MOUSE}), flag_mistakes=True
    )
    assert isinstance(both, BroadcastObserver)
    assert [type(o) for o in both.observers] == [ConsoleObserver, MistakeObserver]

    # Flagging is orthogonal to the level, so it survives `--watch none`...
    alone = make_observer(
        "none", mistake_sides=frozenset({Side.MOUSE}), flag_mistakes=True
    )
    assert isinstance(alone, MistakeObserver)

    # ...and asking for neither still means no observer at all.
    assert make_observer("none") is None
    assert isinstance(make_observer("game"), ConsoleObserver)

    # A file alone is reason enough to grade, with nothing printed live.
    recording = make_observer(
        "none", mistake_sides=frozenset({Side.MOUSE}), mistakes_path=Path("m.jsonl")
    )
    assert isinstance(recording, MistakeObserver)

    # Nothing to grade: the flag cannot conjure a side worth watching, so the
    # console observer is left to stand on its own.
    nothing_graded = make_observer(
        "game", mistake_sides=frozenset(), flag_mistakes=True
    )
    assert isinstance(nothing_graded, ConsoleObserver)
