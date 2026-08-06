"""Tests for the console renderer: its ObservationLevel gates what it prints.

The engine always fires every hook (see test_match / test_observer); here we
check that a ConsoleObserver, built at a given level, renders only the detail
that level asks for.
"""

from __future__ import annotations

import pytest

from snakes_and_mice import (
    MoveChoice,
    ObservationLevel,
    ScriptedPlayer,
    TurnOutcome,
    play_match,
)
from snakes_and_mice.core import Move
from snakes_and_mice.cli import ConsoleObserver


def _mouse_wins_row_a() -> tuple[ScriptedPlayer, ScriptedPlayer]:
    mouse = ScriptedPlayer(
        [
            MoveChoice(Move.from_labels("A1", "A2")),
            MoveChoice(Move.from_labels("A3", "A4")),
            MoveChoice(Move.from_labels("B1", "A5"), TurnOutcome.WIN),
        ],
        name="Mouse",
    )
    snake = ScriptedPlayer.from_moves(
        [Move.from_labels("E1", "E2"), Move.from_labels("E3", "E4")],
        name="Snake",
    )
    return mouse, snake


def test_watch_move_shows_every_turn(capsys: pytest.CaptureFixture[str]) -> None:
    mouse, snake = _mouse_wins_row_a()
    play_match(mouse, snake, 1, ConsoleObserver(ObservationLevel.MOVE))
    out = capsys.readouterr().out

    assert "Turn 1" in out  # per-move narration is shown
    assert "1  2  3  4  5" in out  # the board grid is rendered
    assert "(mouse) wins." in out  # and the game result


def test_watch_game_shows_boundaries_not_moves(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mouse, snake = _mouse_wins_row_a()
    play_match(mouse, snake, 2, ConsoleObserver(ObservationLevel.GAME))
    out = capsys.readouterr().out

    assert "Turn 1" not in out  # per-move narration suppressed
    assert "1  2  3  4  5" not in out  # and the board grid is not rendered
    assert "=== Game 1 of 2 ===" in out  # but game headers are shown
    assert "(mouse) wins." in out  # and per-game results
    assert "Match complete" in out  # and the closing tally


def test_watch_match_shows_only_banner_and_tally(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mouse, snake = _mouse_wins_row_a()
    play_match(mouse, snake, 2, ConsoleObserver(ObservationLevel.MATCH))
    out = capsys.readouterr().out

    assert "🐭 Mouse:" in out  # opening banner
    assert "Match complete" in out  # closing tally
    assert "Turn 1" not in out  # no per-move narration
    assert "=== Game" not in out  # no per-game headers
    assert "(mouse) wins." not in out  # no per-game results


def test_single_game_omits_match_scaffolding(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A one-game match reads as a single game: no "Match: N games" line and no
    # match-summary tally, just the game itself.
    mouse, snake = _mouse_wins_row_a()
    play_match(mouse, snake, 1, ConsoleObserver(ObservationLevel.MOVE))
    out = capsys.readouterr().out

    assert "Match:" not in out
    assert "Match complete" not in out
    assert "(mouse) wins." in out
