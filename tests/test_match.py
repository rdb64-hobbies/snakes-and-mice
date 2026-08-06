"""Tests for play_match: tallies, invariants, instance reuse, and observation."""

from __future__ import annotations

import random

import pytest

from snakes_and_mice import (
    Board,
    GameResult,
    MatchResult,
    Move,
    MoveChoice,
    Observer,
    RandomPlayer,
    ScriptedPlayer,
    Side,
    Termination,
    TurnOutcome,
    play_match,
)


def _mouse_wins_row_a() -> tuple[ScriptedPlayer, ScriptedPlayer]:
    """A scripted pair where the mouse completes row A in three moves. The
    scripts are replayed from the top each game, so the pair can play a match."""
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


def _mouse_faults() -> tuple[ScriptedPlayer, ScriptedPlayer]:
    """A scripted pair where the mouse faults on its first move (B3 is seeded)."""
    mouse = ScriptedPlayer([MoveChoice(Move.from_labels("B3", "A1"))], name="Mouse")
    snake = ScriptedPlayer([], name="Snake")
    return mouse, snake


def test_match_tallies_wins() -> None:
    mouse, snake = _mouse_wins_row_a()
    result = play_match(mouse, snake, 3)

    assert result.num_games == 3
    assert result.mouse_wins == 3
    assert result.snake_wins == 0
    assert result.cats_games == 0
    assert result.mouse_faults == 0
    assert result.snake_faults == 0
    assert result.faults == []
    assert result.names == {Side.MOUSE: "Mouse", Side.SNAKE: "Snake"}


def test_match_tallies_faults_and_keeps_faulted_games() -> None:
    mouse, snake = _mouse_faults()
    result = play_match(mouse, snake, 2)

    assert result.mouse_faults == 2
    assert result.snake_faults == 0
    assert result.mouse_wins == result.snake_wins == result.cats_games == 0
    # Only faulted games keep their full result, one per fault.
    assert len(result.faults) == 2
    for game in result.faults:
        assert game.termination is Termination.PLAYER_FAULT
        assert game.fault is not None
        assert game.fault.offender is Side.MOUSE


def test_match_result_invariants_hold() -> None:
    # Random players never fault, and the five tallies always partition the games.
    mouse = RandomPlayer("Mouse", random.Random(1))
    snake = RandomPlayer("Snake", random.Random(2))
    result = play_match(mouse, snake, 20)

    assert (
        result.mouse_wins
        + result.snake_wins
        + result.cats_games
        + result.mouse_faults
        + result.snake_faults
        == result.num_games
        == 20
    )
    assert result.mouse_faults + result.snake_faults == len(result.faults) == 0


def test_match_requires_at_least_one_game() -> None:
    mouse, snake = _mouse_wins_row_a()
    with pytest.raises(ValueError):
        play_match(mouse, snake, 0)


class _CountingPlayer(RandomPlayer):
    """A random player that records the side it is dealt at each game's start."""

    def __init__(self, name: str, rng: random.Random) -> None:
        super().__init__(name, rng)
        self.sides_seen: list[Side] = []
        self.games_ended: int = 0

    def start_game(self, side: Side) -> None:
        self.sides_seen.append(side)
        super().start_game(side)

    def end_game(self, result: GameResult) -> None:
        self.games_ended += 1
        super().end_game(result)


def test_match_reuses_instances_with_fixed_sides() -> None:
    # The same two instances play every game, each pinned to one side — this is
    # what lets a learning player accumulate experience across a match.
    mouse = _CountingPlayer("Mouse", random.Random(5))
    snake = _CountingPlayer("Snake", random.Random(6))
    play_match(mouse, snake, 4)

    assert mouse.sides_seen == [Side.MOUSE] * 4
    assert snake.sides_seen == [Side.SNAKE] * 4
    assert mouse.games_ended == snake.games_ended == 4


class _MatchRecorder(Observer):
    """Records the sequence of hook kinds the runner fires."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []
        self.match_result: MatchResult | None = None

    def on_match_start(self, names: dict[Side, str], num_games: int) -> None:
        self.events.append("match_start")

    def on_match_end(self, result: MatchResult) -> None:
        self.events.append("match_end")
        self.match_result = result

    def on_game_start(self, names: dict[Side, str], board: Board) -> None:
        self.events.append("game_start")

    def on_move_start(self, side: Side, board: Board) -> None:
        self.events.append("move_start")

    def on_move_end(
        self, side: Side, move: Move, board: Board, outcome: TurnOutcome
    ) -> None:
        self.events.append("move_end")

    def on_game_end(self, result: GameResult) -> None:
        self.events.append("game_end")


def test_match_fires_complete_event_stream() -> None:
    # The runner is level-blind: it always fires the full stream — match
    # boundaries wrapping each game's start/end, which wrap that game's moves.
    # An observer decides for itself how much of this to act on.
    recorder = _MatchRecorder()
    mouse, snake = _mouse_wins_row_a()
    result = play_match(mouse, snake, 2, recorder)

    assert recorder.match_result is result
    assert recorder.events[0] == "match_start"
    assert recorder.events[-1] == "match_end"
    # Two games, each five accepted moves, each wrapped in its own boundaries.
    assert recorder.events.count("game_start") == 2
    assert recorder.events.count("game_end") == 2
    assert recorder.events.count("move_start") == recorder.events.count(
        "move_end"
    ) == 10
    # The first game's boundaries enclose its moves, before the second begins.
    assert recorder.events[1] == "game_start"
    first_game_end: int = recorder.events.index("game_end")
    assert recorder.events[2:first_game_end] == ["move_start", "move_end"] * 5
