"""Tests for the console renderer: its ObservationLevel gates what it prints.

The engine always fires every hook (see test_match / test_observer); here we
check that a ConsoleObserver, built at a given level, renders only the detail
that level asks for, and that the match summary tallies faults per side.
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
from snakes_and_mice.console import (
    ConsoleObserver,
    describe_match_result,
    render_fault_tally,
    render_standings,
)
from snakes_and_mice.board import Board
from snakes_and_mice.core import Cell, Move, Side
from snakes_and_mice.faults import PlayerFaultReason
from snakes_and_mice.result import (
    GameResult,
    MatchResult,
    PlayerFaultDetail,
    Termination,
)
from snakes_and_mice.tally import PlayerStanding, StandingsSort


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

    assert "plays" not in out  # the played-move narration is suppressed
    assert "1  2  3  4  5" not in out  # and the board grid is not rendered
    assert "=== Game 1 of 2 ===" in out  # but game headers are shown
    assert "to move…" in out  # an in-place per-turn status line
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
    assert "now in play" in out  # an in-place per-game status line
    assert "🐭 1" in out  # a running scoreboard rides that line
    assert "last game: mouse won" in out  # as does the previous game's outcome
    assert "faults" not in out  # a clean match keeps the scoreboard terse
    assert "Turn 1" not in out  # no per-move narration
    assert "=== Game" not in out  # no per-game headers
    assert "(mouse) wins." not in out  # no per-game results


def test_match_scoreboard_grows_only_for_outcomes_that_occur(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Drive the MATCH-level status line directly with one game of each outcome,
    # then start a further game so its status line reflects the full tally.
    names: dict[Side, str] = {Side.MOUSE: "M", Side.SNAKE: "S"}
    board: Board = Board()
    def fault(side: Side) -> GameResult:
        return GameResult(
            Termination.PLAYER_FAULT,
            fault=PlayerFaultDetail(side, PlayerFaultReason.UNPARSEABLE_OUTPUT),
        )

    results: list[GameResult] = [
        GameResult(Termination.LINE_COMPLETED, winner=Side.MOUSE),
        GameResult(Termination.LINE_COMPLETED, winner=Side.SNAKE),
        GameResult(Termination.CATS_GAME),
        fault(Side.MOUSE),
        fault(Side.MOUSE),  # a second, to show a plural per-side fault count
        GameResult(Termination.ABORTED, error="backend unreachable"),
    ]

    observer: ConsoleObserver = ConsoleObserver(ObservationLevel.MATCH)
    observer.on_match_start(names, len(results) + 1)
    for result in results:
        observer.on_game_start(names, board)
        observer.on_game_end(result)
    observer.on_game_start(names, board)  # its status line shows the full tally
    out: str = capsys.readouterr().out

    # Wins and cat's games always; a side's faults ride its own win token (per
    # side, only once it faulted); aborts append only once they occur.
    assert "🐭 1 (and 2 faults)  🐍 1  🐱 1  aborted 1" in out
    assert "last game: no contest" in out  # the terse phrase for the last game


def _fault(offender: Side, reason: PlayerFaultReason) -> GameResult:
    return GameResult(
        Termination.PLAYER_FAULT,
        fault=PlayerFaultDetail(offender=offender, reason=reason),
    )


def test_match_summary_breaks_down_faults_by_side_and_type() -> None:
    # Three mouse faults (two of one type, one of another) and one snake fault.
    faults: list[GameResult] = [
        _fault(Side.MOUSE, PlayerFaultReason.UNPARSEABLE_OUTPUT),
        _fault(Side.MOUSE, PlayerFaultReason.WRONG_OUTCOME_CLAIM),
        _fault(Side.MOUSE, PlayerFaultReason.UNPARSEABLE_OUTPUT),
        _fault(Side.SNAKE, PlayerFaultReason.CELL_NOT_EMPTY),
    ]
    result: MatchResult = MatchResult(
        names={Side.MOUSE: "Mona", Side.SNAKE: "Sly"},
        num_games=5,
        mouse_wins=1,
        snake_wins=0,
        cats_games=0,
        mouse_faults=3,
        snake_faults=1,
        faults=faults,
        aborted=0,
    )

    summary: str = describe_match_result(result)

    assert "Faults: 3 mouse, 1 snake" in summary
    # Most frequent type first, ties broken by name; only faulted sides appear.
    assert "🐭 mouse: unparseable_output ×2, wrong_outcome_claim ×1" in summary
    assert "🐍 snake: cell_not_empty ×1" in summary


def test_match_summary_omits_fault_breakdown_when_clean() -> None:
    result: MatchResult = MatchResult(
        names={Side.MOUSE: "Mona", Side.SNAKE: "Sly"},
        num_games=2,
        mouse_wins=1,
        snake_wins=1,
        cats_games=0,
        mouse_faults=0,
        snake_faults=0,
        faults=[],
        aborted=0,
    )

    summary: str = describe_match_result(result)

    assert "Faults" not in summary


def test_render_standings_shows_columns_percentages_and_dashes() -> None:
    standings: list[PlayerStanding] = [
        PlayerStanding("opus", played=10, won=6, lost=2, tied=2,
                       faulted=0, opponent_faulted=0),
        PlayerStanding("dud", played=3, won=0, lost=0, tied=0,
                       faulted=3, opponent_faulted=0),
    ]
    table: str = render_standings(standings, StandingsSort.WIN)

    assert "sorted by win%" in table
    assert "Win%" in table and "Fault%" in table
    assert "60.0%" in table  # opus: 6 clean wins of 10
    assert "—" in table  # dud never played a clean game → undefined win/loss rate
    # The header and both players each render on their own line.
    assert "opus" in table and "dud" in table


def test_render_standings_empty_is_a_plain_message() -> None:
    assert render_standings([], StandingsSort.WIN) == "No matches recorded yet."


def test_render_fault_tally_lists_faulty_players_in_the_match_format() -> None:
    standings: list[PlayerStanding] = [
        PlayerStanding("clean", played=4, won=2, lost=2, tied=0,
                       faulted=0, opponent_faulted=0),
        PlayerStanding("messy", played=5, won=1, lost=1, tied=0,
                       faulted=3, opponent_faulted=0,
                       fault_reasons={
                           PlayerFaultReason.UNPARSEABLE_OUTPUT: 2,
                           PlayerFaultReason.CELL_NOT_EMPTY: 1,
                       }),
    ]
    text: str = render_fault_tally(standings)

    assert "Faults by player:" in text
    # Same reason ×n format as a match summary, most frequent first.
    assert "messy: unparseable_output ×2, cell_not_empty ×1" in text
    assert "clean" not in text  # players with no faults are omitted


def test_render_fault_tally_when_no_one_faulted() -> None:
    clean: PlayerStanding = PlayerStanding(
        "clean", played=2, won=1, lost=1, tied=0, faulted=0, opponent_faulted=0
    )
    assert render_fault_tally([clean]) == "No faults recorded."


def test_game_start_announces_the_seed_cell(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The seed line shows the game's actual opening cell, so a randomized (or
    # pinned) opening is visible to anyone watching.
    mouse, snake = _mouse_wins_row_a()
    play_match(
        mouse, snake, 1, ConsoleObserver(ObservationLevel.GAME),
        opening=Cell.from_label("D4"),
    )
    out = capsys.readouterr().out

    assert "Snake seeded at D4." in out


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
