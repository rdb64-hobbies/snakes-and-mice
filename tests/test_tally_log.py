"""Tests for the `--log-llm` outcome reader (``tools/tally_log.py``).

The reader inverts the prose the LLM player writes to its model at the end of each
game, so the thing worth testing is the round trip: every termination and every
:class:`PlayerFaultReason` composed by the player must be recovered exactly. The
motivating bug was a reader that tested for "cat" before testing for a fault, which
silently scored ``WRONG_PIECE_COUNT`` and ``WRONG_OUTCOME_CLAIM`` — whose advice
mentions a cat's game — as draws.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import tally_log  # noqa: E402

from snakes_and_mice import (  # noqa: E402
    GameResult,
    Move,
    PlayerFaultDetail,
    PlayerFaultReason,
    Side,
    Termination,
    TurnOutcome,
)
from snakes_and_mice.players import LLMPlayer  # noqa: E402


def _describe(result: GameResult, side: Side = Side.MOUSE) -> str:
    """The message the player would send its model for ``result``."""
    player: LLMPlayer = LLMPlayer.__new__(LLMPlayer)
    player._side = side
    return player._describe_end(result)


@pytest.mark.parametrize(
    "termination,winner,expected",
    [
        (Termination.LINE_COMPLETED, Side.MOUSE, tally_log.WON),
        (Termination.LINE_COMPLETED, Side.SNAKE, tally_log.LOST),
        (Termination.CATS_GAME, None, tally_log.DREW),
        (Termination.ABORTED, None, tally_log.ABORTED),
    ],
)
def test_plain_outcomes_round_trip(
    termination: Termination, winner: Side | None, expected: str
) -> None:
    result: GameResult = GameResult(termination=termination, winner=winner, fault=None)

    outcome, reason = tally_log.classify(_describe(result))

    assert outcome == expected
    assert reason is None


@pytest.mark.parametrize("reason", list(PlayerFaultReason))
def test_every_fault_reason_round_trips(reason: PlayerFaultReason) -> None:
    # The bug this guards against: a fault whose advice mentions a cat's game read
    # as a draw. Every reason must come back as a fault, with its own reason.
    fault: PlayerFaultDetail = PlayerFaultDetail(
        offender=Side.MOUSE,
        reason=reason,
        attempted_move=Move.from_labels("A1", "A2"),
        claimed_outcome=(
            TurnOutcome.CATS_GAME
            if reason is PlayerFaultReason.WRONG_OUTCOME_CLAIM
            else None
        ),
        actual_outcome=(
            TurnOutcome.IN_PLAY
            if reason is PlayerFaultReason.WRONG_OUTCOME_CLAIM
            else None
        ),
    )
    result: GameResult = GameResult(
        termination=Termination.PLAYER_FAULT, winner=None, fault=fault
    )

    outcome, recovered = tally_log.classify(_describe(result))

    assert outcome == "faulted", f"{reason.name} was read as {outcome!r}"
    assert recovered is reason


def test_opponent_fault_is_not_scored_against_the_logged_player() -> None:
    fault: PlayerFaultDetail = PlayerFaultDetail(
        offender=Side.SNAKE,
        reason=PlayerFaultReason.CELL_NOT_EMPTY,
        attempted_move=Move.from_labels("B2", "B3"),
    )
    result: GameResult = GameResult(
        termination=Termination.PLAYER_FAULT, winner=None, fault=fault
    )

    outcome, reason = tally_log.classify(_describe(result))

    assert outcome == tally_log.OPPONENT_FAULTED
    assert reason is None


def test_unrelated_text_is_not_read_as_an_outcome() -> None:
    # Opponent-move relays and the seed announcement share the same user turn as a
    # game-over message; only the latter may be counted.
    for block in (
        "Your opponent (snake) played E3 C1.",
        "A new game begins. You are playing mouse. The snake is seeded at B3.",
        "It is your turn (mouse). Choose your move.",
    ):
        assert tally_log.classify(block)[0] == tally_log.UNRECOGNIZED


def test_outcomes_reads_a_thread(tmp_path: Path) -> None:
    # One user turn carries the previous game's outcome plus the next game's setup,
    # joined by blank lines — exactly the shape the player flushes.
    drawn: str = _describe(
        GameResult(termination=Termination.CATS_GAME, winner=None, fault=None)
    )
    turn: str = f"{drawn}\n\nA new game begins. You are playing mouse.\n\nYour turn."
    log: Path = tmp_path / "thread.json"
    log.write_text(
        json.dumps(
            [{"kind": "request", "parts": [{"part_kind": "user-prompt", "content": turn}]}]
        )
    )

    assert tally_log.outcomes(log) == [(tally_log.DREW, None)]
