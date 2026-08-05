"""How a game ends: termination reasons, fault detail, and the final result.

The engine reports *facts only*. When a game ends in ``PLAYER_FAULT`` it records
what it observed in :class:`PlayerFaultDetail`; it is up to a player (e.g. an LLM
player) to turn those facts into advice for a future game.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .core import Move, Side, TurnOutcome
from .faults import PlayerFaultReason


class Termination(Enum):
    """Why a game ended."""

    LINE_COMPLETED = "line_completed"  # a player completed a line — normal win
    CATS_GAME = "cats_game"  # every line dead — a draw, no winner
    PLAYER_FAULT = "player_fault"  # a player failed a turn — error, no winner


@dataclass(frozen=True)
class PlayerFaultDetail:
    """Facts about a ``PLAYER_FAULT`` termination, as observed by the engine.

    ``attempted_move`` is set when a well-formed move existed (``CELL_NOT_EMPTY``
    and ``WRONG_OUTCOME_CLAIM``) and ``None`` when the player produced no move
    (``UNPARSEABLE_OUTPUT`` and the structural reasons). ``claimed_outcome`` and
    ``actual_outcome`` are set only for ``WRONG_OUTCOME_CLAIM``.
    """

    offender: Side
    reason: PlayerFaultReason
    attempted_move: Move | None = None
    claimed_outcome: TurnOutcome | None = None
    actual_outcome: TurnOutcome | None = None


@dataclass(frozen=True)
class GameResult:
    """The outcome of a game."""

    termination: Termination
    winner: Side | None = None  # None for both CATS_GAME and PLAYER_FAULT
    fault: PlayerFaultDetail | None = None  # set iff termination == PLAYER_FAULT
