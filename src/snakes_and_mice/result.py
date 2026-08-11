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
    ABORTED = "aborted"  # a no-contest: an environmental failure (e.g. the model
    # backend stayed unreachable after retries) voided the game. Not scored to
    # either side and not a fault; the rest of the match plays on.


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
    winner: Side | None = None  # set iff termination == LINE_COMPLETED
    fault: PlayerFaultDetail | None = None  # set iff termination == PLAYER_FAULT
    error: str | None = None  # a short cause description, set iff ABORTED


@dataclass(frozen=True)
class MatchResult:
    """The outcome of a match: two fixed players over a sequence of games.

    The tallies partition the games: every game is a mouse win, a snake win, a
    cat's game, a fault charged to exactly one side, or a no-contest abort — so
    ``mouse_wins + snake_wins + cats_games + mouse_faults + snake_faults +
    aborted == num_games``. Only faulted games keep their full :class:`GameResult`
    (in :attr:`faults`, with the fault detail); the rest are captured by the
    counts alone, hence ``mouse_faults + snake_faults == len(faults)``. Aborted
    games are counted but not otherwise recorded: they belong to neither player.
    """

    names: dict[Side, str]  # who played each side, fixed for the whole match
    num_games: int
    mouse_wins: int
    snake_wins: int
    cats_games: int
    mouse_faults: int  # games the Mouse-side player faulted
    snake_faults: int  # games the Snake-side player faulted
    faults: list[GameResult]  # the faulted games' full results
    aborted: int  # no-contest games (ABORTED) — charged to neither side
