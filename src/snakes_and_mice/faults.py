"""Fault vocabulary and exceptions.

A *fault* is any way a player fails to complete a valid turn. The engine records
the reason a game ended in ``PLAYER_FAULT`` using :class:`PlayerFaultReason`.

Two exceptions carry a reason so that construction- and apply-time errors map
cleanly onto that vocabulary:

* :class:`IllegalMove` — a move that violates the rules. Raised at *construction*
  time by validated value types (off-board / duplicate / wrong count) and at
  *apply* time by the engine (a target cell is not empty).
* :class:`MoveUnavailable` — a player could not produce a move at all (e.g. an
  LLM emitted output we cannot parse into a move).
"""

from __future__ import annotations

from enum import Enum


class PlayerFaultReason(Enum):
    """Why a player failed to complete a valid turn."""

    # Structural — caught when a Cell/Move value is constructed. An LLM player
    # catches these and reports them; a trusted player never triggers them.
    OFF_BOARD = "off_board"
    DUPLICATE_CELLS = "duplicate_cells"
    WRONG_PIECE_COUNT = "wrong_piece_count"

    # Engine-detected, stateful — the move is well-formed but a target cell is
    # already occupied. The offending move is known (attempted_move is set).
    CELL_NOT_EMPTY = "cell_not_empty"

    # Player-reported — the player could not produce a move at all
    # (attempted_move is None).
    UNPARSEABLE_OUTPUT = "unparseable_output"

    # Engine-detected misread — the move is legal but the player's self-assessed
    # outcome disagrees with ground truth (claimed/actual outcomes are set).
    WRONG_OUTCOME_CLAIM = "wrong_outcome_claim"


class SnakesAndMiceError(Exception):
    """Base class for all game errors."""


class IllegalMove(SnakesAndMiceError):
    """A move that violates the rules of the game.

    Carries the :class:`PlayerFaultReason` so callers (notably an LLM player that
    parses model output into moves) can catch it and report the specific fault.
    """

    def __init__(self, reason: PlayerFaultReason, message: str | None = None) -> None:
        self.reason = reason
        super().__init__(message or reason.name)


class MoveUnavailable(SnakesAndMiceError):
    """A player was unable to produce a move for its turn.

    Raised by a player (not the engine) when it cannot supply a legal move —
    typically an LLM whose output could not be parsed at all.
    """

    def __init__(self, reason: PlayerFaultReason, message: str | None = None) -> None:
        self.reason = reason
        super().__init__(message or reason.name)
