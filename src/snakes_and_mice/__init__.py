"""Snakes and Mice — a game engine and pluggable players.

A 5×5 tic-tac-toe variant used as an LLM reasoning benchmark. The rules engine
(:func:`~snakes_and_mice.game.play_game`) is the single source of truth;
players plug in through the :class:`~snakes_and_mice.players.Player` ABC.
"""

from __future__ import annotations

from .board import Board
from .core import BOARD_SIZE, Cell, Move, MoveChoice, Side, TurnOutcome
from .game import play_game
from .faults import (
    IllegalMove,
    MoveUnavailable,
    PlayerFaultReason,
    SnakesAndMiceError,
)
from .players import Player, ScriptedPlayer
from .result import GameResult, PlayerFaultDetail, Termination

__all__ = [
    "BOARD_SIZE",
    "Board",
    "Cell",
    "GameResult",
    "IllegalMove",
    "Move",
    "MoveChoice",
    "MoveUnavailable",
    "Player",
    "PlayerFaultDetail",
    "PlayerFaultReason",
    "ScriptedPlayer",
    "Side",
    "SnakesAndMiceError",
    "Termination",
    "TurnOutcome",
    "play_game",
]
