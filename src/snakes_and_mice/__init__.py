"""Snakes and Mice — a game engine and pluggable players.

A 5×5 tic-tac-toe variant used as an LLM reasoning benchmark. The rules engine
(:func:`~snakes_and_mice.game.play_game`) is the single source of truth;
players plug in through the :class:`~snakes_and_mice.players.Player` ABC.
"""

from __future__ import annotations

from .board import Board
from .core import BOARD_SIZE, Cell, Move, MoveChoice, Side, TurnOutcome
from .faults import (
    IllegalMove,
    MoveUnavailable,
    PlayerFaultReason,
    PlayerUnavailable,
    SnakesAndMiceError,
)
from .game import play_game
from .match import play_match
from .observer import ObservationLevel, Observer
from .players import (
    HumanPlayer,
    LLMMove,
    LLMPlayer,
    PerfectPlayer,
    Player,
    RandomPlayer,
    ScriptedPlayer,
)
from .result import GameResult, MatchResult, PlayerFaultDetail, Termination

__all__ = [
    "BOARD_SIZE",
    "Board",
    "Cell",
    "GameResult",
    "HumanPlayer",
    "IllegalMove",
    "LLMMove",
    "LLMPlayer",
    "MatchResult",
    "Move",
    "MoveChoice",
    "MoveUnavailable",
    "ObservationLevel",
    "Observer",
    "PerfectPlayer",
    "Player",
    "PlayerFaultDetail",
    "PlayerFaultReason",
    "PlayerUnavailable",
    "RandomPlayer",
    "ScriptedPlayer",
    "Side",
    "SnakesAndMiceError",
    "Termination",
    "TurnOutcome",
    "play_game",
    "play_match",
]
