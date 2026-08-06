"""Pluggable players."""

from __future__ import annotations

from .base import Player
from .human import HumanPlayer
from .random import RandomPlayer
from .scripted import ScriptedPlayer

__all__ = ["HumanPlayer", "Player", "RandomPlayer", "ScriptedPlayer"]
