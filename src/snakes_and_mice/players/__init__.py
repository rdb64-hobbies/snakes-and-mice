"""Pluggable players."""

from __future__ import annotations

from .base import Player
from .random import RandomPlayer
from .scripted import ScriptedPlayer

__all__ = ["Player", "RandomPlayer", "ScriptedPlayer"]
