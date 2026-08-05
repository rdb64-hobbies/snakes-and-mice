"""Pluggable players."""

from __future__ import annotations

from .base import Player
from .scripted import ScriptedPlayer

__all__ = ["Player", "ScriptedPlayer"]
