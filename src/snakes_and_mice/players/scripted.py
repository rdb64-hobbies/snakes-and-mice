"""A scripted player that replays a fixed sequence of choices.

Deterministic and trusted: useful for exercising the engine and rules. It
ignores the moves it observes and simply hands back its next scripted choice
each turn. The script is replayed from the top on every ``start_game``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..core import Move, MoveChoice, Side
from .base import Player


class ScriptedPlayer(Player):
    """Replays a predetermined sequence of :class:`MoveChoice` objects."""

    def __init__(
        self, choices: Iterable[MoveChoice], name: str | None = None
    ) -> None:
        super().__init__(name)
        self._choices: tuple[MoveChoice, ...] = tuple(choices)
        self._index: int = 0
        self._side: Side | None = None

    @classmethod
    def from_moves(
        cls, moves: Sequence[Move], name: str | None = None
    ) -> ScriptedPlayer:
        """Build a scripted player from bare moves (no outcome claims)."""
        return cls([MoveChoice(move) for move in moves], name)

    def start_game(self, side: Side) -> None:
        self._side = side
        self._index = 0

    def observe_move(self, side: Side, move: Move) -> None:
        return None

    def choose_move(self) -> MoveChoice:
        if self._index >= len(self._choices):
            raise RuntimeError(
                f"{self.name}: scripted choices exhausted "
                f"({len(self._choices)} provided)"
            )
        choice: MoveChoice = self._choices[self._index]
        self._index += 1
        return choice
