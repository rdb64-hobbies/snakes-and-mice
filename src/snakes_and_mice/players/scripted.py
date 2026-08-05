"""A scripted player that replays a fixed sequence of decisions.

Deterministic and trusted: useful for exercising the engine and rules. It
ignores the moves it observes and simply hands back its next scripted decision
each turn. The script is replayed from the top on every ``start_game``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..core import Decision, Move, Side
from .base import Player


class ScriptedPlayer(Player):
    """Replays a predetermined sequence of :class:`Decision` objects."""

    def __init__(
        self, decisions: Iterable[Decision], name: str | None = None
    ) -> None:
        super().__init__(name)
        self._decisions: tuple[Decision, ...] = tuple(decisions)
        self._index = 0
        self._side: Side | None = None

    @classmethod
    def from_moves(
        cls, moves: Sequence[Move], name: str | None = None
    ) -> ScriptedPlayer:
        """Build a scripted player from bare moves (no outcome claims)."""
        return cls([Decision(move) for move in moves], name)

    def start_game(self, side: Side) -> None:
        self._side = side
        self._index = 0

    def observe_move(self, side: Side, move: Move) -> None:
        return None

    def choose_move(self) -> Decision:
        if self._index >= len(self._decisions):
            raise RuntimeError(
                f"{self.name}: scripted decisions exhausted "
                f"({len(self._decisions)} provided)"
            )
        decision = self._decisions[self._index]
        self._index += 1
        return decision
