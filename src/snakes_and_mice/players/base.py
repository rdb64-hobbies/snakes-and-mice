"""The ``Player`` abstract base class.

A player owns its own view of the game. The engine drives it through one game's
lifecycle:

1. :meth:`start_game` — told which side it is playing (a player instance may play
   many games and either side across them).
2. :meth:`choose_move` / :meth:`observe_move` — alternating until the game ends.
   ``choose_move`` may raise
   :class:`~snakes_and_mice.faults.MoveUnavailable` to concede it cannot move.
3. :meth:`end_game` — told the result, including full fault detail, so a player
   can learn across games. Default is a no-op.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..core import Cell, Move, MoveChoice, Side
from ..result import GameResult


class Player(ABC):
    """Base class every player implementation extends."""

    def __init__(self, name: str | None = None) -> None:
        self.name: str = name or type(self).__name__

    @abstractmethod
    def start_game(self, side: Side, seed: Cell) -> None:
        """Begin a new game playing ``side``, with the snake seeded on ``seed``.

        Reset any per-game state here. ``seed`` is the snake's starting cell for
        this game, so a player that tracks its own board can seed it correctly.
        """

    @abstractmethod
    def observe_move(self, side: Side, move: Move) -> None:
        """Observe a move just played by ``side`` (including the player's own)."""

    @abstractmethod
    def choose_move(self) -> MoveChoice:
        """Choose this turn's move.

        Raise :class:`~snakes_and_mice.faults.MoveUnavailable` to concede that no
        move can be produced (recorded as a ``PLAYER_FAULT``).
        """

    def assume_own_move(self, side: Side, move: Move) -> None:
        """Observe a move attributed to **this** player that it did not choose.

        Only used to *construct* a position — replaying a move sequence to reach
        some board of interest and then taking a single :meth:`choose_move`
        (SPEC-rl-player.md, "The loop", step 3). It never fires during a real
        game, where every own move came from ``choose_move`` already.

        The default delegates to :meth:`observe_move`, which is right for any
        player that rebuilds the board purely from what it is told, whoever
        moved — every mechanical player here. A player that instead infers its
        own moves from its own past output must override this, or a constructed
        position will be missing all of its own pieces.
        """
        self.observe_move(side, move)

    def end_game(self, result: GameResult) -> None:
        """Be told how the game ended. Default: do nothing."""
        return None
