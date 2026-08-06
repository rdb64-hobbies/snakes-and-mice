"""The observer abstraction: watch a match unfold without influencing it.

An :class:`Observer` is a passive spectator. The engine drives it in lockstep
alongside the players, handing it the authoritative board (read-only) after each
event, but an observer never affects play. The engine is level-blind: it always
fires every hook. Each observer is constructed with an :class:`ObservationLevel`
and decides for itself how much of that event stream to act on — so *how much to
watch* is a property of the watcher, not of the game.
"""

from __future__ import annotations

from enum import IntEnum

from .board import Board
from .core import Move, Side, TurnOutcome
from .result import GameResult, MatchResult


class ObservationLevel(IntEnum):
    """How much of a match an observer acts on, coarsest to finest.

    The levels nest: each covers everything the coarser ones do, plus more.
    ``MATCH`` is just the match boundaries, ``GAME`` adds each game's start and
    end, and ``MOVE`` adds every move. An observer stores the level it was built
    with and gates its own output accordingly; the engine itself is level-blind
    and always fires every hook. Members are ordered so ``self.level >= GAME``
    and the like read naturally.
    """

    MATCH = 0  # match start/end only
    GAME = 1  # + each game's start and end
    MOVE = 2  # + every move


class Observer:
    """A spectator the engine drives alongside the players, so a match can be
    watched or logged as it unfolds.

    Unlike a :class:`~snakes_and_mice.players.Player`, an observer never
    influences play: it only receives the authoritative board after each event.
    The ``Board`` handed to a hook is the engine's live board — treat it as
    read-only. Every hook defaults to a no-op, so a subclass overrides only what
    it needs, and consults :attr:`level` to decide how much to act on.
    """

    def __init__(self, level: ObservationLevel = ObservationLevel.MOVE) -> None:
        self.level: ObservationLevel = level

    def on_match_start(self, names: dict[Side, str], num_games: int) -> None:
        """A match of ``num_games`` games is about to begin, with the given
        side→name mapping (fixed for the whole match)."""
        return None

    def on_match_end(self, result: MatchResult) -> None:
        """The match has ended with ``result`` (the tallies and any faults)."""
        return None

    def on_game_start(self, names: dict[Side, str], board: Board) -> None:
        """The game is about to begin, with the given side→name mapping and the
        seeded starting board."""
        return None

    def on_game_end(self, result: GameResult) -> None:
        """The game has ended with ``result``."""
        return None

    def on_move_start(self, side: Side, board: Board) -> None:
        """``side`` is about to be asked for its move; ``board`` is the current
        pre-move state.

        Fires at the start of every turn, before ``choose_move``. This matters
        when producing a move is slow — e.g. an LLM player querying a model — so
        a watcher can show that the turn has begun and who is thinking, well
        before :meth:`on_move_end` reports what they played.
        """
        return None

    def on_move_end(
        self, side: Side, move: Move, board: Board, outcome: TurnOutcome
    ) -> None:
        """``side`` just played ``move``, yielding ``outcome``; ``board`` now
        reflects it. Fires once per accepted move (including the terminal one),
        and not at all for a turn that ends in a fault."""
        return None
