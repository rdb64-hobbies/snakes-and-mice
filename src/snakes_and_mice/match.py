"""A match: two fixed players over a sequence of games, with tallied results.

A match keeps the *same* two :class:`~snakes_and_mice.players.Player` instances
for every game, each on a fixed side. Reusing the instances is deliberate: a
player that learns across games (e.g. an LLM player digesting each
:class:`~snakes_and_mice.result.GameResult` in ``end_game``) carries that
experience forward through the match. To compare two players with each getting
to play both sides, play two matches with the sides swapped.
"""

from __future__ import annotations

import random

from .core import BOARD_SIZE, Cell, Side
from .game import play_game
from .observer import Observer
from .players.base import Player
from .result import GameResult, MatchResult, Termination

# Every cell is a legal opening seed for the snake (the board is otherwise empty),
# so a randomized opening draws uniformly from all of them.
_ALL_CELLS: tuple[Cell, ...] = tuple(
    Cell(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
)


def play_match(
    mouse: Player,
    snake: Player,
    num_games: int,
    observer: Observer | None = None,
    opening: Cell | random.Random | None = None,
) -> MatchResult:
    """Play ``num_games`` games between ``mouse`` and ``snake``; return the tally.

    ``mouse`` plays Mouse and ``snake`` plays Snake for every game. Each game is
    run by :func:`~snakes_and_mice.game.play_game`, so both players are notified
    of every game's outcome. An optional :class:`Observer` is driven in lockstep:
    the match boundaries and every game's events are fired to it, and it decides
    (from its :class:`~snakes_and_mice.observer.ObservationLevel`) what to show.

    ``opening`` chooses where the snake is seeded each game (see
    :func:`_pick_seed`): a fixed :class:`~snakes_and_mice.core.Cell` uses that
    cell every game; a :class:`random.Random` draws one uniformly per game, so
    games don't replay the same opening — useful when deterministic players (e.g.
    two temperature-0 LLMs) would otherwise produce identical games; ``None``
    uses the board's default seed.

    Like :func:`~snakes_and_mice.game.play_game`, this does not catch
    provider/configuration errors: an LLM player's ``ModelRequestError`` flies
    straight through the match loop to the caller, since a broken model name or
    key dooms every game, not one. The caller (the CLI) reports it once and stops.
    """
    if num_games < 1:
        raise ValueError(f"a match needs at least one game, got {num_games}")

    names: dict[Side, str] = {Side.MOUSE: mouse.name, Side.SNAKE: snake.name}
    if observer is not None:
        observer.on_match_start(names, num_games)

    mouse_wins: int = 0
    snake_wins: int = 0
    cats_games: int = 0
    mouse_faults: int = 0
    snake_faults: int = 0
    aborted: int = 0
    faults: list[GameResult] = []

    for _ in range(num_games):
        result: GameResult = play_game(
            mouse, snake, observer, seed=_pick_seed(opening)
        )
        if result.termination is Termination.LINE_COMPLETED:
            assert result.winner is not None
            if result.winner is Side.MOUSE:
                mouse_wins += 1
            else:
                snake_wins += 1
        elif result.termination is Termination.CATS_GAME:
            cats_games += 1
        elif result.termination is Termination.ABORTED:
            aborted += 1  # no-contest: charged to neither side
        else:  # PLAYER_FAULT
            assert result.fault is not None
            faults.append(result)
            if result.fault.offender is Side.MOUSE:
                mouse_faults += 1
            else:
                snake_faults += 1

    match_result: MatchResult = MatchResult(
        names=names,
        num_games=num_games,
        mouse_wins=mouse_wins,
        snake_wins=snake_wins,
        cats_games=cats_games,
        mouse_faults=mouse_faults,
        snake_faults=snake_faults,
        faults=faults,
        aborted=aborted,
    )
    if observer is not None:
        observer.on_match_end(match_result)
    return match_result


def _pick_seed(opening: Cell | random.Random | None) -> Cell | None:
    """The snake's seed cell for one game, per the ``opening`` policy: a uniformly
    random cell for a :class:`random.Random` (so successive games open
    differently), the given cell for a :class:`~snakes_and_mice.core.Cell`, or
    ``None`` to let :func:`~snakes_and_mice.game.play_game` (and the board) apply
    their default. The board's default lives in one place, not here."""
    if isinstance(opening, random.Random):
        return opening.choice(_ALL_CELLS)
    return opening
