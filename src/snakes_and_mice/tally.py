"""Tallying tournament results into per-player standings (§6).

A **tournament** is simply *any set of matches* (§6). This module selects which
of a bag of :class:`~snakes_and_mice.result.MatchResult`\\s count
(:func:`select_matches`), aggregates those into per-player
:class:`PlayerStanding`\\s (:func:`tally`) and the :class:`HeadToHead` matrix, and
orders the standings for display (:func:`sort_standings`).

Nothing here imports the CLI, Pydantic AI, or the roster loader — and, unlike the
rest of the package, nothing here takes a roster at all (§6, "The results file
stands alone"), which is why ranking ties break by name.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from .core import Side
from .faults import PlayerFaultReason, TournamentError
from .result import MatchResult


@dataclass(frozen=True)
class PlayerStanding:
    """One player's aggregate record across every match it appears in.

    The six counts reconcile: ``played == won + lost + tied + faulted +
    opponent_faulted`` (aborted games are excluded everywhere — they are charged to
    neither side). ``won``/``lost`` count only line-completion wins/losses; a fault
    by either side is its own bucket, never a win or loss.
    """

    name: str
    played: int  # games played, excluding no-contest aborts
    won: int
    lost: int
    tied: int
    faulted: int  # games this player faulted
    opponent_faulted: int  # games the opponent faulted
    # How this player's faults break down by reason; the counts sum to `faulted`.
    # Empty when the player never faulted.
    fault_reasons: Mapping[PlayerFaultReason, int] = field(default_factory=dict)

    @property
    def clean_games(self) -> int:
        """Games decided or drawn with neither side faulting — the win/loss base."""
        return self.won + self.lost + self.tied

    @property
    def win_rate(self) -> float | None:
        """``won / clean_games``, or ``None`` when there were no clean games."""
        return self.won / self.clean_games if self.clean_games else None

    @property
    def loss_rate(self) -> float | None:
        """``lost / clean_games``, or ``None`` when there were no clean games."""
        return self.lost / self.clean_games if self.clean_games else None

    @property
    def fault_rate(self) -> float | None:
        """``faulted / played`` (a fault *is* a played game), ``None`` if none played."""
        return self.faulted / self.played if self.played else None


@dataclass
class _Accumulator:
    """Mutable per-player tallies, folded into a :class:`PlayerStanding` at the end."""

    played: int = 0
    won: int = 0
    lost: int = 0
    tied: int = 0
    faulted: int = 0
    opponent_faulted: int = 0
    fault_reasons: Counter[PlayerFaultReason] = field(default_factory=Counter)


def names_in(results: Iterable[MatchResult]) -> set[str]:
    """Every player name appearing in ``results``, on either side."""
    return {name for result in results for name in result.names.values()}


def select_matches(
    results: Sequence[MatchResult],
    *,
    players: Sequence[str] | None = None,
    excluded: Sequence[str] | None = None,
) -> list[MatchResult]:
    """Keep only the matches played **among** a selected set of players (§6).

    ``players`` names the players to keep, ``excluded`` their complement; the two
    are mutually exclusive (passing both is a :class:`ValueError`) and passing
    neither keeps every match. A match survives only when *both* its players are
    selected, so dropping a player also withdraws the games it contributed to
    every opponent's record.

    Names are checked against ``results``, not against any roster: an unknown one
    raises :class:`~snakes_and_mice.faults.TournamentError` naming the names
    present.
    """
    if players is not None and excluded is not None:
        raise ValueError("select_matches takes 'players' or 'excluded', not both")
    if players is None and excluded is None:
        return list(results)

    present: set[str] = names_in(results)
    if players is not None:
        _check_names(players, present)
        selected: set[str] = set(players)
    else:
        assert excluded is not None  # exactly one of the two is set, checked above
        _check_names(excluded, present)
        selected = present - set(excluded)

    return [
        result for result in results if selected.issuperset(result.names.values())
    ]


def _check_names(named: Sequence[str], present: set[str]) -> None:
    """Raise unless every name in ``named`` appears in the results (§6)."""
    unknown: list[str] = sorted({name for name in named if name not in present})
    if not unknown:
        return
    listing: str = (
        ", ".join(sorted(present)) if present else "(none — the file has no matches)"
    )
    raise TournamentError(
        f"unknown player {'names' if len(unknown) > 1 else 'name'} "
        f"{', '.join(repr(name) for name in unknown)}; "
        f"the results file names: {listing}"
    )


def tally(results: Iterable[MatchResult]) -> list[PlayerStanding]:
    """Aggregate a bag of :class:`MatchResult`\\s into per-player standings.

    Players are keyed by the ``name`` recorded in each match, so a name reused
    across models is merged (a deliberate operator choice, §6). Standings come back
    in first-appearance order; use :func:`sort_standings` to rank them.
    """
    totals: dict[str, _Accumulator] = {}
    for result in results:
        played: int = result.num_games - result.aborted
        mouse_name: str = result.names[Side.MOUSE]
        snake_name: str = result.names[Side.SNAKE]

        mouse: _Accumulator = totals.setdefault(mouse_name, _Accumulator())
        mouse.played += played
        mouse.won += result.mouse_wins
        mouse.lost += result.snake_wins
        mouse.tied += result.cats_games
        mouse.faulted += result.mouse_faults
        mouse.opponent_faulted += result.snake_faults

        snake: _Accumulator = totals.setdefault(snake_name, _Accumulator())
        snake.played += played
        snake.won += result.snake_wins
        snake.lost += result.mouse_wins
        snake.tied += result.cats_games
        snake.faulted += result.snake_faults
        snake.opponent_faulted += result.mouse_faults

        # Attribute each recorded fault to whichever side committed it in this
        # match, so a player's faults break down by reason across the tournament.
        for game in result.faults:
            detail = game.fault
            if detail is None:
                continue
            totals[result.names[detail.offender]].fault_reasons[detail.reason] += 1

    return [
        PlayerStanding(
            name=name,
            played=acc.played,
            won=acc.won,
            lost=acc.lost,
            tied=acc.tied,
            faulted=acc.faulted,
            opponent_faulted=acc.opponent_faulted,
            fault_reasons=dict(acc.fault_reasons),
        )
        for name, acc in totals.items()
    ]


@dataclass(frozen=True)
class HeadToHead:
    """Who beat whom, as an n × n table of losses (§6).

    The cell at row ``A``, column ``B`` counts the games **A lost to B**, so row
    sums are that player's losses and column sums its wins. Wins and losses only:
    cat's games, faults and aborts play no part, and a pairing may have met many
    times and still show ``0`` both ways — which is why :meth:`cell` returns ``0``
    for that and ``None`` for a pairing that never met.
    """

    names: tuple[str, ...]  # row/column order, as ranked by the standings
    # (row, column) -> games the row player lost to the column player. Both
    # directions are present for every pairing that met, and only for those, so
    # the diagonal is absent entirely.
    losses: Mapping[tuple[str, str], int]

    def cell(self, row: str, column: str) -> int | None:
        """Games ``row`` lost to ``column``, or ``None`` if the two never met."""
        return self.losses.get((row, column))

    def met(self, row: str, column: str) -> bool:
        """Whether these two played any match at all (aborts and faults included)."""
        return (row, column) in self.losses


def head_to_head(
    results: Iterable[MatchResult], order: Sequence[str]
) -> HeadToHead:
    """Cross-tabulate ``results`` into a :class:`HeadToHead` over ``order`` (§6).

    ``order`` is the ranked standings' names, which become the matrix's rows and
    columns; matches involving anyone outside it are skipped. A match's
    ``mouse_wins`` are losses charged to its snake-side player and its
    ``snake_wins`` losses charged to its mouse-side one, so several lines for one
    pairing accumulate and its two seatings fold into the same pair of cells.
    """
    ranked: set[str] = set(order)
    losses: dict[tuple[str, str], int] = {}
    for result in results:
        mouse: str = result.names[Side.MOUSE]
        snake: str = result.names[Side.SNAKE]
        if mouse == snake or mouse not in ranked or snake not in ranked:
            continue
        # Seed both directions so a pairing that met is distinguishable from one
        # that never did, even when neither side won a game.
        losses.setdefault((mouse, snake), 0)
        losses.setdefault((snake, mouse), 0)
        losses[(mouse, snake)] += result.snake_wins  # the mouse-side player's losses
        losses[(snake, mouse)] += result.mouse_wins  # the snake-side player's losses
    return HeadToHead(names=tuple(order), losses=losses)


class StandingsSort(Enum):
    """Which rate ranks the standings. All sort **best-on-top**."""

    WIN = "win"  # highest win_rate first
    LOSS = "loss"  # lowest loss_rate first
    FAULT = "fault"  # lowest fault_rate first


def sort_standings(
    standings: Sequence[PlayerStanding],
    sort: StandingsSort,
) -> list[PlayerStanding]:
    """Order ``standings`` best-on-top by the chosen rate (§6).

    ``win_rate`` descends; ``loss_rate`` and ``fault_rate`` ascend (fewest first).
    A player with an undefined rate (no clean/played games) sorts to the end.
    Ties break by **name**, the only order available without a roster (§6).
    """

    def sort_key(standing: PlayerStanding) -> tuple[bool, float, str]:
        if sort is StandingsSort.WIN:
            rate: float | None = standing.win_rate
            primary: float = -(rate or 0.0)  # descending
        elif sort is StandingsSort.LOSS:
            rate = standing.loss_rate
            primary = rate or 0.0  # ascending
        else:
            rate = standing.fault_rate
            primary = rate or 0.0  # ascending
        return (rate is None, primary, standing.name)

    return sorted(standings, key=sort_key)
