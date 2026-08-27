"""Tallying tournament results into per-player standings (§6).

A **tournament** is simply *any set of matches* (§6). This module aggregates a bag
of :class:`~snakes_and_mice.result.MatchResult`\\s — however they were produced —
into per-player :class:`PlayerStanding`\\s and orders them for display.

Nothing here imports the CLI, Pydantic AI, or the roster loader: the logic stays
light and unit-testable, taking the roster only as an ordered list of names.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from .core import Side
from .faults import PlayerFaultReason
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


class StandingsSort(Enum):
    """Which rate ranks the standings. All sort **best-on-top**."""

    WIN = "win"  # highest win_rate first
    LOSS = "loss"  # lowest loss_rate first
    FAULT = "fault"  # lowest fault_rate first


def sort_standings(
    standings: Sequence[PlayerStanding],
    sort: StandingsSort,
    roster_order: Sequence[str],
) -> list[PlayerStanding]:
    """Order ``standings`` best-on-top by the chosen rate (§6).

    ``win_rate`` descends; ``loss_rate`` and ``fault_rate`` ascend (fewest first).
    A player with an undefined rate (no clean/played games) sorts to the end. Ties
    break by ``roster_order``, then name — so the order is fully deterministic.
    """
    index: dict[str, int] = {name: i for i, name in enumerate(roster_order)}
    tail: int = len(roster_order)

    def sort_key(standing: PlayerStanding) -> tuple[bool, float, int, str]:
        if sort is StandingsSort.WIN:
            rate: float | None = standing.win_rate
            primary: float = -(rate or 0.0)  # descending
        elif sort is StandingsSort.LOSS:
            rate = standing.loss_rate
            primary = rate or 0.0  # ascending
        else:
            rate = standing.fault_rate
            primary = rate or 0.0  # ascending
        return (rate is None, primary, index.get(standing.name, tail), standing.name)

    return sorted(standings, key=sort_key)
