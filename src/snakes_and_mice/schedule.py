"""Building a tournament schedule from two player subsets.

A **tournament** is simply *any set of matches* (§6). This module turns two player
*subsets* (plus the roster's declared order) into the ordered list of
``(mouse, snake)`` name pairs to play, via a single "straddle" rule that expresses
round-robins, subset play, and cross-of-two-subsets alike.

Nothing here imports the CLI, Pydantic AI, or the roster loader: the logic stays
light and unit-testable, taking the roster only as an ordered list of names.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import assert_never

from .faults import TournamentError

# A schedule is generated from two player subsets, A and B, each named by one
# selector. The generating rule is a single "straddle" predicate: emit a match
# ``(mouse=x, snake=y)`` for every ordered pair of distinct players whose unordered
# matchup has one endpoint in A and the other in B. Both seatings and the exclusion
# of self-play fall straight out of it (§6, "The one generating operation").


@dataclass(frozen=True)
class AllPlayers:
    """Selector: every player in the roster (``all``)."""


@dataclass(frozen=True)
class SamePlayers:
    """Selector: the same set as the *other* subset (``same``).

    Valid only for the second subset (B); using it for A raises
    :class:`~snakes_and_mice.faults.TournamentError`, since there is no other
    subset to mirror.
    """


@dataclass(frozen=True)
class NamedPlayers:
    """Selector: an explicit list of player names."""

    names: tuple[str, ...]


@dataclass(frozen=True)
class PlayersAbove:
    """Selector: every player listed strictly *above* ``pivot`` in the roster."""

    pivot: str


@dataclass(frozen=True)
class PlayersBelow:
    """Selector: every player listed strictly *below* ``pivot`` in the roster."""

    pivot: str


Selector = AllPlayers | SamePlayers | NamedPlayers | PlayersAbove | PlayersBelow
"""One subset selector. ``above``/``below`` are exclusive of the pivot and rely on
the roster's declared order (a documented contract — order the roster by rough
strength and ``above X`` names the stronger cohort)."""


def resolve_subset(
    selector: Selector,
    roster_order: Sequence[str],
    *,
    other: Sequence[str] | None = None,
) -> list[str]:
    """Resolve a :class:`Selector` to the concrete player names it selects.

    ``roster_order`` is the roster's names in declared order; the result preserves
    that order (so ``above``/``below`` and explicit lists all read consistently).
    ``other`` is the already-resolved *other* subset, required only to resolve
    :class:`SamePlayers`. Raises :class:`~snakes_and_mice.faults.TournamentError`
    for an unknown name or pivot, or for ``same`` used without an ``other``.
    """
    index: dict[str, int] = {name: i for i, name in enumerate(roster_order)}
    if isinstance(selector, AllPlayers):
        return list(roster_order)
    if isinstance(selector, SamePlayers):
        if other is None:
            raise TournamentError(
                "'same' refers to the other subset and cannot be used for the first"
            )
        return list(other)
    if isinstance(selector, NamedPlayers):
        for name in selector.names:
            if name not in index:
                raise TournamentError(_unknown(name, roster_order))
        chosen: set[str] = set(selector.names)
        return [name for name in roster_order if name in chosen]
    if isinstance(selector, PlayersAbove):
        if selector.pivot not in index:
            raise TournamentError(_unknown(selector.pivot, roster_order))
        return list(roster_order[: index[selector.pivot]])
    if isinstance(selector, PlayersBelow):
        if selector.pivot not in index:
            raise TournamentError(_unknown(selector.pivot, roster_order))
        return list(roster_order[index[selector.pivot] + 1 :])
    assert_never(selector)


def build_schedule(
    subset_a: Sequence[str],
    subset_b: Sequence[str],
    roster_order: Sequence[str],
) -> list[tuple[str, str]]:
    """The ordered ``(mouse, snake)`` name pairs for two *resolved* subsets.

    Emits a match for every ordered pair of distinct players ``(x, y)`` drawn from
    ``A ∪ B`` whose matchup straddles the subsets — ``(x∈A and y∈B) or (x∈B and
    y∈A)``. Pairs are enumerated in roster order for determinism. This is the whole
    schedule generator: a round-robin is ``A = B = all``; introducing newcomers is
    ``A = {new}, B = all``.
    """
    set_a: set[str] = set(subset_a)
    set_b: set[str] = set(subset_b)
    universe: list[str] = [n for n in roster_order if n in set_a or n in set_b]
    schedule: list[tuple[str, str]] = []
    for x in universe:
        for y in universe:
            if x == y:
                continue
            if (x in set_a and y in set_b) or (x in set_b and y in set_a):
                schedule.append((x, y))
    return schedule


def plan_schedule(
    selector_a: Selector,
    selector_b: Selector,
    roster_order: Sequence[str],
) -> list[tuple[str, str]]:
    """Resolve both selectors and build the schedule (§6).

    A is resolved first; B is resolved with A supplied as ``other`` so that
    :class:`SamePlayers` mirrors A.
    """
    subset_a: list[str] = resolve_subset(selector_a, roster_order)
    subset_b: list[str] = resolve_subset(selector_b, roster_order, other=subset_a)
    return build_schedule(subset_a, subset_b, roster_order)


def _unknown(name: str, roster_order: Sequence[str]) -> str:
    return (
        f"no player named {name!r} in the roster; "
        f"available: {', '.join(roster_order) or '(none)'}"
    )
