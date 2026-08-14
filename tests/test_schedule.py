"""Tests for schedule building and selector resolution."""

from __future__ import annotations

import pytest

from snakes_and_mice.faults import TournamentError
from snakes_and_mice.schedule import (
    AllPlayers,
    NamedPlayers,
    PlayersAbove,
    PlayersBelow,
    SamePlayers,
    build_schedule,
    plan_schedule,
    resolve_subset,
)

ROSTER: list[str] = ["opus", "gpt5", "gemini", "llama"]


# --------------------------------------------------------------------------- #
# Scheduling
# --------------------------------------------------------------------------- #


def test_round_robin_is_all_ordered_pairs() -> None:
    players: list[str] = ["a", "b", "c"]
    schedule: list[tuple[str, str]] = build_schedule(players, players, players)

    # N*(N-1) matches: every ordered pair of distinct players, both seatings.
    assert len(schedule) == 6
    assert set(schedule) == {
        ("a", "b"), ("b", "a"),
        ("a", "c"), ("c", "a"),
        ("b", "c"), ("c", "b"),
    }


def test_schedule_enumerates_in_roster_order() -> None:
    players: list[str] = ["a", "b", "c"]
    schedule: list[tuple[str, str]] = build_schedule(players, players, players)
    # x iterates the roster order outermost, y innermost.
    assert schedule == [
        ("a", "b"), ("a", "c"),
        ("b", "a"), ("b", "c"),
        ("c", "a"), ("c", "b"),
    ]


def test_new_player_plays_everyone_but_existing_pairs_do_not_replay() -> None:
    # A = {new}, B = all: the newcomer meets each existing player both ways, and
    # the existing players never replay one another.
    schedule: list[tuple[str, str]] = build_schedule(
        ["new"], ["new", "opus", "gpt5"], ["opus", "gpt5", "new"]
    )
    assert set(schedule) == {
        ("new", "opus"), ("opus", "new"),
        ("new", "gpt5"), ("gpt5", "new"),
    }
    assert ("opus", "gpt5") not in schedule
    assert ("gpt5", "opus") not in schedule


def test_disjoint_subsets_give_2nm_matches() -> None:
    schedule: list[tuple[str, str]] = build_schedule(
        ["a", "b"], ["c", "d"], ["a", "b", "c", "d"]
    )
    assert len(schedule) == 2 * 2 * 2  # 2 * N * M
    # Every match straddles the two subsets; none pairs within a subset.
    for mouse, snake in schedule:
        assert {mouse, snake} & {"a", "b"}
        assert {mouse, snake} & {"c", "d"}
    assert ("a", "b") not in schedule


def test_overlapping_subsets_drop_self_play_only() -> None:
    # A = {a, b}, B = {b, c}: b appears in both, but x != y removes the b-vs-b
    # self match while keeping every genuine straddle.
    schedule: list[tuple[str, str]] = build_schedule(
        ["a", "b"], ["b", "c"], ["a", "b", "c"]
    )
    assert ("b", "b") not in schedule
    assert set(schedule) == {
        ("a", "b"), ("b", "a"),
        ("a", "c"), ("c", "a"),
        ("b", "c"), ("c", "b"),
    }


# --------------------------------------------------------------------------- #
# Selector resolution
# --------------------------------------------------------------------------- #


def test_resolve_all() -> None:
    assert resolve_subset(AllPlayers(), ROSTER) == ROSTER


def test_resolve_named_dedupes_and_keeps_roster_order() -> None:
    got: list[str] = resolve_subset(NamedPlayers(("llama", "opus", "opus")), ROSTER)
    assert got == ["opus", "llama"]  # roster order, no duplicate


def test_resolve_named_unknown_raises() -> None:
    with pytest.raises(TournamentError, match="no player named 'nope'"):
        resolve_subset(NamedPlayers(("nope",)), ROSTER)


def test_resolve_above_and_below_are_exclusive() -> None:
    assert resolve_subset(PlayersAbove("gemini"), ROSTER) == ["opus", "gpt5"]
    assert resolve_subset(PlayersBelow("gpt5"), ROSTER) == ["gemini", "llama"]


def test_resolve_above_at_the_top_is_empty() -> None:
    assert resolve_subset(PlayersAbove("opus"), ROSTER) == []


def test_resolve_above_unknown_pivot_raises() -> None:
    with pytest.raises(TournamentError, match="no player named 'ghost'"):
        resolve_subset(PlayersAbove("ghost"), ROSTER)


def test_resolve_same_mirrors_other() -> None:
    other: list[str] = ["opus", "gpt5"]
    assert resolve_subset(SamePlayers(), ROSTER, other=other) == other


def test_resolve_same_without_other_raises() -> None:
    with pytest.raises(TournamentError, match="'same'"):
        resolve_subset(SamePlayers(), ROSTER)


def test_plan_schedule_defaults_all_vs_same_is_round_robin() -> None:
    # The command defaults: A = all, B = same → full round-robin.
    schedule: list[tuple[str, str]] = plan_schedule(AllPlayers(), SamePlayers(), ROSTER)
    assert len(schedule) == len(ROSTER) * (len(ROSTER) - 1)


def test_plan_schedule_new_player_against_all() -> None:
    schedule: list[tuple[str, str]] = plan_schedule(
        NamedPlayers(("llama",)), AllPlayers(), ROSTER
    )
    # llama vs each of the other three, both seatings.
    assert len(schedule) == 2 * 3
    assert all("llama" in pair for pair in schedule)
