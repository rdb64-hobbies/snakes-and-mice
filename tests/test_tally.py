"""Tests for tallying results into standings and sorting them."""

from __future__ import annotations

import pytest

from snakes_and_mice import (
    GameResult,
    MatchResult,
    PlayerFaultDetail,
    PlayerFaultReason,
    Side,
    Termination,
)
from snakes_and_mice.tally import (
    PlayerStanding,
    StandingsSort,
    sort_standings,
    tally,
)


def _match(
    mouse: str,
    snake: str,
    *,
    num_games: int,
    mouse_wins: int = 0,
    snake_wins: int = 0,
    cats_games: int = 0,
    mouse_faults: int = 0,
    snake_faults: int = 0,
    aborted: int = 0,
    faults: list[GameResult] | None = None,
) -> MatchResult:
    return MatchResult(
        names={Side.MOUSE: mouse, Side.SNAKE: snake},
        num_games=num_games,
        mouse_wins=mouse_wins,
        snake_wins=snake_wins,
        cats_games=cats_games,
        mouse_faults=mouse_faults,
        snake_faults=snake_faults,
        faults=faults or [],
        aborted=aborted,
    )


def _standing(
    name: str,
    *,
    won: int = 0,
    lost: int = 0,
    tied: int = 0,
    faulted: int = 0,
    opponent_faulted: int = 0,
    played: int | None = None,
) -> PlayerStanding:
    total: int = (
        played
        if played is not None
        else won + lost + tied + faulted + opponent_faulted
    )
    return PlayerStanding(name, total, won, lost, tied, faulted, opponent_faulted)


# --------------------------------------------------------------------------- #
# Tallying
# --------------------------------------------------------------------------- #


def test_tally_aggregates_both_seats_and_reconciles() -> None:
    # A player takes each seat once, across two matches with swapped seats.
    faults_1: list[GameResult] = [
        GameResult(
            Termination.PLAYER_FAULT,
            fault=PlayerFaultDetail(Side.MOUSE, PlayerFaultReason.UNPARSEABLE_OUTPUT),
        ),
        GameResult(
            Termination.PLAYER_FAULT,
            fault=PlayerFaultDetail(Side.SNAKE, PlayerFaultReason.CELL_NOT_EMPTY),
        ),
    ]
    match_1: MatchResult = _match(
        "A", "B", num_games=10, mouse_wins=4, snake_wins=3, cats_games=1,
        mouse_faults=1, snake_faults=1, faults=faults_1,
    )
    faults_2: list[GameResult] = [
        GameResult(
            Termination.PLAYER_FAULT,
            fault=PlayerFaultDetail(Side.SNAKE, PlayerFaultReason.UNPARSEABLE_OUTPUT),
        ),
    ]
    match_2: MatchResult = _match(
        "B", "A", num_games=5, mouse_wins=2, snake_wins=1, snake_faults=1,
        aborted=1, faults=faults_2,
    )

    standings: dict[str, PlayerStanding] = {s.name: s for s in tally([match_1, match_2])}
    a: PlayerStanding = standings["A"]
    b: PlayerStanding = standings["B"]

    # A: mouse in match 1 (10 games), snake in match 2 (4 non-aborted games).
    assert (a.played, a.won, a.lost, a.tied, a.faulted, a.opponent_faulted) == (
        14, 5, 5, 1, 2, 1,
    )
    assert a.played == a.won + a.lost + a.tied + a.faulted + a.opponent_faulted
    # B mirrors A's record here.
    assert (b.played, b.won, b.lost, b.tied, b.faulted, b.opponent_faulted) == (
        14, 5, 5, 1, 1, 2,
    )
    assert a.win_rate == pytest.approx(5 / 11)  # denominator is clean games (11)
    assert a.fault_rate == pytest.approx(2 / 14)  # denominator is games played


def test_tally_excludes_aborted_games_from_played() -> None:
    match: MatchResult = _match("A", "B", num_games=4, mouse_wins=2, aborted=2)
    standings: dict[str, PlayerStanding] = {s.name: s for s in tally([match])}
    assert standings["A"].played == 2  # 4 games - 2 aborts
    assert standings["B"].played == 2


def test_tally_rates_are_none_without_a_denominator() -> None:
    # Every game a fault: clean_games is 0, and one side never played a clean game.
    faults: list[GameResult] = [
        GameResult(
            Termination.PLAYER_FAULT,
            fault=PlayerFaultDetail(Side.MOUSE, PlayerFaultReason.UNPARSEABLE_OUTPUT),
        )
        for _ in range(3)
    ]
    match: MatchResult = _match("faulter", "victim", num_games=3, mouse_faults=3, faults=faults)
    standings: dict[str, PlayerStanding] = {s.name: s for s in tally([match])}

    faulter: PlayerStanding = standings["faulter"]
    assert faulter.win_rate is None and faulter.loss_rate is None
    assert faulter.fault_rate == pytest.approx(1.0)  # played 3, faulted 3

    victim: PlayerStanding = standings["victim"]
    assert victim.win_rate is None  # no clean games
    assert victim.fault_rate == pytest.approx(0.0)  # played 3, faulted 0


# --------------------------------------------------------------------------- #
# Sorting standings
# --------------------------------------------------------------------------- #


def test_sort_by_win_rate_is_highest_first_then_undefined_last() -> None:
    high: PlayerStanding = _standing("high", won=4, lost=1)  # 0.8
    mid: PlayerStanding = _standing("mid", won=1, lost=1)  # 0.5
    none: PlayerStanding = _standing("none", faulted=3)  # no clean games
    ranked: list[PlayerStanding] = sort_standings(
        [none, mid, high], StandingsSort.WIN, ["high", "mid", "none"]
    )
    assert [s.name for s in ranked] == ["high", "mid", "none"]


def test_sort_by_loss_rate_is_lowest_first() -> None:
    low: PlayerStanding = _standing("low", won=9, lost=1)  # loss 0.1
    high: PlayerStanding = _standing("high", won=1, lost=1)  # loss 0.5
    none: PlayerStanding = _standing("none", faulted=2)
    ranked: list[PlayerStanding] = sort_standings(
        [high, none, low], StandingsSort.LOSS, ["low", "high", "none"]
    )
    assert [s.name for s in ranked] == ["low", "high", "none"]


def test_sort_by_fault_rate_is_lowest_first() -> None:
    clean: PlayerStanding = _standing("clean", won=1, lost=1)  # fault 0.0
    faulty: PlayerStanding = _standing(
        "faulty", won=5, lost=1, tied=1, faulted=3, played=10
    )  # fault 0.3
    none: PlayerStanding = _standing("none", played=0)  # nothing played
    ranked: list[PlayerStanding] = sort_standings(
        [faulty, none, clean], StandingsSort.FAULT, ["clean", "faulty", "none"]
    )
    assert [s.name for s in ranked] == ["clean", "faulty", "none"]


def test_sort_ties_break_by_roster_order_then_name() -> None:
    alpha: PlayerStanding = _standing("alpha", won=1, lost=1)  # 0.5
    bravo: PlayerStanding = _standing("bravo", won=1, lost=1)  # 0.5
    outsider: PlayerStanding = _standing("zzz", won=1, lost=1)  # 0.5, not in roster
    ranked: list[PlayerStanding] = sort_standings(
        [alpha, bravo, outsider], StandingsSort.WIN, ["bravo", "alpha"]
    )
    # bravo before alpha by roster order; the non-roster player sorts last.
    assert [s.name for s in ranked] == ["bravo", "alpha", "zzz"]
