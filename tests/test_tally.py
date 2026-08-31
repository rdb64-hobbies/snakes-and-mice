"""Tests for selecting matches, tallying them into standings, cross-tabulating
the head-to-head matrix, and sorting the standings."""

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
from snakes_and_mice.faults import TournamentError
from snakes_and_mice.tally import (
    HeadToHead,
    PlayerStanding,
    StandingsSort,
    head_to_head,
    names_in,
    select_matches,
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


def _fault(offender: Side, reason: PlayerFaultReason) -> GameResult:
    return GameResult(
        Termination.PLAYER_FAULT,
        fault=PlayerFaultDetail(offender=offender, reason=reason),
    )


def test_tally_breaks_down_fault_reasons_per_player() -> None:
    faults: list[GameResult] = [
        _fault(Side.MOUSE, PlayerFaultReason.UNPARSEABLE_OUTPUT),
        _fault(Side.MOUSE, PlayerFaultReason.UNPARSEABLE_OUTPUT),
        _fault(Side.SNAKE, PlayerFaultReason.CELL_NOT_EMPTY),
    ]
    match: MatchResult = _match(
        "A", "B", num_games=5, mouse_wins=1, snake_wins=1,
        mouse_faults=2, snake_faults=1, faults=faults,
    )
    standings: dict[str, PlayerStanding] = {s.name: s for s in tally([match])}

    # Each side's faults are attributed to that side's player, by reason.
    assert dict(standings["A"].fault_reasons) == {
        PlayerFaultReason.UNPARSEABLE_OUTPUT: 2
    }
    assert dict(standings["B"].fault_reasons) == {PlayerFaultReason.CELL_NOT_EMPTY: 1}
    # The breakdown reconciles with the faulted total.
    assert sum(standings["A"].fault_reasons.values()) == standings["A"].faulted


def test_tally_leaves_fault_reasons_empty_for_a_clean_player() -> None:
    match: MatchResult = _match("A", "B", num_games=2, mouse_wins=1, snake_wins=1)
    standings: dict[str, PlayerStanding] = {s.name: s for s in tally([match])}
    assert standings["A"].fault_reasons == {}


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
    ranked: list[PlayerStanding] = sort_standings([none, mid, high], StandingsSort.WIN)
    assert [s.name for s in ranked] == ["high", "mid", "none"]


def test_sort_by_loss_rate_is_lowest_first() -> None:
    low: PlayerStanding = _standing("low", won=9, lost=1)  # loss 0.1
    high: PlayerStanding = _standing("high", won=1, lost=1)  # loss 0.5
    none: PlayerStanding = _standing("none", faulted=2)
    ranked: list[PlayerStanding] = sort_standings([high, none, low], StandingsSort.LOSS)
    assert [s.name for s in ranked] == ["low", "high", "none"]


def test_sort_by_fault_rate_is_lowest_first() -> None:
    clean: PlayerStanding = _standing("clean", won=1, lost=1)  # fault 0.0
    faulty: PlayerStanding = _standing(
        "faulty", won=5, lost=1, tied=1, faulted=3, played=10
    )  # fault 0.3
    none: PlayerStanding = _standing("none", played=0)  # nothing played
    ranked: list[PlayerStanding] = sort_standings([faulty, none, clean], StandingsSort.FAULT)
    assert [s.name for s in ranked] == ["clean", "faulty", "none"]


def test_sort_ties_break_by_name() -> None:
    # Equal rates, given out of order: only the name can separate them, since the
    # tally never reads a roster (§6, "The results file stands alone").
    alpha: PlayerStanding = _standing("alpha", won=1, lost=1)  # 0.5
    bravo: PlayerStanding = _standing("bravo", won=1, lost=1)  # 0.5
    zulu: PlayerStanding = _standing("zulu", won=1, lost=1)  # 0.5
    ranked: list[PlayerStanding] = sort_standings(
        [zulu, bravo, alpha], StandingsSort.WIN
    )
    assert [s.name for s in ranked] == ["alpha", "bravo", "zulu"]


# --------------------------------------------------------------------------- #
# Selecting which matches count
# --------------------------------------------------------------------------- #


def test_names_in_collects_both_sides() -> None:
    results: list[MatchResult] = [
        _match("a", "b", num_games=1),
        _match("c", "a", num_games=1),
    ]
    assert names_in(results) == {"a", "b", "c"}


def test_no_filter_keeps_every_match() -> None:
    results: list[MatchResult] = [
        _match("a", "b", num_games=1), _match("b", "c", num_games=1)
    ]
    assert select_matches(results) == results


def test_players_keeps_only_matches_with_both_sides_selected() -> None:
    ab: MatchResult = _match("a", "b", num_games=1)
    bc: MatchResult = _match("b", "c", num_games=1)
    ca: MatchResult = _match("c", "a", num_games=1)
    selected: list[MatchResult] = select_matches([ab, bc, ca], players=["a", "b"])
    assert selected == [ab]  # b-c and c-a each have one unselected player


def test_except_is_the_complement_of_players() -> None:
    results: list[MatchResult] = [
        _match("a", "b", num_games=1),
        _match("b", "c", num_games=1),
        _match("c", "a", num_games=1),
    ]
    assert select_matches(results, excluded=["c"]) == select_matches(
        results, players=["a", "b"]
    )


def test_dropping_a_player_withdraws_its_games_from_every_opponent() -> None:
    # a beats b 3-0, and loses to c 0-4. Excluding c must take back those four
    # losses, not merely hide c's row.
    results: list[MatchResult] = [
        _match("a", "b", num_games=3, mouse_wins=3),
        _match("a", "c", num_games=4, snake_wins=4),
    ]
    whole: dict[str, PlayerStanding] = {s.name: s for s in tally(results)}
    assert (whole["a"].played, whole["a"].won, whole["a"].lost) == (7, 3, 4)

    without_c: list[MatchResult] = select_matches(results, excluded=["c"])
    narrowed: dict[str, PlayerStanding] = {s.name: s for s in tally(without_c)}
    assert set(narrowed) == {"a", "b"}
    a: PlayerStanding = narrowed["a"]
    assert (a.played, a.won, a.lost) == (3, 3, 0)  # as though c had never entered
    # The smaller tournament still reconciles exactly as the whole file does.
    assert a.played == a.won + a.lost + a.tied + a.faulted + a.opponent_faulted


def test_a_single_players_name_selects_nothing() -> None:
    # Self-play is excluded (§6), so no match has "a" on both sides: a record
    # exists only relative to opponents, and naming none selects no games.
    results: list[MatchResult] = [
        _match("a", "b", num_games=1), _match("c", "a", num_games=1)
    ]
    assert select_matches(results, players=["a"]) == []
    assert tally(select_matches(results, players=["a"])) == []


def test_a_selected_player_whose_opponents_were_all_dropped_gets_no_row() -> None:
    # "c" survives the filter but every match it played was against "b".
    results: list[MatchResult] = [
        _match("a", "b", num_games=1, mouse_wins=1),
        _match("b", "c", num_games=1, mouse_wins=1),
    ]
    standings: list[PlayerStanding] = tally(
        select_matches(results, players=["a", "c"])
    )
    # No row at all, rather than a row of zeros asserting it played and did nothing.
    assert standings == []


def test_unknown_name_is_an_error_naming_the_names_present() -> None:
    results: list[MatchResult] = [_match("a", "b", num_games=1)]
    with pytest.raises(TournamentError) as excinfo:
        select_matches(results, players=["a", "typo"])
    message: str = str(excinfo.value)
    assert "'typo'" in message
    assert "a, b" in message  # the names actually in the file


def test_unknown_name_is_checked_for_except_too() -> None:
    with pytest.raises(TournamentError):
        select_matches([_match("a", "b", num_games=1)], excluded=["nobody"])


def test_both_selectors_at_once_is_a_programming_error() -> None:
    with pytest.raises(ValueError):
        select_matches(
            [_match("a", "b", num_games=1)], players=["a"], excluded=["b"]
        )


def test_selection_ignores_the_order_names_are_given_in() -> None:
    results: list[MatchResult] = [
        _match("a", "b", num_games=1), _match("b", "c", num_games=1)
    ]
    assert select_matches(results, players=["b", "a"]) == select_matches(
        results, players=["a", "b"]
    )


# --------------------------------------------------------------------------- #
# The head-to-head matrix
# --------------------------------------------------------------------------- #


def test_head_to_head_rows_sum_to_losses_and_columns_to_wins() -> None:
    # The matrix's defining property: it decomposes two standings columns by
    # opponent, so the two views cannot disagree.
    results: list[MatchResult] = [
        _match("a", "b", num_games=6, mouse_wins=4, snake_wins=1, cats_games=1),
        _match("c", "a", num_games=5, mouse_wins=3, snake_wins=2),
        _match("b", "c", num_games=4, mouse_wins=1, snake_wins=2, cats_games=1),
    ]
    standings: list[PlayerStanding] = sort_standings(tally(results), StandingsSort.WIN)
    order: list[str] = [s.name for s in standings]
    matrix: HeadToHead = head_to_head(results, order)

    for standing in standings:
        row_sum: int = sum(
            matrix.cell(standing.name, other) or 0 for other in order
        )
        column_sum: int = sum(
            matrix.cell(other, standing.name) or 0 for other in order
        )
        assert row_sum == standing.lost
        assert column_sum == standing.won


def test_head_to_head_counts_only_wins_and_losses() -> None:
    # Cat's games, faults and aborts contribute nothing: this pairing played six
    # games and shows 0 both ways.
    faults: list[GameResult] = [_fault(Side.MOUSE, PlayerFaultReason.CELL_NOT_EMPTY)]
    results: list[MatchResult] = [
        _match(
            "a", "b", num_games=6, cats_games=4, mouse_faults=1, faults=faults,
            aborted=1,
        )
    ]
    matrix: HeadToHead = head_to_head(results, ["a", "b"])
    assert matrix.cell("a", "b") == 0
    assert matrix.cell("b", "a") == 0


def test_head_to_head_distinguishes_never_met_from_a_goalless_pairing() -> None:
    results: list[MatchResult] = [
        _match("a", "b", num_games=2, cats_games=2),  # met, no decisive game
    ]
    matrix: HeadToHead = head_to_head(results, ["a", "b", "c"])
    assert matrix.cell("a", "b") == 0 and matrix.met("a", "b")
    assert matrix.cell("a", "c") is None and not matrix.met("a", "c")


def test_head_to_head_has_no_diagonal_entries() -> None:
    results: list[MatchResult] = [_match("a", "b", num_games=2, mouse_wins=2)]
    matrix: HeadToHead = head_to_head(results, ["a", "b"])
    assert matrix.cell("a", "a") is None  # not 0 — no match has a player on both sides
    assert not any(row == column for row, column in matrix.losses)


def test_head_to_head_folds_both_seatings_into_one_pairing() -> None:
    # a wins 3 as Mouse and 2 as Snake; b's losses to a are all five.
    results: list[MatchResult] = [
        _match("a", "b", num_games=3, mouse_wins=3),
        _match("b", "a", num_games=2, snake_wins=2),
    ]
    matrix: HeadToHead = head_to_head(results, ["a", "b"])
    assert matrix.cell("b", "a") == 5
    assert matrix.cell("a", "b") == 0


def test_head_to_head_skips_players_outside_the_ranked_order() -> None:
    # The matrix follows the standings, so a match with an unranked player
    # contributes nothing — matching a filtered-out player's absence.
    results: list[MatchResult] = [
        _match("a", "b", num_games=2, mouse_wins=2),
        _match("a", "gone", num_games=2, mouse_wins=2),
    ]
    matrix: HeadToHead = head_to_head(results, ["a", "b"])
    assert matrix.names == ("a", "b")
    assert set(matrix.losses) == {("a", "b"), ("b", "a")}
