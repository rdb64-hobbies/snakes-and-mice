"""Tests for the results-file encoding and I/O."""

from __future__ import annotations

from pathlib import Path

import pytest

from snakes_and_mice import (
    GameResult,
    MatchResult,
    Move,
    PlayerFaultDetail,
    PlayerFaultReason,
    Side,
    Termination,
    TurnOutcome,
)
from snakes_and_mice.faults import TournamentError
from snakes_and_mice.serialize import (
    append_match_result,
    decode_match_result,
    dump_match_result,
    encode_match_result,
    load_match_result,
    read_match_results,
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


def _match_with_faults() -> MatchResult:
    wrong_claim = GameResult(
        Termination.PLAYER_FAULT,
        fault=PlayerFaultDetail(
            offender=Side.MOUSE,
            reason=PlayerFaultReason.WRONG_OUTCOME_CLAIM,
            attempted_move=Move.from_labels("A1", "A2"),
            claimed_outcome=TurnOutcome.WIN,
            actual_outcome=TurnOutcome.IN_PLAY,
        ),
    )
    occupied = GameResult(
        Termination.PLAYER_FAULT,
        fault=PlayerFaultDetail(
            offender=Side.SNAKE,
            reason=PlayerFaultReason.CELL_NOT_EMPTY,
            attempted_move=Move.from_labels("B3"),
        ),
    )
    garbage = GameResult(
        Termination.PLAYER_FAULT,
        fault=PlayerFaultDetail(
            offender=Side.MOUSE,
            reason=PlayerFaultReason.UNPARSEABLE_OUTPUT,
        ),
    )
    return _match(
        "opus",
        "gpt5",
        num_games=10,
        mouse_wins=3,
        snake_wins=2,
        cats_games=1,
        mouse_faults=2,
        snake_faults=1,
        aborted=1,
        faults=[wrong_claim, occupied, garbage],
    )


# --------------------------------------------------------------------------- #
# Results file encoding
# --------------------------------------------------------------------------- #


def test_encode_decode_round_trips_a_rich_result() -> None:
    result: MatchResult = _match_with_faults()
    assert decode_match_result(encode_match_result(result)) == result


def test_dump_load_round_trips_via_a_single_line() -> None:
    result: MatchResult = _match_with_faults()
    line: str = dump_match_result(result)
    assert "\n" not in line
    assert load_match_result(line) == result


def test_encoding_uses_side_values_and_cell_labels() -> None:
    encoded: dict[str, object] = encode_match_result(_match_with_faults())
    assert encoded["names"] == {"mouse": "opus", "snake": "gpt5"}
    faults = encoded["faults"]
    assert isinstance(faults, list)
    fault = faults[0]["fault"]
    assert fault["attempted_move"] == ["A1", "A2"]
    assert fault["claimed_outcome"] == "win"


def test_load_rejects_non_json() -> None:
    with pytest.raises(TournamentError, match="malformed"):
        load_match_result("{not json")


def test_load_rejects_missing_fields() -> None:
    with pytest.raises(TournamentError, match="malformed"):
        load_match_result('{"num_games": 1}')


def test_load_rejects_bad_enum_value() -> None:
    with pytest.raises(TournamentError, match="malformed"):
        load_match_result('{"names": {"frog": "a", "snake": "b"}}')


# --------------------------------------------------------------------------- #
# Results file I/O
# --------------------------------------------------------------------------- #


def test_append_then_read_round_trips(tmp_path: Path) -> None:
    path: Path = tmp_path / "results.jsonl"
    first: MatchResult = _match("opus", "gpt5", num_games=3, mouse_wins=3)
    second: MatchResult = _match_with_faults()

    append_match_result(first, path)
    append_match_result(second, path)

    assert read_match_results(path) == [first, second]


def test_read_ignores_blank_lines(tmp_path: Path) -> None:
    path: Path = tmp_path / "results.jsonl"
    result: MatchResult = _match("opus", "gpt5", num_games=1, cats_games=1)
    path.write_text(dump_match_result(result) + "\n\n   \n", encoding="utf-8")
    assert read_match_results(path) == [result]


def test_append_creates_parent_directories(tmp_path: Path) -> None:
    path: Path = tmp_path / "nested" / "dir" / "results.jsonl"
    result: MatchResult = _match("opus", "gpt5", num_games=1, mouse_wins=1)
    append_match_result(result, path)
    assert read_match_results(path) == [result]


def test_read_missing_file_raises() -> None:
    with pytest.raises(TournamentError, match="not found"):
        read_match_results(Path("does-not-exist.jsonl"))


def test_read_reports_the_offending_line_number(tmp_path: Path) -> None:
    path: Path = tmp_path / "results.jsonl"
    good: str = dump_match_result(_match("opus", "gpt5", num_games=1, mouse_wins=1))
    path.write_text(good + "\n{broken\n", encoding="utf-8")
    with pytest.raises(TournamentError, match=r":2:"):
        read_match_results(path)
