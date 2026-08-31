"""Tests for the ``tally-tournament`` command.

The rendering itself is covered in test_console; here we check the command's own
behavior — reading the file, mapping ``--sort``, wiring the ``--faults``,
``--head-to-head`` and ``--players`` / ``--except`` flags, and reporting a missing
or empty file. Nothing here patches a roster: the command reads the results file
and nothing else (§6, "The results file stands alone"), which these tests rely on
by never providing a ``players.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from snakes_and_mice import (
    GameResult,
    MatchResult,
    PlayerFaultDetail,
    PlayerFaultReason,
    Side,
    Termination,
)
from snakes_and_mice.serialize import append_match_result
from snakes_and_mice.tally_cli import main


def _match(mouse: str, snake: str, *, num_games: int, mouse_wins: int = 0,
           snake_wins: int = 0, cats_games: int = 0) -> MatchResult:
    return MatchResult(
        names={Side.MOUSE: mouse, Side.SNAKE: snake},
        num_games=num_games,
        mouse_wins=mouse_wins,
        snake_wins=snake_wins,
        cats_games=cats_games,
        mouse_faults=0,
        snake_faults=0,
        faults=[],
        aborted=0,
    )


def test_prints_a_standings_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path: Path = tmp_path / "results.jsonl"
    append_match_result(_match("a", "b", num_games=4, mouse_wins=3, snake_wins=1), path)

    main(["--tournament-results", str(path)])
    out: str = capsys.readouterr().out
    assert "sorted by win%" in out
    assert "Win%" in out
    assert "a" in out and "b" in out


def test_missing_file_errors(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--tournament-results", str(tmp_path / "nope.jsonl")])


def test_sort_flag_is_reflected_in_the_header(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path: Path = tmp_path / "results.jsonl"
    append_match_result(_match("a", "b", num_games=2, mouse_wins=1, snake_wins=1), path)

    main(["--tournament-results", str(path), "--sort", "fault%"])
    assert "sorted by fault%" in capsys.readouterr().out


def _faulty_match() -> MatchResult:
    faults: list[GameResult] = [
        GameResult(
            Termination.PLAYER_FAULT,
            fault=PlayerFaultDetail(Side.MOUSE, PlayerFaultReason.UNPARSEABLE_OUTPUT),
        )
        for _ in range(2)
    ]
    return MatchResult(
        names={Side.MOUSE: "a", Side.SNAKE: "b"},
        num_games=5, mouse_wins=2, snake_wins=1, cats_games=0,
        mouse_faults=2, snake_faults=0, faults=faults, aborted=0,
    )


def test_faults_flag_appends_the_breakdown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path: Path = tmp_path / "results.jsonl"
    append_match_result(_faulty_match(), path)

    main(["--tournament-results", str(path), "--faults"])
    out: str = capsys.readouterr().out
    assert "Faults by player:" in out
    assert "a: unparseable_output ×2" in out


def test_faults_omitted_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path: Path = tmp_path / "results.jsonl"
    append_match_result(_faulty_match(), path)

    main(["--tournament-results", str(path)])  # no --faults
    assert "Faults by player:" not in capsys.readouterr().out


def test_empty_file_reports_no_matches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path: Path = tmp_path / "results.jsonl"
    path.write_text("", encoding="utf-8")

    main(["--tournament-results", str(path)])
    assert "No matches recorded yet." in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Restricting the tally to some players
# --------------------------------------------------------------------------- #


def _three_player_file(path: Path) -> None:
    """a beats b 3-0, c beats a 4-0, b and c never meet."""
    append_match_result(_match("a", "b", num_games=3, mouse_wins=3), path)
    append_match_result(_match("c", "a", num_games=4, mouse_wins=4), path)


def test_except_re_scores_as_though_the_player_never_entered(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path: Path = tmp_path / "results.jsonl"
    _three_player_file(path)

    main(["--tournament-results", str(path), "--except", "c"])
    out: str = capsys.readouterr().out
    assert "2 players" in out  # c's row is gone
    assert not any(line.startswith("c ") for line in out.splitlines())
    # a's four losses to c go with it: 3 played, 3 won, 0 lost.
    row: str = next(line for line in out.splitlines() if line.startswith("a "))
    assert row.split()[1:4] == ["3", "3", "0"]


def test_players_selects_the_same_matches_as_the_complementary_except(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path: Path = tmp_path / "results.jsonl"
    _three_player_file(path)

    main(["--tournament-results", str(path), "--players", "a", "b"])
    by_players: str = capsys.readouterr().out
    main(["--tournament-results", str(path), "--except", "c"])
    assert capsys.readouterr().out == by_players


def test_players_and_except_together_is_an_error(tmp_path: Path) -> None:
    path: Path = tmp_path / "results.jsonl"
    _three_player_file(path)
    with pytest.raises(SystemExit):
        main(["--tournament-results", str(path), "--players", "a", "--except", "b"])


def test_unknown_name_errors_rather_than_tallying_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path: Path = tmp_path / "results.jsonl"
    _three_player_file(path)
    with pytest.raises(SystemExit):
        main(["--tournament-results", str(path), "--players", "a", "typo"])
    assert "'typo'" in capsys.readouterr().err


def test_a_filter_that_leaves_nothing_reads_differently_from_an_empty_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path: Path = tmp_path / "results.jsonl"
    _three_player_file(path)

    main(["--tournament-results", str(path), "--players", "b", "c"])  # never met
    filtered: str = capsys.readouterr().out
    assert "No matches among the selected players." in filtered
    assert "No matches recorded yet." not in filtered


# --------------------------------------------------------------------------- #
# The head-to-head matrix
# --------------------------------------------------------------------------- #


def test_head_to_head_flag_appends_the_matrix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path: Path = tmp_path / "results.jsonl"
    _three_player_file(path)

    main(["--tournament-results", str(path), "--head-to-head"])
    out: str = capsys.readouterr().out
    assert "Head-to-head" in out
    # Ranked by win%: c (1.0), a (3/7), b (0.0). b never met c, so that cell is ·.
    matrix: list[str] = out.splitlines()[-3:]
    # Columns are numbered in the standings' order (c, a, b); · = never met,
    # — = the diagonal, which has no entries at all.
    assert matrix[0].split() == ["1.", "c", "—", "0", "·"]
    assert matrix[1].split() == ["2.", "a", "4", "—", "0"]
    assert matrix[2].split() == ["3.", "b", "·", "3", "—"]


def test_head_to_head_omitted_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path: Path = tmp_path / "results.jsonl"
    _three_player_file(path)

    main(["--tournament-results", str(path)])
    assert "Head-to-head" not in capsys.readouterr().out


def test_head_to_head_is_omitted_with_fewer_than_two_players(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path: Path = tmp_path / "results.jsonl"
    _three_player_file(path)

    # One name selects no matches at all, so there is nothing to cross-tabulate.
    main(["--tournament-results", str(path), "--players", "a", "--head-to-head"])
    assert "Head-to-head" not in capsys.readouterr().out


def test_head_to_head_follows_the_sort_and_the_filter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path: Path = tmp_path / "results.jsonl"
    _three_player_file(path)

    main([
        "--tournament-results", str(path), "--except", "b",
        "--sort", "loss%", "--head-to-head",
    ])
    out: str = capsys.readouterr().out
    matrix: list[str] = out.splitlines()[-2:]
    # Only a and c survive; by loss% c (0.0) leads a (1.0).
    # a lost all four to c; c lost none to a. b's games are gone with b.
    assert matrix[0].split() == ["1.", "c", "—", "0"]
    assert matrix[1].split() == ["2.", "a", "4", "—"]
