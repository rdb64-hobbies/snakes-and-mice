"""Tests for the ``tally-tournament`` command.

The rendering itself is covered in test_console; here we check the command's own
behavior — reading the file, mapping ``--sort``, and reporting a missing or empty
file. The roster load is patched so tie-breaking order does not depend on a real
``players.yaml``.
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
from snakes_and_mice.config import PlayerSpec, Roster
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


def _patch_roster(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    roster = Roster(
        players={n: PlayerSpec(name=n, provider="p", model="m") for n in names},
        providers={},
    )
    monkeypatch.setattr("snakes_and_mice.tally_cli.load_roster", lambda: roster)


def test_prints_a_standings_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_roster(monkeypatch, "a", "b")
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_roster(monkeypatch)  # empty roster; ties fall back to name order
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_roster(monkeypatch, "a", "b")
    path: Path = tmp_path / "results.jsonl"
    append_match_result(_faulty_match(), path)

    main(["--tournament-results", str(path), "--faults"])
    out: str = capsys.readouterr().out
    assert "Faults by player:" in out
    assert "a: unparseable_output ×2" in out


def test_faults_omitted_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_roster(monkeypatch, "a", "b")
    path: Path = tmp_path / "results.jsonl"
    append_match_result(_faulty_match(), path)

    main(["--tournament-results", str(path)])  # no --faults
    assert "Faults by player:" not in capsys.readouterr().out


def test_empty_file_reports_no_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_roster(monkeypatch)
    path: Path = tmp_path / "results.jsonl"
    path.write_text("", encoding="utf-8")

    main(["--tournament-results", str(path)])
    assert "No matches recorded yet." in capsys.readouterr().out
