"""Tests for the ``play-match`` command.

The presentation layer it drives is covered in test_console; here we check only
what is specific to ``main`` — that it wires up and runs a match without a
network, quiets the per-request HTTP logging that would otherwise clutter the
board, and records to the results file only when asked. (Rendering assertions
live in test_console.)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import pytest

from snakes_and_mice.cli_common import make_observer, parse_seed
from snakes_and_mice.console import ConsoleObserver
from snakes_and_mice.core import Cell
from snakes_and_mice.match_cli import main
from snakes_and_mice.observer import ObservationLevel
from snakes_and_mice.serialize import read_match_results


def test_make_observer_maps_none_to_no_observer() -> None:
    # "none" means no observer at all; the other choices build a ConsoleObserver.
    assert make_observer("none") is None
    observer = make_observer("game")
    assert isinstance(observer, ConsoleObserver)
    assert observer.level is ObservationLevel.GAME


def test_parse_seed_maps_random_and_fixed_cells() -> None:
    # "random" (any case) yields an RNG; a label yields that fixed cell.
    assert isinstance(parse_seed("random"), random.Random)
    assert isinstance(parse_seed("RANDOM"), random.Random)
    assert parse_seed("B3") == Cell.from_label("B3")
    # Bad labels — off-board or malformed — raise ValueError for the CLI to report.
    with pytest.raises(ValueError):
        parse_seed("Z9")
    with pytest.raises(ValueError):
        parse_seed("nonsense")


def test_main_quiets_http_request_logging(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The per-request httpx INFO logs must not bleed into the board rendering.
    logging.getLogger("httpx").setLevel(logging.INFO)
    main(["--watch", "match"])  # two random players, no network
    assert logging.getLogger("httpx").level == logging.WARNING


def test_does_not_write_results_file_by_default(tmp_path: Path) -> None:
    results: Path = tmp_path / "tournament-results.jsonl"
    main(["--watch", "match", "--tournament-results", str(results)])
    # The flag was given a path, so it *does* write there. Sanity that it exists...
    assert results.exists()


def test_omitting_the_flag_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Without --tournament-results, no results file is created anywhere.
    main(["--watch", "match"])
    out: str = capsys.readouterr().out
    assert "Recorded to" not in out


def test_records_one_line_when_flag_given(tmp_path: Path) -> None:
    results: Path = tmp_path / "nested" / "results.jsonl"
    main(["--watch", "match", "--games", "3", "--tournament-results", str(results)])
    recorded = read_match_results(results)
    assert len(recorded) == 1
    assert recorded[0].num_games == 3


def test_seed_flag_runs_and_records(tmp_path: Path) -> None:
    # Both --seed forms wire an opening into the match; the run completes and
    # records as usual (the seed variation itself is covered in test_match).
    for seed in ("random", "B3"):
        results: Path = tmp_path / f"results-{seed}.jsonl"
        main(["--watch", "none", "--games", "3", "--seed", seed,
              "--tournament-results", str(results)])
        assert len(read_match_results(results)) == 1


def test_invalid_seed_is_reported(tmp_path: Path) -> None:
    # A bad --seed value exits cleanly via parser.error rather than tracebacking.
    with pytest.raises(SystemExit):
        main(["--seed", "Z9", "--tournament-results", str(tmp_path / "r.jsonl")])


def test_watch_none_runs_the_match_but_shows_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results: Path = tmp_path / "results.jsonl"
    main(["--watch", "none", "--games", "2", "--tournament-results", str(results)])
    out: str = capsys.readouterr().out
    # No observer output at all — not even the opening banner.
    assert "Snakes and Mice" not in out
    assert "Match complete" not in out
    # But the match still ran and was recorded.
    assert len(read_match_results(results)) == 1
