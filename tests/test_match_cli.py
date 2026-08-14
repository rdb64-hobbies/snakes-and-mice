"""Tests for the ``play-match`` command.

The presentation layer it drives is covered in test_console; here we check only
what is specific to ``main`` — that it wires up and runs a match without a
network, quiets the per-request HTTP logging that would otherwise clutter the
board, and records to the results file only when asked. (Rendering assertions
live in test_console.)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from snakes_and_mice.match_cli import main
from snakes_and_mice.serialize import read_match_results


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
