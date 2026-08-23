"""Tests for the ``play-tournament-matches`` batch runner.

Selector parsing and the empty/unknown-subset error paths need no network. The
end-to-end batch run is exercised with the roster and player construction patched
so matches play out between local random players instead of real LLMs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from snakes_and_mice.config import PlayerSpec, Roster
from snakes_and_mice.core import Side
from snakes_and_mice.matches_cli import _parse_selector, main
from snakes_and_mice.players import Player, RandomPlayer
from snakes_and_mice.schedule import (
    AllPlayers,
    NamedPlayers,
    PlayersAbove,
    PlayersBelow,
    SamePlayers,
)
from snakes_and_mice.serialize import read_match_results


def _roster(*names: str) -> Roster:
    return Roster(
        players={n: PlayerSpec(name=n, provider="p", model="m") for n in names},
        providers={},
    )


def _patch_runner(monkeypatch: pytest.MonkeyPatch, roster: Roster) -> None:
    """Make the batch run offline: fixed roster, random players, no .env read."""
    monkeypatch.setattr("snakes_and_mice.matches_cli.load_roster", lambda: roster)
    monkeypatch.setattr("snakes_and_mice.matches_cli.load_environment", lambda: None)

    def fake_make_player(
        kind: str, side: Side, roster_: Roster | None, log_dir: Path | None,
        *, prune_thinking: bool = False,
    ) -> Player:
        return RandomPlayer(name=kind)

    monkeypatch.setattr("snakes_and_mice.matches_cli.make_player", fake_make_player)


# --------------------------------------------------------------------------- #
# Selector parsing
# --------------------------------------------------------------------------- #


def test_parse_selector_keywords_names_and_pivots() -> None:
    parser = argparse.ArgumentParser()
    assert _parse_selector(parser, ["all"], allow_same=False) == AllPlayers()
    assert _parse_selector(parser, ["same"], allow_same=True) == SamePlayers()
    assert _parse_selector(parser, ["above", "x"], allow_same=False) == PlayersAbove("x")
    assert _parse_selector(parser, ["below", "y"], allow_same=False) == PlayersBelow("y")
    assert _parse_selector(parser, ["a", "b"], allow_same=False) == NamedPlayers(("a", "b"))


def test_parse_selector_same_is_rejected_for_players_subset() -> None:
    parser = argparse.ArgumentParser()
    with pytest.raises(SystemExit):
        _parse_selector(parser, ["same"], allow_same=False)


def test_parse_selector_pivot_needs_exactly_one_name() -> None:
    parser = argparse.ArgumentParser()
    with pytest.raises(SystemExit):
        _parse_selector(parser, ["above"], allow_same=False)


# --------------------------------------------------------------------------- #
# End-to-end batch run
# --------------------------------------------------------------------------- #


def test_default_round_robin_plays_and_appends_every_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_runner(monkeypatch, _roster("a", "b", "c"))
    results: Path = tmp_path / "results.jsonl"
    main(["--watch", "match", "--games", "1", "--tournament-results", str(results)])
    # all vs same over three players → N*(N-1) = 6 matches, one line each.
    assert len(read_match_results(results)) == 6


def test_empty_selection_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_runner(monkeypatch, _roster("solo"))  # round-robin of one → no matches
    with pytest.raises(SystemExit):
        main(["--tournament-results", str(tmp_path / "results.jsonl")])


def test_unknown_player_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_runner(monkeypatch, _roster("a", "b"))
    with pytest.raises(SystemExit):
        main(["--players", "ghost", "--tournament-results", str(tmp_path / "r.jsonl")])
