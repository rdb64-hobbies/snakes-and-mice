"""The ``tally-tournament`` command: print per-player standings (§7).

Reads a results file (§6), aggregates it into per-player standings, orders them
by the chosen rate, and prints the table. It runs no games and needs no API keys;
it only reads ``players.yaml`` — when present — to know the roster order that
breaks ties (and is tolerant of its absence). Presentation is in :mod:`console`;
the aggregation and ordering are in :mod:`tally`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .cli_common import DEFAULT_RESULTS_PATH
from .roster import ConfigError, load_roster
from .console import render_fault_tally, render_standings
from .faults import TournamentError
from .result import MatchResult
from .serialize import read_match_results
from .tally import PlayerStanding, StandingsSort, sort_standings, tally

# The --sort choices are spelled with the trailing % the standings columns use.
_SORT_CHOICES: dict[str, StandingsSort] = {
    "win%": StandingsSort.WIN,
    "loss%": StandingsSort.LOSS,
    "fault%": StandingsSort.FAULT,
}


def main(argv: list[str] | None = None) -> None:
    """Read a results file and print per-player standings (§6).

    ``--tournament-results FILE`` selects the file (default the shared results
    file) and ``--sort win%|loss%|fault%`` orders the table best-on-top (default
    ``win%``). Ties keep ``players.yaml`` order; if that file is unavailable, ties
    fall back to alphabetical name order. ``--faults`` appends a per-player
    breakdown of fault types for every player that faulted.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="tally-tournament",
        description="Tally a Snakes and Mice tournament results file into standings.",
    )
    parser.add_argument(
        "--tournament-results", type=Path, default=DEFAULT_RESULTS_PATH, metavar="FILE",
        help=f"results file to read (default: {DEFAULT_RESULTS_PATH})",
    )
    parser.add_argument(
        "--sort", choices=list(_SORT_CHOICES), default="win%",
        help="which rate ranks the standings, best-on-top (default: win%%)",
    )
    parser.add_argument(
        "--faults", action="store_true",
        help="append a per-player breakdown of fault types",
    )
    args: argparse.Namespace = parser.parse_args(argv)

    try:
        results: list[MatchResult] = read_match_results(args.tournament_results)
    except TournamentError as exc:
        parser.error(str(exc))

    roster_order: list[str] = _roster_order()
    sort: StandingsSort = _SORT_CHOICES[args.sort]
    standings: list[PlayerStanding] = sort_standings(tally(results), sort, roster_order)
    print(render_standings(standings, sort))
    if args.faults:
        print()
        print(render_fault_tally(standings))


def _roster_order() -> list[str]:
    """The roster's names in ``players.yaml`` order, for tie-breaking.

    A tally should still work off the results file alone, so a missing or malformed
    roster is not fatal: warn and fall back to an empty order (ties then break by
    name), rather than refusing to print.
    """
    try:
        return list(load_roster().players)
    except ConfigError as exc:
        print(f"warning: {exc}; ranking ties by name only", file=sys.stderr)
        return []
