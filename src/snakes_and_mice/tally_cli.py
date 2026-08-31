"""The ``tally-tournament`` command: print per-player standings (§7).

Reads a results file (§6), narrows it to the matches worth counting, aggregates
those into per-player standings, orders them by the chosen rate, and prints the
table — optionally with the fault breakdown and the head-to-head matrix.

Alone among the three commands it runs no games and reads no configuration at all
— no roster, no providers, no keys (§6, "The results file stands alone") — so it
takes from :mod:`cli_common` only the shared results-file default. Presentation is
in :mod:`console`; the selection, aggregation and ordering are in :mod:`tally`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .cli_common import DEFAULT_RESULTS_PATH
from .console import render_fault_tally, render_head_to_head, render_standings
from .faults import TournamentError
from .result import MatchResult
from .serialize import read_match_results
from .tally import (
    HeadToHead,
    PlayerStanding,
    StandingsSort,
    head_to_head,
    select_matches,
    sort_standings,
    tally,
)

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
    ``win%``), ties breaking by name. Three flags, all off by default, add to or
    narrow that table: ``--faults`` appends the per-player breakdown of fault
    reasons, ``--head-to-head`` appends the n × n matrix of who beat whom, and
    ``--players`` / ``--except`` restrict the tally to the matches played among a
    set of players.
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
    parser.add_argument(
        "--head-to-head", action="store_true",
        help="append the matrix of who beat whom: cell (row, column) counts the row "
             "player's losses to the column player",
    )
    # One operation with two spellings, so argparse rejects both at once.
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--players", nargs="+", metavar="NAME",
        help="tally only the matches played among the named players",
    )
    selection.add_argument(
        "--except", dest="excluded", nargs="+", metavar="NAME",
        help="tally only the matches played among everyone else",
    )
    args: argparse.Namespace = parser.parse_args(argv)

    try:
        results: list[MatchResult] = read_match_results(args.tournament_results)
        # Raises on a name the file does not hold, hence inside the try.
        selected: list[MatchResult] = select_matches(
            results, players=args.players, excluded=args.excluded
        )
    except TournamentError as exc:
        parser.error(str(exc))

    sort: StandingsSort = _SORT_CHOICES[args.sort]
    standings: list[PlayerStanding] = sort_standings(tally(selected), sort)
    filtered: bool = args.players is not None or args.excluded is not None
    print(render_standings(standings, sort, filtered=filtered))

    if args.faults:
        print()
        print(render_fault_tally(standings))
    # The matrix's rows and columns are the ranked standings' players.
    if args.head_to_head and len(standings) >= 2:
        matrix: HeadToHead = head_to_head(selected, [s.name for s in standings])
        print()
        print(render_head_to_head(matrix))
