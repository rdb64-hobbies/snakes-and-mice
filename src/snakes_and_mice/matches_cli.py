"""The ``play-tournament-matches`` command: run many matches (§7).

The batch runner. It resolves two player *subsets* into a schedule of ordered
``(mouse, snake)`` pairs (§6), plays each as its own match, and appends every
result to the shared results file. Unlike ``play-match``, appending is intrinsic
here — ``--tournament-results`` only overrides the path. Human players cannot be
selected (a batch runs unattended), and there is no ``--log-llm``: a per-turn
dump across a large batch is not useful.

Selector parsing (turning ``--players``/``--against`` tokens into a
:class:`~snakes_and_mice.schedule.Selector`) is this command's own concern and
lives here; the schedule computation itself is in :mod:`schedule`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .cli_common import (
    DEFAULT_RESULTS_PATH,
    add_watch_argument,
    make_observer,
    make_player,
    quiet_http_logging,
)
from .config import ConfigError, Roster, load_environment, load_roster
from .core import Side
from .faults import TournamentError
from .match import play_match
from .players import ModelRequestError, Player
from .result import MatchResult
from .schedule import (
    AllPlayers,
    NamedPlayers,
    PlayersAbove,
    PlayersBelow,
    SamePlayers,
    Selector,
    plan_schedule,
)
from .serialize import append_match_result

# Common --players / --against combinations, shown at the bottom of --help. The
# defaults (all vs same) give a round-robin; the rest illustrate the other subset
# forms — an explicit list, the newcomer cross, and above/below cohorts.
_EXAMPLES: str = """\
examples (each match pairs a --players entry against an --against entry):
  play-tournament-matches
      full round-robin: every player meets every other, both seats (all vs same)
  play-tournament-matches --players alice bob carol
      round-robin among just those three (--against defaults to same)
  play-tournament-matches --players newbie --against all
      the newcomer against the whole roster, both seats, without making the
      existing players replay one another
  play-tournament-matches --players above gpt5 --against below gpt5
      the stronger cohort vs the weaker one (by players.yaml order)
"""


def main(argv: list[str] | None = None) -> None:
    """Play many matches from two player subsets and append each result (§6).

    ``--players`` selects subset A (default ``all``) and ``--against`` subset B
    (default ``same``); every match straddles the two, both seatings. Each selector
    is one of: an explicit list of roster names, ``all``, ``same`` (``--against``
    only), ``above <name>``, or ``below <name>`` (the pivot exclusive, using
    ``players.yaml`` order). ``--games N`` sets a uniform game count and ``--watch``
    the detail (default ``game``). Results append to the tournament file; pass
    ``--tournament-results FILE`` to point elsewhere.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="play-tournament-matches",
        description="Play and optionally watch multiple Snakes and Mice matches, "
                    "and record them as tournament results.",
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--players", nargs="+", default=["all"], metavar="SELECTOR",
        help="the players to run: roster names, or all | above <name> | below <name> "
        "(default: all)",
    )
    parser.add_argument(
        "--against", nargs="+", default=["same"], metavar="SELECTOR",
        help="their opponents: roster names, or all | same | above <name> | below <name> "
        "(default: same)",
    )
    parser.add_argument(
        "--games", type=int, default=1, metavar="N",
        help="number of games in each match (default: 1)",
    )
    add_watch_argument(parser, default="game")
    parser.add_argument(
        "--tournament-results", type=Path, default=DEFAULT_RESULTS_PATH, metavar="FILE",
        help=f"results file to append to (default: {DEFAULT_RESULTS_PATH})",
    )
    args: argparse.Namespace = parser.parse_args(argv)

    if args.games < 1:
        parser.error("--games must be at least 1")

    selector_a: Selector = _parse_selector(parser, args.players, allow_same=False)
    selector_b: Selector = _parse_selector(parser, args.against, allow_same=True)

    quiet_http_logging()
    results_path: Path = args.tournament_results

    try:
        load_environment()
        roster: Roster = load_roster()
        roster_order: list[str] = list(roster.players)
        schedule: list[tuple[str, str]] = plan_schedule(
            selector_a, selector_b, roster_order
        )
    except (ConfigError, TournamentError) as exc:
        parser.error(str(exc))

    if not schedule:
        parser.error("the selected players and opponents produce no matches to play")

    total: int = len(schedule)
    try:
        for index, (mouse_name, snake_name) in enumerate(schedule, start=1):
            print(f"\n### Match {index}/{total}: {mouse_name} (mouse) "
                  f"vs {snake_name} (snake) ###")
            mouse: Player = make_player(mouse_name, Side.MOUSE, roster, None)
            snake: Player = make_player(snake_name, Side.SNAKE, roster, None)
            # A fresh observer per match, so its counters and scoreboard reset.
            result: MatchResult = play_match(
                mouse, snake, args.games, make_observer(args.watch)
            )
            append_match_result(result, results_path)
    except ConfigError as exc:
        parser.error(str(exc))
    except ModelRequestError as exc:
        # A provider/config failure dooms every remaining match, not just this one.
        # Results appended so far are already on disk (append-only), so stop cleanly.
        parser.error(str(exc))

    print(f"\nPlayed {total} {'match' if total == 1 else 'matches'}; "
          f"appended to {results_path}")


def _parse_selector(
    parser: argparse.ArgumentParser, tokens: list[str], *, allow_same: bool
) -> Selector:
    """Turn one ``--players``/``--against`` token list into a :class:`Selector`.

    ``all`` and ``same`` are single keywords; ``above``/``below`` take exactly one
    pivot name; anything else is an explicit list of names. ``same`` is valid only
    for ``--against`` (``allow_same``). Bad usage calls ``parser.error`` and exits.
    """
    if tokens == ["all"]:
        return AllPlayers()
    if tokens == ["same"]:
        if not allow_same:
            parser.error("'same' is only valid for --against, not --players")
        return SamePlayers()
    if tokens[0] in ("above", "below"):
        if len(tokens) != 2:
            parser.error(f"'{tokens[0]}' takes exactly one player name")
        return PlayersAbove(tokens[1]) if tokens[0] == "above" else PlayersBelow(tokens[1])
    return NamedPlayers(tuple(tokens))
