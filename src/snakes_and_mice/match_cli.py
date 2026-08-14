"""The ``play-match`` command: play or watch a single match (§7).

This is the debugging/casual-play frontend — a human at the board, a quick game
against Random, or an LLM match with a full ``--log-llm`` dump. It writes to the
tournament results file only when ``--tournament-results`` is given (§6); by
default it leaves that file untouched. Shared plumbing (player construction, HTTP
quieting, ``--watch``) lives in :mod:`cli_common`; presentation is in
:mod:`console`. This module keeps only what is specific to *this* command.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .cli_common import (
    DEFAULT_LOG_DIR,
    DEFAULT_RESULTS_PATH,
    add_watch_argument,
    make_player,
    observation_level,
    quiet_http_logging,
)
from .config import ConfigError, Roster, load_environment, load_roster
from .console import ConsoleObserver
from .core import Side
from .match import play_match
from .observer import ObservationLevel
from .players import ModelRequestError, Player
from .result import MatchResult
from .serialize import append_match_result


def main(argv: list[str] | None = None) -> None:
    """Play or watch a match of one or more games.

    By default two random bots play a single game at ``move`` detail. Pass
    ``--games N`` for a longer match and ``--watch match|game|move`` to choose
    how much is shown. ``--mouse`` and ``--snake`` each name who plays that side:
    ``random``, ``human``, or an LLM roster name from ``players.yaml``. A human at
    the board always forces ``move`` detail (with a note), since a human must see
    every move to play it. ``--log-llm [DIR]`` dumps each LLM player's full raw
    message thread as JSON for debugging. ``--tournament-results [FILE]`` records
    the match to the results file (§6); omit it and nothing is written.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="play-match",
        description="Play or watch a single match of Snakes and Mice.",
    )
    parser.add_argument(
        "--mouse", default="random", metavar="WHO",
        help="who plays Mouse: random, human, or an LLM roster name (default: random)",
    )
    parser.add_argument(
        "--snake", default="random", metavar="WHO",
        help="who plays Snake: random, human, or an LLM roster name (default: random)",
    )
    parser.add_argument(
        "--games", type=int, default=1, metavar="N",
        help="number of games in the match (default: 1)",
    )
    add_watch_argument(parser, default="move")
    parser.add_argument(
        "--log-llm", nargs="?", const=DEFAULT_LOG_DIR, default=None, metavar="DIR",
        help=(
            "dump each LLM player's full message thread as JSON for debugging "
            f"(default DIR: {DEFAULT_LOG_DIR}/); no-op for random/human"
        ),
    )
    parser.add_argument(
        "--tournament-results", nargs="?", const=str(DEFAULT_RESULTS_PATH),
        default=None, metavar="FILE",
        help=(
            "record this match to the tournament results file (§6); off by default, "
            f"bare appends to {DEFAULT_RESULTS_PATH}, or give a path"
        ),
    )
    args: argparse.Namespace = parser.parse_args(argv)

    if args.games < 1:
        parser.error("--games must be at least 1")

    quiet_http_logging()

    log_dir: Path | None = Path(args.log_llm) if args.log_llm is not None else None
    results_path: Path | None = (
        Path(args.tournament_results) if args.tournament_results is not None else None
    )
    builtin: set[str] = {"random", "human"}
    needs_roster: bool = args.mouse not in builtin or args.snake not in builtin

    roster: Roster | None = None
    try:
        if needs_roster:
            load_environment()
            roster = load_roster()
        mouse: Player = make_player(args.mouse, Side.MOUSE, roster, log_dir)
        snake: Player = make_player(args.snake, Side.SNAKE, roster, log_dir)
    except ConfigError as exc:
        parser.error(str(exc))

    has_human: bool = "human" in (args.mouse, args.snake)
    level: ObservationLevel = observation_level(args.watch)
    if has_human and level < ObservationLevel.MOVE:
        print("(a human is playing — showing every move)\n")
        level = ObservationLevel.MOVE
    try:
        result: MatchResult = play_match(mouse, snake, args.games, ConsoleObserver(level))
    except ModelRequestError as exc:
        # A provider call failed mid-game (e.g. an unavailable model); report it
        # as a clean message instead of an escaping traceback.
        parser.error(str(exc))

    if results_path is not None:
        append_match_result(result, results_path)
        print(f"\nRecorded to {results_path}")
