"""Command-line entry point: parse arguments, wire up players, run the match.

The presentation layer — board rendering, result summaries, and the stdout
observer — lives in :mod:`console` so it can be shared by other frontends. This
module keeps only what is specific to *this* command: argument parsing, player
construction from the roster, quieting HTTP logging, and ``main``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import (
    ConfigError,
    Roster,
    load_environment,
    load_roster,
    make_llm_player,
)
from .console import ConsoleObserver
from .core import Side
from .match import play_match
from .observer import ObservationLevel
from .players import HumanPlayer, ModelRequestError, Player, RandomPlayer

_RANDOM_NAME: dict[Side, str] = {Side.MOUSE: "Randy", Side.SNAKE: "Ransom"}
_DEFAULT_LOG_DIR: str = "llm-logs"

# HTTP/SDK client loggers that would otherwise print a per-request line (an httpx
# "HTTP Request: POST ..." INFO, or an Anthropic request-id DEBUG) into the middle
# of the board.
_NOISY_LOGGERS: tuple[str, ...] = (
    "httpx",
    "httpcore",
    "openai",
    "anthropic",
    "google_genai",
)


def _quiet_http_logging() -> None:
    """Silence per-request HTTP INFO logs so they don't clutter the board.

    Only the CLI does this: it owns the terminal, whereas the library must not
    reconfigure a host application's logging.
    """
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _make_player(
    kind: str, side: Side, roster: Roster | None, log_dir: Path | None
) -> Player:
    """Build the player for one side. ``kind`` is ``random``, ``human``, or an
    LLM roster name (in which case ``roster`` must be loaded)."""
    if kind == "human":
        return HumanPlayer(name="You")
    if kind == "random":
        return RandomPlayer(name=_RANDOM_NAME[side])
    assert roster is not None  # a roster is loaded whenever an LLM name is used
    return make_llm_player(kind, roster, log_dir=log_dir)


def main(argv: list[str] | None = None) -> None:
    """Play or watch a match of one or more games.

    By default two random bots play a single game at ``move`` detail. Pass
    ``--games N`` for a longer match and ``--watch match|game|move`` to choose
    how much is shown. ``--mouse`` and ``--snake`` each name who plays that side:
    ``random``, ``human``, or an LLM roster name from ``players.yaml``. A human at
    the board always forces ``move`` detail (with a note), since a human must see
    every move to play it. ``--log-llm [DIR]`` dumps each LLM player's full raw
    message thread as JSON for debugging.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="snakes-and-mice",
        description="Play or watch a match of Snakes and Mice.",
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
    parser.add_argument(
        "--watch", choices=["match", "game", "move"], default="move",
        help="how much to show: match, game, or every move (default: move)",
    )
    parser.add_argument(
        "--log-llm", nargs="?", const=_DEFAULT_LOG_DIR, default=None, metavar="DIR",
        help=(
            "dump each LLM player's full message thread as JSON for debugging "
            f"(default DIR: {_DEFAULT_LOG_DIR}/); no-op for random/human"
        ),
    )
    args: argparse.Namespace = parser.parse_args(argv)

    if args.games < 1:
        parser.error("--games must be at least 1")

    _quiet_http_logging()

    log_dir: Path | None = Path(args.log_llm) if args.log_llm is not None else None
    builtin: set[str] = {"random", "human"}
    needs_roster: bool = args.mouse not in builtin or args.snake not in builtin

    roster: Roster | None = None
    try:
        if needs_roster:
            load_environment()
            roster = load_roster()
        mouse: Player = _make_player(args.mouse, Side.MOUSE, roster, log_dir)
        snake: Player = _make_player(args.snake, Side.SNAKE, roster, log_dir)
    except ConfigError as exc:
        parser.error(str(exc))

    has_human: bool = "human" in (args.mouse, args.snake)
    level: ObservationLevel = ObservationLevel[args.watch.upper()]
    if has_human and level < ObservationLevel.MOVE:
        print("(a human is playing — showing every move)\n")
        level = ObservationLevel.MOVE
    try:
        play_match(mouse, snake, args.games, ConsoleObserver(level))
    except ModelRequestError as exc:
        # A provider call failed mid-game (e.g. an unavailable model); report it
        # as a clean message instead of an escaping traceback.
        parser.error(str(exc))
