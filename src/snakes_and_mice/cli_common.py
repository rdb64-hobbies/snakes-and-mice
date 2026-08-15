"""Shared plumbing for the command-line frontends.

The three commands (§7) — ``play-match``, ``play-tournament-matches``, and
``tally-tournament`` — each live in their own thin module, but they overlap on a
few concerns: quieting HTTP request logging, building a player from a roster name,
and offering ``--watch`` (§5). Those pieces live here so no single command owns
them. Presentation stays in :mod:`console`; nothing here parses a full command
line — each frontend builds its own parser and calls these helpers.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import Roster, make_llm_player
from .console import ConsoleObserver
from .core import Side
from .observer import ObservationLevel, Observer
from .players import HumanPlayer, Player, RandomPlayer

RANDOM_NAME: dict[Side, str] = {Side.MOUSE: "Randy", Side.SNAKE: "Ransom"}
DEFAULT_LOG_DIR: str = "llm-logs"
DEFAULT_RESULTS_PATH: Path = Path("tournament-results.jsonl")
"""Where the tournament results file (§6) lives unless a command overrides it."""

WATCH_CHOICES: tuple[str, ...] = ("none", "match", "game", "move")

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


def quiet_http_logging() -> None:
    """Silence per-request HTTP INFO logs so they don't clutter the board.

    Only the CLI does this: it owns the terminal, whereas the library must not
    reconfigure a host application's logging.
    """
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def make_player(
    kind: str, side: Side, roster: Roster | None, log_dir: Path | None
) -> Player:
    """Build the player for one side. ``kind`` is ``random``, ``human``, or an
    LLM roster name (in which case ``roster`` must be loaded)."""
    if kind == "human":
        return HumanPlayer(name="You")
    if kind == "random":
        return RandomPlayer(name=RANDOM_NAME[side])
    assert roster is not None  # a roster is loaded whenever an LLM name is used
    return make_llm_player(kind, roster, log_dir=log_dir)


def add_watch_argument(parser: argparse.ArgumentParser, *, default: str) -> None:
    """Add the shared ``--watch none|match|game|move`` option (§5) with ``default``."""
    parser.add_argument(
        "--watch", choices=list(WATCH_CHOICES), default=default,
        help="how much to show: nothing, match, game, or every move "
             f"(default: {default})",
    )


def observation_level(watch: str) -> ObservationLevel:
    """Map a ``--watch`` choice string to its :class:`ObservationLevel`.

    Only the watching levels map here; ``none`` has no level (it means no
    observer at all) and is handled by :func:`make_observer`.
    """
    return ObservationLevel[watch.upper()]


def make_observer(watch: str) -> Observer | None:
    """The console observer for a ``--watch`` choice, or ``None`` for ``none``.

    ``none`` maps to no observer at all, so the engine runs down its existing
    "no watcher" path and stays silent — cleaner than a do-nothing observer.
    """
    if watch == "none":
        return None
    return ConsoleObserver(observation_level(watch))
