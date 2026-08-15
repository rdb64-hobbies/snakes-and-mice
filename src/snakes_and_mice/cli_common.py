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
import random
from pathlib import Path

from .config import Roster, make_llm_player
from .console import ConsoleObserver
from .core import Cell, Side
from .faults import IllegalMove
from .observer import ObservationLevel, Observer
from .players import HumanPlayer, Player, RandomPlayer

SEED_DEFAULT: str = "random"

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


def add_seed_argument(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--seed`` option: ``random`` (the default) or a fixed cell."""
    parser.add_argument(
        "--seed", default=SEED_DEFAULT, metavar="CELL",
        help="where the snake is seeded each game: 'random' (default) or a fixed "
             "cell like B3",
    )


def parse_seed(value: str) -> Cell | random.Random:
    """Turn a ``--seed`` argument into an opening for :func:`play_match`.

    ``random`` yields a fresh :class:`random.Random` (a new seed cell per game);
    any other value is parsed as a fixed cell label like ``B3``. Raises
    :class:`ValueError` with a CLI-friendly message if the label is not a valid
    on-board cell, so the caller can report it via ``parser.error``.
    """
    if value.lower() == SEED_DEFAULT:
        return random.Random()
    try:
        return Cell.from_label(value)
    except (ValueError, IllegalMove) as exc:
        raise ValueError(
            f"invalid --seed {value!r}: use 'random' or a cell label like B3"
        ) from exc


def make_observer(watch: str) -> Observer | None:
    """The console observer for a ``--watch`` choice, or ``None`` for ``none``.

    ``none`` maps to no observer at all, so the engine runs down its existing
    "no watcher" path and stays silent — cleaner than a do-nothing observer.
    """
    if watch == "none":
        return None
    return ConsoleObserver(observation_level(watch))
