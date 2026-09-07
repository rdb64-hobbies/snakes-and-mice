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

from .roster import Roster
from .console import ConsoleObserver, SECOND_PERSON
from .core import Cell, Side
from .faults import IllegalMove
from .mistakes import DEFAULT_MISTAKES_PATH, MistakeObserver
from .observer import BroadcastObserver, ObservationLevel, Observer
from .players import HumanPlayer, LLMPlayer, PerfectPlayer, Player, RandomPlayer

SEED_DEFAULT: str = "random"

BUILTIN_KINDS: frozenset[str] = frozenset({"human", "random", "perfect"})
"""The player kinds that are *not* roster names — anything else names an LLM.

One definition, used both to build a player and to decide which sides are worth
grading for mistakes (§5), so the two can never disagree about what an LLM is.
"""

RANDOM_NAME: dict[Side, str] = {Side.MOUSE: "Randy", Side.SNAKE: "Ransom"}
PERFECT_NAME: dict[Side, str] = {Side.MOUSE: "Percy", Side.SNAKE: "Perseus"}
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
    kind: str, side: Side, roster: Roster | None, log_dir: Path | None,
    *, prune_thinking: bool = False,
) -> Player:
    """Build the player for one side. ``kind`` is ``random``, ``human``,
    ``perfect``, or an LLM roster name (in which case ``roster`` must be loaded).
    ``prune_thinking`` applies only to LLM players (§4)."""
    if kind == "human":
        return HumanPlayer(name=SECOND_PERSON)
    if kind == "random":
        return RandomPlayer(name=RANDOM_NAME[side])
    if kind == "perfect":
        return PerfectPlayer(name=PERFECT_NAME[side])
    assert roster is not None  # a roster is loaded whenever an LLM name is used
    return LLMPlayer.from_roster(
        kind, roster, prune_thinking=prune_thinking, log_dir=log_dir
    )


def add_prune_thinking_argument(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--prune-thinking`` opt-in (§4, "Pruning re-sent
    reasoning"), off by default."""
    parser.add_argument(
        "--prune-thinking", action="store_true",
        help="strip earlier turns' reasoning text from each LLM request, to slow "
             "context growth over a long match (default: off)",
    )


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


def add_mistake_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the ``--flag-mistakes`` / ``--mistakes-file`` opt-ins (§5).

    Both are off by default and orthogonal to ``--watch``: flagging annotates
    play with something the level scale has no way to express — that a move was
    *wrong* — so it is wanted at any level, or at none.
    """
    parser.add_argument(
        "--flag-mistakes", action="store_true",
        help="call out legal moves that throw away a win or a draw, as they "
             "happen (default: off)",
    )
    parser.add_argument(
        "--mistakes-file", nargs="?", const=str(DEFAULT_MISTAKES_PATH),
        default=None, metavar="FILE",
        help="record every mistake found as a JSON line; bare to append to "
             f"{DEFAULT_MISTAKES_PATH} (default: off)",
    )


def llm_sides(kinds: dict[Side, str]) -> frozenset[Side]:
    """Which sides are played by an LLM, given each side's ``kind``.

    This is what mistake-flagging grades (§5). The built-in players are left out
    for opposite reasons: ``perfect`` can never qualify, being the ground truth
    the grading measures against, while ``random`` would qualify on nearly every
    turn — so callouts from either carry no signal. It is also what decides
    whether the roster needs loading at all.
    """
    return frozenset(
        side for side, kind in kinds.items() if kind not in BUILTIN_KINDS
    )


def make_observer(
    watch: str,
    *,
    mistake_sides: frozenset[Side] = frozenset(),
    mistakes_path: Path | None = None,
    flag_mistakes: bool = False,
) -> Observer | None:
    """The observer for one match, or ``None`` if nothing is watching.

    ``none`` with no grading maps to no observer at all, so the engine runs down
    its existing "no watcher" path and stays silent — cleaner than a do-nothing
    observer. Otherwise the console renderer and the mistake grader are composed
    with :class:`~snakes_and_mice.observer.BroadcastObserver` — or either alone,
    if only one was asked for.
    """
    console: Observer | None = (
        None if watch == "none" else ConsoleObserver(observation_level(watch))
    )
    grader: Observer | None = None
    if mistake_sides and (flag_mistakes or mistakes_path is not None):
        grader = MistakeObserver(
            mistake_sides, report=flag_mistakes, path=mistakes_path
        )
    if console is not None and grader is not None:
        return BroadcastObserver([console, grader])
    return console if grader is None else grader
