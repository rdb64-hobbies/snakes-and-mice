"""Measure what ranking the equally-optimal pool is worth (§10).

The perfect player never loses, so its strength against a fallible opponent shows up
entirely in how often it converts a draw into a win. This plays it against the
**random** player — a maximally fallible opponent, whose loss rate is therefore a
direct estimate of P(the opponent goes wrong) — under every `TieBreak` (§10,
"Selecting a variant"), in both seats, and prints them side by side. This is the
intended way to re-run the comparison whenever the keys or their gates change.

Openings and the opponent's RNG are seeded identically across all policies, so a
difference in the tallies is attributable to the ranking rather than to the draw.

    uv run python tools/bench_tie_break.py [GAMES] [SEED]
"""

from __future__ import annotations

import random
import sys
import time

from snakes_and_mice.core import Side
from snakes_and_mice.match import play_match
from snakes_and_mice.players.perfect import PerfectPlayer, TieBreak
from snakes_and_mice.players.random import RandomPlayer
from snakes_and_mice.result import MatchResult


def main() -> None:
    games: int = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    seed: int = int(sys.argv[2]) if len(sys.argv) > 2 else 7

    print(f"perfect vs random — {games} games per seat, opening seed {seed}\n")
    header: str = (
        f"{'policy':14s} {'seat':6s} {'won':>5s} {'drew':>5s} {'lost':>5s} "
        f"{'win%':>7s} {'secs':>7s}"
    )
    print(header)
    print("-" * len(header))

    for policy in TieBreak:
        label: str = policy.value
        total_won: int = 0
        for seat in (Side.MOUSE, Side.SNAKE):
            # Fresh instances per match, identically seeded, so every policy meets
            # the same openings and the same opponent behaviour.
            perfect: PerfectPlayer = PerfectPlayer(
                name="perfect", rng=random.Random(99), tie_break=policy
            )
            opponent: RandomPlayer = RandomPlayer(name="random", rng=random.Random(1234))
            mouse, snake = (
                (perfect, opponent) if seat is Side.MOUSE else (opponent, perfect)
            )
            started: float = time.perf_counter()
            result: MatchResult = play_match(
                mouse, snake, games, opening=random.Random(seed)
            )
            elapsed: float = time.perf_counter() - started

            won: int = (
                result.mouse_wins if seat is Side.MOUSE else result.snake_wins
            )
            lost: int = (
                result.snake_wins if seat is Side.MOUSE else result.mouse_wins
            )
            total_won += won
            print(
                f"{label:14s} {seat.name.lower():6s} {won:5d} {result.cats_games:5d} "
                f"{lost:5d} {won / games:6.1%} {elapsed:7.1f}",
                flush=True,
            )
        played: int = games * 2
        print(
            f"{'':14s} {'both':6s} {total_won:5d} of {played} = "
            f"{total_won / played:.1%}\n"
        )


if __name__ == "__main__":
    main()
