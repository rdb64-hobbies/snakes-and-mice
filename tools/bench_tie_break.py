"""Measure what ranking the equally-optimal pool is worth (§10).

The perfect player never loses, so its strength against a fallible opponent shows up
entirely in how often it converts a draw into a win. This plays it against the
**random** player — a maximally fallible opponent, whose loss rate is therefore a
direct estimate of P(the opponent goes wrong) — with ranking on and off, in both
seats, and prints the two side by side.

Ranking is deliberately not configurable on the player itself: a strength knob would
make `perfect` mean two different things under one name, and a results file
identifies a player by name alone (§6). So the unranked baseline is reconstructed
here, in a throwaway subclass that restores the pre-1.4 pick. That is the intended
way to re-run this comparison whenever the keys or their gates change.

Openings and the opponent's RNG are seeded identically across both policies, so a
difference in the tallies is attributable to the ranking rather than to the draw.

    uv run python tools/bench_tie_break.py [GAMES] [SEED]
"""

from __future__ import annotations

import random
import sys
import time

from snakes_and_mice.core import Move, Side
from snakes_and_mice.match import play_match
from snakes_and_mice.players.perfect import PerfectPlayer, _Candidate
from snakes_and_mice.players.random import RandomPlayer
from snakes_and_mice.result import MatchResult


class UnrankedPerfectPlayer(PerfectPlayer):
    """The perfect player as it was before 1.4: uniform over the whole pool.

    Still perfect — every candidate shares the exact minimax value — just with no
    preference among them, so it gives the opponent no help in going wrong.
    """

    def _pick(
        self, pool: list[_Candidate], best: int, empties: int, depth: int
    ) -> Move:
        return self._rng.choice(pool).move


def main() -> None:
    games: int = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    seed: int = int(sys.argv[2]) if len(sys.argv) > 2 else 7

    print(f"perfect vs random — {games} games per seat, opening seed {seed}\n")
    header: str = (
        f"{'policy':10s} {'seat':6s} {'won':>5s} {'drew':>5s} {'lost':>5s} "
        f"{'win%':>7s} {'secs':>7s}"
    )
    print(header)
    print("-" * len(header))

    policies: tuple[tuple[str, type[PerfectPlayer]], ...] = (
        ("unranked", UnrankedPerfectPlayer),
        ("ranked", PerfectPlayer),
    )
    for label, player_type in policies:
        total_won: int = 0
        for seat in (Side.MOUSE, Side.SNAKE):
            # Fresh instances per match, identically seeded, so the two policies meet
            # the same openings and the same opponent behaviour.
            perfect: PerfectPlayer = player_type(name="perfect", rng=random.Random(99))
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
                f"{label:10s} {seat.name.lower():6s} {won:5d} {result.cats_games:5d} "
                f"{lost:5d} {won / games:6.1%} {elapsed:7.1f}",
                flush=True,
            )
        played: int = games * 2
        print(
            f"{'':10s} {'both':6s} {total_won:5d} of {played} = "
            f"{total_won / played:.1%}\n"
        )


if __name__ == "__main__":
    main()
