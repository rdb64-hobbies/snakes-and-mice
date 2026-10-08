"""Compare several players against one LLM over an identical set of openings.

Every arm plays **the same 25 seeds, in both seats**, so whatever difficulty an
opening carries cancels between arms instead of adding to the spread, and the
comparison to make is the per-unit difference rather than two marginal rates
(`tally_paired_eval.py`).

**How much that is worth: very little, measured.** This was built on the belief
that opening difficulty was the dominant noise source, citing one RL checkpoint
scoring 4, 5, 12 and 11 losses per 100 games against `perfect`. That reasoning
was wrong. Those six samples have an SD of 3.44 where plain binomial noise at
the observed rate already predicts 2.76 — nearly all of the spread was ordinary
sampling variance, with little left for seeds to explain. On the first real run
(150 games, three `perfect` variants) the paired standard error came out 0.0665
against an unpaired 0.0670, and *worse* than unpaired on another of the three
comparisons.

So pairing is kept because it is free and cannot hurt a comparison, not because
it substitutes for sample size. The honest fix for noise here is more games:
separating a 6-point difference at two standard errors needs ~250 games an arm,
which no design trick avoids.

Output is **one JSON line per game**, so any aggregation is possible afterwards
and a crash mid-run loses nothing already played. Each line records the arm, the
seed, the seat, and the outcome from the arm's point of view.

    uv run python tools/paired_llm_eval.py --opponent qwen-3-8-rtx \\
        --arm rl-move-pref=rl-models/rl-move-pref.pt \\
        --arm rl=rl-models/rl.pt \\
        --arm perfect \\
        --games-per-seed 1 --out paired-eval.jsonl

An ``--arm`` is either a built-in name (`perfect`, `perfect-trappiness`,
`perfect-mistake-model`, `random`) or ``label=path`` naming an RL checkpoint, so
two checkpoints can be compared in one run without juggling
``SNAKES_AND_MICE_RL_MODEL``.

Games are played in **balanced passes** — every arm, every seed, both seats —
so a run interrupted after hours still yields a fair comparison rather than
starving whichever arms the loop had not reached yet. The first attempt at this
ordered seat-outer and arm-inner, and a run that stopped 63 games in had given
all of them to one arm in one seat.

A fresh LLM player is built every ``--games-per-thread`` games per arm and seat,
default 20 to match how the baseline data was collected (`play-match --games
20`). This matters for more than comparability: the message thread grows with
every game an instance plays, so one instance spanning a whole run would both
face a history no baseline ever did and eventually exceed the model's context —
the likely cause of that same 63-game stop, which came after ~378 turns against
the baseline's ~120.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from snakes_and_mice.core import BOARD_SIZE, Cell, Side
from snakes_and_mice.game import play_game
from snakes_and_mice.players import ModelRequestError, Player
from snakes_and_mice.players.llm import LLMPlayer
from snakes_and_mice.players.perfect import PerfectPlayer, TieBreak
from snakes_and_mice.players.random import RandomPlayer
from snakes_and_mice.players.rl import RLPlayer
from snakes_and_mice.result import GameResult, Termination
from snakes_and_mice.roster import ConfigError, load_environment, load_roster

ALL_SEEDS: Final[tuple[Cell, ...]] = tuple(
    Cell(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
)

TIE_BREAKS: Final[dict[str, TieBreak]] = {
    "perfect": TieBreak.NONE,
    "perfect-trappiness": TieBreak.TRAPPINESS,
    "perfect-mistake-model": TieBreak.MISTAKE_MODEL,
}


@dataclass
class Thread:
    """One LLM instance and how many games it has served.

    Tracked so a fresh one replaces it every ``--games-per-thread`` games: the
    message thread grows with each game, and an unbounded one diverges from the
    baseline's conditions and eventually overruns the model's context.
    """

    player: Player
    games: int = 0


@dataclass(frozen=True)
class Arm:
    """One player under comparison: a label, and how to build an instance."""

    label: str
    checkpoint: Path | None

    def build(self, rng: random.Random) -> Player:
        child = random.Random(rng.getrandbits(64))
        if self.checkpoint is not None:
            return RLPlayer.from_checkpoint(self.checkpoint, name=self.label,
                                            rng=child)
        if self.label == "random":
            return RandomPlayer(name=self.label, rng=child)
        return PerfectPlayer(name=self.label, rng=child,
                             tie_break=TIE_BREAKS[self.label])


def parse_arm(text: str) -> Arm:
    """``label=path`` for an RL checkpoint, or a built-in player's name."""
    if "=" in text:
        label, _, path = text.partition("=")
        return Arm(label, Path(path))
    if text not in TIE_BREAKS and text != "random":
        raise ValueError(
            f"unknown arm {text!r}: use a built-in name "
            f"({', '.join(sorted(TIE_BREAKS))}, random) or label=path"
        )
    return Arm(text, None)


def outcome(result: GameResult, side: Side) -> str:
    """The game's result from ``side``'s point of view, as one word."""
    if result.termination is Termination.LINE_COMPLETED:
        return "win" if result.winner is side else "loss"
    if result.termination is Termination.CATS_GAME:
        return "draw"
    if result.termination is Termination.ABORTED:
        return "abort"
    assert result.fault is not None
    return "fault_ours" if result.fault.offender is side else "fault_theirs"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="paired_llm_eval.py",
        description="Compare players against one LLM over identical openings.",
    )
    parser.add_argument("--opponent", required=True, metavar="ROSTER_NAME")
    parser.add_argument("--arm", action="append", required=True, metavar="ARM")
    parser.add_argument("--games-per-seed", type=int, default=1, metavar="N",
                        help="games per seed per seat (default: 1, so 50 per arm)")
    parser.add_argument("--out", type=Path, default=Path("paired-eval.jsonl"))
    parser.add_argument(
        "--games-per-thread", type=int, default=20, metavar="N",
        help="games before a fresh LLM instance is built, per arm and seat "
             "(default: 20, matching `play-match --games 20`, which is how "
             "the baseline data was collected)",
    )
    parser.add_argument("--rng-seed", type=int, default=0)
    args: argparse.Namespace = parser.parse_args(argv)

    try:
        arms: list[Arm] = [parse_arm(text) for text in args.arm]
    except ValueError as exc:
        parser.error(str(exc))

    try:
        load_environment()
        roster = load_roster()
    except ConfigError as exc:
        parser.error(str(exc))
    if args.opponent not in roster.players:
        parser.error(f"{args.opponent!r} is not in the roster: "
                     f"{', '.join(roster.players)}")

    per_arm: int = len(ALL_SEEDS) * 2 * args.games_per_seed
    total: int = len(arms) * per_arm
    print(f"{len(arms)} arms x {per_arm} games = {total} games against "
          f"{args.opponent}, in {args.games_per_seed} balanced pass(es)",
          file=sys.stderr)

    rng = random.Random(args.rng_seed)
    # Arm players are built once per seat and reused: no player here carries
    # state across games (`perfect` clears its table, `rl` holds none), so this
    # only avoids rebuilding them.
    players: dict[tuple[str, Side], Player] = {
        (arm.label, seat): arm.build(rng)
        for seat in (Side.MOUSE, Side.SNAKE)
        for arm in arms
    }
    # LLM threads, with the count of games each has served. A fresh instance
    # every `--games-per-thread` games: the thread grows with every game it
    # plays, and an unbounded one both diverges from how the baseline data was
    # collected and eventually exceeds the model's context.
    threads: dict[tuple[str, Side], Thread] = {}

    def opponent_for(arm_label: str, seat: Side) -> Player:
        key = (arm_label, seat)
        thread = threads.get(key)
        if thread is None or thread.games >= args.games_per_thread:
            thread = Thread(LLMPlayer.from_roster(args.opponent, roster))
            threads[key] = thread
        thread.games += 1
        return thread.player

    started: float = time.time()
    played: int = 0
    with args.out.open("a") as handle:
        # Passes outermost, arms innermost: an interrupted run then leaves whole
        # balanced passes plus one partial seed, rather than starving whichever
        # arms the loop had not reached. A run that stops early is still a fair
        # comparison, which for a job measured in hours is the difference
        # between salvageable and wasted.
        for _pass in range(args.games_per_seed):
            for seat in (Side.MOUSE, Side.SNAKE):
                for seed in ALL_SEEDS:
                    for arm in arms:
                        player = players[(arm.label, seat)]
                        try:
                            opponent = opponent_for(arm.label, seat)
                        except ConfigError as exc:
                            parser.error(str(exc))
                        mouse, snake = (
                            (player, opponent) if seat is Side.MOUSE
                            else (opponent, player)
                        )
                        try:
                            result = play_game(mouse, snake, seed=seed)
                        except ModelRequestError as exc:
                            print(f"\nprovider error after {played} games, "
                                  f"stopping: {exc}", file=sys.stderr)
                            return
                        played += 1
                        handle.write(json.dumps({
                            "arm": arm.label,
                            "opponent": args.opponent,
                            "seed": seed.label,
                            "seat": seat.value,
                            "outcome": outcome(result, seat),
                        }) + "\n")
                        handle.flush()
                        elapsed = time.time() - started
                        print(f"\rpass {_pass + 1}/{args.games_per_seed}  "
                              f"{played}/{total} games  {elapsed / 60:.0f} min  "
                              f"{elapsed / played:.0f}s/game  "
                              f"[{arm.label} {seat.value} {seed.label}] ",
                              end="", file=sys.stderr, flush=True)
    print(f"\nwrote {played} games to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
