"""Read a `paired_llm_eval.py` run: marginal rates, and the paired differences.

The harness writes one JSON line per game and nothing that reads it, which is
half a tool. This is the other half.

Two things are reported, and the second is the one to quote. Marginal win rates
answer "how did each arm do"; the **paired difference** answers "did arm A beat
arm B", by comparing them on each matched (seed, seat) unit and taking the mean
and standard error of those per-unit differences. A difference smaller than
about two standard errors is unresolved, whatever the rates look like — this
project has twice produced a confident-looking lead at ~1 SE that a larger
sample erased (`SPEC-rl-player.md`, "Shaping on move preference instead").

The `unpaired +-` column is printed beside the paired one as a check on whether
pairing bought anything. On the first real run it did not: 0.0665 against 0.0670,
and worse than unpaired on another comparison. The design is free and cannot hurt,
but the fix for noise here is games, not pairing.

A game is scored +1 for a win, -1 for a loss, 0 for a draw. An opponent fault
counts as a win and the arm's own fault as a loss, since a fault decides the game
exactly as a completed line does (SPEC.md §3). Aborted games are dropped: a
no-contest is charged to neither side.

    uv run python tools/tally_paired_eval.py [paired-eval.jsonl]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Final

SCORE: Final[dict[str, float]] = {
    "win": 1.0, "fault_theirs": 1.0,
    "draw": 0.0,
    "loss": -1.0, "fault_ours": -1.0,
}

WINS: Final[frozenset[str]] = frozenset({"win", "fault_theirs"})
LOSSES: Final[frozenset[str]] = frozenset({"loss", "fault_ours"})


@dataclass(frozen=True)
class Marginal:
    """One arm's overall record."""

    games: int
    wins: int
    draws: int
    losses: int

    @property
    def rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def error(self) -> float:
        """Binomial standard error of :attr:`rate`."""
        p = self.rate
        return math.sqrt(p * (1 - p) / self.games) if self.games else 0.0


def load(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def marginals(rows: list[dict[str, str]]) -> dict[str, Marginal]:
    out: dict[str, Marginal] = {}
    for arm in sorted({r["arm"] for r in rows}):
        outcomes = [r["outcome"] for r in rows
                    if r["arm"] == arm and r["outcome"] != "abort"]
        wins = sum(1 for o in outcomes if o in WINS)
        losses = sum(1 for o in outcomes if o in LOSSES)
        out[arm] = Marginal(len(outcomes), wins,
                            len(outcomes) - wins - losses, losses)
    return out


def units(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, list[float]]]:
    """Scores per matched ``(seed, seat)`` unit, per arm."""
    got: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        if r["outcome"] == "abort":
            continue
        got[(r["seed"], r["seat"])][r["arm"]].append(SCORE[r["outcome"]])
    return got


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="tally_paired_eval.py",
        description="Tally a paired LLM comparison, pairing included.",
    )
    parser.add_argument("path", nargs="?", type=Path,
                        default=Path("paired-eval.jsonl"))
    args: argparse.Namespace = parser.parse_args(argv)

    rows = load(args.path)
    if not rows:
        parser.error(f"{args.path} has no games")
    per_arm = marginals(rows)
    matched = units(rows)
    # Tolerate a missing opponent field so hand-made fixtures work.
    opponents = sorted({r["opponent"] for r in rows if "opponent" in r})

    against: str = ", ".join(opponents) if opponents else "(unnamed)"
    print(f"{len(rows)} games vs {against}; "
          f"{len(per_arm)} arms over {len(matched)} (seed, seat) units\n")
    print(f"{'arm':<24}{'games':>6}{'W':>5}{'D':>6}{'L':>4}{'win%':>8}{'+-':>7}")
    for arm, m in per_arm.items():
        print(f"{arm:<24}{m.games:>6}{m.wins:>5}{m.draws:>6}{m.losses:>4}"
              f"{100 * m.rate:>7.2f}%{100 * m.error:>6.2f}%")

    print(f"\n{'paired comparison':<40}{'n':>4}{'mean diff':>11}{'+-':>8}"
          f"{'SE':>6}{'unpaired +-':>13}{'verdict':>12}")
    for a, b in combinations(per_arm, 2):
        diffs = [
            sum(got[a]) / len(got[a]) - sum(got[b]) / len(got[b])
            for got in matched.values() if a in got and b in got
        ]
        if len(diffs) < 2:
            continue
        mean = sum(diffs) / len(diffs)
        spread = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
        error = math.sqrt(spread / len(diffs))
        unpaired = math.hypot(per_arm[a].error, per_arm[b].error)
        sigmas = abs(mean) / error if error else math.inf
        verdict = "resolved" if sigmas >= 2.0 else "unresolved"
        print(f"{a + ' - ' + b:<40}{len(diffs):>4}{mean:>+11.4f}{error:>8.4f}"
              f"{sigmas:>6.2f}{unpaired:>13.4f}{verdict:>12}")


if __name__ == "__main__":
    main()
