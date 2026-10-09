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

Scoring follows `tally.py`, which is the project's canonical definition and
disagrees with the obvious guess: a **fault is not a win for the opponent**
(SPEC.md §3), and a faulted game leaves the win-rate base altogether —
`win_rate = won / (won + lost + tied)`. So faults are counted and reported in
their own column, never folded into wins. An earlier version of this tool scored
a forced fault as +1 and kept it in the denominator, which inflated every arm
that provoked faults; the RL player provokes them at three times a `perfect`
variant's rate, so that was not a small error.

Aborted games are dropped: a no-contest is charged to neither side.

    uv run python tools/tally_paired_eval.py [paired-eval.jsonl]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Final

SCORE: Final[dict[str, float]] = {"win": 1.0, "draw": 0.0, "loss": -1.0}
"""Per-game score for the paired difference. Faults are excluded, not scored:
they are their own outcome (SPEC.md §3) and belong in neither direction."""

FAULTS: Final[frozenset[str]] = frozenset({"fault_theirs", "fault_ours"})


@dataclass(frozen=True)
class Marginal:
    """One arm's record, split the way `tally.py` splits it."""

    wins: int
    draws: int
    losses: int
    forced: int   # games the opponent faulted
    conceded: int  # games this arm faulted

    @property
    def clean(self) -> int:
        """Games decided or drawn with neither side faulting — the win-rate base."""
        return self.wins + self.draws + self.losses

    @property
    def played(self) -> int:
        return self.clean + self.forced + self.conceded

    @property
    def rate(self) -> float:
        return self.wins / self.clean if self.clean else 0.0

    @property
    def error(self) -> float:
        """Binomial standard error of :attr:`rate`."""
        p = self.rate
        return math.sqrt(p * (1 - p) / self.clean) if self.clean else 0.0


def load(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def marginals(rows: list[dict[str, str]]) -> dict[str, Marginal]:
    out: dict[str, Marginal] = {}
    for arm in sorted({r["arm"] for r in rows}):
        got = Counter(r["outcome"] for r in rows
                      if r["arm"] == arm and r["outcome"] != "abort")
        out[arm] = Marginal(got["win"], got["draw"], got["loss"],
                            got["fault_theirs"], got["fault_ours"])
    return out


def units(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, list[float]]]:
    """Scores per matched ``(seed, seat)`` unit, per arm."""
    got: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        if r["outcome"] == "abort" or r["outcome"] in FAULTS:
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
    print(f"{'arm':<24}{'played':>7}{'clean':>6}{'W':>5}{'D':>6}{'L':>4}"
          f"{'forced':>7}{'win%':>8}{'+-':>7}")
    for arm, m in per_arm.items():
        print(f"{arm:<24}{m.played:>7}{m.clean:>6}{m.wins:>5}{m.draws:>6}"
              f"{m.losses:>4}{m.forced:>7}{100 * m.rate:>7.2f}%"
              f"{100 * m.error:>6.2f}%")

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
