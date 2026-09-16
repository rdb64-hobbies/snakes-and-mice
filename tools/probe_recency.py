"""Does a cell placed early in the constructed history get forgotten?

`probe_multi_threats.py` turned up something unplanned: `gpt-5-6-terra`
attempted to replay an already-occupied cell in 3 of 8 responses across two
runs, and in every one of the three, the reoccupied cell was placed on that
side's *first* move of the constructed replay sequence — the oldest fact in
the history, several turns before the decision point. That is suggestive of a
recency effect, not a general "loses track of the board" failure, but three
incidental observations from a probe designed to test something else is not
evidence of it.

Two positions where this already happened get a "late" variant here: the
**identical final board** — same cells, same threats, same everything an
`evaluate`-based grader would see — with only the *order* cells were placed
changed, moving the previously-first-placed target cell to that side's *last*
move instead. Their "early" condition is not re-run, to save cost; it already
exists (`probe_multi_threats` output, 2026-09-16):

    mouse-defends-double-row-col: A1 placed move 1 of 2 -- replayed illegally
      (gpt-5-6-terra, 2/2 runs; qwen-3-8-rtx and gemini-3-8 not yet tested early)
    snake-defends-single-row:     B4 placed move 1 of 2 -- replayed illegally
      (qwen-3-8-rtx 1/1, gpt-5-6-terra 1/1 -- two-model replication already)

Two more pairs are fresh — neither half has been run before — to get a
deliberately-designed sample rather than relying only on positions found by
accident. Both conditions are included for these.

    uv run python tools/probe_recency.py LLM_NAME
"""

from __future__ import annotations

import argparse
import sys

from snakes_and_mice.cli_common import make_player
from snakes_and_mice.core import Side
from snakes_and_mice.faults import MoveUnavailable
from snakes_and_mice.roster import ConfigError, Roster, load_environment, load_roster

from threat_scenarios import Scenario, run

# --- Scenarios ----------------------------------------------------------------
#
# Two kinds of pair. The first two reuse positions already run "early" by
# accident (probe_multi_threats.py / probe_line_types.py output, not re-run
# here to save cost) -- only the "late" variant is new. The second two are
# fresh matched pairs, run both ways here, since no incidental "early" data
# exists for them yet. In every case the target cell belongs to the
# *defender's own* filler cells, matching the pattern observed so far.

SCENARIOS: tuple[tuple[Scenario, str], ...] = (
    (
        Scenario(  # early condition already observed: A1 replayed illegally,
                   # 2 of 2 times (gpt-5-6-terra) -- not re-run here.
            "mouse-defends-double-row-col-late-A1",
            Side.MOUSE, "C3",
            mouse_cells=frozenset({"A1", "B4", "E1", "E5"}),
            snake_cells=frozenset({"C1", "C2", "C3", "D3", "E3"}),
            mouse_order=("B4", "E1", "E5", "A1"),  # A1 now on move 2 of 2
        ),
        "A1",
    ),
    (
        Scenario(  # early condition already observed: B4 replayed illegally,
                   # 2 of 2 times (qwen-3-8-rtx once, gpt-5-6-terra once) --
                   # not re-run here.
            "snake-defends-single-row-late-B4",
            Side.SNAKE, "E1",
            mouse_cells=frozenset({"A1", "A2", "A3", "C2", "E4", "E5"}),
            snake_cells=frozenset({"B2", "B4", "E1", "E2", "E3"}),
            snake_order=("B2", "E2", "E3", "B4"),  # B4 now on move 2 of 2
        ),
        "B4",
    ),
    (
        Scenario(  # fresh pair: early half, run here for the first time.
            "mouse-defends-row-B-early-D1",
            Side.MOUSE, "B1",
            mouse_cells=frozenset({"D1", "D2", "E4", "E5"}),
            snake_cells=frozenset({"B1", "B2", "B3", "C4", "C5"}),
        ),
        "D1",
    ),
    (
        Scenario(  # fresh pair: late half -- same final board, D1 moved to
                   # Mouse's last move instead of its first.
            "mouse-defends-row-B-late-D1",
            Side.MOUSE, "B1",
            mouse_cells=frozenset({"D1", "D2", "E4", "E5"}),
            snake_cells=frozenset({"B1", "B2", "B3", "C4", "C5"}),
            mouse_order=("D2", "E4", "E5", "D1"),
        ),
        "D1",
    ),
    (
        Scenario(  # fresh pair: early half.
            "snake-defends-col-1-early-B4",
            Side.SNAKE, "E5",
            mouse_cells=frozenset({"A1", "B1", "C1", "D2", "D3", "E4"}),
            snake_cells=frozenset({"B4", "C2", "D5", "E2", "E5"}),
        ),
        "B4",
    ),
    (
        Scenario(  # fresh pair: late half -- same final board, B4 moved to
                   # Snake's last move instead of its first.
            "snake-defends-col-1-late-B4",
            Side.SNAKE, "E5",
            mouse_cells=frozenset({"A1", "B1", "C1", "D2", "D3", "E4"}),
            snake_cells=frozenset({"B4", "C2", "D5", "E2", "E5"}),
            snake_order=("C2", "D5", "E2", "B4"),
        ),
        "B4",
    ),
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="probe_recency",
        description=__doc__.strip().splitlines()[0],
    )
    parser.add_argument("llm", metavar="LLM_NAME", help="a roster name from players.yaml")
    args = parser.parse_args(argv)

    load_environment()
    try:
        roster: Roster = load_roster()
    except ConfigError as exc:
        parser.error(str(exc))

    for s, target_cell in SCENARIOS:
        player = make_player(args.llm, s.defender, roster, log_dir=None)
        try:
            r = run(player, s)
        except MoveUnavailable as exc:
            print(f"{s.name:38s} FAULT (excluded): {exc.reason.name} -- {exc}")
            continue
        played = frozenset(c.label for c in r.move.cells)
        replayed = target_cell in played
        print(
            f"{s.name:38s} played {r.move}: "
            f"{'REPLAYED ' + target_cell + ' (still forgotten)' if replayed else f'{target_cell} correctly avoided'}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
