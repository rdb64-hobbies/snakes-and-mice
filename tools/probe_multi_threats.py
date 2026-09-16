"""Does a second simultaneous threat make a defender more likely to err?

Two three-piece live lines are not a forced win even against a fallible
defender: every move places two pieces, so a defender that notices both gaps
can plug them in a single move. The interesting question is not the game tree —
it is whether tracking two developing threats at once is harder for a model
than tracking one, given it never sees a rendered board and has to reconstruct
occupancy from the move history alone (SPEC.md §4).

This constructs matched pairs of positions — one live three-piece threat, or
two — and puts a named LLM roster player on move as the defender, using the
same technique SPEC-rl-player.md's training strategy calls "densifying": build
the position by replaying a chosen move sequence through `start_game` /
`observe_move`, then take exactly one `choose_move()`. No real game is played,
and nothing about the sequence needs to be plausible — only legal and free of
any line completed before the position of interest.

Each scenario is validated before it is ever sent to a model: the *only* live
three-or-more-piece lines are the intended threats (for either side — the
defender must have no threat of its own, or blocking would not be its only
good option), no line is already complete, and the two conditions in a pair are
matched: same defender, same total pieces on the board, one shares a threat
cell (fair — a real double threat can hardly avoid sharing lines it can win
outright on) and one does not.

A response is graded by whether it places at least one piece in *every*
originally-live threat line — not by whether it is the move `evaluate` would
call optimal, since a defender could satisfy this and still not be playing the
solved-optimal move elsewhere on the board; what is being measured here is
"did it attend to every threat," not overall move quality.

    uv run python tools/probe_multi_threats.py LLM_NAME
"""

from __future__ import annotations

import argparse
import sys

from snakes_and_mice.cli_common import make_player
from snakes_and_mice.core import Side
from snakes_and_mice.faults import MoveUnavailable
from snakes_and_mice.roster import ConfigError, Roster, load_environment, load_roster

from threat_scenarios import Scenario, run

# --- Scenarios --------------------------------------------------------------
#
# Four matched pairs: two with Mouse defending, two with Snake defending (the
# side split matters -- qwen-3-8-rtx's own threat-type bias turned out to
# differ sharply by side). Each pair's single- and double-threat halves share
# the same defender and the same total piece count. The double half's two
# threats always cross at one shared cell (the natural shape a real double
# threat takes -- two lines built through a common point) and mix line types
# (row+column, column+diagonal, ...) rather than repeating the row/column bias
# already found, since that is a separate effect this probe is not about.

SCENARIOS: tuple[Scenario, ...] = (
    Scenario(  # single: Snake holds row C, Mouse to defend
        "mouse-defends-single-row",
        Side.MOUSE, "C3",
        mouse_cells=frozenset({"A2", "B4", "D3", "D4"}),
        snake_cells=frozenset({"B2", "C1", "C2", "C3", "E1"}),
    ),
    Scenario(  # double: Snake holds row C AND column 3, sharing C3
        "mouse-defends-double-row-col",
        Side.MOUSE, "C3",
        mouse_cells=frozenset({"A1", "B4", "E1", "E5"}),
        snake_cells=frozenset({"C1", "C2", "C3", "D3", "E3"}),
    ),
    Scenario(  # single: Mouse holds column 4, Snake to defend
        "snake-defends-single-col",
        Side.SNAKE, "E5",
        mouse_cells=frozenset({"A5", "B2", "B4", "C4", "C5", "D4"}),
        snake_cells=frozenset({"A2", "A3", "D1", "D3", "E5"}),
    ),
    Scenario(  # double: Mouse holds column 4 AND row B, sharing B4
        "snake-defends-double-col-row",
        Side.SNAKE, "E5",
        mouse_cells=frozenset({"B1", "B2", "B4", "C4", "D4", "E2"}),
        snake_cells=frozenset({"A1", "C3", "E1", "E3", "E5"}),
    ),
    Scenario(  # single: Snake holds the anti-diagonal, Mouse to defend
        "mouse-defends-single-diag",
        Side.MOUSE, "B4",
        mouse_cells=frozenset({"A2", "C1", "D1", "D3"}),
        snake_cells=frozenset({"B3", "B4", "D2", "D5", "E1"}),
    ),
    Scenario(  # double: Snake holds the anti-diagonal AND row D, sharing D2
        "mouse-defends-double-diag-row",
        Side.MOUSE, "B4",
        mouse_cells=frozenset({"A1", "B3", "E2", "E5"}),
        snake_cells=frozenset({"B4", "D1", "D2", "D5", "E1"}),
    ),
    Scenario(  # single: Mouse holds row A, Snake to defend
        "snake-defends-single-row",
        Side.SNAKE, "E1",
        mouse_cells=frozenset({"A1", "A2", "A3", "C2", "E4", "E5"}),
        snake_cells=frozenset({"B2", "B4", "E1", "E2", "E3"}),
    ),
    Scenario(  # double: Mouse holds row A AND column 3, sharing A3
        "snake-defends-double-row-col",
        Side.SNAKE, "E1",
        mouse_cells=frozenset({"A1", "A2", "A3", "B3", "C3", "E4"}),
        snake_cells=frozenset({"B1", "C4", "E1", "E2", "E5"}),
    ),
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="probe_multi_threats",
        description=__doc__.strip().splitlines()[0],
    )
    parser.add_argument("llm", metavar="LLM_NAME", help="a roster name from players.yaml")
    args = parser.parse_args(argv)

    load_environment()
    try:
        roster: Roster = load_roster()
    except ConfigError as exc:
        parser.error(str(exc))

    results: list[tuple[Scenario, bool]] = []
    illegal: list[Scenario] = []
    for s in SCENARIOS:
        player = make_player(args.llm, s.defender, roster, log_dir=None)
        try:
            r = run(player, s)
        except MoveUnavailable as exc:
            print(f"{s.name:30s} FAULT (excluded): {exc.reason.name} -- {exc}")
            continue
        kind = "single" if len(r.threats) == 1 else "double"
        if not r.legal:
            illegal.append(s)
            print(f"{s.name:30s} [{kind}] played {r.move}: ILLEGAL -- reoccupied a cell")
            continue
        ok = all(r.hits)
        results.append((s, ok))
        print(
            f"{s.name:30s} [{kind}] played {r.move}: "
            f"{sum(r.hits)}/{len(r.threats)} threats addressed "
            f"({'OK' if ok else 'MISSED ONE'})"
        )

    single = [ok for s, ok in results if "single" in s.name]
    double = [ok for s, ok in results if "double" in s.name]
    print()
    print(
        f"single-threat: {sum(single)}/{len(single)} fully defended  |  "
        f"double-threat: {sum(double)}/{len(double)} fully defended"
    )
    if illegal:
        print(
            f"({len(illegal)} scenario(s) excluded above -- reoccupied a cell: "
            f"{', '.join(s.name for s in illegal)})"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
