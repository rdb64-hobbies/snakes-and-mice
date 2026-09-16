"""Does a lone diagonal threat get missed more often than a lone row or column?

SPEC-rl-player.md ("What the mistakes look like") found `qwen-3-8-rtx` missing
column threats far more than row threats as Mouse, nearly evenly between the two
as Snake, and apparently not checking for diagonal threats at all in two
observed cases. Neither reading isolates line type cleanly: the column-vs-row
numbers come from whatever threats happened to arise in real games, not a
designed comparison, and the diagonal evidence is two anecdotes.

This probes it directly: for each defending side, six matched single-threat
positions — two replicates each of a row, a column, and a diagonal threat, all
with the same total piece count for that side — put a named LLM roster player
on move as the defender and check whether its move touches the threatened line.
Same technique as `probe_multi_threats.py` (see `threat_scenarios.py`): replay
a chosen sequence through `start_game` / `observe_move`, take one
`choose_move()`, no real game played.

    uv run python tools/probe_line_types.py LLM_NAME
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from snakes_and_mice.cli_common import make_player
from snakes_and_mice.core import Side
from snakes_and_mice.roster import ConfigError, Roster, load_environment, load_roster

from threat_scenarios import Scenario, line_kind, run

# --- Scenarios ----------------------------------------------------------------
#
# Two matched groups, one per defending side (attacker is always the other
# side). Within a group every scenario has the same total piece count, so
# "which line type" is the only thing varying: Mouse defends at 9 pieces total,
# Snake at 11 (Snake needs one more piece to reach a to-move position, since
# its threat's 3 required cells are an odd count needing a padding piece --
# see `threat_scenarios.validate`'s parity check). Found with a generator
# under a "filler must not touch the threat line" constraint, the same
# approach `probe_multi_threats.py`'s scenarios needed.

SCENARIOS: tuple[Scenario, ...] = (
    # Mouse defends (Snake attacks), 9 pieces total.
    Scenario(
        "mouse-defends-row-A",
        Side.MOUSE, "A1",
        mouse_cells=frozenset({"B2", "C4", "D3", "D4"}),
        snake_cells=frozenset({"A1", "A2", "A3", "C2", "E1"}),
    ),
    Scenario(
        "mouse-defends-row-E",
        Side.MOUSE, "E2",
        mouse_cells=frozenset({"A2", "B4", "C3", "C4"}),
        snake_cells=frozenset({"B2", "D1", "E1", "E2", "E3"}),
    ),
    Scenario(
        "mouse-defends-col-1",
        Side.MOUSE, "A1",
        mouse_cells=frozenset({"A3", "C2", "D2", "D3"}),
        snake_cells=frozenset({"A1", "B1", "B4", "C1", "D5"}),
    ),
    Scenario(
        "mouse-defends-col-5",
        Side.MOUSE, "B5",
        mouse_cells=frozenset({"A2", "C1", "D1", "D2"}),
        snake_cells=frozenset({"A5", "B3", "B5", "C5", "D4"}),
    ),
    Scenario(
        "mouse-defends-main-diag",
        Side.MOUSE, "A1",
        mouse_cells=frozenset({"A3", "C1", "D1", "D2"}),
        snake_cells=frozenset({"A1", "B2", "B4", "C3", "D5"}),
    ),
    Scenario(
        "mouse-defends-anti-diag",
        Side.MOUSE, "B4",
        mouse_cells=frozenset({"A2", "C1", "D1", "D3"}),
        snake_cells=frozenset({"A5", "B3", "B4", "C3", "D5"}),
    ),
    # Snake defends (Mouse attacks), 11 pieces total.
    Scenario(
        "snake-defends-row-B",
        Side.SNAKE, "E5",
        mouse_cells=frozenset({"B1", "B2", "B3", "C1", "D5", "E4"}),
        snake_cells=frozenset({"A3", "A4", "D2", "E3", "E5"}),
    ),
    Scenario(
        "snake-defends-row-D",
        Side.SNAKE, "E1",
        mouse_cells=frozenset({"B1", "C5", "D1", "D2", "D3", "E5"}),
        snake_cells=frozenset({"A3", "A4", "C2", "E1", "E4"}),
    ),
    Scenario(
        "snake-defends-col-2",
        Side.SNAKE, "E5",
        mouse_cells=frozenset({"A2", "B2", "B4", "C2", "C5", "D1"}),
        snake_cells=frozenset({"A1", "A4", "A5", "E3", "E5"}),
    ),
    Scenario(
        "snake-defends-col-4",
        Side.SNAKE, "E1",
        mouse_cells=frozenset({"A2", "A4", "B4", "C4", "D1", "D2"}),
        snake_cells=frozenset({"B3", "B5", "C1", "D5", "E1"}),
    ),
    Scenario(
        "snake-defends-main-diag",
        Side.SNAKE, "B4",
        mouse_cells=frozenset({"A1", "B2", "B3", "C3", "D5", "E4"}),
        snake_cells=frozenset({"A4", "A5", "B4", "D1", "E3"}),
    ),
    Scenario(
        "snake-defends-anti-diag",
        Side.SNAKE, "D1",
        mouse_cells=frozenset({"A2", "A5", "B4", "C3", "D3", "D4"}),
        snake_cells=frozenset({"B3", "B5", "C1", "D1", "E2"}),
    ),
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="probe_line_types",
        description=__doc__.strip().splitlines()[0],
    )
    parser.add_argument("llm", metavar="LLM_NAME", help="a roster name from players.yaml")
    args = parser.parse_args(argv)

    load_environment()
    try:
        roster: Roster = load_roster()
    except ConfigError as exc:
        parser.error(str(exc))

    tally: dict[tuple[Side, str], list[bool]] = defaultdict(list)
    for s in SCENARIOS:
        player = make_player(args.llm, s.defender, roster, log_dir=None)
        move, threats, hits = run(player, s)
        assert len(threats) == 1, f"{s.name}: expected exactly one threat"
        ok = hits[0]
        kind = line_kind(threats[0])
        tally[(s.defender, kind)].append(ok)
        print(f"{s.name:24s} played {move}: {'OK' if ok else 'MISSED'}")

    print()
    for side in (Side.MOUSE, Side.SNAKE):
        parts = []
        for kind in ("row", "col", "diag"):
            results = tally[(side, kind)]
            parts.append(f"{kind}={sum(results)}/{len(results)}")
        print(f"{side.value:5s} defending: {'  '.join(parts)}")


if __name__ == "__main__":
    main(sys.argv[1:])
