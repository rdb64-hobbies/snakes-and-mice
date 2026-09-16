"""Hand-built, validated board positions for probing a defender's move choice.

Shared by the `probe_*` scripts in this directory: each constructs a `Scenario`
(a target occupancy, a seed, and which side is on move to defend), replays a
legal move sequence into a live `Player` to reach it (the "densifying"
technique from SPEC-rl-player.md's training strategy — a `Player` only ever
learns the board through `start_game` / `observe_move`, so any reachable
position can be set up directly, no real game needed), takes exactly one
`choose_move()`, and checks whether the response touches every line the
defender needed to attend to.

A `Scenario` is validated before it is ever sent to a model (`validate`): the
*only* live three-or-more-piece lines are the intended threats — the defender
has no winning line of its own, or blocking would not be its only good
option — no line is already complete, and the occupancy is one an alternating,
Mouse-first sequence could actually reach. Building these by hand is
error-prone (a filler cell can silently kill or duplicate an intended threat);
see `find_scenarios.py`-style generation in each probe script's own history
for how they were found.

A response is graded by whether it places at least one piece in *every*
originally-live threat line — not by whether it is the move `evaluate` would
call optimal, since a defender could satisfy this and still not be playing the
solved-optimal move elsewhere on the board. What is measured is "did it attend
to every threat," not overall move quality.
"""

from __future__ import annotations

from dataclasses import dataclass

from snakes_and_mice.board import LINES
from snakes_and_mice.core import Cell, Move, Side
from snakes_and_mice.players import Player


@dataclass(frozen=True)
class Scenario:
    """A hand-built position, validated before use (see `validate`)."""

    name: str
    defender: Side
    seed: str
    mouse_cells: frozenset[str]
    snake_cells: frozenset[str]  # includes the seed


def _line_labels() -> tuple[frozenset[str], ...]:
    return tuple(frozenset(cell.label for cell in line) for line in LINES)


ALL_LINES: tuple[frozenset[str], ...] = _line_labels()


def line_kind(line: frozenset[str]) -> str:
    """'row', 'col', or 'diag', from a line's cell labels."""
    rows = {label[0] for label in line}
    cols = {label[1] for label in line}
    if len(rows) == 1:
        return "row"
    if len(cols) == 1:
        return "col"
    return "diag"


def live_threats(
    mouse: frozenset[str], snake: frozenset[str], owner: Side
) -> list[frozenset[str]]:
    """Lines where `owner` holds >=3 cells and the other side holds none."""
    theirs, ours = (snake, mouse) if owner is Side.SNAKE else (mouse, snake)
    return [
        line for line in ALL_LINES if len(line & theirs) >= 3 and not (line & ours)
    ]


def validate(s: Scenario) -> list[frozenset[str]]:
    """The defender's threats to defend against, or raise if the scenario is
    unsound (a completed line, an extra threat, a defender who could also just
    win, or an occupancy no alternating sequence could reach)."""
    attacker = s.defender.other
    if any(len(line & s.mouse_cells) == 5 or len(line & s.snake_cells) == 5
           for line in ALL_LINES):
        raise ValueError(f"{s.name}: a line is already complete")
    if live_threats(s.mouse_cells, s.snake_cells, s.defender):
        raise ValueError(f"{s.name}: defender already has a winning line of its own")
    threats = live_threats(s.mouse_cells, s.snake_cells, attacker)
    if not threats:
        raise ValueError(f"{s.name}: attacker has no threat at all")

    mouse_n, snake_extra = len(s.mouse_cells), len(s.snake_cells) - 1
    if mouse_n % 2 or snake_extra % 2:
        raise ValueError(f"{s.name}: an odd cell count -- moves place two at a time")
    expected_gap = 0 if s.defender is Side.MOUSE else 2
    if mouse_n - snake_extra != expected_gap:
        raise ValueError(
            f"{s.name}: {mouse_n} mouse / {snake_extra} snake-beyond-seed cells "
            f"cannot reach a {s.defender.value}-to-move position"
        )
    return threats


def replay_sequence(
    mouse_cells: frozenset[str], snake_cells: frozenset[str], seed: str
) -> list[tuple[Side, tuple[str, ...]]]:
    """One alternating, Mouse-first sequence reaching this occupancy.

    Order within a side never matters for the final position (module
    docstring: no intermediate state can complete a line the final one
    doesn't), so cells are just consumed in sorted order.
    """
    mouse_left = sorted(mouse_cells)
    snake_left = sorted(c for c in snake_cells if c != seed)
    moves: list[tuple[Side, tuple[str, ...]]] = []
    while mouse_left or snake_left:
        if mouse_left:
            moves.append((Side.MOUSE, (mouse_left.pop(0), mouse_left.pop(0))))
        if snake_left:
            moves.append((Side.SNAKE, (snake_left.pop(0), snake_left.pop(0))))
    return moves


def run(player: Player, s: Scenario) -> tuple[Move, list[frozenset[str]], list[bool]]:
    """Replay `s` into `player`, take its move, and check every threat."""
    threats = validate(s)
    player.start_game(s.defender, Cell.from_label(s.seed))
    for side, labels in replay_sequence(s.mouse_cells, s.snake_cells, s.seed):
        move = Move.of(*(Cell.from_label(label) for label in labels))
        player.observe_move(side, move)
    choice = player.choose_move()
    played: frozenset[str] = frozenset(c.label for c in choice.move.cells)
    return choice.move, threats, [bool(t & played) for t in threats]
