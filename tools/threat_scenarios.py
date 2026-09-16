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
    """A hand-built position, validated before use (see `validate`).

    ``mouse_order`` / ``snake_order``, if given, fix the order cells are
    consumed into moves during replay (see `replay_sequence`) — same final
    occupancy, different position in the constructed history. Used to test
    whether *when* a cell was placed, not just that it was, affects whether a
    model still tracks it as occupied several turns later.
    """

    name: str
    defender: Side
    seed: str
    mouse_cells: frozenset[str]
    snake_cells: frozenset[str]  # includes the seed
    mouse_order: tuple[str, ...] | None = None
    snake_order: tuple[str, ...] | None = None


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
    mouse_cells: frozenset[str],
    snake_cells: frozenset[str],
    seed: str,
    *,
    mouse_order: tuple[str, ...] | None = None,
    snake_order: tuple[str, ...] | None = None,
) -> list[tuple[Side, tuple[str, ...]]]:
    """One alternating, Mouse-first sequence reaching this occupancy.

    Which final position each side's moves land in never matters (module
    docstring: no intermediate state can complete a line the final one
    doesn't), so cells are consumed in sorted order by default. ``mouse_order``
    / ``snake_order`` override that — the same cells, consumed (and so paired
    into moves) in a caller-chosen order instead, to control which move a
    particular cell falls on without changing the final position at all.
    """
    mouse_left = list(mouse_order) if mouse_order is not None else sorted(mouse_cells)
    snake_left = (
        list(snake_order)
        if snake_order is not None
        else sorted(c for c in snake_cells if c != seed)
    )
    assert set(mouse_left) == mouse_cells, "mouse_order must be a permutation of mouse_cells"
    assert set(snake_left) == snake_cells - {seed}, (
        "snake_order must be a permutation of snake_cells minus the seed"
    )
    moves: list[tuple[Side, tuple[str, ...]]] = []
    while mouse_left or snake_left:
        if mouse_left:
            moves.append((Side.MOUSE, (mouse_left.pop(0), mouse_left.pop(0))))
        if snake_left:
            moves.append((Side.SNAKE, (snake_left.pop(0), snake_left.pop(0))))
    return moves


@dataclass(frozen=True)
class Response:
    """One scenario's outcome.

    ``legal`` is `False` if the move reoccupied a cell the scenario already had
    filled — the model losing track of its own board, not a judgement about
    which threat it addressed. `hits` is computed regardless (it costs nothing
    and an illegal move's *other* cell may still be informative), but a caller
    should report `legal is False` distinctly rather than folding it into a
    hit/miss tally: a "hit" earned by luck on the one real cell of an otherwise
    illegal move is not the same finding as a clean defense.
    """

    move: Move
    threats: list[frozenset[str]]
    legal: bool
    hits: list[bool]


def run(player: Player, s: Scenario) -> Response:
    """Replay `s` into `player`, take its move, and check every threat."""
    threats = validate(s)
    occupied = s.mouse_cells | s.snake_cells
    player.start_game(s.defender, Cell.from_label(s.seed))
    sequence = replay_sequence(
        s.mouse_cells, s.snake_cells, s.seed,
        mouse_order=s.mouse_order, snake_order=s.snake_order,
    )
    for side, labels in sequence:
        move = Move.of(*(Cell.from_label(label) for label in labels))
        player.observe_move(side, move)
    choice = player.choose_move()
    played: frozenset[str] = frozenset(c.label for c in choice.move.cells)
    legal = not (played & occupied)
    return Response(choice.move, threats, legal, [bool(t & played) for t in threats])
