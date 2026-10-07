"""The mistake model: how hard a position is for a *fallible* defender to hold.

A score over board features, specified in `SPEC-mistake-model.md` ("The mistake
model", "The score, as implemented"). Positions are the pair of 25-bit masks the
perfect player searches with, one bit per cell at ``5·row + col``.
"""

from __future__ import annotations

from itertools import combinations
from math import exp
from typing import Final

from .board import LINES
from .core import BOARD_SIZE

_CELL_COUNT: Final[int] = BOARD_SIZE * BOARD_SIZE
_FULL_MASK: Final[int] = (1 << _CELL_COUNT) - 1

# The three levels of the score, worst to best for the attacker.
SCORE_NONE: Final[int] = 0
"""Fewer than two live threats."""

SCORE_ALIGNED: Final[int] = 1
"""Two or more, all answerable by one row- or column-aligned move."""

SCORE_SPLIT: Final[int] = 2
"""Two or more that no aligned move answers — the defender must split."""

_LINE_MASKS: Final[tuple[int, ...]] = tuple(
    sum(1 << (cell.row * BOARD_SIZE + cell.col) for cell in line) for line in LINES
)
"""The 12 winning lines as bit-masks."""

_ALIGNED_MASKS: Final[tuple[int, ...]] = tuple(
    sum(1 << (r * BOARD_SIZE + c) for c in range(BOARD_SIZE))
    for r in range(BOARD_SIZE)
) + tuple(
    sum(1 << (r * BOARD_SIZE + c) for r in range(BOARD_SIZE))
    for c in range(BOARD_SIZE)
)
"""The 5 rows and 5 columns — the groups a two-piece move counts as aligned in.

Deliberately not the diagonals; the measured habit is a shared row or column.
"""


def threat_score(attacker: int, defender: int) -> int:
    """How hard ``attacker``'s live threats are for ``defender`` to answer.

    Returns :data:`SCORE_NONE`, :data:`SCORE_ALIGNED` or :data:`SCORE_SPLIT`,
    larger being harder to defend. ``defender`` is the side to move; both
    arguments are cell bit-masks.

    A **live threat** is a line holding three or more of the attacker's pieces
    and none of the defender's — one or two gaps, either of which wins next turn
    (§2.5). Playing any empty cell of a line kills it for good (§2.7), so the
    defender's two pieces must between them touch every threatened line.
    """
    threats: list[int] = [
        line
        for line in _LINE_MASKS
        if not line & defender and 1 <= (line & ~attacker).bit_count() <= 2
    ]
    if len(threats) < 2:
        return SCORE_NONE

    # Per empty cell, which threats it kills, as a bitset over `threats`. Cells
    # killing nothing are dropped: they can never be half of an answer.
    covers: dict[int, int] = {}
    remaining: int = _FULL_MASK & ~(attacker | defender)
    while remaining:
        cell: int = remaining & -remaining
        remaining ^= cell
        killed: int = sum(1 << i for i, line in enumerate(threats) if line & cell)
        if killed:
            covers[cell] = killed

    every: int = (1 << len(threats)) - 1
    if any(killed == every for killed in covers.values()):
        return SCORE_ALIGNED  # the threats cross: one piece answers them all
    for group in _ALIGNED_MASKS:
        within: list[int] = [
            killed for cell, killed in covers.items() if cell & group
        ]
        for first, second in combinations(within, 2):
            if first | second == every:
                return SCORE_ALIGNED
    return SCORE_SPLIT


# --- The learned move-preference model -------------------------------------
#
# The score above reads a *position*. This reads a *choice*: which move a
# fallible player picks from the ones available. It is fitted, not hand-made —
# a conditional logit over 588 recorded `draw->loss` blunders, specified in
# SPEC-mistake-model.md, "Validation against 588 real blunders". The score above
# cannot express what it captures, because the mechanism it found (this player
# blocks threats but shuns the move that blocks them *all*) is a property of the
# choice and not of the board.

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "aligned",
    "adjacent",
    "advances_own",
    "blocks_threat",
    "answers_all",
    "creates_split",
    "creates_threat",
    "wins_now",
    "on_diagonal",
    "center",
    "corner",
)
"""The move features, in the order :data:`MOVE_PREFERENCE` weights them."""

MOVE_PREFERENCE: Final[tuple[float, ...]] = (
    1.6306,   # aligned        — the habit a shared row or column is named for
    0.2652,   # adjacent
    1.0617,   # advances_own
    2.8945,   # blocks_threat  — it does block threats...
    -4.6746,  # answers_all    — ...but shuns the move that blocks them all
    0.0347,   # creates_split
    2.5140,   # creates_threat — offence preferred over defence
    0.0000,   # wins_now       — never varies in the fitted data; see the spec
    0.1230,   # on_diagonal
    0.4697,   # center
    -0.3895,  # corner
)
"""Weights fitted to ``qwen-3-8-rtx``, 588 blunders, held-out top-1 9x chance.

Specific to one model, which SPEC-rl-player.md's Goal permits as a fallback
("a player specialized to a single target LLM is an acceptable, and possibly
necessary, fallback"). Refit for another target with
``tools/fit_move_preference.py --player NAME``.
"""

_DIAGONAL_CELLS: Final[int] = _LINE_MASKS[-1] | _LINE_MASKS[-2]
_CENTER_CELL: Final[int] = 1 << (2 * BOARD_SIZE + 2)
_CORNER_CELLS: Final[int] = sum(
    1 << (r * BOARD_SIZE + c) for r in (0, BOARD_SIZE - 1) for c in (0, BOARD_SIZE - 1)
)


def live_threats(attacker: int, defender: int) -> list[int]:
    """``attacker``'s lines holding 3+ of its pieces and none of ``defender``'s.

    One or two gaps, either of which wins next turn (SPEC.md §2.5). Playing any
    empty cell of such a line kills it for good (§2.7).
    """
    return [
        line
        for line in _LINE_MASKS
        if not line & defender and 1 <= (line & ~attacker).bit_count() <= 2
    ]


def move_features(mine: int, theirs: int, move: int) -> tuple[float, ...]:
    """Features of playing ``move`` (a mask) from ``(mine, theirs)``.

    Every feature is a property of the *move*: a position-level one would be
    identical across the candidates it is compared against and cancel in the
    softmax, so it could carry no information about the choice.
    """
    cells: list[int] = [i for i in range(_CELL_COUNT) if (move >> i) & 1]
    after: int = mine | move
    threats: list[int] = live_threats(theirs, mine)

    aligned: float = 0.0
    adjacent: float = 0.0
    if len(cells) == 2:
        rows = (cells[0] // BOARD_SIZE, cells[1] // BOARD_SIZE)
        cols = (cells[0] % BOARD_SIZE, cells[1] % BOARD_SIZE)
        aligned = float(rows[0] == rows[1] or cols[0] == cols[1])
        adjacent = float(
            max(abs(rows[0] - rows[1]), abs(cols[0] - cols[1])) == 1
        )
    blocked: int = sum(1 for line in threats if line & move)
    return (
        aligned,
        adjacent,
        float(sum(1 for line in _LINE_MASKS
                  if line & move and not line & theirs and line & mine)),
        float(blocked),
        float(bool(threats) and blocked == len(threats)),
        float(threat_score(after, theirs) >= SCORE_SPLIT),
        float(bool(live_threats(after, theirs))),
        float(any(after & line == line for line in _LINE_MASKS)),
        float((move & _DIAGONAL_CELLS).bit_count()),
        float(bool(move & _CENTER_CELL)),
        float((move & _CORNER_CELLS).bit_count()),
    )


def move_preference(mine: int, theirs: int, move: int) -> float:
    """How much the modelled player prefers ``move``; higher is likelier."""
    return sum(
        w * f for w, f in zip(MOVE_PREFERENCE, move_features(mine, theirs, move))
    )


def blunder_probability(attacker: int, defender: int) -> float:
    """Predicted chance that ``defender``, to move, fails to answer ``attacker``.

    A reply is an *answer* when it kills every one of ``attacker``'s live
    threats, which is pure bit arithmetic — so this needs no game-tree search
    and is affordable inside a training loop, unlike asking
    :func:`~snakes_and_mice.players.perfect.evaluate` for each reply.

    Returns 0 when ``attacker`` holds no live threat (nothing to fail to answer)
    and when ``defender`` has a move that wins outright, since a position the
    defender simply wins is no trap however it is scored. The fitted weights
    carry no signal about winning moves — ``wins_now`` never varied in the
    blunder-only data it was fitted to — so that case is excluded here rather
    than mispredicted.

    Against a split threat no reply answers everything and this tends to 1,
    matching :func:`threat_score`'s top level. Its reason for existing is the
    *single*-threat case, where answers exist but the modelled player may well
    not choose one: 62.4% of real blunders are of that kind, and the three-level
    score rates every one of them 0 (SPEC-mistake-model.md, "Validation against
    588 real blunders").
    """
    threats: list[int] = live_threats(attacker, defender)
    if not threats:
        return 0.0
    free: int = _FULL_MASK & ~(attacker | defender)
    empties: list[int] = [i for i in range(_CELL_COUNT) if (free >> i) & 1]
    if len(empties) < 2:
        return 0.0

    scores: list[float] = []
    answers: list[bool] = []
    for first, second in combinations(empties, 2):
        move: int = (1 << first) | (1 << second)
        features = move_features(defender, attacker, move)
        if features[FEATURE_NAMES.index("wins_now")]:
            return 0.0
        scores.append(
            sum(w * f for w, f in zip(MOVE_PREFERENCE, features))
        )
        answers.append(bool(features[FEATURE_NAMES.index("answers_all")]))

    top: float = max(scores)
    weights: list[float] = [exp(s - top) for s in scores]
    total: float = sum(weights)
    answered: float = sum(w for w, a in zip(weights, answers) if a)
    return 1.0 - answered / total
