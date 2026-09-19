"""The mistake model: how hard a position is for a *fallible* defender to hold.

A score over board features, specified in `SPEC-mistake-model.md` ("The mistake
model", "The score, as implemented"). Positions are the pair of 25-bit masks the
perfect player searches with, one bit per cell at ``5·row + col``.
"""

from __future__ import annotations

from itertools import combinations
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
