"""The board's line-preserving symmetry group and the canonical-position key.

The value of a position is invariant under every transformation that maps the 12
winning lines onto themselves. These form a group **G of order 32** — strictly
larger than the 8 rigid rotations and reflections (**D₄**) of the square (§10).
Every element has the form ``(r, c) ↦ (σr, τc)`` or ``(r, c) ↦ (σc, τr)`` (the
latter transposing rows and columns), where ``σ`` is one of the 8 permutations of
``{0..4}`` that commute with the reversal ``ρ = (0 4)(1 3)`` and ``τ ∈ {σ, ρσ}``.
D₄ is the slice with ``σ ∈ {identity, ρ}``; the other 24 elements are "warps"
(such as *swap rows B↔D and columns 2↔4*) that preserve every line without being
rigid motions.

Each symmetry is materialized as an **index permutation** of the 25 cells, so
applying it to a bit-mask is a bit permutation. :func:`canonical_key` folds a
position (a mouse mask and a snake mask) to the numerically smallest of its 32
transforms — a true canonical form, since the 32 transforms are exactly the
position's orbit under G. A self-check at import time asserts that the group has
exactly 32 distinct elements and that each maps the set of 12 line-index-sets
onto itself.
"""

from __future__ import annotations

from itertools import permutations

from ..board import LINES
from ..core import BOARD_SIZE

CELL_COUNT: int = BOARD_SIZE * BOARD_SIZE
"""Number of cells (and the width, in bits, of a single-piece mask)."""


def _index(row: int, col: int) -> int:
    """The 0..24 index of a cell, row-major (``5·row + col``)."""
    return row * BOARD_SIZE + col


def _reversal() -> tuple[int, ...]:
    """The coordinate reversal ``ρ`` on ``{0..4}`` (``i ↦ 4 − i``)."""
    return tuple(BOARD_SIZE - 1 - i for i in range(BOARD_SIZE))


def _compose(outer: tuple[int, ...], inner: tuple[int, ...]) -> tuple[int, ...]:
    """The composition ``outer ∘ inner`` of two coordinate permutations."""
    return tuple(outer[inner[i]] for i in range(len(inner)))


def _centralizer_of_reversal() -> tuple[tuple[int, ...], ...]:
    """The 8 permutations of ``{0..4}`` that commute with the reversal ``ρ``.

    Equivalently: those fixing the center coordinate 2 and preserving the pairs
    ``{0,4}`` and ``{1,3}``. Found by filtering all 120 permutations — cheap, and
    it needs no hand-derived list to be trusted.
    """
    rho: tuple[int, ...] = _reversal()
    commuting: list[tuple[int, ...]] = [
        sigma
        for sigma in permutations(range(BOARD_SIZE))
        if all(sigma[rho[i]] == rho[sigma[i]] for i in range(BOARD_SIZE))
    ]
    return tuple(commuting)


def _build_group() -> tuple[tuple[int, ...], ...]:
    """Materialize G as a sorted tuple of 25-element index permutations."""
    rho: tuple[int, ...] = _reversal()
    perms: set[tuple[int, ...]] = set()
    for sigma in _centralizer_of_reversal():
        for tau in (sigma, _compose(rho, sigma)):
            for transpose in (False, True):
                perm: list[int] = [0] * CELL_COUNT
                for r in range(BOARD_SIZE):
                    for c in range(BOARD_SIZE):
                        if transpose:
                            new_row, new_col = sigma[c], tau[r]
                        else:
                            new_row, new_col = sigma[r], tau[c]
                        perm[_index(r, c)] = _index(new_row, new_col)
                perms.add(tuple(perm))
    return tuple(sorted(perms))


ALL_PERMS: tuple[tuple[int, ...], ...] = _build_group()
"""The 32 symmetries of G, each a 0..24 → 0..24 index permutation."""


def _line_index_sets() -> frozenset[frozenset[int]]:
    """The 12 winning lines as sets of cell indices."""
    return frozenset(
        frozenset(_index(cell.row, cell.col) for cell in line) for line in LINES
    )


def _validate() -> None:
    """Assert G has exactly 32 elements, each preserving the set of 12 lines."""
    assert len(ALL_PERMS) == 32, f"expected 32 symmetries, got {len(ALL_PERMS)}"
    lines: frozenset[frozenset[int]] = _line_index_sets()
    for perm in ALL_PERMS:
        for line in lines:
            image: frozenset[int] = frozenset(perm[i] for i in line)
            assert image in lines, "a symmetry did not map lines onto lines"


_validate()


def permute(perm: tuple[int, ...], mask: int) -> int:
    """Apply an index permutation to a bit-mask, moving each set bit to its image."""
    result: int = 0
    remaining: int = mask
    while remaining:
        low: int = remaining & -remaining
        result |= 1 << perm[low.bit_length() - 1]
        remaining ^= low
    return result


_CHUNK: int = 5
"""Width, in bits, of one lookup chunk; ``CELL_COUNT`` is a whole number of these."""

_CHUNKS: int = (CELL_COUNT + _CHUNK - 1) // _CHUNK


def _build_permute_tables() -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Precompute, per symmetry and per 5-bit chunk, the permuted contribution of
    every chunk value — so :func:`canonical_key` permutes a mask with a handful of
    table lookups instead of iterating its set bits."""
    tables: list[tuple[tuple[int, ...], ...]] = []
    for perm in ALL_PERMS:
        per_chunk: list[tuple[int, ...]] = []
        for chunk in range(_CHUNKS):
            base: int = chunk * _CHUNK
            values: list[int] = []
            for value in range(1 << _CHUNK):
                image: int = 0
                for bit in range(_CHUNK):
                    if (value >> bit) & 1:
                        image |= 1 << perm[base + bit]
                values.append(image)
            per_chunk.append(tuple(values))
        tables.append(tuple(per_chunk))
    return tuple(tables)


_PERMUTE_TABLES: tuple[tuple[tuple[int, ...], ...], ...] = _build_permute_tables()
_MASK5: int = (1 << _CHUNK) - 1


def canonical_key(mouse: int, snake: int) -> int:
    """The canonical key of a position: the smallest of its 32 symmetric transforms.

    Mouse and snake masks are transformed *together* by the same symmetry and
    packed into one integer (``mouse`` in the high 25 bits, ``snake`` in the low
    25) so the comparison ranks whole positions. The minimum is identical for
    every member of a position's orbit and distinct across orbits.
    """
    best: int = -1
    for table in _PERMUTE_TABLES:
        t0, t1, t2, t3, t4 = table
        pm: int = (
            t0[mouse & _MASK5]
            | t1[(mouse >> 5) & _MASK5]
            | t2[(mouse >> 10) & _MASK5]
            | t3[(mouse >> 15) & _MASK5]
            | t4[(mouse >> 20) & _MASK5]
        )
        ps: int = (
            t0[snake & _MASK5]
            | t1[(snake >> 5) & _MASK5]
            | t2[(snake >> 10) & _MASK5]
            | t3[(snake >> 15) & _MASK5]
            | t4[(snake >> 20) & _MASK5]
        )
        packed: int = (pm << CELL_COUNT) | ps
        if best < 0 or packed < best:
            best = packed
    return best
