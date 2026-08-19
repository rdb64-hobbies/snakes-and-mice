"""Vectorized position enumeration shared by the solver's two passes.

Positions are a pair of 25-bit masks (mouse, snake) packed into one ``uint64``
canonical key. Everything here works on whole arrays of positions rather than one at
a time: within a layer every position has the same fan-out, so children are generated
a layer at a time and canonicalized with precomputed 5-bit-chunk permutation tables.
That is what makes a pure-Python solver viable — the interpreter is entered once per
*million* positions, and the real ceiling is memory bandwidth, not Python.

Solver-only: this imports numpy, which is a dev-group dependency and is not used
anywhere in the runtime player.
"""

from __future__ import annotations

import numpy as np

from snakes_and_mice.players.perfect import _LINE_MASKS
from snakes_and_mice.players.symmetry import ALL_PERMS, CELL_COUNT

FULL: int = (1 << CELL_COUNT) - 1
"""A mask with every cell bit set."""


def _build_tables() -> np.ndarray:
    """The permutation lookup tables as a ``(32, 5, 32)`` uint32 array.

    ``tables[p, k, v]`` is the image, under symmetry ``p``, of chunk ``k`` (bits
    ``5k..5k+4``) holding the 5-bit value ``v``.
    """
    tables: np.ndarray = np.zeros((len(ALL_PERMS), 5, 32), dtype=np.uint32)
    for p, perm in enumerate(ALL_PERMS):
        for k in range(5):
            base: int = k * 5
            for v in range(32):
                image: int = 0
                for bit in range(5):
                    if (v >> bit) & 1:
                        image |= 1 << perm[base + bit]
                tables[p, k, v] = image
    return tables


_TABLES: np.ndarray = _build_tables()


def _permute(masks: np.ndarray, p: int) -> np.ndarray:
    """Apply symmetry ``p`` to a uint32 array of 25-bit masks (vectorized)."""
    t: np.ndarray = _TABLES[p]
    image: np.ndarray = (
        t[0][masks & 31]
        | t[1][(masks >> 5) & 31]
        | t[2][(masks >> 10) & 31]
        | t[3][(masks >> 15) & 31]
        | t[4][(masks >> 20) & 31]
    )
    return image


def canon(mouse: np.ndarray, snake: np.ndarray) -> np.ndarray:
    """Canonical packed keys (uint64) for arrays of (mouse, snake) masks."""
    best: np.ndarray | None = None
    for p in range(len(ALL_PERMS)):
        pm: np.ndarray = _permute(mouse, p).astype(np.uint64)
        ps: np.ndarray = _permute(snake, p).astype(np.uint64)
        packed: np.ndarray = (pm << np.uint64(CELL_COUNT)) | ps
        best = packed if best is None else np.minimum(best, packed)
    assert best is not None
    return best


def wins(mine: np.ndarray) -> np.ndarray:
    """Boolean array: whether each mask completes some winning line."""
    result: np.ndarray = np.zeros(mine.shape, dtype=bool)
    for line in _LINE_MASKS:
        result |= (mine & line) == line
    return result


def cats(mouse: np.ndarray, snake: np.ndarray) -> np.ndarray:
    """Boolean array: whether every line is dead (both colors present)."""
    result: np.ndarray = np.ones(mouse.shape, dtype=bool)
    for line in _LINE_MASKS:
        result &= ((mouse & line) != 0) & ((snake & line) != 0)
    return result


_PAIRS: tuple[int, ...] = tuple(
    (1 << a) | (1 << b) for a in range(CELL_COUNT) for b in range(a + 1, CELL_COUNT)
)


def child_keys(m: np.ndarray, s: np.ndarray, mouse_to_move: bool) -> np.ndarray:
    """Unique canonical keys of the *non-terminal* children of a batch of positions.

    Generates every child across all 300 cell-pairs at once, drops the terminal ones
    (a win or a cat's game), then canonicalizes and dedups the whole batch — keeping
    the per-pair Python overhead well below the vectorized work.
    """
    occ: np.ndarray = m | s
    nm_parts: list[np.ndarray] = []
    ns_parts: list[np.ndarray] = []
    for bitab in _PAIRS:
        elig: np.ndarray = (occ & bitab) == 0
        if not elig.any():
            continue
        if mouse_to_move:
            nm_parts.append(m[elig] | bitab)
            ns_parts.append(s[elig])
        else:
            nm_parts.append(m[elig])
            ns_parts.append(s[elig] | bitab)
    if not nm_parts:
        return np.empty(0, dtype=np.uint64)
    nm: np.ndarray = np.concatenate(nm_parts)
    ns: np.ndarray = np.concatenate(ns_parts)
    mine: np.ndarray = nm if mouse_to_move else ns
    keep: np.ndarray = ~(wins(mine) | cats(nm, ns))
    if not keep.any():
        return np.empty(0, dtype=np.uint64)
    return np.unique(canon(nm[keep], ns[keep]))
