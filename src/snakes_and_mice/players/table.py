"""Loading the offline perfect-play table that covers the opening plies (§10).

Full-depth search is exact but far too slow in the opening, so an offline retrograde
solve (``tools/``) precomputes the exact value of every reachable position down to a
threshold, and :class:`~.perfect.PerfectPlayer` looks the opening up instead of
searching it. Below the threshold the live search is fast, so the table stops there.

The table is keyed by :func:`~.symmetry.canonical_key`, which is invariant under the
board's order-32 symmetry group. That is what makes one table per *seed orbit* enough
rather than one per seed, and it is also why nothing here maps moves between frames:
the player generates children on the real board, canonicalizes each, and looks up its
**value**, which the symmetry preserves.

Standard library only: the payload is read with :mod:`array` and searched with
:mod:`bisect`. numpy would do this too, but it is a dev-group dependency used by the
solver, and nothing here needs it — so the player stays installable with no
dependencies beyond the engine's. A preference rather than a rule; a real benefit
would be worth the dependency. Files may be plain or gzipped; gzip cuts a table from
~72 MB to ~11 MB and costs well under a second at load, so the compressed form is
tried first.
"""

from __future__ import annotations

import bisect
import gzip
import os
import struct
import sys
from array import array
from pathlib import Path
from typing import BinaryIO, cast

from ..core import BOARD_SIZE, Cell
from .symmetry import ALL_PERMS, CELL_COUNT

MAGIC: bytes = b"SNM-PERF"
VERSION: int = 1

TABLE_DIR_ENV: str = "SNAKES_AND_MICE_TABLE_DIR"
"""Environment variable naming the directory of ``.table`` / ``.table.gz`` files."""

DEFAULT_TABLE_DIR: Path = Path(__file__).resolve().parents[3] / "perfect-tables"
"""Where tables live when the environment says nothing — the repo's own directory."""

_HEADER: struct.Struct = struct.Struct("<8sBBBB")
_LAYER: struct.Struct = struct.Struct("<BBI")


def seed_representative(seed: Cell) -> Cell:
    """The seed of ``seed``'s orbit that a table is stored under.

    The 25 seeds fall into 4 orbits under the symmetry group, so 4 tables cover every
    opening. The representative is the orbit's lowest cell index, which makes the
    choice canonical and independent of how the orbits were discovered.
    """
    index: int = seed.row * BOARD_SIZE + seed.col
    orbit: int = min(perm[index] for perm in ALL_PERMS)
    return Cell(orbit // BOARD_SIZE, orbit % BOARD_SIZE)


class PerfectTable:
    """Exact values for every solved position, by layer and canonical key."""

    def __init__(self, layers: dict[int, tuple[array[int], array[int]]]) -> None:
        self._layers: dict[int, tuple[array[int], array[int]]] = layers

    @property
    def empties_covered(self) -> frozenset[int]:
        """The layers this table can answer for, by empty-cell count."""
        return frozenset(self._layers)

    def covers(self, empties: int) -> bool:
        """Whether a position with ``empties`` empty cells can be looked up."""
        return empties in self._layers

    def value(self, empties: int, key: int) -> int | None:
        """The exact value of a canonical position, or ``None`` if absent.

        Absent means the position is not in the solved set — terminal positions were
        never enumerated, so the caller must detect wins and cat's games itself.
        """
        layer: tuple[array[int], array[int]] | None = self._layers.get(empties)
        if layer is None:
            return None
        keys, values = layer
        at: int = bisect.bisect_left(keys, key)
        if at < len(keys) and keys[at] == key:
            return values[at]
        return None

    @classmethod
    def load(cls, path: Path) -> PerfectTable:
        """Read a table file, transparently handling gzip."""
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as handle:
            return cls._read(cast(BinaryIO, handle))

    @classmethod
    def _read(cls, handle: BinaryIO) -> PerfectTable:
        magic, version, _seed_bit, count, _pad = _HEADER.unpack(
            handle.read(_HEADER.size)
        )
        if magic != MAGIC:
            raise ValueError(f"not a perfect-play table (magic {magic!r})")
        if version != VERSION:
            raise ValueError(f"unsupported table version {version}")

        sizes: list[tuple[int, int]] = []
        for _ in range(count):
            empties, _reserved, entries = _LAYER.unpack(handle.read(_LAYER.size))
            sizes.append((empties, entries))

        layers: dict[int, tuple[array[int], array[int]]] = {}
        for empties, entries in sizes:
            keys: array[int] = array("Q")
            values: array[int] = array("h")
            # gzip streams do not support `fromfile`, so read bytes and unpack.
            keys.frombytes(handle.read(entries * keys.itemsize))
            values.frombytes(handle.read(entries * values.itemsize))
            if len(keys) != entries or len(values) != entries:
                raise ValueError(f"table truncated in layer {empties}")
            # The file is little-endian; swap on a big-endian host.
            if struct.pack("<H", 1) != struct.pack("=H", 1):
                keys.byteswap()
                values.byteswap()
            layers[empties] = (keys, values)
        return cls(layers)


def table_dir() -> Path:
    """Where to look for tables: the environment first, then the default."""
    configured: str | None = os.environ.get(TABLE_DIR_ENV)
    return Path(configured) if configured else DEFAULT_TABLE_DIR


_CACHE: dict[str, PerfectTable | None] = {}
"""Loaded tables, by representative label.

Reading ~72 MB per game would dominate a tournament, and a solved table is immutable
public fact about the game, not per-game state — so caching it does not give a player
memory across games the way a transposition table would.
"""


def _warn(message: str) -> None:
    """Report a table problem on stderr, matching the CLI's warning style."""
    print(f"warning: {message}", file=sys.stderr, flush=True)


def load_for_seed(seed: Cell) -> PerfectTable | None:
    """The table covering ``seed``'s orbit, or ``None`` if it could not be loaded.

    A missing table is not an error: the player falls back to searching the opening,
    which is correct but slow. That keeps the player usable — and testable — on a
    checkout that has not been given the solver's output.

    It is, however, always **reported**. The fallback is not a mild degradation: the
    opening is the part the table exists to avoid, and searching it can take hours per
    move, which is indistinguishable from a hang to anyone watching a match. Warning
    once per table (the cache makes that natural) says what is wrong and how to fix it.
    """
    representative: Cell = seed_representative(seed)
    directory: Path = table_dir()
    cache_key: str = f"{directory}:{representative.label}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    loaded: PerfectTable | None = None
    for suffix in (".table.gz", ".table"):
        path: Path = directory / f"{representative.label}{suffix}"
        if not path.exists():
            continue
        try:
            loaded = PerfectTable.load(path)
        except (OSError, ValueError) as exc:
            # A damaged table must not stop play, but must not pass unnoticed either.
            _warn(
                f"could not read perfect-play table {path}: {exc}. "
                f"PerfectPlayer will search the opening instead, which is correct "
                f"but can take hours per move."
            )
        break

    if loaded is None and not any(
        (directory / f"{representative.label}{suffix}").exists()
        for suffix in (".table.gz", ".table")
    ):
        _warn(
            f"no perfect-play table for seed {seed.label} "
            f"(orbit representative {representative.label}) in {directory} — "
            f"expected {representative.label}.table.gz or {representative.label}.table."
        )
        _warn(
            "PerfectPlayer will search the opening instead: still perfect, but the "
            "first moves can take hours. Build tables with "
            "tools/solver/build_table.py, or point "
            f"{TABLE_DIR_ENV} at a directory that has them."
        )

    _CACHE[cache_key] = loaded
    return loaded


__all__ = [
    "CELL_COUNT",
    "PerfectTable",
    "load_for_seed",
    "seed_representative",
    "table_dir",
]
