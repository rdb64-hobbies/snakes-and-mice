"""Building synthetic perfect-play tables for tests.

The solved tables are large and are not part of a checkout, so no test may
depend on them being installed. Everything that needs a table builds a tiny one
holding exactly the positions that test cares about.
"""

from __future__ import annotations

import struct
from pathlib import Path

from snakes_and_mice.players.table import MAGIC, VERSION


def write_table(path: Path, layers: dict[int, list[tuple[int, int]]]) -> None:
    """Write a table file holding the given (key, value) pairs per layer."""
    ordered: list[int] = sorted(layers, reverse=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<8sBBBB", MAGIC, VERSION, 12, len(ordered), 0))
        for empties in ordered:
            handle.write(struct.pack("<BBI", empties, 0, len(layers[empties])))
        for empties in ordered:
            entries = sorted(layers[empties])
            for key, _value in entries:
                handle.write(struct.pack("<Q", key))
            for _key, value in entries:
                handle.write(struct.pack("<h", value))
