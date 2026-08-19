"""Read a solved table and print it in human-readable form.

The shipped tables are binary — packed keys and values, sized for a fast
standard-library loader rather than for reading (see ``SPEC.md``, "The table file"). This turns them
back into something a person can inspect: summaries, filtered listings, board
diagrams, and lookups of a specific position.

    uv run python tools/solver/dump_table.py TABLE [options]

    # what is in it, and how the values are distributed
    uv run python tools/solver/dump_table.py perfect-tables/C3.table.gz

    # the mouse's twenty distinct opening replies, as boards
    uv run python tools/solver/dump_table.py perfect-tables/C3.table.gz -e 22 --board

    # positions at 16 empties that are already lost for the side to move
    uv run python tools/solver/dump_table.py perfect-tables/A1.table.gz -e 16 --lost

    # the value of one position, given as the cells each side holds
    uv run python tools/solver/dump_table.py perfect-tables/C3.table.gz \\
        --mouse A1,A2 --snake C3
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from snakes_and_mice.core import BOARD_SIZE, Cell, Side
from snakes_and_mice.players.symmetry import CELL_COUNT, canonical_key
from snakes_and_mice.players.table import PerfectTable

WIN: int = 1000
FULL: int = (1 << CELL_COUNT) - 1


def label(index: int) -> str:
    """The ``A1``-style label of a cell index."""
    return f"{chr(ord('A') + index // BOARD_SIZE)}{index % BOARD_SIZE + 1}"


def cells(mask: int) -> list[str]:
    """The labels of every set bit, in board order."""
    return [label(i) for i in range(CELL_COUNT) if (mask >> i) & 1]


def unpack(key: int) -> tuple[int, int]:
    """The (mouse, snake) masks of a canonical key."""
    return (key >> CELL_COUNT) & FULL, key & FULL


def empties_of(mouse: int, snake: int) -> int:
    return CELL_COUNT - (mouse.bit_count() + snake.bit_count())


def side_to_move(empties: int) -> Side:
    """Whose turn it is, which is fixed by the layer: the mouse moves first."""
    return Side.MOUSE if ((24 - empties) // 2) % 2 == 0 else Side.SNAKE


def side_from_counts(mouse: int, snake: int) -> Side | None:
    """Whose turn it is by piece count, or ``None`` if the position is impossible.

    Every position between complete moves has ``snake == mouse + 1`` (mouse to move)
    or ``snake == mouse - 1`` (snake to move). Both counts are also fixed in parity:
    the mouse places two pieces a move so it holds an **even** number, and the snake
    starts on the seed and adds two at a time so it holds an **odd** one. Checking
    only the difference is not enough — mouse 3 / snake 2 differs by one and is still
    impossible. (Terminal positions can break the parity via a one-piece winning move,
    but those are never in the table.)
    """
    mice: int = mouse.bit_count()
    snakes: int = snake.bit_count()
    if mice % 2 != 0 or snakes % 2 != 1:
        return None
    if snakes - mice == 1:
        return Side.MOUSE
    if snakes - mice == -1:
        return Side.SNAKE
    return None


def describe(value: int, empties: int) -> str:
    """A value in words, from the perspective of the side to move."""
    if value == 0:
        return "draw"
    depth: int = (CELL_COUNT - empties - 1) // 2
    # A win scores WIN - (depth of the position the winning move is made from), and a
    # loss is that negated, so either way the winning ply is recoverable.
    winning_depth: int = WIN - value if value > 0 else WIN + value
    plies: int = winning_depth - depth
    outcome: str = "win" if value > 0 else "loss"
    if plies == 0:
        return f"{outcome} now"
    return f"{outcome} in {plies} {'ply' if plies == 1 else 'plies'}"


def board_diagram(mouse: int, snake: int) -> str:
    """The position as a 5x5 grid: ``M`` mouse, ``S`` snake, ``.`` empty."""
    lines: list[str] = ["    1 2 3 4 5"]
    for row in range(BOARD_SIZE):
        marks: list[str] = []
        for col in range(BOARD_SIZE):
            bit: int = row * BOARD_SIZE + col
            if (mouse >> bit) & 1:
                marks.append("M")
            elif (snake >> bit) & 1:
                marks.append("S")
            else:
                marks.append(".")
        lines.append(f"  {chr(ord('A') + row)} " + " ".join(marks))
    return "\n".join(lines)


def summarise(table: PerfectTable, path: Path) -> None:
    """Layer sizes and the value distribution of each."""
    print(f"{path}  ({path.stat().st_size / 2**20:.2f} MB on disk)")
    print(f"{'empties':>8} {'to move':>8} {'positions':>12}   values")
    for empties in sorted(table.empties_covered, reverse=True):
        keys, values = table._layers[empties]
        counts: Counter[int] = Counter(values)
        parts: list[str] = [
            f"{describe(value, empties)}: {count:,}"
            for value, count in sorted(counts.items(), reverse=True)
        ]
        print(
            f"{empties:>8} {side_to_move(empties).name.lower():>8} "
            f"{len(keys):>12,}   " + ", ".join(parts)
        )


def listing(table: PerfectTable, empties: int, args: argparse.Namespace) -> None:
    """Entries of one layer, filtered and formatted."""
    keys, values = table._layers[empties]
    mover: Side = side_to_move(empties)
    shown: int = 0
    print(f"layer {empties} empties, {mover.name.lower()} to move, {len(keys):,} entries")
    for i in range(len(keys)):
        value: int = values[i]
        if args.won and value <= 0:
            continue
        if args.lost and value >= 0:
            continue
        if args.drawn and value != 0:
            continue
        if args.value is not None and value != args.value:
            continue
        mouse, snake = unpack(keys[i])
        if args.board:
            print(f"\n[{i}] {describe(value, empties)}  (value {value})")
            print(board_diagram(mouse, snake))
        else:
            print(
                f"[{i:>9}] {keys[i]:#014x}  {value:>5}  {describe(value, empties):<14}"
                f" mouse={','.join(cells(mouse)) or '-'}"
                f" snake={','.join(cells(snake))}"
            )
        shown += 1
        if shown >= args.limit:
            print(f"\n... stopping at --limit {args.limit}")
            return
    if shown == 0:
        print("(no entries matched)")


def lookup(table: PerfectTable, mouse_arg: str, snake_arg: str) -> int:
    """Print the value of one position, given as cell labels. Returns an exit code."""
    mouse: int = 0
    snake: int = 0
    for text in filter(None, mouse_arg.split(",")):
        cell: Cell = Cell.from_label(text.strip().upper())
        mouse |= 1 << (cell.row * BOARD_SIZE + cell.col)
    for text in filter(None, snake_arg.split(",")):
        cell = Cell.from_label(text.strip().upper())
        snake |= 1 << (cell.row * BOARD_SIZE + cell.col)
    if mouse & snake:
        print("error: a cell is claimed by both sides")
        return 2

    empties: int = empties_of(mouse, snake)
    print(board_diagram(mouse, snake))
    mover: Side | None = side_from_counts(mouse, snake)
    if mover is None:
        print(
            f"\nimpossible position: mouse holds {mouse.bit_count()}, "
            f"snake holds {snake.bit_count()}. The mouse's count must be even, the "
            "snake's odd, and the two must differ by exactly one."
        )
        return 2

    key: int = canonical_key(mouse, snake)
    print(f"\nempties {empties}, {mover.name.lower()} to move")
    print(f"canonical key {key:#014x}")
    if not table.covers(empties):
        print(f"not covered: this table holds {sorted(table.empties_covered)} empties")
        return 1
    value: int | None = table.value(empties, key)
    if value is None:
        print("not in the table — unreachable, or already terminal (won or a cat's game)")
        return 1
    print(f"value {value}  ({describe(value, empties)})")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a solved perfect-play table in readable form.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("table", type=Path, help="a .table or .table.gz file")
    parser.add_argument(
        "-e", "--empties", type=int, help="list this layer instead of summarising"
    )
    parser.add_argument("-n", "--limit", type=int, default=20, help="max rows (20)")
    parser.add_argument("--board", action="store_true", help="draw each as a grid")
    parser.add_argument("--value", type=int, help="only entries with this exact value")
    parser.add_argument("--won", action="store_true", help="only wins for the mover")
    parser.add_argument("--lost", action="store_true", help="only losses for the mover")
    parser.add_argument("--drawn", action="store_true", help="only draws")
    parser.add_argument("--mouse", default="", help="look up: mouse cells, e.g. A1,A2")
    parser.add_argument("--snake", default="", help="look up: snake cells, e.g. C3")
    args = parser.parse_args()

    table: PerfectTable = PerfectTable.load(args.table)

    if args.mouse or args.snake:
        raise SystemExit(lookup(table, args.mouse, args.snake))
    if args.empties is None:
        summarise(table, args.table)
        return
    if not table.covers(args.empties):
        raise SystemExit(
            f"layer {args.empties} is not in this table; "
            f"it holds {sorted(table.empties_covered)}"
        )
    listing(table, args.empties, args)


if __name__ == "__main__":
    main()
