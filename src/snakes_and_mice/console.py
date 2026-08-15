"""Text console presentation: board rendering, result summaries, and a
stdout observer that narrates a match as it plays.

This is the reusable presentation layer, kept separate from any single command-
line entry point so more than one frontend can share it. It knows nothing about
argument parsing, rosters, or process wiring — those belong to :mod:`cli` (and
to any other CLI that renders through this module).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from .board import Board
from .core import BOARD_SIZE, Cell, Move, Side, TurnOutcome
from .faults import PlayerFaultReason
from .observer import ObservationLevel, Observer
from .result import GameResult, MatchResult, PlayerFaultDetail, Termination
from .tally import PlayerStanding, StandingsSort

# The piece glyphs are emoji, which occupy TWO display columns in a terminal.
# Every rendered cell is therefore normalized to a two-column token so the board
# lines up: emoji are left as-is, and the single-column empty marker is padded
# with a trailing space. Header labels and the row-label margin are sized to
# match (two columns each, joined by a single-space gutter).
GLYPH: dict[Side, str] = {Side.MOUSE: "🐭", Side.SNAKE: "🐍"}
EMPTY: str = "·"
_CELL_WIDTH: int = 2


def _cell_token(board: Board, cell: Cell) -> str:
    """A two-display-column token for one cell: the piece emoji, or a padded dot."""
    occupant: Side | None = board.occupant(cell)
    if occupant is not None:
        return GLYPH[occupant]
    return EMPTY.ljust(_CELL_WIDTH)


def render_board(board: Board) -> str:
    """Render the board as text, row ``A`` at the top and column ``1`` at left."""
    margin: str = " " * _CELL_WIDTH  # sits above the "<row> " row-label column
    header: str = margin + " ".join(
        str(c + 1).ljust(_CELL_WIDTH) for c in range(BOARD_SIZE)
    )
    rows: list[str] = [header]
    for r in range(BOARD_SIZE):
        row_label: str = chr(ord("A") + r)
        cells: list[str] = [
            _cell_token(board, Cell(r, c)) for c in range(BOARD_SIZE)
        ]
        rows.append(f"{row_label} " + " ".join(cells))
    return "\n".join(rows)


def describe_game_result(result: GameResult, players: dict[Side, str]) -> str:
    """A one-line human-readable summary of a game result."""
    if result.termination is Termination.LINE_COMPLETED:
        assert result.winner is not None
        return f"{players[result.winner]} ({result.winner.value}) wins."
    if result.termination is Termination.CATS_GAME:
        return "Cat's game — a draw."
    if result.termination is Termination.ABORTED:
        abort: str = "Game abandoned (no contest)"
        if result.error is not None:
            abort += f": {result.error}"
        return abort
    fault: PlayerFaultDetail | None = result.fault
    assert fault is not None
    detail: str = f"{players[fault.offender]} ({fault.offender.value}) faulted: {fault.reason.value}"
    if fault.attempted_move is not None:
        detail += f" on {fault.attempted_move}"
    if fault.claimed_outcome is not None and fault.actual_outcome is not None:
        detail += (
            f" (claimed {fault.claimed_outcome.value}, "
            f"actually {fault.actual_outcome.value})"
        )
    return detail


def describe_match_result(result: MatchResult) -> str:
    """A multi-line human-readable summary of a match's tallies.

    When either side faulted, the fault line is followed by a per-side breakdown
    of the fault types that side incurred (only the sides that faulted appear).
    """
    names: dict[Side, str] = result.names
    lines: list[str] = [
        f"Match complete — {result.num_games} "
        f"{'game' if result.num_games == 1 else 'games'}",
        f"  🐭 {names[Side.MOUSE]} (mouse): {result.mouse_wins} "
        f"{'win' if result.mouse_wins == 1 else 'wins'}",
        f"  🐍 {names[Side.SNAKE]} (snake): {result.snake_wins} "
        f"{'win' if result.snake_wins == 1 else 'wins'}",
        f"  Cat's games: {result.cats_games}",
    ]
    if result.faults:
        lines.append(
            f"  Faults: {result.mouse_faults} mouse, {result.snake_faults} snake"
        )
        for side in Side:
            breakdown: str = _fault_breakdown(result.faults, side)
            if breakdown:
                lines.append(f"    {GLYPH[side]} {side.value}: {breakdown}")
    if result.aborted:
        # A no-contest tally, shown only when it happened: these games belong to
        # neither player, so they sit apart from the wins and faults above.
        lines.append(f"  Abandoned (no contest): {result.aborted}")
    return "\n".join(lines)


def _fault_breakdown(faults: list[GameResult], side: Side) -> str:
    """A ``reason ×n`` tally of one side's faults, most frequent first (ties
    broken by name), or an empty string if that side never faulted."""
    counts: Counter[PlayerFaultReason] = Counter(
        game.fault.reason
        for game in faults
        if game.fault is not None and game.fault.offender is side
    )
    return _format_reason_counts(counts)


def _format_reason_counts(counts: Mapping[PlayerFaultReason, int]) -> str:
    """Format fault-reason counts as ``reason ×n, …``, most frequent first (ties
    broken by reason name), or an empty string if there are none."""
    parts: list[str] = [
        f"{reason.value} ×{count}"
        for reason, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0].value)
        )
    ]
    return ", ".join(parts)


# Standings columns: header, and how to read the value off a PlayerStanding.
# The count columns render as-is; the three rate columns go through _percent.
_STANDING_COLUMNS: tuple[tuple[str, str], ...] = (
    ("Player", "name"),
    ("Played", "played"),
    ("Won", "won"),
    ("Lost", "lost"),
    ("Tied", "tied"),
    ("Faulted", "faulted"),
    ("OppFaulted", "opponent_faulted"),
    ("Win%", "win_rate"),
    ("Loss%", "loss_rate"),
    ("Fault%", "fault_rate"),
)
_RATE_COLUMNS: frozenset[str] = frozenset({"win_rate", "loss_rate", "fault_rate"})


def _percent(rate: float | None) -> str:
    """A rate in ``[0, 1]`` as a percentage, or ``—`` when it is undefined (§6)."""
    return "—" if rate is None else f"{rate * 100:.1f}%"


def render_standings(
    standings: Sequence[PlayerStanding], sort: StandingsSort
) -> str:
    """Render per-player standings (§6) as an aligned text table.

    ``standings`` are shown in the order given (already ranked by
    :func:`~snakes_and_mice.tally.sort_standings`); ``sort`` only labels the header
    so the reader knows which column ordered the table. The name column is
    left-aligned, every count and rate column right-aligned.
    """
    if not standings:
        return "No matches recorded yet."

    rows: list[list[str]] = []
    for standing in standings:
        cells: list[str] = []
        for _, attr in _STANDING_COLUMNS:
            value: object = getattr(standing, attr)
            if attr in _RATE_COLUMNS:
                assert value is None or isinstance(value, float)
                cells.append(_percent(value))
            else:
                cells.append(str(value))
        rows.append(cells)

    headers: list[str] = [header for header, _ in _STANDING_COLUMNS]
    widths: list[int] = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def format_row(cells: Sequence[str]) -> str:
        # First column (the name) left-aligned; the numeric columns right-aligned.
        padded: list[str] = [cells[0].ljust(widths[0])]
        padded += [cells[i].rjust(widths[i]) for i in range(1, len(cells))]
        return "  ".join(padded).rstrip()

    lines: list[str] = [
        f"Standings — {len(standings)} "
        f"{'player' if len(standings) == 1 else 'players'}, sorted by {sort.value}%",
        "",
        format_row(headers),
        *(format_row(row) for row in rows),
    ]
    return "\n".join(lines)


def render_fault_tally(standings: Sequence[PlayerStanding]) -> str:
    """Render a per-player fault breakdown for every player that faulted (§6).

    Uses the same ``reason ×n`` format as a match's own fault breakdown, one line
    per player, in the order given (so it follows the ranked standings). Players
    with no faults are omitted; if no one faulted, a plain note is returned.
    """
    faulted: list[PlayerStanding] = [s for s in standings if s.faulted]
    if not faulted:
        return "No faults recorded."
    lines: list[str] = ["Faults by player:"]
    for standing in faulted:
        lines.append(f"  {standing.name}: {_format_reason_counts(standing.fault_reasons)}")
    return "\n".join(lines)


class ConsoleObserver(Observer):
    """Renders a match to stdout, showing as much as its level asks for.

    The engine fires every hook; this observer gates its own output against the
    :class:`~snakes_and_mice.observer.ObservationLevel` it was built with —
    ``move`` shows every turn, ``game`` just each game's board and result, and
    ``match`` only the opening banner and the closing tally. The two coarser
    levels also show a single line of dots for progress: ``game`` a dot per move
    within a game, ``match`` a dot per finished game. Per-game
    bookkeeping (the game counter, the turn counter) is kept up to date at every
    level, so it is correct whenever a finer level does render. It never pauses:
    with a human player, that player's own input paces the game; otherwise play
    runs straight through.
    """

    def __init__(self, level: ObservationLevel = ObservationLevel.MOVE) -> None:
        super().__init__(level)
        self._names: dict[Side, str] = {}
        self._num_games: int = 1
        self._game: int = 0
        self._turn: int = 0

    def on_match_start(self, names: dict[Side, str], num_games: int) -> None:
        self._names = names
        self._num_games = num_games
        print("Snakes and Mice\n")
        print(f"🐭 Mouse: {names[Side.MOUSE]}    🐍 Snake: {names[Side.SNAKE]}")
        if num_games > 1:
            print(f"Match: {num_games} games")

    def on_game_start(self, names: dict[Side, str], board: Board) -> None:
        self._names = names
        self._game += 1
        self._turn = 0
        if self.level < ObservationLevel.GAME:
            return
        if self._num_games > 1:
            print(f"\n=== Game {self._game} of {self._num_games} ===")
        # The starting board is only worth showing when the moves that follow
        # will update it — i.e. at MOVE level. At GAME level it would be a lone
        # seeded board no one watches change, so skip it.
        if self.level >= ObservationLevel.MOVE:
            print()
            print(render_board(board))

    def on_move_start(self, side: Side, board: Board) -> None:
        if self.level < ObservationLevel.MOVE:
            return
        self._turn += 1
        # Printed before the move is produced, so a slow player (e.g. an LLM)
        # visibly "thinks" here before on_move_end reports what it played.
        print(f"\nTurn {self._turn} — {self._names[side]} ({side.value}) to move…")

    def on_move_end(
        self, side: Side, move: Move, board: Board, outcome: TurnOutcome
    ) -> None:
        if self.level >= ObservationLevel.MOVE:
            print(f"  plays {move}:\n")
            print(render_board(board))
        elif self.level == ObservationLevel.GAME:
            # One dot per move, on a single line, so a game's progress is visible
            # without the full per-move narration. The line is terminated by the
            # leading newline of the game result in on_game_end.
            print(".", end="", flush=True)

    def on_game_end(self, result: GameResult) -> None:
        if self.level >= ObservationLevel.GAME:
            print(f"\n{describe_game_result(result, self._names)}")
        elif self._num_games > 1:
            # MATCH level: one dot per finished game, on a single line, so match
            # progress is visible without any per-game detail. The line is
            # terminated by the leading newline of the tally in on_match_end
            # (which runs on the same num_games > 1 condition).
            print(".", end="", flush=True)

    def on_match_end(self, result: MatchResult) -> None:
        if self._num_games > 1:
            print(f"\n{describe_match_result(result)}")
