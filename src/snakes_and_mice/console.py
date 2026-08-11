"""Text console presentation: board rendering, result summaries, and a
stdout observer that narrates a match as it plays.

This is the reusable presentation layer, kept separate from any single command-
line entry point so more than one frontend can share it. It knows nothing about
argument parsing, rosters, or process wiring — those belong to :mod:`cli` (and
to any other CLI that renders through this module).
"""

from __future__ import annotations

from collections import Counter

from .board import Board
from .core import BOARD_SIZE, Cell, Move, Side, TurnOutcome
from .faults import PlayerFaultReason
from .observer import ObservationLevel, Observer
from .result import GameResult, MatchResult, PlayerFaultDetail, Termination

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
    parts: list[str] = [
        f"{reason.value} ×{count}"
        for reason, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0].value)
        )
    ]
    return ", ".join(parts)


class ConsoleObserver(Observer):
    """Renders a match to stdout, showing as much as its level asks for.

    The engine fires every hook; this observer gates its own output against the
    :class:`~snakes_and_mice.observer.ObservationLevel` it was built with —
    ``move`` shows every turn, ``game`` just each game's board and result, and
    ``match`` only the opening banner and the closing tally. Per-game
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
        if self.level < ObservationLevel.MOVE:
            return
        print(f"  plays {move}:\n")
        print(render_board(board))

    def on_game_end(self, result: GameResult) -> None:
        if self.level < ObservationLevel.GAME:
            return
        print(f"\n{describe_game_result(result, self._names)}")

    def on_match_end(self, result: MatchResult) -> None:
        if self._num_games > 1:
            print(f"\n{describe_match_result(result)}")
