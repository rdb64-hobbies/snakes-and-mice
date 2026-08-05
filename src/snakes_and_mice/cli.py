"""Text CLI: board rendering, a result summary, and a watchable demo game."""

from __future__ import annotations

from .board import Board
from .core import BOARD_SIZE, Cell, Move, Side, TurnOutcome
from .game import GameObserver, play_game
from .players import RandomPlayer
from .result import GameResult, PlayerFaultDetail, Termination

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
    """Render the board as text, rank ``A`` at the top and file ``1`` at left."""
    margin: str = " " * _CELL_WIDTH  # sits above the "<rank> " row-label column
    header: str = margin + " ".join(
        str(c + 1).ljust(_CELL_WIDTH) for c in range(BOARD_SIZE)
    )
    rows: list[str] = [header]
    for r in range(BOARD_SIZE):
        rank: str = chr(ord("A") + r)
        cells: list[str] = [
            _cell_token(board, Cell(r, c)) for c in range(BOARD_SIZE)
        ]
        rows.append(f"{rank} " + " ".join(cells))
    return "\n".join(rows)


def describe_result(result: GameResult, players: dict[Side, str]) -> str:
    """A one-line human-readable summary of a game result."""
    if result.termination is Termination.LINE_COMPLETED:
        assert result.winner is not None
        return f"{players[result.winner]} ({result.winner.value}) wins."
    if result.termination is Termination.CATS_GAME:
        return "Cat's game — a draw."
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


class ConsoleObserver(GameObserver):
    """Renders a game to stdout as it is played.

    When ``pause`` is set, it waits for the user to press Enter between turns so
    the game can be watched move by move; a non-interactive stdin (piped input,
    CI) simply plays straight through.
    """

    def __init__(self, pause: bool = True) -> None:
        self._pause: bool = pause
        self._names: dict[Side, str] = {}
        self._turn: int = 0

    def _wait(self, prompt: str) -> None:
        if not self._pause:
            return
        try:
            input(prompt)
        except EOFError:  # non-interactive stdin — stop pausing, play on
            self._pause = False

    def on_game_start(self, names: dict[Side, str], board: Board) -> None:
        self._names = names
        print("Snakes and Mice — a random game\n")
        print(f"🐭 Mouse: {names[Side.MOUSE]}    🐍 Snake: {names[Side.SNAKE]}\n")
        print(render_board(board))
        self._wait("\nPress Enter to start… ")

    def on_move_start(self, side: Side, board: Board) -> None:
        self._turn += 1
        # Printed before the move is produced, so a slow player (e.g. an LLM)
        # visibly "thinks" here before on_move_end reports what it played.
        print(f"\nTurn {self._turn} — {self._names[side]} ({side.value}) to move…")

    def on_move_end(
        self, side: Side, move: Move, board: Board, outcome: TurnOutcome
    ) -> None:
        print(f"  plays {move}:\n")
        print(render_board(board))
        if outcome is TurnOutcome.IN_PLAY:
            self._wait("\nPress Enter for the next turn… ")

    def on_game_end(self, result: GameResult) -> None:
        print(f"\n{describe_result(result, self._names)}")


def main() -> None:
    """Play a random game between two bots and render it turn by turn."""
    mouse: RandomPlayer = RandomPlayer(name="Randy")
    snake: RandomPlayer = RandomPlayer(name="Ransom")
    play_game(mouse, snake, ConsoleObserver(pause=True))
