"""Text CLI: board rendering, a result summary, and a playable/watchable game."""

from __future__ import annotations

import argparse

from .board import Board
from .core import BOARD_SIZE, Cell, Move, Side, TurnOutcome
from .game import GameObserver, play_game
from .players import HumanPlayer, Player, RandomPlayer
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
        print("Snakes and Mice\n")
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


_RANDOM_NAME: dict[Side, str] = {Side.MOUSE: "Randy", Side.SNAKE: "Ransom"}


def _make_player(kind: str, side: Side) -> Player:
    if kind == "human":
        return HumanPlayer(name="You")
    return RandomPlayer(name=_RANDOM_NAME[side])


def main(argv: list[str] | None = None) -> None:
    """Play a game and render it turn by turn.

    By default two random bots play (watch mode, pausing between turns). Pass
    ``--mouse human`` and/or ``--snake human`` to take a seat; with a human in
    the game there is no between-turns pause — the human's own input paces it.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="snakes-and-mice", description="Play or watch a game of Snakes and Mice."
    )
    parser.add_argument(
        "--mouse", choices=["random", "human"], default="random",
        help="who plays Mouse (default: random)",
    )
    parser.add_argument(
        "--snake", choices=["random", "human"], default="random",
        help="who plays Snake (default: random)",
    )
    args: argparse.Namespace = parser.parse_args(argv)

    mouse: Player = _make_player(args.mouse, Side.MOUSE)
    snake: Player = _make_player(args.snake, Side.SNAKE)
    has_human: bool = "human" in (args.mouse, args.snake)
    play_game(mouse, snake, ConsoleObserver(pause=not has_human))
