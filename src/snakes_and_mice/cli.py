"""Text CLI: board rendering, a result summary, and a demo entry point."""

from __future__ import annotations

from .board import Board
from .core import BOARD_SIZE, Cell, Move, MoveChoice, Side, TurnOutcome
from .players import ScriptedPlayer
from .game import play_game
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


class _DisplayPlayer(ScriptedPlayer):
    """A scripted player that also records the board as it observes moves.

    Used by the demo so the CLI can show the final position without reaching
    into the engine's private state.
    """

    def __init__(self, choices: list[MoveChoice], name: str) -> None:
        super().__init__(choices, name)
        self.board: Board = Board()

    def observe_move(self, side: Side, move: Move) -> None:
        for cell in move.cells:
            if self.board.is_empty(cell):
                self.board.place(cell, side)


def main() -> None:
    """Play a scripted demo game and print the final board and result."""
    mouse: _DisplayPlayer = _DisplayPlayer(
        [
            MoveChoice(Move.from_labels("A1", "A2")),
            MoveChoice(Move.from_labels("A3", "A4")),
            MoveChoice(Move.from_labels("B1", "A5"), TurnOutcome.WIN),
        ],
        name="Mouse",
    )
    snake: _DisplayPlayer = _DisplayPlayer(
        [
            MoveChoice(Move.from_labels("E1", "E2")),
            MoveChoice(Move.from_labels("E3", "E4")),
        ],
        name="Snake",
    )
    names: dict[Side, str] = {Side.MOUSE: mouse.name, Side.SNAKE: snake.name}

    result: GameResult = play_game(mouse, snake)

    print("Snakes and Mice — demo game\n")
    print(render_board(mouse.board))
    print()
    print(describe_result(result, names))
