"""Text CLI: board rendering, a result summary, and a demo entry point."""

from __future__ import annotations

from .board import Board
from .core import BOARD_SIZE, Cell, Decision, Move, Side, TurnOutcome
from .players import ScriptedPlayer
from .engine import play_game
from .result import GameResult, Termination

GLYPH: dict[Side, str] = {Side.MOUSE: "🐭", Side.SNAKE: "🐍"}
EMPTY = "·"


def render_board(board: Board) -> str:
    """Render the board as text, rank ``A`` at the top and file ``1`` at left."""
    header = "   " + " ".join(str(c + 1) for c in range(BOARD_SIZE))
    rows = [header]
    for r in range(BOARD_SIZE):
        rank = chr(ord("A") + r)
        cells = [
            GLYPH[occ] if (occ := board.occupant(Cell(r, c))) is not None else EMPTY
            for c in range(BOARD_SIZE)
        ]
        rows.append(f"{rank}  " + " ".join(cells))
    return "\n".join(rows)


def describe_result(result: GameResult, players: dict[Side, str]) -> str:
    """A one-line human-readable summary of a game result."""
    if result.termination is Termination.LINE_COMPLETED:
        assert result.winner is not None
        return f"{players[result.winner]} ({result.winner.value}) wins."
    if result.termination is Termination.CATS_GAME:
        return "Cat's game — a draw."
    fault = result.fault
    assert fault is not None
    detail = f"{players[fault.offender]} ({fault.offender.value}) faulted: {fault.reason.value}"
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

    def __init__(self, decisions: list[Decision], name: str) -> None:
        super().__init__(decisions, name)
        self.board = Board()

    def observe_move(self, side: Side, move: Move) -> None:
        for cell in move.cells:
            if self.board.is_empty(cell):
                self.board.place(cell, side)


def main() -> None:
    """Play a scripted demo game and print the final board and result."""
    mouse = _DisplayPlayer(
        [
            Decision(Move.from_labels("A1", "A2")),
            Decision(Move.from_labels("A3", "A4")),
            Decision(Move.from_labels("B1", "A5"), TurnOutcome.WIN),
        ],
        name="Mouse",
    )
    snake = _DisplayPlayer(
        [
            Decision(Move.from_labels("E1", "E2")),
            Decision(Move.from_labels("E3", "E4")),
        ],
        name="Snake",
    )
    names = {Side.MOUSE: mouse.name, Side.SNAKE: snake.name}

    result = play_game(mouse, snake)

    print("Snakes and Mice — demo game\n")
    print(render_board(mouse.board))
    print()
    print(describe_result(result, names))
