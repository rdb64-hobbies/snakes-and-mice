"""Tests for the game loop and turn evaluation."""

from __future__ import annotations

from snakes_and_mice import (
    Board,
    Cell,
    GameResult,
    Move,
    MoveChoice,
    MoveUnavailable,
    Player,
    PlayerFaultReason,
    PlayerUnavailable,
    ScriptedPlayer,
    Side,
    Termination,
    TurnOutcome,
    play_game,
)
from snakes_and_mice.game import _apply_and_evaluate


class RecordingPlayer(ScriptedPlayer):
    """A scripted player that records lifecycle callbacks for assertions."""

    def __init__(self, choices: list[MoveChoice], name: str) -> None:
        super().__init__(choices, name)
        self.started_side: Side | None = None
        self.observed: list[tuple[Side, Move]] = []
        self.result: GameResult | None = None

    def start_game(self, side: Side, seed: Cell) -> None:
        super().start_game(side, seed)
        self.started_side = side

    def observe_move(self, side: Side, move: Move) -> None:
        self.observed.append((side, move))

    def end_game(self, result: GameResult) -> None:
        self.result = result


class SilentPlayer(Player):
    """A player that cannot produce a move."""

    def start_game(self, side: Side, seed: Cell) -> None:
        return None

    def observe_move(self, side: Side, move: Move) -> None:
        return None

    def choose_move(self) -> MoveChoice:
        raise MoveUnavailable(PlayerFaultReason.UNPARSEABLE_OUTPUT, "no output")


class UnreachablePlayer(Player):
    """A player whose backend is unreachable — it can never take a turn, but
    through no fault of its own (see :class:`PlayerUnavailable`)."""

    def start_game(self, side: Side, seed: Cell) -> None:
        return None

    def observe_move(self, side: Side, move: Move) -> None:
        return None

    def choose_move(self) -> MoveChoice:
        raise PlayerUnavailable("model unreachable after retries")


def _mouse_wins_row_a() -> tuple[ScriptedPlayer, ScriptedPlayer]:
    """Mouse completes row A; snake plays harmlessly along row E."""
    mouse = ScriptedPlayer(
        [
            MoveChoice(Move.from_labels("A1", "A2")),
            MoveChoice(Move.from_labels("A3", "A4")),
            MoveChoice(Move.from_labels("B1", "A5"), TurnOutcome.WIN),
        ],
        name="Mouse",
    )
    snake = ScriptedPlayer.from_moves(
        [Move.from_labels("E1", "E2"), Move.from_labels("E3", "E4")],
        name="Snake",
    )
    return mouse, snake


def test_mouse_wins_by_completing_a_line() -> None:
    mouse, snake = _mouse_wins_row_a()
    result = play_game(mouse, snake)
    assert result.termination is Termination.LINE_COMPLETED
    assert result.winner is Side.MOUSE
    assert result.fault is None


def test_correct_win_claim_does_not_fault() -> None:
    # The winning choice claims WIN; ground truth agrees, so no fault.
    mouse, snake = _mouse_wins_row_a()
    result = play_game(mouse, snake)
    assert result.termination is Termination.LINE_COMPLETED


def test_lifecycle_callbacks_are_delivered() -> None:
    mouse = RecordingPlayer(
        [
            MoveChoice(Move.from_labels("A1", "A2")),
            MoveChoice(Move.from_labels("A3", "A4")),
            MoveChoice(Move.from_labels("B1", "A5"), TurnOutcome.WIN),
        ],
        name="Mouse",
    )
    snake = RecordingPlayer(
        [MoveChoice(Move.from_labels("E1", "E2")), MoveChoice(Move.from_labels("E3", "E4"))],
        name="Snake",
    )
    result = play_game(mouse, snake)

    assert mouse.started_side is Side.MOUSE
    assert snake.started_side is Side.SNAKE
    # Both players observe every accepted move, including their own.
    assert mouse.observed == snake.observed
    assert mouse.observed[0] == (Side.MOUSE, Move.from_labels("A1", "A2"))
    assert mouse.observed[-1] == (Side.MOUSE, Move.from_labels("B1", "A5"))
    # Both are told the final result.
    assert mouse.result is result
    assert snake.result is result


def test_win_on_first_piece_skips_second_placement() -> None:
    board = Board()
    for col in range(4):
        board.place(Cell(0, col), Side.MOUSE)  # A1..A4
    outcome = _apply_and_evaluate(board, Side.MOUSE, Move.from_labels("A5", "B1"))
    assert outcome is TurnOutcome.WIN
    assert board.occupant(Cell.from_label("A5")) is Side.MOUSE
    assert board.is_empty(Cell.from_label("B1"))  # second piece not placed


def test_win_on_second_piece_places_both() -> None:
    board = Board()
    for col in range(4):
        board.place(Cell(0, col), Side.MOUSE)  # A1..A4
    outcome = _apply_and_evaluate(board, Side.MOUSE, Move.from_labels("B1", "A5"))
    assert outcome is TurnOutcome.WIN
    assert board.occupant(Cell.from_label("B1")) is Side.MOUSE
    assert board.occupant(Cell.from_label("A5")) is Side.MOUSE


def test_single_piece_move_that_wins_is_legal() -> None:
    board = Board()
    for col in range(4):
        board.place(Cell(0, col), Side.MOUSE)  # A1..A4
    outcome = _apply_and_evaluate(board, Side.MOUSE, Move.from_labels("A5"))
    assert outcome is TurnOutcome.WIN
    assert board.occupant(Cell.from_label("A5")) is Side.MOUSE


def test_single_piece_move_that_completes_cats_game_is_legal() -> None:
    # The whole board is filled except E5. Every line is already dead except the
    # main diagonal (A1,B2,C3,D4 are all snake, E5 empty). Placing a mouse at E5
    # makes that last line mixed too — a cat's game completed by a single piece,
    # and not a win (the diagonal becomes 4 snake + 1 mouse).
    grid = ["SMSMM", "MSSMS", "MMSSM", "SMMSM", "MSMS."]
    board = Board()
    for row, glyphs in enumerate(grid):
        for col, glyph in enumerate(glyphs):
            cell = Cell(row, col)
            if glyph != "." and board.is_empty(cell):
                board.place(cell, Side.SNAKE if glyph == "S" else Side.MOUSE)
    assert not board.is_cats_game()  # not yet — the main diagonal is still alive
    assert board.winner() is None
    outcome = _apply_and_evaluate(board, Side.MOUSE, Move.from_labels("E5"))
    assert outcome is TurnOutcome.CATS_GAME


def test_single_piece_move_still_in_play_is_a_fault() -> None:
    mouse = ScriptedPlayer([MoveChoice(Move.from_labels("A1"))], name="Mouse")
    snake = ScriptedPlayer([], name="Snake")
    result = play_game(mouse, snake)
    assert result.termination is Termination.PLAYER_FAULT
    assert result.fault is not None
    assert result.fault.offender is Side.MOUSE
    assert result.fault.reason is PlayerFaultReason.WRONG_PIECE_COUNT
    assert result.fault.attempted_move == Move.from_labels("A1")


def test_cell_not_empty_is_a_fault() -> None:
    # Mouse's first move targets B3, which the snake already occupies.
    mouse = ScriptedPlayer([MoveChoice(Move.from_labels("B3", "A1"))], name="Mouse")
    snake = ScriptedPlayer([], name="Snake")
    result = play_game(mouse, snake)
    assert result.termination is Termination.PLAYER_FAULT
    assert result.winner is None
    assert result.fault is not None
    assert result.fault.offender is Side.MOUSE
    assert result.fault.reason is PlayerFaultReason.CELL_NOT_EMPTY
    assert result.fault.attempted_move == Move.from_labels("B3", "A1")


def test_wrong_outcome_claim_is_a_fault() -> None:
    # A legal, non-winning move claimed as a WIN.
    mouse = ScriptedPlayer(
        [MoveChoice(Move.from_labels("A1", "A2"), TurnOutcome.WIN)], name="Mouse"
    )
    snake = ScriptedPlayer([], name="Snake")
    result = play_game(mouse, snake)
    assert result.termination is Termination.PLAYER_FAULT
    assert result.fault is not None
    assert result.fault.reason is PlayerFaultReason.WRONG_OUTCOME_CLAIM
    assert result.fault.claimed_outcome is TurnOutcome.WIN
    assert result.fault.actual_outcome is TurnOutcome.IN_PLAY


def test_move_unavailable_is_a_fault() -> None:
    mouse = SilentPlayer(name="Mouse")
    snake = ScriptedPlayer([], name="Snake")
    result = play_game(mouse, snake)
    assert result.termination is Termination.PLAYER_FAULT
    assert result.fault is not None
    assert result.fault.offender is Side.MOUSE
    assert result.fault.reason is PlayerFaultReason.UNPARSEABLE_OUTPUT
    assert result.fault.attempted_move is None


def test_unreachable_player_ends_game_as_no_contest() -> None:
    # A PlayerUnavailable is not a fault: the game ends ABORTED (no winner, no
    # fault), it carries the cause, and both players are still notified.
    mouse = UnreachablePlayer(name="Mouse")
    snake = RecordingPlayer([], name="Snake")
    result = play_game(mouse, snake)
    assert result.termination is Termination.ABORTED
    assert result.winner is None
    assert result.fault is None
    assert result.error == "model unreachable after retries"
    # end_game still fires for both sides, so a learning player sees the outcome.
    assert snake.result is result


def test_game_ends_in_cats_game() -> None:
    # A full game engineered so every line ends mixed: no winner, a draw.
    mouse = ScriptedPlayer.from_moves(
        [
            Move.from_labels("A2", "A4"),
            Move.from_labels("B2", "B4"),
            Move.from_labels("C1", "C3"),
            Move.from_labels("C5", "D2"),
            Move.from_labels("D4", "E1"),
            Move.from_labels("E3", "E5"),
        ],
        name="Mouse",
    )
    snake = ScriptedPlayer.from_moves(
        [
            Move.from_labels("A1", "A3"),
            Move.from_labels("A5", "B1"),
            Move.from_labels("B5", "C2"),
            Move.from_labels("C4", "D1"),
            Move.from_labels("D3", "D5"),
            Move.from_labels("E2", "E4"),
        ],
        name="Snake",
    )
    result = play_game(mouse, snake)
    assert result.termination is Termination.CATS_GAME
    assert result.winner is None
    assert result.fault is None
