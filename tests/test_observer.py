"""Tests for the GameObserver hook: the engine drives it in lockstep."""

from __future__ import annotations

from snakes_and_mice import (
    Board,
    Cell,
    GameObserver,
    GameResult,
    Move,
    MoveChoice,
    ScriptedPlayer,
    Side,
    Termination,
    TurnOutcome,
    play_game,
)


class _Recorder(GameObserver):
    """Records every hook call, snapshotting board facts at call time."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.names: dict[Side, str] = {}
        self.start_piece_count: int = -1
        self.move_starts: list[Side] = []
        self.moves: list[tuple[Side, Move, TurnOutcome]] = []
        self.placed_ok: list[bool] = []
        self.result: GameResult | None = None

    def on_game_start(self, names: dict[Side, str], board: Board) -> None:
        self.events.append("start")
        self.names = dict(names)
        self.start_piece_count = len(board.empty_cells())

    def on_move_start(self, side: Side, board: Board) -> None:
        self.events.append("move_start")
        # Fires before the move is applied: the mover's cells are still empty.
        self.move_starts.append(side)

    def on_move_end(
        self, side: Side, move: Move, board: Board, outcome: TurnOutcome
    ) -> None:
        self.events.append("move_end")
        self.moves.append((side, move, outcome))
        # The live board must already reflect the pieces just placed.
        self.placed_ok.append(
            all(board.occupant(cell) is side for cell in move.cells)
        )

    def on_game_end(self, result: GameResult) -> None:
        self.events.append("end")
        self.result = result


def _mouse_wins_row_a() -> tuple[ScriptedPlayer, ScriptedPlayer]:
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


def test_observer_receives_ordered_lifecycle() -> None:
    recorder = _Recorder()
    mouse, snake = _mouse_wins_row_a()
    play_game(mouse, snake, recorder)

    # start, then a move_start/move_end pair per accepted move, then end.
    assert recorder.events[0] == "start"
    assert recorder.events[-1] == "end"
    assert recorder.events[1:-1] == ["move_start", "move_end"] * 5  # 3 mouse + 2 snake


def test_observer_move_start_fires_before_fault_without_move_end() -> None:
    # The mouse's first move targets the occupied B3 — an apply-time fault. The
    # turn still opens with on_move_start, but on_move_end never fires because no
    # move was accepted. This is the timing the split exists for.
    recorder = _Recorder()
    mouse = ScriptedPlayer([MoveChoice(Move.from_labels("B3", "A1"))], name="Mouse")
    snake = ScriptedPlayer([], name="Snake")
    result = play_game(mouse, snake, recorder)

    assert result.termination is Termination.PLAYER_FAULT
    assert recorder.events == ["start", "move_start", "end"]
    assert recorder.move_starts == [Side.MOUSE]


def test_observer_start_sees_seeded_board_and_names() -> None:
    recorder = _Recorder()
    mouse, snake = _mouse_wins_row_a()
    play_game(mouse, snake, recorder)

    assert recorder.names == {Side.MOUSE: "Mouse", Side.SNAKE: "Snake"}
    # Only the snake seed at B3 is on the board at the start (24 empties).
    assert recorder.start_piece_count == 24


def test_observer_board_reflects_each_move() -> None:
    recorder = _Recorder()
    mouse, snake = _mouse_wins_row_a()
    play_game(mouse, snake, recorder)

    assert all(recorder.placed_ok)
    # First move was the mouse's A1/A2; last was the winning move.
    assert recorder.moves[0][0] is Side.MOUSE
    assert recorder.moves[0][1] == Move.from_labels("A1", "A2")
    assert recorder.moves[-1][2] is TurnOutcome.WIN


def test_observer_end_carries_result() -> None:
    recorder = _Recorder()
    mouse, snake = _mouse_wins_row_a()
    result = play_game(mouse, snake, recorder)

    assert recorder.result is result
    assert recorder.result.termination is Termination.LINE_COMPLETED
    assert recorder.result.winner is Side.MOUSE


def test_observer_is_optional() -> None:
    # Omitting the observer must not change the game outcome.
    mouse, snake = _mouse_wins_row_a()
    result = play_game(mouse, snake)
    assert result.termination is Termination.LINE_COMPLETED
