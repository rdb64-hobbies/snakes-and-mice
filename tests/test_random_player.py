"""Tests for the RandomPlayer: legality, determinism, and endgame behavior."""

from __future__ import annotations

import random

import pytest

from snakes_and_mice import (
    Board,
    Cell,
    Move,
    Observer,
    RandomPlayer,
    Side,
    Termination,
    TurnOutcome,
    play_game,
)


class _MoveRecorder(Observer):
    """Records the (side, move) of every accepted move."""

    def __init__(self) -> None:
        self.moves: list[tuple[Side, Move]] = []

    def on_move_end(
        self, side: Side, move: Move, board: Board, outcome: TurnOutcome
    ) -> None:
        self.moves.append((side, move))


@pytest.mark.parametrize("seed", range(25))
def test_random_game_always_ends_legally(seed: int) -> None:
    # Two random players never fault: every move they make is legal, so a game
    # can only end in a completed line or a cat's game.
    mouse = RandomPlayer("Mouse", random.Random(seed))
    snake = RandomPlayer("Snake", random.Random(seed + 1000))
    result = play_game(mouse, snake)
    assert result.termination in {
        Termination.LINE_COMPLETED,
        Termination.CATS_GAME,
    }
    assert result.fault is None


def test_same_seed_yields_identical_game() -> None:
    def run() -> list[tuple[Side, Move]]:
        recorder = _MoveRecorder()
        play_game(
            RandomPlayer("Mouse", random.Random(7)),
            RandomPlayer("Snake", random.Random(8)),
            recorder,
        )
        return recorder.moves

    assert run() == run()


def test_chosen_cells_are_always_empty_and_legal() -> None:
    # Drive one player directly: every move it returns places 1–2 distinct,
    # currently-empty cells on the shared board it is tracking.
    player = RandomPlayer("Mouse", random.Random(3))
    player.start_game(Side.MOUSE)
    board = Board()
    for _ in range(10):
        empties_before = set(board.empty_cells())
        move = player.choose_move().move
        assert 1 <= len(move.cells) <= 2
        assert len(set(move.cells)) == len(move.cells)  # distinct
        for cell in move.cells:
            assert cell in empties_before
            board.place(cell, Side.MOUSE)
        # Keep the player's view in step with the board we are building.
        player.observe_move(Side.MOUSE, move)


def test_single_piece_move_when_one_cell_remains() -> None:
    # Fill every cell but one, then the random player must (and does) play a
    # single-piece move on the last empty cell.
    player = RandomPlayer("Mouse", random.Random(0))
    player.start_game(Side.MOUSE)
    leftover = Cell.from_label("E5")
    for cell in Board().empty_cells():
        if cell != leftover:
            player.observe_move(Side.MOUSE, Move.of(cell))
    choice = player.choose_move()
    assert choice.move.cells == (leftover,)
    assert choice.claimed_outcome is None  # random player never claims


def test_no_empty_cells_raises() -> None:
    # A defensive invariant: the engine never asks a player to move on a full
    # board, but if it did, the random player refuses loudly rather than
    # returning an illegal empty move.
    player = RandomPlayer("Mouse", random.Random(0))
    player.start_game(Side.MOUSE)
    for cell in Board().empty_cells():
        player.observe_move(Side.MOUSE, Move.of(cell))
    with pytest.raises(RuntimeError):
        player.choose_move()
