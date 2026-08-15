"""Tests for the HumanPlayer: parsing, re-prompting, and endgame handling."""

from __future__ import annotations

import pytest

from snakes_and_mice import (
    Cell,
    HumanPlayer,
    Move,
    MoveUnavailable,
    PlayerFaultReason,
    ScriptedPlayer,
    Side,
    Termination,
    play_game,
)


class _FakeConsole:
    """Feeds a queued list of input lines and captures everything written."""

    def __init__(self, lines: list[str]) -> None:
        self._lines: list[str] = list(lines)
        self.out: list[str] = []

    def read_line(self, prompt: str) -> str:
        if not self._lines:
            raise EOFError
        return self._lines.pop(0)

    def write(self, line: str) -> None:
        self.out.append(line)

    def _player(self, side: Side = Side.MOUSE) -> HumanPlayer:
        player = HumanPlayer("You", read_line=self.read_line, write=self.write)
        player.start_game(side, Cell.from_label("B3"))
        return player

    @property
    def errors(self) -> str:
        return "\n".join(self.out)


def test_reads_a_two_cell_move() -> None:
    console = _FakeConsole(["C3 D4"])
    player = console._player()
    assert player.choose_move().move == Move.from_labels("C3", "D4")


def test_accepts_comma_separated_and_lowercase() -> None:
    console = _FakeConsole(["c3, d4"])
    player = console._player()
    assert player.choose_move().move == Move.from_labels("C3", "D4")


def test_makes_no_outcome_claim() -> None:
    console = _FakeConsole(["A1 A2"])
    player = console._player()
    assert player.choose_move().claimed_outcome is None


def test_reprompts_on_unparseable_label() -> None:
    console = _FakeConsole(["hello", "C3 D4"])
    player = console._player()
    assert player.choose_move().move == Move.from_labels("C3", "D4")
    assert len(console.out) == 1  # one rejection before the good move


def test_reprompts_on_wrong_count() -> None:
    console = _FakeConsole(["A1 A2 A3", "A1 A2"])
    player = console._player()
    assert player.choose_move().move == Move.from_labels("A1", "A2")
    assert "one or two cells" in console.errors


def test_reprompts_on_off_board() -> None:
    console = _FakeConsole(["A9 A1", "A1 A2"])
    player = console._player()
    assert player.choose_move().move == Move.from_labels("A1", "A2")
    assert "off the board" in console.errors


def test_reprompts_on_duplicate_cells() -> None:
    console = _FakeConsole(["A1 A1", "A1 A2"])
    player = console._player()
    assert player.choose_move().move == Move.from_labels("A1", "A2")
    assert "must be different" in console.errors


def test_reprompts_on_occupied_cell() -> None:
    # B3 is the snake's seed, so it is occupied from the start.
    console = _FakeConsole(["B3 A1", "A1 A2"])
    player = console._player()
    assert player.choose_move().move == Move.from_labels("A1", "A2")
    assert "occupied" in console.errors


def test_reprompts_on_single_piece_that_does_not_end_game() -> None:
    console = _FakeConsole(["A1", "A1 A2"])
    player = console._player()
    assert player.choose_move().move == Move.from_labels("A1", "A2")
    assert "single piece" in console.errors


def test_single_piece_accepted_when_it_wins() -> None:
    # Mouse already holds A1..A4; the lone A5 completes row A, so it is legal.
    console = _FakeConsole(["A5"])
    player = console._player()
    player.observe_move(Side.MOUSE, Move.from_labels("A1", "A2"))
    player.observe_move(Side.MOUSE, Move.from_labels("A3", "A4"))
    choice = player.choose_move()
    assert choice.move == Move.from_labels("A5")


def test_end_of_input_concedes_the_turn() -> None:
    console = _FakeConsole([])  # no input → EOF on first read
    player = console._player()
    with pytest.raises(MoveUnavailable) as info:
        player.choose_move()
    assert info.value.reason is PlayerFaultReason.UNPARSEABLE_OUTPUT


def test_human_player_wins_a_full_game() -> None:
    # Drive a whole game through the engine with a scripted human's typed moves.
    console = _FakeConsole(["A1 A2", "A3 A4", "B1 A5"])
    mouse = HumanPlayer("You", read_line=console.read_line, write=console.write)
    snake = ScriptedPlayer.from_moves(
        [Move.from_labels("E1", "E2"), Move.from_labels("E3", "E4")],
        name="Snake",
    )
    result = play_game(mouse, snake)
    assert result.termination is Termination.LINE_COMPLETED
    assert result.winner is Side.MOUSE


def test_human_concession_is_a_fault_in_a_game() -> None:
    console = _FakeConsole([])  # the human gives no input on its first turn
    mouse = HumanPlayer("You", read_line=console.read_line, write=console.write)
    snake = ScriptedPlayer([], name="Snake")
    result = play_game(mouse, snake)
    assert result.termination is Termination.PLAYER_FAULT
    assert result.fault is not None
    assert result.fault.offender is Side.MOUSE
    assert result.fault.reason is PlayerFaultReason.UNPARSEABLE_OUTPUT
