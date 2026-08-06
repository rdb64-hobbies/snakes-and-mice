"""The game loop: drives two players through one game and reports the result.

The engine is the single source of truth. It applies moves to its own
:class:`Board`, evaluates the outcome from ground truth, and reports facts — it
never guesses intent or assigns blame beyond what it observes.
"""

from __future__ import annotations

from .board import Board
from .core import Move, MoveChoice, Side, TurnOutcome
from .faults import IllegalMove, MoveUnavailable, PlayerFaultReason
from .observer import Observer
from .players.base import Player
from .result import GameResult, PlayerFaultDetail, Termination


def play_game(
    mouse: Player, snake: Player, observer: Observer | None = None
) -> GameResult:
    """Play one game between ``mouse`` and ``snake``; return the result.

    ``mouse`` moves first. Every termination — win, cat's game, or fault — is
    reported to both players via ``end_game`` before this returns. An optional
    :class:`Observer` is driven in lockstep: the engine fires every hook it has,
    in order, and leaves it to the observer to decide (from its
    :class:`~snakes_and_mice.observer.ObservationLevel`) how much to act on.
    """
    board: Board = Board()
    players: dict[Side, Player] = {Side.MOUSE: mouse, Side.SNAKE: snake}
    player: Player
    for side, player in players.items():
        player.start_game(side)
    if observer is not None:
        names: dict[Side, str] = {side: players[side].name for side in Side}
        observer.on_game_start(names, board)

    to_move: Side = Side.MOUSE
    while True:
        player = players[to_move]
        if observer is not None:
            observer.on_move_start(to_move, board)

        try:
            choice: MoveChoice = player.choose_move()
        except MoveUnavailable as exc:
            return _finish(players, _fault(to_move, exc.reason), observer)

        try:
            actual_outcome: TurnOutcome = _apply_and_evaluate(
                board, to_move, choice.move
            )
        except IllegalMove as exc:
            return _finish(
                players,
                _fault(to_move, exc.reason, attempted_move=choice.move),
                observer,
            )

        if (
            choice.claimed_outcome is not None
            and choice.claimed_outcome is not actual_outcome
        ):
            return _finish(
                players,
                _fault(
                    to_move,
                    PlayerFaultReason.WRONG_OUTCOME_CLAIM,
                    attempted_move=choice.move,
                    claimed_outcome=choice.claimed_outcome,
                    actual_outcome=actual_outcome,
                ),
                observer,
            )

        for member in players.values():
            member.observe_move(to_move, choice.move)
        if observer is not None:
            observer.on_move_end(to_move, choice.move, board, actual_outcome)

        if actual_outcome is TurnOutcome.WIN:
            return _finish(
                players,
                GameResult(Termination.LINE_COMPLETED, winner=to_move),
                observer,
            )
        if actual_outcome is TurnOutcome.CATS_GAME:
            return _finish(
                players, GameResult(Termination.CATS_GAME), observer
            )

        to_move = to_move.other


def _apply_and_evaluate(board: Board, side: Side, move: Move) -> TurnOutcome:
    """Apply ``move`` for ``side`` and report the resulting outcome.

    Emptiness of every target cell is checked *before* any placement, so a
    ``CELL_NOT_EMPTY`` fault never leaves the board partially mutated. Pieces are
    then placed in order and the outcome re-evaluated after each: the moment a
    placement wins or completes a cat's game, the turn ends and any remaining
    piece is not placed (see the rules).

    A single-piece move is only legal if that piece ends the game. If every piece
    is placed and the game is still in play, a one-piece move is a
    ``WRONG_PIECE_COUNT`` fault (the player owed a second piece).
    """
    for cell in move.cells:
        if not board.is_empty(cell):
            raise IllegalMove(
                PlayerFaultReason.CELL_NOT_EMPTY, f"cell not empty: {cell}"
            )

    for cell in move.cells:
        board.place(cell, side)
        if board.winner() is side:
            return TurnOutcome.WIN
        if board.is_cats_game():
            return TurnOutcome.CATS_GAME

    if len(move.cells) < 2:
        raise IllegalMove(
            PlayerFaultReason.WRONG_PIECE_COUNT,
            "single-piece move did not end the game",
        )
    return TurnOutcome.IN_PLAY


def _fault(
    offender: Side,
    reason: PlayerFaultReason,
    *,
    attempted_move: Move | None = None,
    claimed_outcome: TurnOutcome | None = None,
    actual_outcome: TurnOutcome | None = None,
) -> GameResult:
    """Build a ``PLAYER_FAULT`` result."""
    return GameResult(
        Termination.PLAYER_FAULT,
        winner=None,
        fault=PlayerFaultDetail(
            offender=offender,
            reason=reason,
            attempted_move=attempted_move,
            claimed_outcome=claimed_outcome,
            actual_outcome=actual_outcome,
        ),
    )


def _finish(
    players: dict[Side, Player],
    result: GameResult,
    observer: Observer | None = None,
) -> GameResult:
    """Notify both players (and any observer) that the game ended, then return
    the result."""
    for player in players.values():
        player.end_game(result)
    if observer is not None:
        observer.on_game_end(result)
    return result
