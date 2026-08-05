"""The game loop: drives two players through one game and reports the result.

The engine is the single source of truth. It applies moves to its own
:class:`Board`, evaluates the outcome from ground truth, and reports facts — it
never guesses intent or assigns blame beyond what it observes.
"""

from __future__ import annotations

from .board import Board
from .core import Move, MoveChoice, Side, TurnOutcome
from .faults import IllegalMove, MoveUnavailable, PlayerFaultReason
from .players.base import Player
from .result import GameResult, PlayerFaultDetail, Termination


class GameObserver:
    """A spectator the engine drives alongside the players, so a game can be
    watched or logged turn by turn.

    Unlike a :class:`~snakes_and_mice.players.Player`, an observer never
    influences play: it only receives the authoritative board after each event.
    Every hook defaults to a no-op, so a subclass overrides only what it needs.
    The ``Board`` handed to a hook is the engine's live board — treat it as
    read-only.
    """

    def on_game_start(self, names: dict[Side, str], board: Board) -> None:
        """The game is about to begin, with the given side→name mapping and the
        seeded starting board."""
        return None

    def on_move_start(self, side: Side, board: Board) -> None:
        """``side`` is about to be asked for its move; ``board`` is the current
        pre-move state.

        Fires at the start of every turn, before ``choose_move``. This matters
        when producing a move is slow — e.g. an LLM player querying a model — so
        a watcher can show that the turn has begun and who is thinking, well
        before :meth:`on_move_end` reports what they played.
        """
        return None

    def on_move_end(
        self, side: Side, move: Move, board: Board, outcome: TurnOutcome
    ) -> None:
        """``side`` just played ``move``, yielding ``outcome``; ``board`` now
        reflects it. Fires once per accepted move (including the terminal one),
        and not at all for a turn that ends in a fault."""
        return None

    def on_game_end(self, result: GameResult) -> None:
        """The game has ended with ``result``."""
        return None


def play_game(
    mouse: Player, snake: Player, observer: GameObserver | None = None
) -> GameResult:
    """Play one game between ``mouse`` and ``snake``; return the result.

    ``mouse`` moves first. Every termination — win, cat's game, or fault — is
    reported to both players via ``end_game`` before this returns. An optional
    :class:`GameObserver` is driven in lockstep so a caller (e.g. the CLI) can
    watch the game unfold turn by turn.
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
    observer: GameObserver | None = None,
) -> GameResult:
    """Notify both players (and any observer) that the game ended, then return
    the result."""
    for player in players.values():
        player.end_game(result)
    if observer is not None:
        observer.on_game_end(result)
    return result
