"""The LLM player: chooses moves by querying a large language model.

This is the player type the project exists to compare (§11). It wraps a single
Pydantic AI :class:`~pydantic_ai.Agent` bound to one model and keeps **one
message thread that spans every game it plays**, so the model accumulates
context across games — including how earlier games ended — and can learn from a
mistake with no change to the :class:`Player` interface.

The model is given as little help as possible: it sees only the opponent's moves
(via :meth:`observe_move`) and must track the board, find lines, and assess the
outcome of its own move itself. No network call is made except when a move is
actually needed — :meth:`start_game`, :meth:`observe_move`, and :meth:`end_game`
only *enqueue* messages, which are flushed as the next user turn on the following
:meth:`choose_move`. There are no retries within a game: an illegal or unusable
move ends the game, and the consequence is delivered as feedback in the *next*
game's opening (§11, "No retries").
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent, ModelMessagesTypeAdapter, UnexpectedModelBehavior
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UserError
from pydantic_ai.messages import ModelMessage

from ..core import Move, MoveChoice, Side, TurnOutcome
from ..faults import (
    IllegalMove,
    MoveUnavailable,
    PlayerFaultReason,
    SnakesAndMiceError,
)
from ..result import GameResult, PlayerFaultDetail, Termination
from .base import Player


class ModelRequestError(SnakesAndMiceError):
    """The call to a model's provider failed for a reason that is not a game
    fault — a misspelled or unavailable model, a rejected key, a rate limit, or a
    network problem. It is an environment/configuration error, so it aborts the
    run with a clear message rather than ending a game against the player."""


class LLMMove(BaseModel):
    """The structured object the model returns for each move request (§11).

    ``move_rationale`` is placed first so the model articulates a reason before
    committing to a move; it is log-only and never validated. ``claimed_outcome``
    is required: the model must assess, every turn, whether its move wins, draws,
    or leaves the game in play — a wrong claim is a ``WRONG_OUTCOME_CLAIM`` fault.
    """

    move_rationale: str  # a SHORT justification; logged, never validated
    cells: list[str]  # one or two cell labels, e.g. ["C3", "D4"]
    claimed_outcome: TurnOutcome  # required self-assessment: in_play | win | cats_game


# The rules preamble, enqueued once (lazily) ahead of the first move request. It
# explains the game and the response protocol; everything else the model learns
# it must infer from the opponent moves it is told about.
RULES_PREAMBLE: str = """\
You are playing Snakes and Mice, a two-player game on a 5x5 board.

The board. Rows are labeled A (top) to E (bottom); columns 1 (left) to 5 \
(right). A cell is a row letter then a column digit, e.g. C3. The valid cells \
are A1 through E5.

Sides. One player is the mouse, the other the snake. The mouse moves first. \
Before the first move the snake already occupies cell B3 (it is seeded there, \
which is a starting position, not the snake's move); every other cell starts \
empty.

A turn. On your turn you place TWO of your own pieces on two different empty \
cells — UNLESS a single piece already ends the game (completes a line for you, \
or leaves every line dead), in which case you may place just ONE. Pieces are \
placed in the order you list them and the position is checked after each, so a \
line completed by your first piece wins before the second is placed.

Winning. There are 12 lines: the 5 rows, the 5 columns, and the 2 main \
diagonals. You WIN the instant a line is fully occupied by your own pieces.

Cat's game. If every one of the 12 lines contains at least one piece from each \
side, no line can ever be completed: the game is a draw (a "cat's game").

Illegal moves lose the game immediately — there are no second chances within a \
game. A move is illegal if it: names a cell off the board; repeats a cell; \
places the wrong number of pieces (not one or two); plays on an already-occupied \
cell; plays a single piece that does not end the game; or misreports the outcome.

What you see. You are told only your opponent's moves, as they happen. You must \
track the full board yourself from the seeded snake at B3, your own moves, and \
your opponent's.

Your response. Each turn, return the structured fields: move_rationale (a short \
justification), cells (one or two labels like ["C3","D4"]), and claimed_outcome \
(exactly one of in_play, win, or cats_game — your honest assessment of the \
position after your move)."""


# How to explain each fault back to the model, so the next game's opening can
# tell it what it did wrong and how to avoid repeating it (§11, end_game).
_FAULT_ADVICE: dict[PlayerFaultReason, str] = {
    PlayerFaultReason.OFF_BOARD: (
        "you named a cell that is off the board — cells range from A1 to E5 "
        "(rows A–E, columns 1–5)."
    ),
    PlayerFaultReason.DUPLICATE_CELLS: (
        "you named the same cell twice — your two cells must be different."
    ),
    PlayerFaultReason.WRONG_PIECE_COUNT: (
        "you placed the wrong number of pieces — place exactly two cells, unless "
        "a single cell already wins or completes a cat's game."
    ),
    PlayerFaultReason.CELL_NOT_EMPTY: (
        "you played on a cell that was already occupied — only empty cells may be "
        "played, so track every piece already on the board."
    ),
    PlayerFaultReason.UNPARSEABLE_OUTPUT: (
        "your response could not be read as a move — return the required "
        "structured fields with one or two valid cell labels."
    ),
    PlayerFaultReason.WRONG_OUTCOME_CLAIM: (
        "you misjudged the outcome of your own move — assess win, cats_game, or "
        "in_play carefully before committing each turn."
    ),
}


class LLMPlayer(Player):
    """A player that queries one LLM over a single cross-game message thread."""

    def __init__(
        self,
        agent: Agent[None, LLMMove],
        name: str | None = None,
        *,
        log_dir: Path | None = None,
    ) -> None:
        super().__init__(name)
        # The agent is built and configured by the config layer, which alone knows
        # the provider and so can pick the right output mode and settings for it
        # (e.g. NativeOutput for Anthropic). The player stays provider-agnostic: it
        # only drives the message thread and reads back an LLMMove.
        self._agent: Agent[None, LLMMove] = agent
        self._log_dir: Path | None = log_dir
        # The running conversation (spans every game) and the messages queued for
        # the next user turn, flushed on the following choose_move. The one-time
        # rules preamble seeds the queue, so it leads the very first user turn.
        self._history: list[ModelMessage] = []
        self._pending: list[str] = [RULES_PREAMBLE]
        self._side: Side | None = None

    def start_game(self, side: Side) -> None:
        # Any feedback from a game we just faulted was enqueued by end_game and is
        # already ahead of this message in the pending queue (§11).
        self._side = side
        self._pending.append(f"A new game begins. You are playing {side.value}.")

    def observe_move(self, side: Side, move: Move) -> None:
        # Our own move is already in the thread as the model's structured response;
        # only the opponent's moves need to be relayed.
        if side == self._side:
            return
        self._pending.append(f"Your opponent ({side.value}) played {move}.")

    def choose_move(self) -> MoveChoice:
        assert self._side is not None, "choose_move called before start_game"
        self._pending.append(
            f"It is your turn ({self._side.value}). Choose your move."
        )
        user_prompt: str = "\n\n".join(self._pending)
        self._pending = []

        try:
            result = self._agent.run_sync(user_prompt, message_history=self._history)
        except UnexpectedModelBehavior as exc:
            # The model's output could not be validated into an LLMMove even once;
            # with no retries that ends the game as an unparseable-output fault.
            raise MoveUnavailable(
                PlayerFaultReason.UNPARSEABLE_OUTPUT, str(exc)
            ) from exc
        except (ModelAPIError, UserError) as exc:
            # The provider call could not be made or failed: a bad model name, a
            # rejected key, a rate limit, a network error (ModelAPIError), or a
            # model/capability mismatch caught client-side before any request
            # (UserError — e.g. an unknown Anthropic model rejecting native
            # output). None of these is a game fault, so surface a clear message
            # instead of letting a raw traceback escape.
            raise ModelRequestError(self._describe_backend_error(exc)) from exc

        self._history = list(result.all_messages())
        self._write_log()

        output: LLMMove = result.output
        move: Move = self._parse_move(output.cells)
        return MoveChoice(move, output.claimed_outcome)

    def end_game(self, result: GameResult) -> None:
        # Composed now, sent only if there is a next game: it stays queued and is
        # flushed on the next game's first choose_move (§11).
        self._pending.append(self._describe_end(result))

    def _parse_move(self, cells: list[str]) -> Move:
        """Turn the model's ``cells`` into a :class:`Move`, mapping any structural
        problem to the matching :class:`MoveUnavailable` fault (§11)."""
        try:
            return Move.from_labels(*cells)
        except IllegalMove as exc:
            raise MoveUnavailable(exc.reason, str(exc)) from exc
        except ValueError as exc:
            raise MoveUnavailable(
                PlayerFaultReason.UNPARSEABLE_OUTPUT, str(exc)
            ) from exc

    def _describe_backend_error(self, exc: ModelAPIError | UserError) -> str:
        """A short, actionable message for a failed provider call (see
        :class:`ModelRequestError`)."""
        if isinstance(exc, ModelHTTPError):
            hint: str
            if exc.status_code == 404:
                hint = (
                    "the model name may be misspelled or unavailable from this "
                    "provider — check it in players.yaml"
                )
            elif exc.status_code in (401, 403):
                hint = "the provider rejected the API key — check it in your .env"
            elif exc.status_code == 429:
                hint = "the provider is rate-limiting — wait and try again"
            else:
                hint = "check the model name in players.yaml and your provider setup"
            return (
                f"player {self.name!r} could not use model {exc.model_name!r}: "
                f"provider returned HTTP {exc.status_code} — {hint}"
            )
        if isinstance(exc, ModelAPIError):
            return (
                f"player {self.name!r} could not reach its model ({exc}); "
                f"check your network and provider configuration"
            )
        # UserError: a client-side model/capability mismatch (e.g. an unknown
        # model name that the provider profile can't confirm supports the output
        # mode). Surface the underlying message and point at the likely cause.
        return (
            f"player {self.name!r} could not use its model ({exc}); "
            f"check the model name in players.yaml is correct for this provider"
        )

    def _describe_end(self, result: GameResult) -> str:
        """A message telling the model how the game ended, from its perspective.

        Reports facts from the :class:`GameResult` — and, when the *opponent* won
        or faulted on their own turn, names their game-ending move, which the
        model was never handed via ``choose_move``. When the model itself faulted,
        it says what went wrong and how to avoid repeating it.
        """
        assert self._side is not None
        if result.termination is Termination.LINE_COMPLETED:
            if result.winner is self._side:
                return "That game is over: you completed a line and won."
            return "That game is over: your opponent completed a line and won."
        if result.termination is Termination.CATS_GAME:
            return (
                "That game is over: every line is dead, so it was a cat's game "
                "(a draw)."
            )

        fault: PlayerFaultDetail | None = result.fault
        assert fault is not None
        advice: str = _FAULT_ADVICE[fault.reason]
        if fault.offender is self._side:
            detail: str = (
                f"That game is over: you failed your turn, ending the game — "
                f"{advice}"
            )
            if fault.reason is PlayerFaultReason.WRONG_OUTCOME_CLAIM:
                detail += (
                    f" You claimed {fault.claimed_outcome.value} but it was "  # type: ignore[union-attr]
                    f"actually {fault.actual_outcome.value}."  # type: ignore[union-attr]
                )
            elif fault.attempted_move is not None:
                detail += f" (You attempted {fault.attempted_move}.)"
            detail += " Play a legal, well-assessed move this game."
            return detail

        move_note: str = (
            f" (attempting {fault.attempted_move})"
            if fault.attempted_move is not None
            else ""
        )
        return (
            f"That game is over: your opponent failed their turn{move_note}, "
            f"ending the game."
        )

    def _write_log(self) -> None:
        """Overwrite this player's log file with the full thread so far (§11).

        Rewritten after every model call so an interrupted run still leaves the
        whole conversation up to the point of failure. A no-op when logging is off.
        """
        if self._log_dir is None or self._side is None:
            return
        self._log_dir.mkdir(parents=True, exist_ok=True)
        safe_name: str = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in self.name
        )
        path: Path = self._log_dir / f"{safe_name}-{self._side.value}.json"
        path.write_bytes(ModelMessagesTypeAdapter.dump_json(self._history))
