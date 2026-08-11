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

import time
from pathlib import Path

import httpx
from pydantic import BaseModel
from pydantic_ai import (
    Agent,
    ModelMessagesTypeAdapter,
    UnexpectedModelBehavior,
    capture_run_messages,
)
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse

from ..core import Move, MoveChoice, Side, TurnOutcome
from ..faults import (
    IllegalMove,
    MoveUnavailable,
    PlayerFaultReason,
    PlayerUnavailable,
    SnakesAndMiceError,
)
from ..result import GameResult, PlayerFaultDetail, Termination
from .base import Player
from .prompts import FAULT_ADVICE, RULES_PREAMBLE

# Transient transport failures (read/connect timeouts, dropped connections) are
# not the model's doing, so we retry the call before giving up. These live at
# httpx's transport layer, below pydantic-ai's ModelAPIError, and so would
# otherwise escape as a raw traceback. HTTP *status* errors (4xx/5xx) are not
# retried here: pydantic-ai wraps those into ModelAPIError, and a bad model name
# or rejected key is a configuration problem no retry will fix.
_MAX_ATTEMPTS: int = 3
# Seconds to wait before each retry; one entry per gap between attempts, so its
# length is _MAX_ATTEMPTS - 1. Exponential so a brief blip clears quickly while a
# longer wobble still gets a real pause.
_RETRY_BACKOFF: tuple[float, ...] = (1.0, 4.0)


class ModelRequestError(SnakesAndMiceError):
    """The call to a model's provider failed for a reason that no retry fixes and
    that is not a game fault — a misspelled or unavailable model, a rejected key,
    or a capability mismatch. It is an environment/configuration error, broken for
    every game rather than one, so it does *not* end a single game: it propagates
    past :func:`~snakes_and_mice.game.play_game` and
    :func:`~snakes_and_mice.match.play_match` uncaught, up to the entry point,
    which catches it and reports one clear message (see :func:`..cli.main`).

    Contrast the two conditions the engine *does* turn into a
    :class:`~snakes_and_mice.result.GameResult`: :class:`MoveUnavailable` (a
    fault) and :class:`PlayerUnavailable` (a transient transport failure that
    outlasted its retries — a no-contest abort, not run-fatal)."""


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
    # The "Your response" paragraph of RULES_PREAMBLE (players/prompts.py)
    # describes these fields to the model; keep the two in sync.


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

        # A transient transport failure is retried (see _MAX_ATTEMPTS); every
        # other outcome resolves on the first attempt, returning or raising below.
        for attempt in range(_MAX_ATTEMPTS):
            # capture_run_messages records the exchange even when the run raises,
            # so a response that fails validation is still available to us here —
            # see the UnexpectedModelBehavior handler. It captures only the first
            # run within its scope; a fresh scope per attempt keeps each retry's
            # exchange separate.
            with capture_run_messages() as captured:
                try:
                    result = self._agent.run_sync(
                        user_prompt, message_history=self._history
                    )
                except UnexpectedModelBehavior as exc:
                    # The model produced no usable move; with no retries that ends
                    # the game. Persist the exchange first — including the response
                    # captured above — so it reaches the log for debugging and
                    # stays in the thread, letting the next game's fault feedback
                    # point at what actually happened rather than at nothing. A
                    # response truncated at the output-token limit (finish_reason
                    # "length", e.g. thinking that ran to the cap) is a distinct
                    # fault from genuinely malformed output, and gets its own
                    # feedback (think more briefly).
                    self._history = list(captured)
                    self._write_log()
                    reason: PlayerFaultReason = (
                        PlayerFaultReason.THINKING_LIMIT_EXCEEDED
                        if self._hit_token_limit(captured)
                        else PlayerFaultReason.UNPARSEABLE_OUTPUT
                    )
                    raise MoveUnavailable(reason, str(exc)) from exc
                except (ModelAPIError, UserError) as exc:
                    # The provider call failed for a reason no retry fixes: a bad
                    # model name, a rejected key (ModelAPIError/ModelHTTPError), or
                    # a model/capability mismatch caught client-side before any
                    # request (UserError — e.g. an unknown Anthropic model
                    # rejecting native output). A configuration problem, not a
                    # transient one, so it aborts the run with a clear message
                    # instead of letting a raw traceback escape.
                    raise ModelRequestError(self._describe_backend_error(exc)) from exc
                except httpx.TransportError as exc:
                    # A transport-level failure (timeout, dropped connection): not
                    # the model's doing. Retry a few times with backoff, then give
                    # up and let the engine void the game as a no-contest — never
                    # charging the player and never aborting the whole match.
                    if attempt + 1 < _MAX_ATTEMPTS:
                        time.sleep(_RETRY_BACKOFF[attempt])
                        continue
                    if captured:
                        self._history = list(captured)
                    self._write_log()
                    raise PlayerUnavailable(self._describe_unreachable(exc)) from exc

            self._history = list(result.all_messages())
            self._write_log()

            output: LLMMove = result.output
            move: Move = self._parse_move(output.cells)
            return MoveChoice(move, output.claimed_outcome)

        # Unreachable: the final attempt either returns or raises above.
        raise AssertionError("choose_move exhausted its retries without resolving")

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

    @staticmethod
    def _hit_token_limit(messages: list[ModelMessage]) -> bool:
        """Whether the run's final response was truncated at the output-token
        limit — the signal (``finish_reason == "length"``) that distinguishes a
        model that ran out of budget (usually mid-thinking) from one that returned
        genuinely unparseable output. ``messages`` are those captured from the
        failed run; the model's response, if any, is the last of them."""
        last: ModelMessage | None = messages[-1] if messages else None
        return isinstance(last, ModelResponse) and last.finish_reason == "length"

    def _describe_unreachable(self, exc: httpx.TransportError) -> str:
        """A short message for a transport failure that outlasted every retry
        (see :class:`PlayerUnavailable`). It names the player and the failure kind
        so a voided game is explicable, and carries the class name rather than the
        message because bare timeouts often stringify to nothing."""
        return (
            f"player {self.name!r} could not reach its model "
            f"({type(exc).__name__}) after {_MAX_ATTEMPTS} attempts"
        )

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
        if result.termination is Termination.ABORTED:
            # A no-contest: a technical problem, not the model's play, voided the
            # game. Say so plainly and give no fault advice — there is nothing for
            # the model to do differently.
            return (
                "That game was abandoned because of a technical problem reaching "
                "you, not anything about your play — it does not count."
            )

        fault: PlayerFaultDetail | None = result.fault
        assert fault is not None
        advice: str = FAULT_ADVICE[fault.reason]
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
