"""The LLM player: chooses moves by querying a large language model.

This is the player type the project exists to compare (§4). It wraps a single
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
game's opening (§4, "No retries").

All Pydantic AI knowledge lives here, and nowhere else in the project: the
``(provider, model)`` → model mapping (:func:`resolve_model`), the agent's output
mode, effort level, and history processing (:func:`resolve_agent`,
:func:`strip_prior_thinking`), and the alternate constructor that turns a roster
entry into a ready player (:meth:`LLMPlayer.from_roster`). The
:mod:`~snakes_and_mice.config` module supplies only the parsed
:class:`~snakes_and_mice.config.Roster`; it neither imports ``pydantic_ai`` nor
knows what a model is.
"""

from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pydantic_ai import (
    Agent,
    ModelMessagesTypeAdapter,
    NativeOutput,
    UnexpectedModelBehavior,
    capture_run_messages,
)
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.exceptions import ModelHTTPError, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.settings import ModelSettings

from ..config import ConfigError, PlayerSpec, ProviderSpec, Roster
from ..core import Cell, Move, MoveChoice, Side, TurnOutcome
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

# Built-in providers Pydantic AI supports directly, each with the environment
# variable that holds its key (§4, "Model selection"). Custom OpenAI-compatible
# providers are declared in providers.yaml instead and are not listed here.
_BUILTIN_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

ThinkingLevel = Literal["minimal", "low", "medium", "high", "xhigh"]
"""Pydantic AI's unified reasoning-effort levels, coarsest to finest."""

DEFAULT_THINKING: ThinkingLevel = "high"
"""The global default reasoning effort every LLM player uses (§4). ``high``
rather than ``xhigh`` keeps strong reasoning without the steep cost of the top
tier."""

MAX_OUTPUT_TOKENS: int = 16384
"""Upper bound on tokens per response. Pydantic AI's default cap is 4096, and on
Anthropic that ceiling covers the thinking *and* the answer together — at high
effort the reasoning alone can approach it and clip the trailing JSON, which then
fails to parse and faults as UNPARSEABLE_OUTPUT. A generous cap leaves room for
both; only tokens actually produced are billed, so raising it costs nothing on a
short answer."""

# Providers whose structured output must use the model's native JSON-schema
# response format instead of an output tool. Anthropic forbids combining an
# output tool with thinking, so it needs native output; the others were validated
# with Pydantic AI's default tool-based output and keep it.
_NATIVE_OUTPUT_PROVIDERS: frozenset[str] = frozenset({"anthropic"})

# Some failures of a provider call are operational — neither the model's play nor
# a configuration error: a dropped connection or timeout, a rate-limit or server
# (5xx) error, or a malformed/truncated response body. All are worth retrying, so
# choose_move retries a few times with exponential backoff before giving up and
# voiding the game as a no-contest. They reach us in different shapes across
# providers — a raw httpx error (Gemini's SDK), a bare ModelAPIError wrapping one
# (the OpenAI/Anthropic SDKs), an HTTP-status ModelHTTPError, or a raw
# json.JSONDecodeError from the HTTP layer — so choose_move classifies them by
# *behavior* rather than by exception type.
_MAX_ATTEMPTS: int = 3
# Seconds to wait before each retry; one entry per gap between attempts, so its
# length is _MAX_ATTEMPTS - 1. Exponential so a brief blip clears quickly while a
# longer wobble still gets a real pause.
_RETRY_BACKOFF: tuple[float, ...] = (1.0, 4.0)

# HTTP statuses worth retrying: a request timeout (408), a rate-limit (429), or
# any server error (5xx). Other 4xx are client/config problems no retry fixes.
_TRANSIENT_STATUS_CODES: frozenset[int] = frozenset({408, 429})


def _is_transient_status(status_code: int) -> bool:
    """Whether an HTTP status from the provider is worth retrying (see
    :data:`_TRANSIENT_STATUS_CODES`)."""
    return status_code in _TRANSIENT_STATUS_CODES or status_code >= 500


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
    """The structured object the model returns for each move request (§4).

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


def resolve_model(spec: PlayerSpec, providers: dict[str, ProviderSpec]) -> Model:
    """Resolve one :class:`PlayerSpec` to a Pydantic AI model.

    Built-in providers read their key from the environment; a custom provider is
    looked up in ``providers`` and reached at its configured base URL. Raises
    :class:`ConfigError` for an unknown provider or a missing key.
    """
    provider: str = spec.provider
    if provider == "anthropic":
        return AnthropicModel(
            spec.model, provider=AnthropicProvider(api_key=_require_key(provider))
        )
    if provider == "openai":
        # The Responses API (not Chat Completions) is what supports OpenAI's
        # reasoning effort together with the function/output tool our structured
        # output relies on.
        return OpenAIResponsesModel(
            spec.model, provider=OpenAIProvider(api_key=_require_key(provider))
        )
    if provider == "gemini":
        return GoogleModel(
            spec.model, provider=GoogleProvider(api_key=_require_key(provider))
        )
    if provider == "openrouter":
        return OpenRouterModel(
            spec.model, provider=OpenRouterProvider(api_key=_require_key(provider))
        )

    custom: ProviderSpec | None = providers.get(provider)
    if custom is None:
        known: list[str] = sorted(_BUILTIN_KEY_ENV) + sorted(providers)
        raise ConfigError(
            f"player {spec.name!r} names unknown provider {provider!r}; "
            f"known providers: {', '.join(known)}"
        )
    # A local endpoint (e.g. ollama) may accept no key; the OpenAI client still
    # wants a non-empty string, so pass a harmless placeholder when none is set.
    api_key: str = (
        _require_key_env(custom.api_key_env)
        if custom.api_key_env is not None
        else "unused"
    )
    return OpenAIChatModel(
        spec.model, provider=OpenAIProvider(base_url=custom.base_url, api_key=api_key)
    )


def _require_key(provider: str) -> str:
    """The API key for a built-in ``provider``, or a :class:`ConfigError`."""
    return _require_key_env(_BUILTIN_KEY_ENV[provider])


def _require_key_env(env_var: str) -> str:
    """The value of ``env_var``, or a :class:`ConfigError` naming it. The keys
    themselves never appear in the config files — :func:`..config.load_environment`
    puts them in the environment, and they are read from there here."""
    value: str | None = os.environ.get(env_var)
    if not value:
        raise ConfigError(
            f"environment variable {env_var} is not set — add it to your .env or "
            f"export it before running"
        )
    return value


def _is_bulk_reasoning(part: ThinkingPart) -> bool:
    """Whether ``part`` is verbatim reasoning text safe to drop from a re-sent
    history — as opposed to a linked reasoning *item* that must be kept.

    The distinction is by shape, not provider, and the logs bear out three cases:

    * Gemini emits its thought summary as plain text (hundreds–thousands of chars)
      with no ``id`` or ``signature``. This is the payload worth stripping: it is
      self-contained, so removing it leaves a valid history.
    * OpenAI's Responses API and Anthropic emit an *empty-content* part carrying an
      ``id`` and/or ``signature`` that a following item references. Stripping it
      saves nothing (there is no text) and breaks the re-sent history — OpenAI
      rejects the dangling reference with HTTP 400 — so it must be kept.

    So strip only a part that has text *and* no provider id/signature."""
    return bool(part.content) and part.id is None and part.signature is None


def strip_prior_thinking(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Drop bulky reasoning text from earlier turns before each model request.

    A single LLM player keeps one message thread spanning the whole match (§4), and
    every provider re-sends that entire thread as input on each turn. A reasoning
    model's own past chain-of-thought can dominate that payload — for Gemini, which
    returns its full thought summary as text, it is the bulk of the accumulated input
    tokens, growing super-linearly over a match and capping how many games fit — yet
    it is not needed to keep playing: the board is fully reconstructible from the move
    list, and the model reasons afresh (still at full effort) every turn. Stripping
    that prior reasoning from what is *sent* cuts the context back to near-flat growth
    without lowering the thinking level.

    Only self-contained reasoning *text* is stripped (see :func:`_is_bulk_reasoning`);
    linked reasoning items (OpenAI/Anthropic, carrying an id/signature) are left in
    place — they cost nothing to keep and removing them would invalidate the history.

    This is attached as a :class:`ProcessHistory` capability, so it rewrites only the
    request payload; the player's stored thread — and thus the ``--log-llm`` dump —
    keeps the full thinking for debugging.
    """
    stripped: list[ModelMessage] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            kept = [
                p
                for p in message.parts
                if not (isinstance(p, ThinkingPart) and _is_bulk_reasoning(p))
            ]
            if len(kept) != len(message.parts):
                message = replace(message, parts=kept)
        stripped.append(message)
    return stripped


def resolve_agent(
    spec: PlayerSpec,
    providers: dict[str, ProviderSpec],
    *,
    thinking: ThinkingLevel = DEFAULT_THINKING,
) -> Agent[None, LLMMove]:
    """Build the Pydantic AI :class:`Agent` an :class:`LLMPlayer` will drive.

    Provider knowledge lives here, not in the player's own methods: the model is
    resolved and then wrapped in an agent whose output mode fits the provider —
    native JSON-schema output where an output tool cannot coexist with thinking
    (Anthropic), the default tool-based output everywhere else — so the player
    itself only drives the message thread and reads back an :class:`LLMMove`.
    ``retries=0`` (both branches) enforces §4's "no re-prompting within a game".
    Every agent carries the :func:`strip_prior_thinking` history processor to keep
    a long match's context from ballooning with re-sent past reasoning.
    """
    model: Model = resolve_model(spec, providers)
    settings: ModelSettings = ModelSettings(
        thinking=thinking, max_tokens=MAX_OUTPUT_TOKENS
    )
    capabilities: list[ProcessHistory[None]] = [ProcessHistory(strip_prior_thinking)]
    if spec.provider in _NATIVE_OUTPUT_PROVIDERS:
        return Agent(
            model=model,
            output_type=NativeOutput(LLMMove),
            model_settings=settings,
            retries=0,
            capabilities=capabilities,
        )
    return Agent(
        model=model,
        output_type=LLMMove,
        model_settings=settings,
        retries=0,
        capabilities=capabilities,
    )


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
        # The agent arrives ready-made — from from_roster (via resolve_agent, which
        # picks the output mode and settings the provider needs) or, in tests, from
        # a FunctionModel. Once constructed, an instance is provider-agnostic: it
        # only drives the message thread and reads back an LLMMove.
        self._agent: Agent[None, LLMMove] = agent
        self._log_dir: Path | None = log_dir
        # The running conversation (spans every game) and the messages queued for
        # the next user turn, flushed on the following choose_move. The one-time
        # rules preamble seeds the queue, so it leads the very first user turn.
        self._history: list[ModelMessage] = []
        self._pending: list[str] = [RULES_PREAMBLE]
        self._side: Side | None = None

    @classmethod
    def from_roster(
        cls,
        name: str,
        roster: Roster,
        *,
        thinking: ThinkingLevel = DEFAULT_THINKING,
        log_dir: Path | None = None,
    ) -> LLMPlayer:
        """Build the player for roster entry ``name`` — the usual constructor.

        The roster comes from :mod:`~snakes_and_mice.config` as parsed specs; this
        is where one is resolved to a live model and agent (:func:`resolve_agent`),
        so the config layer never touches Pydantic AI. Raises :class:`ConfigError`
        if no such player is in the roster, if its provider is unknown, or if the
        provider's API key is not in the environment.
        """
        spec: PlayerSpec | None = roster.players.get(name)
        if spec is None:
            raise ConfigError(
                f"no player named {name!r} in the roster; "
                f"available: {', '.join(sorted(roster.players)) or '(none)'}"
            )
        agent: Agent[None, LLMMove] = resolve_agent(
            spec, roster.providers, thinking=thinking
        )
        return cls(agent, name=spec.name, log_dir=log_dir)

    def start_game(self, side: Side, seed: Cell) -> None:
        # Any feedback from a game we just faulted was enqueued by end_game and is
        # already ahead of this message in the pending queue (§4). The seed cell
        # is announced here, per game, since it can vary (the preamble describes
        # the rule but names no cell — see RULES_PREAMBLE).
        self._side = side
        self._pending.append(
            f"A new game begins. You are playing {side.value}. "
            f"The snake is seeded at {seed}."
        )

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

        # A transient operational failure is retried (see _MAX_ATTEMPTS); every
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
                    # the game as a fault. Persist the exchange first — including the
                    # response captured above — so it reaches the log for debugging
                    # and stays in the thread, letting the next game's fault feedback
                    # point at what actually happened rather than at nothing. A
                    # response truncated at the output-token limit (finish_reason
                    # "length", e.g. thinking that ran to the cap) is a distinct
                    # fault from genuinely malformed output, and gets its own
                    # feedback (think more briefly).
                    self._record_failed_turn(user_prompt, captured)
                    reason: PlayerFaultReason = (
                        PlayerFaultReason.THINKING_LIMIT_EXCEEDED
                        if self._hit_token_limit(captured)
                        else PlayerFaultReason.UNPARSEABLE_OUTPUT
                    )
                    raise MoveUnavailable(reason, str(exc)) from exc
                except Exception as exc:
                    # Any other failure from the provider call, split two ways by
                    # *behavior* — because pydantic-ai and the provider SDKs surface
                    # the same underlying condition in different exception shapes:
                    #
                    #  - A configuration error no retry can fix — a rejected key, an
                    #    unknown/unavailable model, a capability mismatch (UserError),
                    #    or any non-transient HTTP status — is broken for every game,
                    #    so it aborts the whole run with one clear message
                    #    (ModelRequestError), not a per-game outcome.
                    #  - Everything else is operational and transient: a connection
                    #    drop or timeout (a raw httpx error from Gemini's SDK, or a
                    #    bare ModelAPIError wrapping one from the OpenAI/Anthropic
                    #    SDKs), a rate-limit or 5xx, or a malformed/truncated response
                    #    body (a raw json.JSONDecodeError from the HTTP layer, seen
                    #    from flaky gateways). None is the model's play and none is a
                    #    config problem, so we retry with backoff and, if it persists,
                    #    void just this game as a no-contest (PlayerUnavailable ->
                    #    ABORTED) — never charging the player, never aborting the
                    #    match, and never letting a raw traceback escape.
                    if self._is_config_fatal(exc):
                        raise ModelRequestError(
                            self._describe_backend_error(exc)
                        ) from exc
                    if self._backoff_and_retry(attempt):
                        continue
                    self._record_failed_turn(user_prompt, captured)
                    raise PlayerUnavailable(self._describe_unreachable(exc)) from exc

            # Append only this turn's new messages to the thread we already hold.
            # We must NOT rebuild from result.all_messages(): the
            # strip_prior_thinking capability rewrites the history all_messages()
            # reports too, so overwriting our thread with it would progressively
            # erase earlier turns' thinking from the stored thread and the log.
            # new_messages() carries the current turn's response with its thinking
            # intact — only *prior* turns are stripped, and only on the wire.
            self._history.extend(result.new_messages())
            self._write_log()

            output: LLMMove = result.output
            move: Move = self._parse_move(output.cells)
            return MoveChoice(move, output.claimed_outcome)

        # Unreachable: the final attempt either returns or raises above.
        raise AssertionError("choose_move exhausted its retries without resolving")

    def _record_failed_turn(
        self, user_prompt: str, captured: list[ModelMessage]
    ) -> None:
        """Persist a turn that raised — a fault or a transient failure — to the
        thread and the log, on a path where no ``result`` object is available.

        We must NOT recover the new turn by slicing ``captured`` at
        ``len(self._history)``. ``capture_run_messages`` reports the *wire* history,
        which pydantic-ai builds by running :func:`strip_prior_thinking` and
        then dropping the now-empty reasoning-only responses it leaves behind — so
        ``captured`` can be strictly *shorter* than our unstripped thread, making a
        length-based slice select nothing and silently lose the turn (a mid-match
        unparseable fault vanished from the log this way). Instead reconstruct the
        turn from what we know it was: the prompt we sent, plus the model's response,
        which is always the tail of ``captured``. The response is kept verbatim (its
        thinking intact) so the log stays complete and the next game's fault feedback
        can point at what the model actually produced.
        """
        self._history.append(ModelRequest(parts=[UserPromptPart(content=user_prompt)]))
        if captured and isinstance(captured[-1], ModelResponse):
            response: ModelResponse = captured[-1]
            self._history.append(response)
            self._close_dangling_tool_calls(response)
        self._write_log()

    def _close_dangling_tool_calls(self, response: ModelResponse) -> None:
        """Follow any tool-call in a just-recorded faulting ``response`` with a
        synthetic tool-return, so the stored thread stays a well-formed message
        history the provider will accept when it is re-sent next game.

        An unparseable-output fault can take the shape of a ``final_result`` tool-call
        whose arguments failed validation (e.g. a mistyped field name). Because the
        player runs with no retries (§4), pydantic-ai raises *before* emitting the
        tool-return that would normally close that call, so the response we persist
        ends in a tool-call with no matching return. Left unclosed, re-sending the
        thread makes the provider reject the whole request ("cannot provide a new user
        prompt when the message history contains unprocessed tool calls"), which would
        abort the next game — and, misread as a backend error, the whole match.

        ``_record_failed_turn`` is the *only* place a dangling call can enter the
        thread (the success path extends from ``new_messages()``, always well-formed),
        and the offending call is in the response we just appended — so this is the one
        spot that needs the repair, done once, here. The faulting response itself is
        kept verbatim (its broken args intact) for the log and the next game's fault
        feedback; only a closing tool-return is added after it.
        """
        closers: list[ModelRequestPart] = [
            ToolReturnPart(
                tool_name=part.tool_name,
                content="This move could not be processed; the game ended.",
                tool_call_id=part.tool_call_id,
            )
            for part in response.parts
            if isinstance(part, ToolCallPart)
        ]
        if closers:
            self._history.append(ModelRequest(parts=closers))

    def end_game(self, result: GameResult) -> None:
        # Composed now, sent only if there is a next game: it stays queued and is
        # flushed on the next game's first choose_move (§4).
        self._pending.append(self._describe_end(result))

    def _parse_move(self, cells: list[str]) -> Move:
        """Turn the model's ``cells`` into a :class:`Move`, mapping any structural
        problem to the matching :class:`MoveUnavailable` fault (§4)."""
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

    @staticmethod
    def _is_config_fatal(exc: Exception) -> bool:
        """Whether ``exc`` from the provider call is a configuration error that no
        retry can fix — a rejected key, an unknown/unavailable model, a capability
        mismatch (``UserError``), or any non-transient HTTP status. Such errors are
        broken for every game and abort the whole run (:class:`ModelRequestError`);
        everything else is treated as transient (see :meth:`choose_move`)."""
        if isinstance(exc, UserError):
            return True
        if isinstance(exc, ModelHTTPError):
            return not _is_transient_status(exc.status_code)
        return False

    @staticmethod
    def _backoff_and_retry(attempt: int) -> bool:
        """Sleep before the next attempt and report whether one remains. Returns
        ``False`` on the final attempt (with no sleep), so the caller gives up."""
        if attempt + 1 < _MAX_ATTEMPTS:
            time.sleep(_RETRY_BACKOFF[attempt])
            return True
        return False

    def _describe_unreachable(self, exc: Exception) -> str:
        """A short message for an operational failure that outlasted every retry
        (see :class:`PlayerUnavailable`): a connection drop or timeout, a rate-limit
        or server error, or a malformed response body. It names the player and the
        failure kind — the class name rather than the message, because bare timeouts
        often stringify to nothing — so a voided game is explicable."""
        return (
            f"player {self.name!r} could not complete a call to its model "
            f"({type(exc).__name__}) after {_MAX_ATTEMPTS} attempts"
        )

    def _describe_backend_error(self, exc: Exception) -> str:
        """A short, actionable message for a configuration error from the provider
        call (see :class:`ModelRequestError` and :meth:`_is_config_fatal`)."""
        if isinstance(exc, ModelHTTPError):
            hint: str
            if exc.status_code == 404:
                hint = (
                    "the model name may be misspelled or unavailable from this "
                    "provider — check it in players.yaml"
                )
            elif exc.status_code in (401, 403):
                hint = "the provider rejected the API key — check it in your .env"
            else:
                hint = (
                    "the request was rejected — the model may not support a required "
                    "feature (structured output or thinking); check players.yaml and "
                    "your provider setup"
                )
            return (
                f"player {self.name!r} could not use model {exc.model_name!r}: "
                f"provider returned HTTP {exc.status_code} — {hint}"
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
        """Overwrite this player's log file with the full thread so far (§4).

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
