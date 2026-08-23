"""Tests for the LLM player, driven by Pydantic AI's :class:`FunctionModel`.

No network is used: a scripted model returns a fixed sequence of structured
``LLMMove`` responses, so we can exercise move parsing, fault mapping, the
cross-game message thread (rules once, opponent moves relayed, feedback carried
forward), and message logging deterministically. The player uses
``NativeOutput``, so a response is a plain JSON ``TextPart`` (no output tool).

The last section covers the module's other half — resolving a roster entry to a
model, an agent, and a player (``resolve_model`` / ``resolve_agent`` /
``LLMPlayer.from_roster``) — which is also network-free: constructing a
provider/model only stores the key and endpoint, so we can assert the resolved
type and that a missing key or unknown provider is reported as a ``ConfigError``.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pydantic_ai import Agent, ModelMessagesTypeAdapter, NativeOutput
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.models.openrouter import OpenRouterModel

from snakes_and_mice import (
    Cell,
    GameResult,
    Move,
    MoveUnavailable,
    PlayerFaultDetail,
    PlayerFaultReason,
    PlayerUnavailable,
    Side,
    Termination,
    TurnOutcome,
    play_game,
)
from snakes_and_mice.config import ConfigError, PlayerSpec, ProviderSpec, Roster
from snakes_and_mice.players import LLMPlayer
from snakes_and_mice.players.llm import (
    LLMMove,
    ModelRequestError,
    resolve_agent,
    resolve_model,
    strip_prior_thinking,
)
from snakes_and_mice.players.prompts import RULES_PREAMBLE

# An arbitrary fixed seed cell for start_game calls that don't care where the
# snake begins (the board default is private to board.py).
_SEED: Cell = Cell.from_label("B3")


def _move(
    cells: list[str], outcome: str = "in_play", rationale: str = "because"
) -> dict[str, object]:
    """One scripted structured response for the model to return."""
    return {"move_rationale": rationale, "cells": cells, "claimed_outcome": outcome}


def _scripted(
    moves: list[dict[str, object]], captured: list[str] | None = None
) -> Agent[None, LLMMove]:
    """An agent whose model returns ``moves`` in order, optionally recording the
    user prompt (the flushed messages) it was handed on each call.

    The player no longer builds its own agent — the config layer does, choosing
    the output mode per provider — so tests inject one here. It mirrors the
    Anthropic path (``NativeOutput``): the scripted model returns each move as
    JSON in a text response rather than an output-tool call.
    """
    state: dict[str, int] = {"i": 0}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if captured is not None:
            request: ModelMessage = messages[-1]
            texts: list[str] = [
                str(part.content)
                for part in request.parts
                if isinstance(part, UserPromptPart)
            ]
            captured.append("\n".join(texts))
        args: dict[str, object] = moves[state["i"]]
        state["i"] += 1
        return ModelResponse(parts=[TextPart(content=json.dumps(args))])

    return Agent(
        model=FunctionModel(respond), output_type=NativeOutput(LLMMove), retries=0
    )


def _failing_agent(exc: Exception) -> Agent[None, LLMMove]:
    """An agent whose model raises ``exc`` on the first call."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise exc

    return Agent(
        model=FunctionModel(respond), output_type=NativeOutput(LLMMove), retries=0
    )


def _malformed_agent(content: str) -> Agent[None, LLMMove]:
    """An agent whose model returns ``content`` that cannot validate into an
    LLMMove, so run_sync raises UnexpectedModelBehavior."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=content)])

    return Agent(
        model=FunctionModel(respond), output_type=NativeOutput(LLMMove), retries=0
    )


def _truncated_agent() -> Agent[None, LLMMove]:
    """An agent whose model returns a thinking-only response truncated at the
    output-token limit (finish_reason 'length'), reproducing the adaptive-thinking
    overrun that yields no move at all."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ThinkingPart(content="still reasoning when the budget ran out")],
            finish_reason="length",
        )

    return Agent(
        model=FunctionModel(respond), output_type=NativeOutput(LLMMove), retries=0
    )


def _timeout_then_move_agent(
    timeouts: int, move: dict[str, object]
) -> Agent[None, LLMMove]:
    """An agent that raises a transport timeout on its first ``timeouts`` calls,
    then returns ``move`` — used to exercise the retry path in choose_move."""
    state: dict[str, int] = {"calls": 0}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        state["calls"] += 1
        if state["calls"] <= timeouts:
            raise httpx.ReadTimeout("timed out")
        return ModelResponse(parts=[TextPart(content=json.dumps(move))])

    return Agent(
        model=FunctionModel(respond), output_type=NativeOutput(LLMMove), retries=0
    )


def test_choose_move_returns_parsed_move_and_claim() -> None:
    player: LLMPlayer = LLMPlayer(_scripted([_move(["A1", "A2"], "in_play")]))
    player.start_game(Side.MOUSE, _SEED)

    choice = player.choose_move()

    assert str(choice.move) == "A1 A2"
    assert choice.claimed_outcome is TurnOutcome.IN_PLAY


@pytest.mark.parametrize(
    "cells,reason",
    [
        (["Z9"], PlayerFaultReason.OFF_BOARD),
        (["A1", "A1"], PlayerFaultReason.DUPLICATE_CELLS),
        (["A1", "A2", "A3"], PlayerFaultReason.WRONG_PIECE_COUNT),
        ([], PlayerFaultReason.WRONG_PIECE_COUNT),
        (["nonsense"], PlayerFaultReason.UNPARSEABLE_OUTPUT),
    ],
)
def test_bad_cells_map_to_move_unavailable(
    cells: list[str], reason: PlayerFaultReason
) -> None:
    player: LLMPlayer = LLMPlayer(_scripted([_move(cells)]))
    player.start_game(Side.MOUSE, _SEED)

    with pytest.raises(MoveUnavailable) as excinfo:
        player.choose_move()
    assert excinfo.value.reason is reason


def test_provider_error_becomes_model_request_error() -> None:
    # A failed provider call (here a 404 for an unavailable model) is not a game
    # fault: it surfaces as a ModelRequestError naming the player, model, and
    # status, so the CLI can print a clean message instead of a traceback.
    player: LLMPlayer = LLMPlayer(
        _failing_agent(ModelHTTPError(status_code=404, model_name="bad-model")),
        name="opus",
    )
    player.start_game(Side.MOUSE, _SEED)

    with pytest.raises(ModelRequestError) as excinfo:
        player.choose_move()
    message: str = str(excinfo.value)
    assert "opus" in message
    assert "bad-model" in message
    assert "404" in message


@pytest.mark.parametrize("status", [400, 401, 403])
def test_config_http_status_aborts_the_run(status: int) -> None:
    # Non-transient 4xx statuses — a rejected key, a bad request the model can't
    # satisfy — are configuration problems no retry fixes, so they abort the whole
    # run with a ModelRequestError rather than ending a single game.
    player: LLMPlayer = LLMPlayer(
        _failing_agent(ModelHTTPError(status_code=status, model_name="m")), name="bot"
    )
    player.start_game(Side.MOUSE, _SEED)

    with pytest.raises(ModelRequestError):
        player.choose_move()


def test_user_error_aborts_the_run() -> None:
    # A client-side capability mismatch (raised before any request) is a config
    # problem, not a game outcome: it aborts the run with a clear message.
    player: LLMPlayer = LLMPlayer(
        _failing_agent(UserError("model does not support native output")), name="bot"
    )
    player.start_game(Side.MOUSE, _SEED)

    with pytest.raises(ModelRequestError):
        player.choose_move()


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(
            json.JSONDecodeError("Expecting value", "", 0), id="malformed-body"
        ),
        pytest.param(ModelAPIError(model_name="m", message="connection reset"), id="connection"),
        pytest.param(ModelHTTPError(status_code=429, model_name="m"), id="rate-limit"),
        pytest.param(ModelHTTPError(status_code=503, model_name="m"), id="server-error"),
    ],
)
def test_operational_failures_are_transient_not_crashes(
    monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    # Operational failures reach us in different shapes — a raw json.JSONDecodeError
    # from a truncated response body, a bare ModelAPIError wrapping a dropped
    # connection, a 429 rate-limit, a 5xx server error. None is the model's play or
    # a config error, so none may escape as a traceback: each is retried and, when
    # it persists, voids just this game as a no-contest (PlayerUnavailable ->
    # ABORTED), never a fault and never a run-aborting ModelRequestError.
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    player: LLMPlayer = LLMPlayer(_failing_agent(exc), name="bot")
    player.start_game(Side.MOUSE, _SEED)

    with pytest.raises(PlayerUnavailable) as excinfo:
        player.choose_move()
    assert "bot" in str(excinfo.value)


def test_unparseable_output_is_logged_and_kept_in_history(tmp_path: Path) -> None:
    # When the model's response can't be validated into an LLMMove, it's an
    # unparseable-output fault — but the bad response must still be persisted: to
    # the log (for debugging) and into the thread, so the next game's fault
    # feedback can point at the actual output the model produced.
    log_dir: Path = tmp_path / "logs"
    player: LLMPlayer = LLMPlayer(
        _malformed_agent("I refuse to answer in the required format."),
        name="bot",
        log_dir=log_dir,
    )
    player.start_game(Side.MOUSE, _SEED)

    with pytest.raises(MoveUnavailable) as excinfo:
        player.choose_move()
    assert excinfo.value.reason is PlayerFaultReason.UNPARSEABLE_OUTPUT

    # The malformed exchange survives in the thread the next game will replay…
    assert len(player._history) >= 2  # the user turn and the bad model response
    # …and it was written to the log, bad content and all, for debugging.
    log_file: Path = log_dir / "bot-mouse.json"
    assert log_file.exists()
    assert "I refuse to answer in the required format." in log_file.read_text()


def test_failed_turn_recorded_when_captured_is_shorter_than_thread(
    tmp_path: Path,
) -> None:
    # Regression: capture_run_messages reports the strip-processed WIRE history,
    # which pydantic-ai compacts (dropping the empty reasoning-only responses
    # strip_prior_thinking leaves behind) below our unstripped thread's length. The
    # old recovery — self._history.extend(captured[len(self._history):]) — then
    # selected an empty slice and silently dropped the faulting turn from the thread
    # and the --log-llm dump (a real mid-match unparseable fault vanished this way).
    # The turn must be recorded from the prompt we sent plus the captured response,
    # regardless of how short the wire history was compacted to.
    log_dir: Path = tmp_path / "logs"
    player: LLMPlayer = LLMPlayer(
        _scripted([_move(["A1", "A2"])]), name="bot", log_dir=log_dir
    )
    player.start_game(Side.MOUSE, _SEED)
    # A thread longer than the (compacted) wire history we will hand the recovery.
    player._history = [
        ModelRequest(parts=[UserPromptPart(content="turn 1")]),
        ModelResponse(parts=[ThinkingPart(content="game-1 fault: thinking only")]),
        ModelRequest(parts=[UserPromptPart(content="new game")]),
        ModelResponse(parts=[TextPart(content=json.dumps(_move(["C3", "D2"])))]),
    ]
    before: int = len(player._history)
    # capture_run_messages, strictly shorter than the thread, ending in the failed
    # response (a reasoning-only turn that never produced a move).
    failed: ModelResponse = ModelResponse(parts=[ThinkingPart(content="no move")])
    captured: list[ModelMessage] = [
        ModelResponse(parts=[]),
        ModelRequest(parts=[UserPromptPart(content="your turn (mouse).")]),
        failed,
    ]
    assert len(captured) < before  # the shortfall the old slice mishandled

    player._record_failed_turn("your turn (mouse). Choose your move.", captured)

    # The turn is recorded: the prompt we sent, then the model's failed response…
    assert len(player._history) == before + 2
    assert isinstance(player._history[-2], ModelRequest)
    assert player._history[-1] is failed
    # …and it reached the log, not just the thread.
    assert (log_dir / "bot-mouse.json").exists()
    assert "no move" in (log_dir / "bot-mouse.json").read_text()


def test_dangling_tool_call_is_closed_when_recorded(tmp_path: Path) -> None:
    # Regression: an unparseable-output fault can be a final_result tool-call whose
    # args failed validation; with no retries pydantic-ai raises before emitting the
    # tool-return that closes it, so the response we persist ends in an unreturned
    # tool-call. Left as-is, re-sending the thread next game makes the provider reject
    # the whole request ("unprocessed tool calls") and abort the game. The faulting
    # turn must be recorded verbatim (for the log/feedback) but immediately followed by
    # a synthetic tool-return, so the stored thread is a valid history to re-send.
    log_dir: Path = tmp_path / "logs"
    player: LLMPlayer = LLMPlayer(
        _scripted([_move(["A1", "A2"])]), name="bot", log_dir=log_dir
    )
    player.start_game(Side.MOUSE, _SEED)
    failed: ModelResponse = ModelResponse(
        parts=[
            ThinkingPart(content="reasoning"),
            ToolCallPart(
                tool_name="final_result",
                args='{"move_rationally": "typo"}',  # mistyped field: won't validate
                tool_call_id="call-xyz",
            ),
        ]
    )
    captured: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="your turn (mouse).")]),
        failed,
    ]

    player._record_failed_turn("your turn (mouse). Choose your move.", captured)

    # The faulting response is stored verbatim, its broken tool-call intact (the log
    # must show what the model actually produced)…
    assert failed in player._history
    assert "move_rationally" in (log_dir / "bot-mouse.json").read_text()
    # …immediately followed by a synthetic tool-return closing that call.
    tail: ModelMessage = player._history[-1]
    assert isinstance(tail, ModelRequest)
    assert any(
        isinstance(part, ToolReturnPart) and part.tool_call_id == "call-xyz"
        for part in tail.parts
    )
    # The thread is now well-formed: no tool-call anywhere is left unreturned.
    call_ids: set[str] = {
        part.tool_call_id
        for msg in player._history
        if isinstance(msg, ModelResponse)
        for part in msg.parts
        if isinstance(part, ToolCallPart)
    }
    returned_ids: set[str] = {
        part.tool_call_id
        for msg in player._history
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, ToolReturnPart)
    }
    assert call_ids <= returned_ids


def test_failed_turn_without_tool_call_gets_no_synthetic_return() -> None:
    # The common unparseable/thinking-limit shape is a thinking- or text-only response
    # with no tool-call at all. There is nothing to close, so no synthetic tool-return
    # is appended — the thread must not grow a spurious empty request.
    player: LLMPlayer = LLMPlayer(_scripted([_move(["A1", "A2"])]), name="bot")
    player.start_game(Side.MOUSE, _SEED)
    failed: ModelResponse = ModelResponse(parts=[ThinkingPart(content="no move")])
    captured: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="your turn (mouse).")]),
        failed,
    ]

    player._record_failed_turn("your turn (mouse). Choose your move.", captured)

    # Exactly the prompt and the failed response — no trailing tool-return request.
    assert player._history[-1] is failed
    assert not any(
        isinstance(part, ToolReturnPart)
        for msg in player._history
        if isinstance(msg, ModelRequest)
        for part in msg.parts
    )


def test_token_limit_truncation_is_a_distinct_fault(tmp_path: Path) -> None:
    # A response truncated at the output-token limit (the model thought until it
    # ran out, emitting no move) is THINKING_LIMIT_EXCEEDED, not UNPARSEABLE_OUTPUT
    # — so the next game can tell the model to think more briefly. The truncated
    # response is still persisted to the log and the thread.
    log_dir: Path = tmp_path / "logs"
    player: LLMPlayer = LLMPlayer(_truncated_agent(), name="bot", log_dir=log_dir)
    player.start_game(Side.MOUSE, _SEED)

    with pytest.raises(MoveUnavailable) as excinfo:
        player.choose_move()
    assert excinfo.value.reason is PlayerFaultReason.THINKING_LIMIT_EXCEEDED

    assert player._history and isinstance(player._history[-1], ModelResponse)
    assert (log_dir / "bot-mouse.json").exists()


def test_transient_timeout_is_retried_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A transport timeout is not the model's doing: the call is retried, and a
    # move that arrives on a later attempt is returned normally. Backoff sleeps
    # are stubbed so the test does not actually wait.
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    player: LLMPlayer = LLMPlayer(
        _timeout_then_move_agent(2, _move(["A1", "A2"], "in_play"))
    )
    player.start_game(Side.MOUSE, _SEED)

    choice = player.choose_move()

    assert str(choice.move) == "A1 A2"
    assert choice.claimed_outcome is TurnOutcome.IN_PLAY


def test_unreachable_model_raises_player_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # When every attempt times out, the player gives up with PlayerUnavailable
    # (not a fault, not a run-aborting ModelRequestError). The thread so far is
    # still logged for debugging.
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    log_dir: Path = tmp_path / "logs"
    # More timeouts than the player will ever attempt: it always fails.
    player: LLMPlayer = LLMPlayer(
        _timeout_then_move_agent(99, _move(["A1", "A2"])),
        name="bot",
        log_dir=log_dir,
    )
    player.start_game(Side.MOUSE, _SEED)

    with pytest.raises(PlayerUnavailable) as excinfo:
        player.choose_move()
    assert "bot" in str(excinfo.value)
    assert (log_dir / "bot-mouse.json").exists()


def test_prior_thinking_stripped_from_requests_but_kept_in_history() -> None:
    # Over a match a reasoning model's own past thinking would otherwise be re-sent
    # as input every turn, ballooning the context. The strip_prior_thinking history
    # processor (attached to every agent by resolve_agent) removes it from what the
    # model receives, while the stored thread — and so the --log-llm dump — keeps the
    # full thinking for debugging. This does not touch the current turn's thinking.
    saw_prior_thinking: list[bool] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        saw_prior_thinking.append(
            any(
                isinstance(part, ThinkingPart)
                for msg in messages
                if isinstance(msg, ModelResponse)
                for part in msg.parts
            )
        )
        return ModelResponse(
            parts=[
                ThinkingPart(content="reasoning about the board"),
                TextPart(content=json.dumps(_move(["A1", "A2"]))),
            ]
        )

    agent: Agent[None, LLMMove] = Agent(
        model=FunctionModel(respond),
        output_type=NativeOutput(LLMMove),
        retries=0,
        capabilities=[ProcessHistory(strip_prior_thinking)],
    )
    player: LLMPlayer = LLMPlayer(agent, name="bot")
    player.start_game(Side.MOUSE, _SEED)
    player.choose_move()  # turn 1
    player.observe_move(Side.SNAKE, Move.from_labels("E1", "E2"))
    player.choose_move()  # turn 2: its request carries turn 1's response

    # Neither request carried a prior thinking part — turn 2 saw turn 1's response
    # with the reasoning stripped, so the model never re-reads its own thinking.
    assert saw_prior_thinking == [False, False]
    # Yet the stored thread retains EVERY turn's thinking, so the log stays
    # complete: rebuilding history from the strip-processed all_messages() each
    # turn would have erased the earlier turns' thinking, leaving only the last.
    responses_with_thinking: int = sum(
        any(isinstance(part, ThinkingPart) for part in msg.parts)
        for msg in player._history
        if isinstance(msg, ModelResponse)
    )
    assert responses_with_thinking == 2  # both turns, not just the most recent


def test_full_game_relays_opponent_moves_only() -> None:
    # Mouse takes the whole of row A over three turns (the last a single, winning
    # piece); snake plays harmlessly along row E.
    seen: list[str] = []
    mouse: LLMPlayer = LLMPlayer(
        _scripted(
            [_move(["A1", "A2"]), _move(["A3", "A4"]), _move(["A5"], "win")], seen
        ),
        name="Mona",
    )
    snake: LLMPlayer = LLMPlayer(
        _scripted([_move(["E1", "E2"]), _move(["E3", "E4"])]), name="Sly"
    )

    result: GameResult = play_game(mouse, snake)

    assert result.termination is Termination.LINE_COMPLETED
    assert result.winner is Side.MOUSE
    # By its second turn, the mouse was told the snake's first move…
    assert "Your opponent (snake) played E1 E2." in seen[1]
    # …but its own moves are never relayed back to it as opponent moves.
    assert all("opponent (mouse)" not in prompt for prompt in seen)


def test_rules_sent_once_and_fault_feedback_carried_forward() -> None:
    seen: list[str] = []
    player: LLMPlayer = LLMPlayer(
        _scripted([_move(["A1", "A2"], "win"), _move(["B1", "B2"])], seen)
    )

    player.start_game(Side.MOUSE, _SEED)
    player.choose_move()  # game 1
    player.end_game(
        GameResult(
            Termination.PLAYER_FAULT,
            fault=PlayerFaultDetail(
                offender=Side.MOUSE,
                reason=PlayerFaultReason.WRONG_OUTCOME_CLAIM,
                attempted_move=Move.from_labels("A1", "A2"),
                claimed_outcome=TurnOutcome.WIN,
                actual_outcome=TurnOutcome.IN_PLAY,
            ),
        )
    )
    player.start_game(Side.MOUSE, _SEED)  # game 2
    player.choose_move()

    assert RULES_PREAMBLE in seen[0]  # opening rules preamble, on the first turn…
    assert RULES_PREAMBLE not in seen[1]  # …and never again
    assert "claimed win but it was actually in_play" in seen[1]  # the feedback


def test_start_game_announces_the_seed_cell() -> None:
    seen: list[str] = []
    player: LLMPlayer = LLMPlayer(_scripted([_move(["A1", "A2"])], seen))

    player.start_game(Side.MOUSE, Cell.from_label("D4"))
    player.choose_move()

    # The seed is announced per game (the preamble names no cell), so the model
    # can track the board even when the opening is randomized.
    assert "The snake is seeded at D4." in seen[0]


def test_preamble_names_no_specific_seed_cell() -> None:
    # The rules preamble must stay opening-agnostic: it describes the seeding rule
    # but names no cell, since the seed can vary game to game and is given in each
    # game's start message instead.
    assert "B3" not in RULES_PREAMBLE


def test_message_logging_writes_replayable_thread(tmp_path: Path) -> None:
    log_dir: Path = tmp_path / "logs"
    player: LLMPlayer = LLMPlayer(
        _scripted([_move(["A1", "A2"])]), name="bot", log_dir=log_dir
    )
    player.start_game(Side.SNAKE, _SEED)
    player.choose_move()

    log_file: Path = log_dir / "bot-snake.json"
    assert log_file.exists()
    messages: list[ModelMessage] = ModelMessagesTypeAdapter.validate_json(
        log_file.read_bytes()
    )
    assert len(messages) >= 2  # at least the user turn and the model response


# --- Resolving a roster entry to a model, an agent, and a player ---------------


def test_resolve_builtin_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")

    anthropic = resolve_model(PlayerSpec(name="a", provider="anthropic", model="m"), {})
    openai = resolve_model(PlayerSpec(name="o", provider="openai", model="m"), {})
    gemini = resolve_model(PlayerSpec(name="g", provider="gemini", model="m"), {})
    # OpenRouter requires an upstream-provider-prefixed model name.
    router = resolve_model(
        PlayerSpec(name="r", provider="openrouter", model="openai/gpt-4o"), {}
    )

    assert isinstance(anthropic, AnthropicModel)
    assert isinstance(openai, OpenAIResponsesModel)  # Responses API, not Chat
    assert isinstance(gemini, GoogleModel)
    assert isinstance(router, OpenRouterModel)


def test_resolve_custom_provider_uses_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    providers: dict[str, ProviderSpec] = {
        "my-ollama": ProviderSpec(
            name="my-ollama", base_url="http://localhost:11434/v1"
        )
    }
    # A keyless local endpoint resolves without any API key in the environment.
    model = resolve_model(
        PlayerSpec(name="local", provider="my-ollama", model="llama3.3"), providers
    )
    assert isinstance(model, OpenAIChatModel)


def test_resolve_missing_key_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        resolve_model(PlayerSpec(name="a", provider="anthropic", model="m"), {})


def test_resolve_unknown_provider_is_an_error() -> None:
    with pytest.raises(ConfigError, match="unknown provider"):
        resolve_model(PlayerSpec(name="x", provider="nope", model="m"), {})


def test_resolve_agent_output_mode_by_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    # Anthropic cannot combine an output tool with thinking, so its agent must use
    # the provider's native JSON-schema output; other providers keep Pydantic AI's
    # default (auto/tool-based) output that they were validated against.
    anthropic = resolve_agent(PlayerSpec(name="a", provider="anthropic", model="m"), {})
    openai = resolve_agent(PlayerSpec(name="o", provider="openai", model="m"), {})

    assert type(anthropic._output_schema).__name__ == "NativeOutputSchema"
    assert type(openai._output_schema).__name__ != "NativeOutputSchema"


def test_strip_prior_thinking_removes_thinking_keeps_the_rest() -> None:
    # The history processor drops reasoning parts from prior responses — which
    # dominate a long match's re-sent context — while leaving the move text and the
    # user turns untouched, and without mutating the input it was given.
    request: ModelRequest = ModelRequest(parts=[UserPromptPart(content="your turn")])
    response: ModelResponse = ModelResponse(
        parts=[ThinkingPart(content="deep"), TextPart(content="A1 A2")]
    )

    out: list[object] = list(strip_prior_thinking([request, response]))

    # The thinking part is gone from the response, the move text stays…
    assert isinstance(out[1], ModelResponse)
    assert [type(p).__name__ for p in out[1].parts] == ["TextPart"]
    # …the user turn is passed through unchanged, and the original is not mutated.
    assert out[0] is request
    assert [type(p).__name__ for p in response.parts] == ["ThinkingPart", "TextPart"]


def test_strip_prior_thinking_keeps_linked_reasoning_items() -> None:
    # OpenAI's Responses API and Anthropic emit empty-content reasoning parts that
    # carry an id/signature a following item references. These must survive the
    # stripper: removing them saves nothing (no text) and invalidates the re-sent
    # history — OpenAI rejects the dangling reference with HTTP 400. Only Gemini's
    # self-contained thought *text* (no id, no signature) is dropped.
    openai_like: ModelResponse = ModelResponse(
        parts=[ThinkingPart(content="", id="rs_1", signature="sig"), TextPart("A1 A2")]
    )
    anthropic_like: ModelResponse = ModelResponse(
        parts=[ThinkingPart(content="", signature="sig"), TextPart("B1 B2")]
    )
    gemini_like: ModelResponse = ModelResponse(
        parts=[ThinkingPart(content="long summary"), TextPart("C1 C2")]
    )

    out: list[object] = list(
        strip_prior_thinking([openai_like, anthropic_like, gemini_like])
    )

    # The linked/signed reasoning items are kept intact…
    assert isinstance(out[0], ModelResponse)
    assert [type(p).__name__ for p in out[0].parts] == ["ThinkingPart", "TextPart"]
    assert isinstance(out[1], ModelResponse)
    assert [type(p).__name__ for p in out[1].parts] == ["ThinkingPart", "TextPart"]
    # …while the self-contained thought text is the only reasoning dropped.
    assert isinstance(out[2], ModelResponse)
    assert [type(p).__name__ for p in out[2].parts] == ["TextPart"]


def test_resolve_agent_attaches_thinking_stripper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every resolved agent, on either output-mode branch, carries the history
    # processor so a long match's context does not balloon with re-sent reasoning.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    for provider in ("anthropic", "openai"):
        agent = resolve_agent(PlayerSpec(name="p", provider=provider, model="m"), {})
        processors = [
            c.processor
            for c in agent.root_capability.capabilities
            if isinstance(c, ProcessHistory)
        ]
        assert strip_prior_thinking in processors


def test_from_roster_builds_named_player(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    roster: Roster = Roster(
        players={"opus": PlayerSpec(name="opus", provider="anthropic", model="m")},
        providers={},
    )
    player: LLMPlayer = LLMPlayer.from_roster("opus", roster)
    assert isinstance(player, LLMPlayer)
    assert player.name == "opus"


def test_from_roster_unknown_name_is_an_error() -> None:
    roster: Roster = Roster(players={}, providers={})
    with pytest.raises(ConfigError, match="no player named"):
        LLMPlayer.from_roster("ghost", roster)
