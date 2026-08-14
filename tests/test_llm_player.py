"""Tests for the LLM player, driven by Pydantic AI's :class:`FunctionModel`.

No network is used: a scripted model returns a fixed sequence of structured
``LLMMove`` responses, so we can exercise move parsing, fault mapping, the
cross-game message thread (rules once, opponent moves relayed, feedback carried
forward), and message logging deterministically. The player uses
``NativeOutput``, so a response is a plain JSON ``TextPart`` (no output tool).
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pydantic_ai import Agent, ModelMessagesTypeAdapter, NativeOutput
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
from pydantic_ai.models.function import AgentInfo, FunctionModel

from snakes_and_mice import (
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
from snakes_and_mice.players import LLMPlayer
from snakes_and_mice.players.llm import LLMMove, ModelRequestError
from snakes_and_mice.players.prompts import RULES_PREAMBLE


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
    player.start_game(Side.MOUSE)

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
    player.start_game(Side.MOUSE)

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
    player.start_game(Side.MOUSE)

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
    player.start_game(Side.MOUSE)

    with pytest.raises(ModelRequestError):
        player.choose_move()


def test_user_error_aborts_the_run() -> None:
    # A client-side capability mismatch (raised before any request) is a config
    # problem, not a game outcome: it aborts the run with a clear message.
    player: LLMPlayer = LLMPlayer(
        _failing_agent(UserError("model does not support native output")), name="bot"
    )
    player.start_game(Side.MOUSE)

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
    player.start_game(Side.MOUSE)

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
    player.start_game(Side.MOUSE)

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
    player.start_game(Side.MOUSE)
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
    player.start_game(Side.MOUSE)
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
    player.start_game(Side.MOUSE)
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
    player.start_game(Side.MOUSE)

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
    player.start_game(Side.MOUSE)

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
    player.start_game(Side.MOUSE)

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
    from pydantic_ai.capabilities import ProcessHistory

    from snakes_and_mice.config import strip_prior_thinking

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
    player.start_game(Side.MOUSE)
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

    player.start_game(Side.MOUSE)
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
    player.start_game(Side.MOUSE)  # game 2
    player.choose_move()

    assert RULES_PREAMBLE in seen[0]  # opening rules preamble, on the first turn…
    assert RULES_PREAMBLE not in seen[1]  # …and never again
    assert "claimed win but it was actually in_play" in seen[1]  # the feedback


def test_message_logging_writes_replayable_thread(tmp_path: Path) -> None:
    log_dir: Path = tmp_path / "logs"
    player: LLMPlayer = LLMPlayer(
        _scripted([_move(["A1", "A2"])]), name="bot", log_dir=log_dir
    )
    player.start_game(Side.SNAKE)
    player.choose_move()

    log_file: Path = log_dir / "bot-snake.json"
    assert log_file.exists()
    messages: list[ModelMessage] = ModelMessagesTypeAdapter.validate_json(
        log_file.read_bytes()
    )
    assert len(messages) >= 2  # at least the user turn and the model response
