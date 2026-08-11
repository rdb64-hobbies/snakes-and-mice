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

import pytest
from pydantic_ai import Agent, ModelMessagesTypeAdapter, NativeOutput
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ThinkingPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from snakes_and_mice import (
    GameResult,
    Move,
    MoveUnavailable,
    PlayerFaultDetail,
    PlayerFaultReason,
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
