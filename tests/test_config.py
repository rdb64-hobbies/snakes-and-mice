"""Tests for the roster/provider config loader and model resolution.

These exercise the YAML parsing and the ``(provider, model)`` → Pydantic AI
model mapping without any network: constructing a provider/model just stores the
key and endpoint, so we can assert the resolved type and that a missing key or
unknown provider is reported as a :class:`ConfigError`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    UserPromptPart,
)
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.models.openrouter import OpenRouterModel

from snakes_and_mice.config import (
    ConfigError,
    PlayerSpec,
    ProviderSpec,
    Roster,
    load_environment,
    load_roster,
    make_llm_player,
    resolve_agent,
    resolve_model,
    strip_prior_thinking,
)
from snakes_and_mice.players import LLMPlayer

_PLAYERS_YAML: str = """\
players:
  - name: opus
    provider: anthropic
    model: claude-opus-4-8
  - name: llama-local
    provider: my-ollama
    model: llama3.3
"""

_PROVIDERS_YAML: str = """\
providers:
  - name: my-ollama
    base_url: http://localhost:11434/v1
"""


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_load_roster_indexes_players_and_providers(tmp_path: Path) -> None:
    players: Path = _write(tmp_path / "players.yaml", _PLAYERS_YAML)
    providers: Path = _write(tmp_path / "providers.yaml", _PROVIDERS_YAML)

    roster: Roster = load_roster(players, providers)

    assert set(roster.players) == {"opus", "llama-local"}
    assert roster.players["opus"].provider == "anthropic"
    assert roster.players["opus"].model == "claude-opus-4-8"
    assert set(roster.providers) == {"my-ollama"}
    assert roster.providers["my-ollama"].base_url == "http://localhost:11434/v1"
    assert roster.providers["my-ollama"].api_key_env is None


def test_missing_providers_file_is_allowed(tmp_path: Path) -> None:
    players: Path = _write(tmp_path / "players.yaml", _PLAYERS_YAML)

    roster: Roster = load_roster(players, tmp_path / "providers.yaml")

    assert roster.providers == {}


def test_missing_players_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_roster(tmp_path / "players.yaml", tmp_path / "providers.yaml")


def test_malformed_players_file_is_an_error(tmp_path: Path) -> None:
    bad: Path = _write(tmp_path / "players.yaml", "players:\n  - name: oops\n")
    with pytest.raises(ConfigError, match="could not parse"):
        load_roster(bad, tmp_path / "providers.yaml")


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


def test_make_llm_player_builds_named_player(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    roster: Roster = Roster(
        players={"opus": PlayerSpec(name="opus", provider="anthropic", model="m")},
        providers={},
    )
    player: LLMPlayer = make_llm_player("opus", roster)
    assert isinstance(player, LLMPlayer)
    assert player.name == "opus"


def test_make_llm_player_unknown_name_is_an_error() -> None:
    roster: Roster = Roster(players={}, providers={})
    with pytest.raises(ConfigError, match="no player named"):
        make_llm_player("ghost", roster)


def test_load_environment_reads_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SAM_TEST_ENV_KEY", raising=False)
    env: Path = _write(tmp_path / ".env", "SAM_TEST_ENV_KEY=hello\n")

    load_environment(env)

    assert os.environ["SAM_TEST_ENV_KEY"] == "hello"
    monkeypatch.delenv("SAM_TEST_ENV_KEY", raising=False)
