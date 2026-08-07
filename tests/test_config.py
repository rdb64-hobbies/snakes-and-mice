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
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openrouter import OpenRouterModel

from snakes_and_mice.config import (
    ConfigError,
    PlayerSpec,
    ProviderSpec,
    Roster,
    load_environment,
    load_roster,
    make_llm_player,
    resolve_model,
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
    assert isinstance(openai, OpenAIChatModel)
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
