"""Tests for the roster/provider config loader.

These exercise the configuration *files* and nothing else: parsing
``players.yaml`` / ``providers.yaml`` into a :class:`Roster`, tolerating a
missing providers file, reporting a missing or malformed roster as a
:class:`ConfigError`, and loading ``.env`` into the environment. Resolving a
:class:`PlayerSpec` to a model, an agent, and a player belongs to the LLM player
and is tested in ``test_llm_player.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from snakes_and_mice.config import (
    ConfigError,
    Roster,
    load_environment,
    load_roster,
)

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


def test_load_environment_reads_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SAM_TEST_ENV_KEY", raising=False)
    env: Path = _write(tmp_path / ".env", "SAM_TEST_ENV_KEY=hello\n")

    load_environment(env)

    assert os.environ["SAM_TEST_ENV_KEY"] == "hello"
    monkeypatch.delenv("SAM_TEST_ENV_KEY", raising=False)


def test_provider_output_mode_defaults_to_tool_and_parses_native(
    tmp_path: Path,
) -> None:
    # output_mode is optional, so every existing providers.yaml keeps the tool-based
    # output it was validated against; "native" is opt-in per endpoint (§4).
    providers_path: Path = tmp_path / "providers.yaml"
    providers_path.write_text(
        "providers:\n"
        "  - name: plain\n"
        "    base_url: http://h/v1\n"
        "  - name: declared\n"
        "    base_url: http://h/v1\n"
        "    output_mode: native\n"
    )
    players_path: Path = tmp_path / "players.yaml"
    players_path.write_text(
        "players:\n  - name: p\n    provider: plain\n    model: m\n"
    )

    roster: Roster = load_roster(players_path, providers_path)

    assert roster.providers["plain"].output_mode == "tool"
    assert roster.providers["declared"].output_mode == "native"


def test_unknown_output_mode_is_an_error(tmp_path: Path) -> None:
    providers_path: Path = tmp_path / "providers.yaml"
    providers_path.write_text(
        "providers:\n"
        "  - name: bad\n"
        "    base_url: http://h/v1\n"
        "    output_mode: grammar\n"
    )
    players_path: Path = tmp_path / "players.yaml"
    players_path.write_text(
        "players:\n  - name: p\n    provider: bad\n    model: m\n"
    )

    with pytest.raises(ConfigError, match="output_mode"):
        load_roster(players_path, providers_path)
