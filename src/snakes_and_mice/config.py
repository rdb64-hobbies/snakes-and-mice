"""Loading the LLM roster and provider settings from the configuration files.

This is the "config loader" of §4/§8: it parses the three configuration
sources — ``players.yaml`` (the roster), ``providers.yaml`` (custom
OpenAI-compatible endpoints only), and ``.env`` (API keys) — into typed
specifications, so the rest of the project sees a :class:`Roster` of named
:class:`PlayerSpec` / :class:`ProviderSpec` entries instead of YAML.

The module deliberately stops there: it knows about *files*, not about models.
Turning a :class:`PlayerSpec` into a Pydantic AI model, an agent, and an
:class:`~snakes_and_mice.players.llm.LLMPlayer` is the player's own business
(:meth:`~snakes_and_mice.players.llm.LLMPlayer.from_roster`) — so nothing here
imports ``pydantic_ai``, and the player never touches YAML. API keys are read
from the environment at that point too; this module only *loads* ``.env`` into
the environment and records which variable a custom provider uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from .faults import SnakesAndMiceError

DEFAULT_PLAYERS_PATH: Path = Path("players.yaml")
DEFAULT_PROVIDERS_PATH: Path = Path("providers.yaml")
DEFAULT_ENV_PATH: Path = Path(".env")


class ConfigError(SnakesAndMiceError):
    """A roster or provider configuration could not be loaded or resolved."""


class PlayerSpec(BaseModel):
    """One roster entry: a free-form name, a provider, and a model name."""

    name: str
    provider: str
    model: str


OutputMode = Literal["tool", "native"]
"""How an endpoint is asked for the structured move (§4, "Structured output"):
``tool`` is Pydantic AI's default output tool, ``native`` the model's own
JSON-schema response format."""


class ProviderSpec(BaseModel):
    """One custom OpenAI-compatible endpoint: a name, a base URL, optionally the
    environment variable holding its key (a local endpoint may need none), and how
    it is asked for structured output.
    """

    name: str
    base_url: str
    api_key_env: str | None = None
    output_mode: OutputMode = "tool"


class _PlayersFile(BaseModel):
    players: list[PlayerSpec]


class _ProvidersFile(BaseModel):
    providers: list[ProviderSpec] = []


@dataclass(frozen=True)
class Roster:
    """The resolved configuration: named players and named custom providers."""

    players: dict[str, PlayerSpec]
    providers: dict[str, ProviderSpec]


def load_environment(env_path: Path = DEFAULT_ENV_PATH) -> None:
    """Load API keys from a ``.env`` file into the environment, if present.

    A missing file is not an error: the keys may already be exported in the
    environment. Existing environment variables are never overwritten.
    """
    load_dotenv(dotenv_path=env_path if env_path.exists() else None, override=False)


def load_roster(
    players_path: Path = DEFAULT_PLAYERS_PATH,
    providers_path: Path = DEFAULT_PROVIDERS_PATH,
) -> Roster:
    """Parse ``players.yaml`` and (if present) ``providers.yaml`` into a roster.

    ``providers.yaml`` is optional — it is needed only for custom endpoints — so
    a missing file yields an empty provider set. Raises :class:`ConfigError` for a
    missing roster file or malformed content.
    """
    players_file: _PlayersFile = _parse_file(players_path, _PlayersFile, required=True)
    providers_file: _ProvidersFile = _parse_file(
        providers_path, _ProvidersFile, required=False
    )
    return Roster(
        players={spec.name: spec for spec in players_file.players},
        providers={spec.name: spec for spec in providers_file.providers},
    )


def _parse_file[T: BaseModel](path: Path, schema: type[T], *, required: bool) -> T:
    """Load ``path`` as YAML and validate it against ``schema``.

    A missing optional file yields an empty instance; a missing required file, or
    any malformed content, raises :class:`ConfigError`.
    """
    if not path.exists():
        if required:
            raise ConfigError(
                f"configuration file {path} not found — copy the tracked "
                f"{path.stem}.example{path.suffix} template and edit it"
            )
        return schema()
    try:
        data: object = yaml.safe_load(path.read_text()) or {}
        return schema.model_validate(data)
    except (yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"could not parse {path}: {exc}") from exc
