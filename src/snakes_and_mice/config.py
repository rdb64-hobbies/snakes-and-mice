"""Loading the LLM roster and resolving it to Pydantic AI models.

This is the "config loader" of §11/§14: it parses the three configuration
sources — ``players.yaml`` (the roster), ``providers.yaml`` (custom
OpenAI-compatible endpoints only), and ``.env`` (API keys) — into typed
specifications and resolves each ``(provider, model)`` pair to a Pydantic AI
model, from which an :class:`LLMPlayer` is built. Keeping all of this out of the
player means the player itself never touches YAML and API keys live only in the
environment, never in the tracked config files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

from .faults import SnakesAndMiceError
from .players.llm import DEFAULT_THINKING, LLMPlayer, ThinkingLevel

DEFAULT_PLAYERS_PATH: Path = Path("players.yaml")
DEFAULT_PROVIDERS_PATH: Path = Path("providers.yaml")
DEFAULT_ENV_PATH: Path = Path(".env")

# Built-in providers Pydantic AI supports directly, each with the environment
# variable that holds its key (§11, "Model selection"). Custom OpenAI-compatible
# providers are declared in providers.yaml instead and are not listed here.
_BUILTIN_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


class ConfigError(SnakesAndMiceError):
    """A roster or provider configuration could not be loaded or resolved."""


class PlayerSpec(BaseModel):
    """One roster entry: a free-form name, a provider, and a model name."""

    name: str
    provider: str
    model: str


class ProviderSpec(BaseModel):
    """One custom OpenAI-compatible endpoint: a name, a base URL, and optionally
    the environment variable holding its key (a local endpoint may need none)."""

    name: str
    base_url: str
    api_key_env: str | None = None


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
        return OpenAIChatModel(
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


def make_llm_player(
    name: str,
    roster: Roster,
    *,
    thinking: ThinkingLevel = DEFAULT_THINKING,
    log_dir: Path | None = None,
) -> LLMPlayer:
    """Build the :class:`LLMPlayer` for roster entry ``name``.

    Raises :class:`ConfigError` if no such player is in the roster.
    """
    spec: PlayerSpec | None = roster.players.get(name)
    if spec is None:
        raise ConfigError(
            f"no player named {name!r} in the roster; "
            f"available: {', '.join(sorted(roster.players)) or '(none)'}"
        )
    model: Model = resolve_model(spec, roster.providers)
    return LLMPlayer(model, name=spec.name, thinking=thinking, log_dir=log_dir)


def _require_key(provider: str) -> str:
    """The API key for a built-in ``provider``, or a :class:`ConfigError`."""
    return _require_key_env(_BUILTIN_KEY_ENV[provider])


def _require_key_env(env_var: str) -> str:
    value: str | None = os.environ.get(env_var)
    if not value:
        raise ConfigError(
            f"environment variable {env_var} is not set — add it to your .env or "
            f"export it before running"
        )
    return value


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
