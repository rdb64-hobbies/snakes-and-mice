"""Does an OpenAI-compatible endpoint stop generating once the move is written?

The LLM player asks for its move as a structured object, and an OpenAI-compatible
endpoint can be asked for one two ways (§4, "Structured output"): through an output
*tool*, which Pydantic AI forces with `tool_choice`, or through the model's *native*
JSON-schema `response_format`. On a modern vLLM the two take very different paths —
a forced tool choice is compiled into a decoding grammar by the server's
`--tool-call-parser`, while a response format is compiled from the schema itself.

That distinction bites when the served model has no tool-call parser of its own.
vLLM implements a tool-call format per model family, so a model outside that set is
served with a parser borrowed from another family; on vLLM 0.17 the parser only
*scraped* the finished text, which was harmless, but from the structural-tag refactor
onward it also *constrains generation*. A model held to a foreign tool-call syntax
never reaches a state where stopping is allowed: it writes one correct call, is
denied its end token, writes it again, and runs to `max_tokens`. Measured on
`NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` served with Qwen's parser, every single
turn came back `finish_reason: length` carrying ~190 byte-identical tool calls.

This tool asks one endpoint for one move, both ways, and reports how each ended. A
healthy path finishes on `tool_calls` or `stop` with one call; the failure is
unmistakable — `length`, hundreds of calls, output tokens pinned at the cap. Where
`native` terminates and `tool` does not, `output_mode: native` on that provider in
`providers.yaml` is the fix.

The answer is a property of the deployment — the server version, and whether it has a
parser for the served model — not of this project, so re-run it whenever the container
or the model changes.

    uv run python tools/probe_tool_termination.py PROVIDER

`PROVIDER` names an entry in `providers.yaml`; with no argument the known ones are
listed.
"""

from __future__ import annotations

import sys
from typing import Any, Final

import httpx

from snakes_and_mice.config import ProviderSpec, load_roster

# A cap far below the player's own 16384 (§4): a looping model reaches it in seconds,
# and a healthy one never comes close, so the probe stays quick either way.
MAX_TOKENS: Final[int] = 2048

# The move schema the player asks for, spelled out rather than derived from LLMMove
# so the probe sends exactly this and stays readable as a wire-level record.
MOVE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "move_rationale": {"type": "string"},
        "cells": {"type": "array", "items": {"type": "string"}},
        "claimed_outcome": {
            "type": "string",
            "enum": ["in_play", "win", "cats_game"],
        },
    },
    "required": ["move_rationale", "cells", "claimed_outcome"],
    "additionalProperties": False,
}

PROMPT: Final[str] = (
    "You are playing Snakes and Mice on a 5x5 board, rows A-E and columns 1-5. "
    "You are the mouse and move first. The snake is seeded on B3; every other "
    "cell is empty. Place two of your pieces on two different empty cells. "
    "Report the move as the structured object you have been given: a short "
    "move_rationale, the two cells, and claimed_outcome (in_play, win, or "
    "cats_game)."
)


def _tool_request(model: str) -> dict[str, Any]:
    """The output-tool form: one tool, forced by name. This mirrors what Pydantic AI
    sends for a single output tool — not the bare `"required"` — so the probe
    exercises the same server path the player does."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "final_result",
                    "description": "The move you have chosen.",
                    "parameters": MOVE_SCHEMA,
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "final_result"}},
    }


def _native_request(model: str) -> dict[str, Any]:
    """The native form: the same schema as a JSON-schema response format, with no
    tools at all, so the server's tool-call parser is never consulted."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "final_result",
                "schema": MOVE_SCHEMA,
                "strict": True,
            },
        },
    }


_VARIANTS: Final[tuple[tuple[str, str], ...]] = (
    ("tool", "output tool, forced by name"),
    ("native", "json_schema response format"),
)


def _resolve_provider(name: str | None) -> ProviderSpec:
    """The named custom provider, or exit listing the known ones."""
    providers: dict[str, ProviderSpec] = load_roster().providers
    spec: ProviderSpec | None = providers.get(name) if name else None
    if spec is None:
        known: str = ", ".join(sorted(providers)) or "(none in providers.yaml)"
        raise SystemExit(
            f"usage: probe_tool_termination.py PROVIDER\nknown providers: {known}"
        )
    return spec


def _served_model(client: httpx.Client, base_url: str) -> str:
    """The model the endpoint is currently serving."""
    try:
        listing: dict[str, Any] = client.get(f"{base_url}/models").json()
        return str(listing["data"][0]["id"])
    except httpx.HTTPError as exc:
        raise SystemExit(
            f"could not reach {base_url} ({type(exc).__name__}) — is the endpoint up?"
        ) from exc
    except (KeyError, IndexError, ValueError) as exc:
        raise SystemExit(f"{base_url} served no model yet — still starting up?") from exc


def _attempt(
    client: httpx.Client, base_url: str, body: dict[str, Any]
) -> tuple[str, int, int, str]:
    """Send one request and report how it ended: the finish reason, how many tool
    calls came back, the output-token count, and a one-line sample of the content."""
    response: httpx.Response = client.post(f"{base_url}/chat/completions", json=body)
    if response.is_error:
        return f"HTTP {response.status_code}", 0, 0, response.text[:200]
    payload: dict[str, Any] = response.json()
    choice: dict[str, Any] = payload["choices"][0]
    message: dict[str, Any] = choice["message"]
    calls: list[dict[str, Any]] = message.get("tool_calls") or []
    output_tokens: int = int(payload.get("usage", {}).get("completion_tokens", 0))
    if calls:
        sample: str = str(calls[0]["function"].get("arguments", ""))
    else:
        sample = str(message.get("content") or "")
    return str(choice.get("finish_reason", "?")), len(calls), output_tokens, sample


def _verdict(finish: str, calls: int, output_tokens: int) -> str:
    """What one attempt's ending means. Only a run to the token cap is a true
    failure to terminate; anything else is reported as-is for the reader to judge."""
    if finish.startswith("HTTP"):
        return "REJECTED — the server would not take this form"
    if finish == "length" and output_tokens >= MAX_TOKENS:
        repeats: str = f", {calls} tool calls" if calls > 1 else ""
        return f"DID NOT TERMINATE — ran to the {MAX_TOKENS}-token cap{repeats}"
    if calls > 1:
        return f"terminated, but emitted {calls} calls where one was asked for"
    return "OK — terminated on its own"


def main() -> None:
    spec: ProviderSpec = _resolve_provider(sys.argv[1] if len(sys.argv) > 1 else None)
    base_url: str = spec.base_url.rstrip("/")
    root: str = base_url.removesuffix("/v1")
    with httpx.Client(timeout=300.0) as client:
        model: str = _served_model(client, base_url)
        try:
            version: str = client.get(f"{root}/version").json().get("version", "?")
        except httpx.HTTPError:
            version = "?"
        print(f"server      {base_url}")
        print(f"vllm        {version}")
        print(f"model       {model}")
        print(f"output_mode {spec.output_mode}  (as configured in providers.yaml)\n")

        requests: dict[str, dict[str, Any]] = {
            "tool": _tool_request(model),
            "native": _native_request(model),
        }
        endings: dict[str, tuple[str, int, int, str]] = {}
        for label, description in _VARIANTS:
            finish, calls, output_tokens, sample = _attempt(
                client, base_url, requests[label]
            )
            endings[label] = (finish, calls, output_tokens, sample)
            print(f"  {label:7s} {description}")
            print(
                f"          finish_reason={finish}  tool_calls={calls}  "
                f"output_tokens={output_tokens}"
            )
            print(f"          {_verdict(finish, calls, output_tokens)}")
            print(f"          returned: {sample[:120]}\n")

        tool_ok: bool = "DID NOT" not in _verdict(*endings["tool"][:3])
        native_ok: bool = "DID NOT" not in _verdict(*endings["native"][:3])
        if tool_ok and native_ok:
            print("Both forms work; keep output_mode: tool.")
        elif native_ok:
            print(
                "Only the native form terminates — set output_mode: native on "
                f"provider {spec.name!r} in providers.yaml, and serve this model "
                "with no --tool-call-parser."
            )
        elif tool_ok:
            print("Only the tool form works; keep output_mode: tool.")
        else:
            print(
                "Neither form terminates — the cause is not the tool-call parser. "
                "Suspect the served model, its quantization, or the container."
            )


if __name__ == "__main__":
    main()
