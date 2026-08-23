"""Does an OpenAI-compatible endpoint tokenize the reasoning we send back to it?

The LLM player keeps one thread across a match, so every prior turn is re-sent on
each request — including, for a reasoning model, that turn's thinking. Pydantic AI
returns it to an OpenAI-compatible endpoint as a `reasoning` field on the assistant
message, which is **not** part of the chat-completions schema. Whether it reaches the
prompt is therefore decided by the served model's chat template, per model build, and
the project cannot detect which case it is in (§4, "Pruning re-sent reasoning"). This
tool answers it for one endpoint by measuring.

It sends the same short conversation three ways — prior reasoning carried in
`reasoning`, in `reasoning_content`, and omitted — and compares the resulting prompt
length. A difference is the template rendering that field; zero difference means the
reasoning costs no context and nothing against `max_model_len`, so `--prune-thinking`
would buy nothing there.

Two endpoints measured this way differed purely by model: a vLLM-served `qwen3.8`
renders `reasoning` at ~2.9 chars per input token (and ignores `reasoning_content`),
while `nemotron-3-super` ignores both. Re-run it whenever a model or container
changes — the answer is a property of the deployment, not of this project.

vLLM's `/tokenize` does the work with no generation, so the model need only be loaded.
Endpoints without it (ollama) fall back to a one-token completion, comparing the
`prompt_tokens` the server reports; that measures the same quantity.

    uv run python tools/probe_reasoning_field.py PROVIDER

`PROVIDER` names an entry in `providers.yaml`; with no argument the known ones are
listed.
"""

from __future__ import annotations

import sys
from typing import Any, Final

import httpx

from snakes_and_mice.config import ProviderSpec, load_roster

# Long enough that any tokenization stands far above the noise of the surrounding
# messages, and distinctive enough to find verbatim in a rendered prompt.
_NEEDLE: Final[str] = "Column 3 is dead because the snake sits on E3"
REASONING: Final[str] = (
    f"The mouse holds C3 and B5. {_NEEDLE}. Row B is still live: B5 is mine and "
    f"B1, B2, B3, B4 are empty. "
) * 40

# Fields a server might carry prior reasoning in, plus the control case.
_VARIANTS: Final[tuple[tuple[str, str | None], ...]] = (
    ("reasoning", "reasoning"),
    ("reasoning_content", "reasoning_content"),
    ("omitted", None),
)


def _conversation(field: str | None) -> list[dict[str, Any]]:
    """The two-turn thread the LLM player would re-send, with prior reasoning
    carried in ``field`` (or dropped entirely when ``None``). The assistant turn
    mirrors what Pydantic AI sends: no content, one structured-output tool call."""
    assistant: dict[str, Any] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "final_result",
                    "arguments": '{"move_rationale":"take the center",'
                    '"cells":["C3","B5"],"claimed_outcome":"in_play"}',
                },
            }
        ],
    }
    if field is not None:
        assistant[field] = REASONING
    return [
        {"role": "user", "content": "You are playing Snakes and Mice. Your turn."},
        assistant,
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        {"role": "user", "content": "Your opponent played E3 E4. Your turn."},
    ]


def _resolve_provider(name: str | None) -> str:
    """The server root of the named custom provider, or exit listing the known ones.

    ``providers.yaml`` holds the OpenAI-compatible base URL, which ends in ``/v1``;
    ``/tokenize`` and ``/version`` sit beside that prefix rather than under it, so
    the root is what the probe needs.
    """
    providers: dict[str, ProviderSpec] = load_roster().providers
    spec: ProviderSpec | None = providers.get(name) if name else None
    if spec is None:
        known: str = ", ".join(sorted(providers)) or "(none in providers.yaml)"
        raise SystemExit(
            f"usage: probe_reasoning_field.py PROVIDER\nknown providers: {known}"
        )
    return spec.base_url.rstrip("/").removesuffix("/v1")


def _prompt_tokens(
    client: httpx.Client, root: str, model: str, field: str | None
) -> tuple[int, str]:
    """The prompt length for one variant, and the rendered prompt when the server
    will report it. Prefers ``/tokenize`` (no generation); falls back to a one-token
    completion, whose ``prompt_tokens`` measures the same thing."""
    body: dict[str, Any] = {
        "model": model,
        "messages": _conversation(field),
        "return_token_strs": True,
    }
    response: httpx.Response = client.post(f"{root}/tokenize", json=body)
    if response.status_code == 404:
        body = {"model": model, "messages": _conversation(field), "max_tokens": 1}
        response = client.post(f"{root}/v1/chat/completions", json=body)
        response.raise_for_status()
        return int(response.json()["usage"]["prompt_tokens"]), ""
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    # The token strings joined back together are the prompt the template produced,
    # so the reasoning can be looked for directly rather than inferred from a count.
    strs: list[str] | None = payload.get("token_strs")
    rendered: str = "".join(strs).replace("Ġ", " ") if strs else ""
    return int(payload["count"]), rendered


def main() -> None:
    root: str = _resolve_provider(sys.argv[1] if len(sys.argv) > 1 else None)
    with httpx.Client(timeout=60.0) as client:
        try:
            listing: dict[str, Any] = client.get(f"{root}/v1/models").json()
            served: dict[str, Any] = listing["data"][0]
        except httpx.HTTPError as exc:
            raise SystemExit(
                f"could not reach {root} ({type(exc).__name__}) — is the endpoint up?"
            ) from exc
        except (KeyError, IndexError, ValueError) as exc:
            # A server still loading its model answers /v1/models without a model.
            raise SystemExit(
                f"{root} served no model yet — still starting up?"
            ) from exc
        model: str = served["id"]
        # Containers differ per model, so record which one answered.
        try:
            version: str = client.get(f"{root}/version").json().get("version", "?")
        except httpx.HTTPError:
            version = "?"
        print(f"server   {root}")
        print(f"vllm     {version}")
        print(f"model    {model}")
        print(f"max_len  {served.get('max_model_len', '?')}")
        print(f"payload  {len(REASONING):,} chars of reasoning\n")

        counts: dict[str, int] = {}
        rendered: dict[str, str] = {}
        for label, field in _VARIANTS:
            counts[label], rendered[label] = _prompt_tokens(
                client, root, model, field
            )
            print(f"  {label:20s} prompt = {counts[label]:6,d} tokens")

        print()
        base: int = counts["omitted"]
        for label, _ in _VARIANTS[:-1]:
            delta: int = counts[label] - base
            verdict: str = (
                "DROPPED — never reaches the prompt"
                if delta == 0
                else f"RENDERED — {len(REASONING) / delta:.1f} chars/token"
            )
            print(f"  {label:20s} Δ = {delta:+6,d}   {verdict}")

        if any(rendered.values()):
            print()
            for label, _ in _VARIANTS:
                seen: bool = _NEEDLE in rendered[label]
                print(
                    f"  {label:20s} reasoning text in rendered prompt: "
                    f"{'YES' if seen else 'no'}"
                )
            if rendered["reasoning"]:
                head: str = rendered["reasoning"].replace("Ċ", "\n")[:400]
                print(f"\n  rendered prompt, first 400 chars:\n{head}")


if __name__ == "__main__":
    main()
