"""Do two endpoints serving the same model draw from the same distribution?

A tournament may be split across machines — one model on a fast box, another on a slow
one — but only if a model plays the same wherever it runs. Win rates cannot establish
that: at the game counts a slow machine can afford the comparison has almost no power,
and randomized openings (§5) add noise on top.

The obvious alternative — replay one fixed conversation to both at temperature 0 and
check they answer identically — does not work either, and the reason is worth stating
because it looks like it should. **vLLM is not reproducible at temperature 0.** Three
identical requests to one endpoint produced three different chains of reasoning and two
different moves, one of them a single-cell move. `temperature: 0` makes each step
greedy; it does not make the arithmetic reproducible, and with CUDA graphs, chunked
prefill and a mixture-of-experts model the reduction order varies between otherwise
identical requests. Tiny logit differences then compound over thousands of reasoning
tokens. A single sample is noise even on one machine.

So sample. This asks each endpoint the same question many times and compares the
resulting *distributions* — which moves came back and how often, and how long the
reasoning ran. Crucially it reports each endpoint's own spread first: a difference
between machines only means something if it is larger than the difference a machine has
with itself, and that noise floor is what a one-shot comparison hides.

    uv run python tools/compare_endpoints.py PROVIDER_A PROVIDER_B [-n SAMPLES] [-p POSITION]

The conversations are built by driving a real :class:`LLMPlayer` over a scripted game,
so the prompts are byte-identical to what a match would send — there is no second copy
of the prompt format here to drift out of step with the player.

Read the result against the noise floor it prints. Distributions that overlap heavily,
with reasoning lengths differing by less than each machine varies from itself, are
consistent with the same model behaving the same way; you can pool results. A modal
move that differs with little overlap, or reasoning lengths separated by more than the
within-machine spread, is evidence the machines are not equivalent — keep each model's
games on one of them. Sampling error is large at small `-n`, so treat a close call as
"not established" rather than "the same".
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Final

import httpx
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from snakes_and_mice.core import Cell, Move, Side
from snakes_and_mice.players.llm import LLMMove
from snakes_and_mice.roster import ProviderSpec, load_roster

# A scripted game deep enough to offer positions at several stages, chosen so no move
# completes a line — the point is a realistic mid-game prompt, not a finished game.
SEED: Final[str] = "B3"
OUR_MOVES: Final[tuple[tuple[str, str], ...]] = (
    ("A1", "A2"),
    ("C1", "C2"),
    ("D1", "D2"),
    ("E1", "E2"),
)
THEIR_MOVES: Final[tuple[tuple[str, str], ...]] = (
    ("A4", "A5"),
    ("C4", "C5"),
    ("D4", "D5"),
    ("E4", "E5"),
)

# Well above a normal answer, low enough that one runaway does not stall the run.
MAX_TOKENS: Final[int] = 8192

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


@dataclass(frozen=True)
class Sample:
    """One answer: the cells as a canonical label, and how long the reasoning ran."""

    move: str
    reasoning_chars: int


@dataclass(frozen=True)
class Summary:
    """What one endpoint did over its samples, including its own variability.

    The samples are kept, not just their tally, because the noise floor is measured by
    splitting them: how much an endpoint agrees with *itself* is the only scale on
    which a difference between endpoints can be judged.
    """

    samples: list[Sample]
    failures: int

    @property
    def moves(self) -> Counter[str]:
        return Counter(s.move for s in self.samples)

    @property
    def reasoning(self) -> list[int]:
        return [s.reasoning_chars for s in self.samples]

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def self_overlap(self) -> float:
        """Overlap of the first half of the samples with the second — the noise floor.

        Zero when there are too few samples to split, which the verdict reads as "no
        baseline" rather than "no agreement".
        """
        if self.n < 4:
            return 0.0
        half: int = self.n // 2
        return overlap(
            Counter(s.move for s in self.samples[:half]),
            Counter(s.move for s in self.samples[half:]),
        )

    @property
    def modal(self) -> tuple[str, int]:
        return self.moves.most_common(1)[0] if self.moves else ("—", 0)

    @property
    def mean_reasoning(self) -> float:
        return statistics.fmean(self.reasoning) if self.reasoning else 0.0

    @property
    def reasoning_spread(self) -> float:
        """Standard deviation, the machine's disagreement with itself."""
        return statistics.stdev(self.reasoning) if len(self.reasoning) > 1 else 0.0


def summarize(samples: list[Sample], failures: int = 0) -> Summary:
    """Collect samples into a distribution."""
    return Summary(samples=list(samples), failures=failures)


def overlap(a: Counter[str], b: Counter[str]) -> float:
    """Fraction of probability mass the two distributions share (0 to 1).

    The sum of per-move minimum frequencies — 1.0 when the two agree exactly on how
    often every move appears, 0.0 when they have no move in common. It needs no
    distributional assumptions, which suits a handful of samples.
    """
    n_a, n_b = sum(a.values()), sum(b.values())
    if not n_a or not n_b:
        return 0.0
    return sum(min(a[m] / n_a, b[m] / n_b) for m in set(a) | set(b))


def cross_overlap(a: Summary, b: Summary) -> float:
    """Overlap between two endpoints, measured on half-samples.

    The comparison is against :attr:`Summary.self_overlap`, which necessarily splits
    an endpoint in half — and smaller samples overlap less by construction. Measuring
    the cross-endpoint overlap on full samples would therefore flatter it against that
    baseline, so both are computed at the same sample size, averaged over the four
    ways the halves pair up.
    """
    if a.n < 4 or b.n < 4:
        return overlap(a.moves, b.moves)
    a_half, b_half = a.n // 2, b.n // 2
    parts: list[Counter[str]] = [
        Counter(x.move for x in a.samples[:a_half]),
        Counter(x.move for x in a.samples[a_half:]),
    ]
    others: list[Counter[str]] = [
        Counter(x.move for x in b.samples[:b_half]),
        Counter(x.move for x in b.samples[b_half:]),
    ]
    pairs: list[float] = [overlap(p, q) for p in parts for q in others]
    return statistics.fmean(pairs)


def verdict(a: Summary, b: Summary) -> str:
    """A plain reading of the two summaries, deliberately conservative.

    The comparison is always against the machines' own variability: a gap smaller than
    the noise floor is not evidence of anything, and at these sample sizes a close call
    means "not established" rather than "the same".
    """
    if not a.n or not b.n:
        return "One endpoint returned no usable answers; nothing to compare."
    shared: float = cross_overlap(a, b)
    # The baseline is how well each endpoint agrees with itself. Comparing against a
    # fixed threshold instead would flag a machine as differing from itself: measured
    # here, one endpoint asked the same question eight times gave five different moves
    # and a self-overlap of 25%.
    # Averaged, not maxed: with a handful of samples each endpoint's self-overlap is
    # itself a noisy estimate, and taking the larger would let one lucky endpoint set
    # a bar the other cannot clear.
    baseline: float = statistics.fmean([a.self_overlap, b.self_overlap])
    if baseline <= 0.0:
        return (
            f"Too few samples to measure the noise floor (move overlap {shared:.0%}). "
            "Re-run with -n 8 or more; a single answer from this model is not "
            "reproducible even on one machine."
        )
    # Reasoning lengths are compared against the larger within-machine spread, so a
    # difference only counts when it exceeds the noise a machine makes by itself.
    noise: float = max(a.reasoning_spread, b.reasoning_spread)
    gap: float = abs(a.mean_reasoning - b.mean_reasoning)
    lengths_differ: bool = noise > 0 and gap > noise

    if shared >= baseline and not lengths_differ:
        return (
            f"Consistent with one distribution: the endpoints overlap {shared:.0%}, "
            f"as much as a single endpoint overlaps itself ({baseline:.0%}), and the "
            "reasoning lengths differ by less than the within-machine spread. Pooling "
            "results is reasonable."
        )
    if shared >= baseline:
        return (
            f"Same move distribution ({shared:.0%} overlap against a {baseline:.0%} "
            "noise floor) but reasoning lengths differ by more than each machine "
            "differs from itself. Keep a given model's games on one machine."
        )
    return (
        f"Move overlap {shared:.0%} is below the {baseline:.0%} an endpoint manages "
        "with itself — suggestive of a real difference, but sampling error is large "
        "at this -n. Re-run with more samples before concluding."
    )


def reference_prefixes() -> list[list[dict[str, str]]]:
    """The conversation as a real match would send it, cut at each of our turns.

    A :class:`LLMPlayer` is driven over the scripted game with a scripted model. That
    model is handed the full history on every call, so capturing its argument yields
    exactly the prefixes we want, already in the player's own wording.
    """
    captured: list[list[ModelMessage]] = []
    step: dict[str, int] = {"i": 0}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured.append(list(messages))
        cells = OUR_MOVES[step["i"]]
        step["i"] += 1
        return ModelResponse(
            parts=[
                TextPart(
                    content=json.dumps(
                        {
                            "move_rationale": "scripted",
                            "cells": list(cells),
                            "claimed_outcome": "in_play",
                        }
                    )
                )
            ]
        )

    from snakes_and_mice.players import LLMPlayer

    agent: Agent[None, LLMMove] = Agent(
        model=FunctionModel(respond), output_type=NativeOutput(LLMMove), retries=0
    )
    player: LLMPlayer = LLMPlayer(agent, name="compare")
    player.start_game(Side.MOUSE, Cell.from_label(SEED))
    for ours, theirs in zip(OUR_MOVES, THEIR_MOVES):
        choice = player.choose_move()
        player.observe_move(Side.MOUSE, choice.move)
        player.observe_move(Side.SNAKE, Move.from_labels(*theirs))
    return [_to_chat(m) for m in captured]


def _to_chat(messages: list[ModelMessage]) -> list[dict[str, str]]:
    """Pydantic AI messages → OpenAI chat messages.

    Thinking parts are dropped: re-sent reasoning is carried in a non-standard field
    whose handling varies per deployment (§4, "Pruning re-sent reasoning"), and both
    endpoints must be given byte-identical input.
    """
    chat: list[dict[str, str]] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            text: str = "\n\n".join(
                str(p.content) for p in message.parts if isinstance(p, UserPromptPart)
            )
            if text:
                chat.append({"role": "user", "content": text})
        elif isinstance(message, ModelResponse):
            answer: str = "".join(
                p.content for p in message.parts if isinstance(p, TextPart)
            )
            if answer:
                chat.append({"role": "assistant", "content": answer})
    return chat


def _resolve(name: str) -> ProviderSpec:
    providers: dict[str, ProviderSpec] = load_roster().providers
    spec: ProviderSpec | None = providers.get(name)
    if spec is None:
        known: str = ", ".join(sorted(providers)) or "(none in providers.yaml)"
        raise SystemExit(f"unknown provider {name!r}\nknown providers: {known}")
    return spec


def _served_model(client: httpx.Client, base_url: str) -> str:
    try:
        return str(client.get(f"{base_url}/models").json()["data"][0]["id"])
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        raise SystemExit(f"could not read a model from {base_url} ({exc})") from exc


def _ask(
    client: httpx.Client, base_url: str, model: str, chat: list[dict[str, str]]
) -> Sample | None:
    """One answer, or ``None`` when the endpoint refused or returned nothing usable.

    No `temperature` is sent: the point is to sample the endpoint as a match would
    drive it, at whatever the served model defaults to.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": chat,
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
    response: httpx.Response = client.post(f"{base_url}/chat/completions", json=body)
    if response.is_error:
        return None
    message: dict[str, Any] = response.json()["choices"][0]["message"]
    reasoning: str = str(
        message.get("reasoning") or message.get("reasoning_content") or ""
    )
    try:
        cells = json.loads(str(message.get("content") or ""))["cells"]
    except (ValueError, KeyError, TypeError):
        return None
    return Sample(move=" ".join(str(c) for c in cells), reasoning_chars=len(reasoning))


def _collect(
    client: httpx.Client, spec: ProviderSpec, model: str,
    chat: list[dict[str, str]], samples: int,
) -> Summary:
    """Sample one endpoint, reporting progress since a slow box takes a while."""
    got: list[Sample] = []
    failures: int = 0
    for i in range(samples):
        sample: Sample | None = _ask(client, spec.base_url.rstrip("/"), model, chat)
        if sample is None:
            failures += 1
        else:
            got.append(sample)
        print(f"    {spec.name}: {i + 1}/{samples}", end="\r", file=sys.stderr)
    print(" " * 60, end="\r", file=sys.stderr)
    return summarize(got, failures)


def _report(spec: ProviderSpec, model: str, s: Summary) -> None:
    print(f"  {spec.name}  ({model})")
    if s.failures:
        print(f"    {s.failures} request(s) returned nothing usable")
    print(f"    distinct moves: {len(s.moves)} over {s.n} samples")
    for move, count in s.moves.most_common():
        print(f"      {move:12s} ×{count}")
    print(
        f"    reasoning: mean {s.mean_reasoning:,.0f}  "
        f"spread ±{s.reasoning_spread:,.0f}  "
        f"range {min(s.reasoning, default=0):,}–{max(s.reasoning, default=0):,}"
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two endpoints' move distributions on identical prompts."
    )
    parser.add_argument("provider_a")
    parser.add_argument("provider_b")
    parser.add_argument(
        "-n", "--samples", type=int, default=8,
        help="answers to draw from each endpoint (default 8)",
    )
    parser.add_argument(
        "-p", "--position", type=int, default=2,
        help=f"which scripted position to ask about, 1–{len(OUR_MOVES)} (default 2)",
    )
    args = parser.parse_args()

    prefixes: list[list[dict[str, str]]] = reference_prefixes()
    if not 1 <= args.position <= len(prefixes):
        raise SystemExit(f"--position must be 1–{len(prefixes)}")
    chat: list[dict[str, str]] = prefixes[args.position - 1]

    specs: list[ProviderSpec] = [_resolve(args.provider_a), _resolve(args.provider_b)]
    with httpx.Client(timeout=900.0) as client:
        models: list[str] = [
            _served_model(client, s.base_url.rstrip("/")) for s in specs
        ]
        if models[0] != models[1]:
            print(
                f"WARNING: {models[0]!r} vs {models[1]!r} — the endpoints serve "
                "different models, so any difference below is not about the machines.\n"
            )
        print(
            f"position {args.position} ({len(chat)} messages), "
            f"{args.samples} samples per endpoint\n"
        )
        summaries: list[Summary] = [
            _collect(client, spec, model, chat, args.samples)
            for spec, model in zip(specs, models)
        ]

    for spec, model, s in zip(specs, models, summaries):
        _report(spec, model, s)

    a, b = summaries
    print("  comparison")
    print(f"    modal move   A: {a.modal[0]} ({a.modal[1]}/{a.n})   "
          f"B: {b.modal[0]} ({b.modal[1]}/{b.n})")
    print(f"    move overlap {cross_overlap(a, b):.0%}   "
          f"(noise floor: A {a.self_overlap:.0%}, B {b.self_overlap:.0%} with itself)")
    print(
        f"    reasoning    A {a.mean_reasoning:,.0f} ±{a.reasoning_spread:,.0f}   "
        f"B {b.mean_reasoning:,.0f} ±{b.reasoning_spread:,.0f}"
    )
    print(f"\n  {verdict(a, b)}")


if __name__ == "__main__":
    main()
