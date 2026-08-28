# The LLM Player

> Part of the Snakes and Mice specification. This document covers **only** the
> LLM player; the game, the `Player` abstraction, matches, tournaments, and the
> CLI are in [`SPEC.md`](SPEC.md), which summarizes this player in §4.

The **LLM player** chooses its moves by querying a large language model. It is the
player type this project exists to compare: pitting models against one another (and
against strong non-LLM players) over many games is the benchmark. Because whether a
model can _reason_ about the game — track the board, find lines, recognize a win or a
dead position, avoid illegal moves — is exactly what we measure, the LLM player is
given as little help as possible: it sees only the opponent's moves and must
maintain everything else itself.

## Interaction: Pydantic AI

The player wraps a single [Pydantic AI](https://ai.pydantic.dev/) `Agent`, bound to
one model (one model per player instance — see "Model selection" below). Two library
features carry the design:

- **Structured output.** The agent is configured to return a typed object rather
  than free text, so the move and the player's self-assessment arrive already parsed
  (see "Structured output" below).
- **A single running message thread.** The player keeps one conversation that
  **spans every game it plays** — not one per game. The model thus accumulates
  context across games (including how earlier games ended), which is what lets it
  learn from a mistake with no change to the `Player` interface (SPEC.md §3, "Feedback
  across games"). The thread is **in-memory for the life of the player instance**; it
  is not persisted across processes.

## Structured output

Each move request returns an object with three fields:

```python
class LLMMove(BaseModel):
    move_rationale: str            # a SHORT justification; logged, never validated
    cells: list[str]               # one or two cell labels, e.g. ["C3", "D4"]
    claimed_outcome: TurnOutcome   # required self-assessment: in_play | win | cats_game
```

- **`cells`** are turned into a `Move` via `Move.from_labels(*cells)`. If that cannot
  form a well-formed move — an unparseable label, an off-board or duplicated cell, or
  the wrong number of cells — the player catches the `ValueError` / `IllegalMove` and
  raises `MoveUnavailable` with the matching reason (SPEC.md §3, "Game results and
  termination"), ending the game as a `PLAYER_FAULT`. Cells that are well-formed but
  _illegal against the current board_ (already occupied, or a lone piece that does
  not end the game) are caught by the engine at apply-time — the same fault, detected
  one layer down.
- **`claimed_outcome` is required** of the LLM (unlike the optional
  `MoveChoice.claimed_outcome` mechanical players may omit): the model must state,
  every turn, whether it believes the move wins, draws, or leaves the game in play.
  Recognizing the outcome is part of the reasoning test, so a wrong claim is a
  `WRONG_OUTCOME_CLAIM` fault (SPEC.md §3). The parsed move and this claim become the
  `MoveChoice` the player returns.
- **`move_rationale`** is a brief, human-readable justification — a _summary_ of why
  the model chose its move, not its full chain of thought. It is **log-only**: never
  validated, never acted on, kept for observation and later analysis. It is placed
  first in the schema so the model articulates a reason before committing to a move.

**How the object is requested is a per-provider choice.** Most providers use Pydantic
AI's default output _tool_; **Anthropic** cannot combine an output tool with thinking,
so its agent asks for the model's **native JSON-schema response format** instead. Both
yield the same `LLMMove`, so nothing downstream varies; only the agent's construction
does, which is why that knowledge sits with model resolution rather than in the player.

A **custom endpoint declares its own mode** in `providers.yaml` (see "Model
selection"), since for a self-hosted server this is a property of the deployment
rather than of the provider kind. Three modes, in decreasing order of how much they
constrain generation:

- **`tool`** (the default) — Pydantic AI's output tool.
- **`native`** — the model's own JSON-schema response format. No tool parser is
  involved, and with a reasoning parser configured the server holds the grammar back
  until the thinking ends.
- **`prompted`** — the schema goes in the prompt and the reply is parsed as text, so
  nothing constrains generation at all.

**When `tool` fails.** vLLM implements a tool-call format _per model family_, so a
served model outside that set must borrow another family's `--tool-call-parser`. On
older vLLM (0.17) the parser only _scraped_ the finished text, and borrowing was
harmless; from the structural-tag refactor (present by 0.24) it also **compiles the
decoding grammar**, and a model held to a foreign tool-call syntax never reaches a
state where stopping is allowed — it rewrites its call until `max_tokens`, and the
thread then grows until the provider rejects it. The fix is `native`, served with no
`--tool-call-parser` at all.
[`tools/probe_tool_termination.py`](tools/probe_tool_termination.py) asks one endpoint
for a move both ways and reports how each ended, which is how the mode is chosen.

**The mode may not be play-neutral.** One model reasoned substantially longer under a
_post-hoc_ tool parser on one machine, though the effect did not reproduce on a second
running the same image. Whether that changes how well it *plays* is unmeasured: the
runs that would have answered it were tallied by a reader that misread two fault
messages as draws, and their logs are gone. Read any such comparison against the
variance in "Comparing two machines" below — a served model's reasoning length varies
threefold between identical requests, so a difference of means across two runs is worth
less than it looks. Treat a mode change as a possible benchmark
change until someone measures it properly: do not pool results across one, and prefer
`native` to `prompted` where both work, since only `native` guarantees parseable
output.

**Responses are capped at a generous output-token budget** (16384, against Pydantic
AI's 4096 default). On Anthropic that ceiling covers the thinking _and_ the answer
together, so at high effort the reasoning alone can approach a small cap and clip the
trailing JSON — which then fails to parse and is scored as an `UNPARSEABLE_OUTPUT`
fault rather than the model's play. Only tokens actually produced are billed, so a
generous cap costs nothing on a short answer. It is not a free parameter, though: a
model whose effort level makes it reason past the cap produces a
`THINKING_LIMIT_EXCEEDED` fault instead (see "Thinking / effort level").

## The message thread

No network call is made except when a move is actually needed: `start_game` and
`end_game` (and the one-time rules preamble) only **enqueue** messages, flushed as
the next user turn on the following `choose_move`.

- **Opening message (once, lazily on the first `choose_move`).** Explains the rules —
  board, coordinates, that the snake starts on one seeded cell (which cell you are
  told at the start of each game, since the opening may be randomized), two pieces
  per turn, the winning lines, cat's game, what counts as illegal — and the
  response protocol (return exactly the structured fields above; you see only your
  opponent's moves and must track the board yourself). Because this message is sent
  once for the whole match, it names **no specific seed cell** — that belongs to
  each game's start.
- **`start_game(side, seed)`** enqueues a "you are playing {side} this game, and
  the snake is seeded at {seed}" message — telling the model the opening cell so it
  can track the board even when the opening is randomized. Per SPEC.md §3 it also prepends
  any stored feedback from a game the player faulted.
- **`observe_move(side, move)`** — for the **opponent's** move, enqueues "opponent
  played {move}"; for the player's **own** move it is a no-op (that move is already
  present as the model's own prior structured response).
- **`choose_move()`** flushes the queued messages as the user turn, runs the agent
  over the accumulated history, appends the response, and returns the `MoveChoice`.
- **`end_game(result)`** composes a message describing how the game ended and
  enqueues it — deferred, sent only if there is a next game. It reports the result
  from the structured `GameResult` / `PlayerFaultDetail`, and in particular:
  - the **opponent's game-ending move(s)**, when the opponent won, drew, or faulted
    on their own turn — in that case the player never got a `choose_move` to be told
    that move, so the thread would otherwise be missing it;
  - if **the player itself faulted**, what it did wrong and how to avoid repeating it,
    composed from the fault detail (SPEC.md §3, "Whoever reports, reports only the facts").

## Pruning re-sent reasoning from the request

Because the thread spans the whole match and every provider re-sends the **entire**
thread as input on each turn, a reasoning model's own accumulated chain-of-thought can
dominate the payload, inflating cost and capping how many games fit before the thread
outgrows the model's input budget. That prior reasoning is not needed to keep playing:
the board is fully reconstructible from the move list, and the model reasons afresh —
still at the full effort level (see "Thinking / effort level") — every turn.

So the player _can_ attach a Pydantic AI **history processor** (the `ProcessHistory`
capability) that strips prior reasoning from what is **sent**, turn by turn, without
lowering the thinking level. The rule is by **shape**, not by provider — a part is
stripped when it **has text** and **carries no signature**:

- **Has text** is the whole payload worth dropping, and it excludes OpenAI's
  Responses API and Anthropic outright: both emit _empty-content_ reasoning parts, so
  there is nothing to save by dropping them.
- **No signature** keeps the history valid. A signature is the provider's own handle
  on the reasoning, needed when the turn is re-sent; dropping a signed part
  **invalidates** the re-sent history — OpenAI rejects the dangling reference with an
  HTTP 400.

The part's `id` is deliberately **not** tested: on an OpenAI-compatible endpoint
Pydantic AI sets it to the name of the field the reasoning arrived in, a constant label
rather than a provider handle, so requiring `id is None` would skip every locally
served model — which is exactly where the bulk sits.

**Whether stripping saves anything is decided downstream, by each model build's chat
template**, and the project cannot detect which case it is in: one served model renders
re-sent reasoning back into the prompt and pays for every token of it, while another
ignores the field entirely and pays nothing — so identical clients against identical
servers can differ purely by model. The rule strips the text either way and lets the
template decide whether that mattered.
[`tools/probe_reasoning_field.py`](tools/probe_reasoning_field.py) answers it for one
endpoint. It measures through the **live** request path rather than `/tokenize`,
because a server can render the same conversation two ways and where the two disagree
it is the live path that decides — a false "dropped" would argue against pruning
exactly where pruning pays most.

**Pruning is off by default, enabled per run by `--prune-thinking` (SPEC.md §7).** The
reason is benchmark validity rather than economy. Anthropic, OpenAI and Gemini all hand
a model its own prior reasoning back regardless of what is stripped, via a signature the
provider reconstructs server-side. Only a model with no such channel actually loses
sight of its earlier thinking. Pruning by default would therefore present a
systematically different conversation to different providers, in a benchmark whose whole
purpose is comparing reasoning across models. The default keeps whatever each provider
natively carries; the flag exists for long tournament runs where context growth is the
binding constraint.

This is a **wire-only** transform: it rewrites only the request payload. The player's
stored thread — and therefore the `--log-llm` dump (see "Message logging") — keeps the
full reasoning of every turn for debugging, and the _current_ turn's thinking is never
touched (only prior turns are pruned, and only in what is sent). It never changes which
moves are legal or how faults are scored.

## No retries

An illegal or unusable move **ends the game** — no retries, no re-prompting within a
game (contrast the human player, SPEC.md §3, which re-prompts a person). The consequence is
delivered to the model as feedback in the _next_ game's opening messages, not
mid-game. This makes "does the model play legal, well-assessed moves" a scored
property of the benchmark rather than something the harness papers over.

(Transient _transport_ failures — read/connect timeouts, dropped connections — are a
separate, operational concern, not a player fault. The LLM player retries the call a
few times with exponential backoff; if the backend stays unreachable it raises
`PlayerUnavailable`, and the engine ends that game as a no-contest (`ABORTED`, SPEC.md §3)
without charging the player or aborting the match. This is distinct from a
_configuration_ failure — a misspelled/unavailable model, a rejected key — which no
retry fixes and which aborts the whole run with a clear message, not a game outcome.)

Because there are no in-run retries, an unparseable-output fault can leave the thread
in a shape the provider will not accept on the _next_ turn. When the model's bad
output is a structured-output tool-call whose arguments fail validation (e.g. a
mistyped field name), Pydantic AI raises before emitting the tool-return it would
normally use to close that call — so the faulting turn, persisted verbatim for the
log and fault feedback, ends in a tool-call with no matching return. Re-sending such a
thread makes the provider reject the whole request ("cannot provide a new user prompt
when the message history contains unprocessed tool calls"), which would abort the next
game and, misread as a backend error, the whole match. So when the player records a
faulting turn it keeps the broken call verbatim but immediately follows it with a
synthetic tool-return, leaving the stored thread a well-formed message history that is
safe to re-send. This is the _only_ point a dangling call can arise (the normal path
appends only Pydantic AI's own well-formed turns), so the repair is done once, where
the turn is recorded, rather than re-scanned on every request.

## Model selection

A player is specified by a **provider** and a **model name**, one model per player
instance. Providers supported to start:

| Provider        | Kind                                                 | Key (env var)         |
| --------------- | ---------------------------------------------------- | --------------------- |
| `anthropic`     | built-in                                             | `ANTHROPIC_API_KEY`   |
| `openai`        | built-in                                             | `OPENAI_API_KEY`      |
| `gemini`        | built-in (Google Gemini API, key-based — not Vertex) | `GEMINI_API_KEY`      |
| `openrouter`    | built-in                                             | `OPENROUTER_API_KEY`  |
| _(custom name)_ | OpenAI-compatible endpoint (e.g. ollama, vLLM)       | declared per provider |

All four built-ins — **including OpenRouter** — are supported directly by Pydantic
AI, so none needs extra endpoint configuration; we resolve `(provider, model)` to the
appropriate Pydantic AI model. Custom providers are OpenAI-compatible endpoints
reached at a configured base URL.

Three sources of configuration, parsed **outside** the player (SPEC.md §8, Architecture):

- **`players.yaml`** — the roster of available players. Each entry is a free-form
  **name** (its identity in matches and standings, independent of the model — in
  common use it will just echo the model), a provider, and a model name:

  ```yaml
  players:
    - name: opus
      provider: anthropic
      model: claude-opus-4-8
    - name: gpt5
      provider: openai
      model: gpt-5
    - name: gemini-pro
      provider: gemini
      model: gemini-3-pro
    - name: llama-local
      provider: my-ollama # a custom provider, defined in providers.yaml
      model: llama3.3
  ```

- **`providers.yaml`** — **only** for custom OpenAI-compatible endpoints; built-in
  providers are never listed. Each entry maps a provider name to a base URL and,
  optionally, the environment variable holding its key (a local ollama may need
  none) and how the endpoint is asked for structured output (`output_mode`, default
  `tool`; see "Structured output"):

  ```yaml
  providers:
    - name: my-ollama
      base_url: http://localhost:11434/v1
      # no api_key_env — local endpoint needs no key
    - name: my-vllm
      base_url: http://gpu-box:8000/v1
      api_key_env: VLLM_API_KEY
    - name: my-vllm-native
      base_url: http://gpu-box:8000/v1 # the same server, asked the other way
      api_key_env: VLLM_API_KEY
      output_mode: native
  ```

  Two entries may share a `base_url` and differ only in `output_mode`, which is how
  the two modes are compared against one endpoint — the roster names them as
  separate providers, so a match can be run each way.

- **`.env`** — API keys, kept out of the YAML and out of version control, read from
  the environment:

  ```
  ANTHROPIC_API_KEY=...
  OPENAI_API_KEY=...
  GEMINI_API_KEY=...
  OPENROUTER_API_KEY=...
  ```

**None of these three files is tracked in git** — they hold local roster choices and
secrets. The repository instead ships tracked templates — `players.example.yaml`,
`providers.example.yaml`, and `.env.example` — that document the format and are
copied and edited into the real (git-ignored) files.

A **roster loader** (`roster`) reads these three sources and produces typed, validated
specifications — a roster of named player and provider entries — and stops there. It
knows about _files_, not about models: resolving a `(provider, model)` spec to a
Pydantic AI model and agent, and building a player from it, is the LLM player's own
job, offered as the alternate constructor `LLMPlayer.from_roster(name, roster)`. So
the split is clean in both directions — the roster loader never imports Pydantic AI,
the player never touches YAML — and **every** Pydantic AI dependency in the project
sits in the LLM player module.

## Thinking / effort level

The one non-default setting is the model's **thinking / reasoning effort**. Pydantic
AI exposes a **unified** effort control (`ModelSettings(thinking=...)` / the
`Thinking(effort=...)` capability) with levels `minimal | low | medium | high |
xhigh`; it translates each to the provider's native mechanism (Anthropic's thinking
budget, OpenAI's reasoning effort, Gemini's thinking budget, …) and maps an
unsupported level to the nearest available one.

For now every LLM player uses the **same** level — a single global default, set to
**`high`** — so each model reasons strongly and comparisons are made on an even
footing, without the steep cost of the top (`xhigh`) tier. "Consistent" here means
_each model at the same effort level_, not a byte-identical configuration across
providers. Per-player effort levels and provider-specific setting overrides are
deliberately deferred.

**The level is best-effort, and for many models it does not arrive.** Pydantic AI
drops the unified setting **silently** for any model whose profile reports no support
— which is every OpenAI-compatible endpoint whose model name it cannot recognize, so
most locally served models. The request still succeeds; the model simply reasons at
whatever its server defaults to, which need not be `high`, need not offer `high` at
all, and on some builds is not settable by any means. An over-large default also
interacts with `MAX_OUTPUT_TOKENS`: reasoning that runs to the cap yields a
`THINKING_LIMIT_EXCEEDED` fault, which would then be scoring the harness rather than
the model — so a level above the roster's is not the safe direction to err in.

The project does **not** fail or fall back in that case — the fix belongs on the
server (e.g. vLLM's `--default-chat-template-kwargs`), and a hard failure would make
perfectly usable models unusable. It **does** print one note per affected player at
construction, so "running at `high`" and "running at the server's default" are
distinguishable without probing the endpoint.

## Message logging (debugging)

Because the benchmark turns on _how a model reasons_, it helps to be able to read
the exact conversation a player had with its model. A debugging option captures
that: with the CLI flag **`--log-llm [DIR]`** (off by default; `DIR` defaults to a
git-ignored `llm-logs/`), each LLM player writes the **full message thread** — the
whole conversation with the model, keeping each turn's complete reasoning even where
later requests prune it (see "Pruning re-sent reasoning") — as JSON under `DIR`.

- **The complete thread, with full reasoning.** The dump is the player's stored
  thread serialized with `ModelMessagesTypeAdapter`: the opening rules preamble, each
  game's "you are {side}" and opponent-move messages, the flushed user turns, and the
  model's structured responses (`move_rationale`, `cells`, `claimed_outcome`) — plus
  the deferred `end_game` feedback — all in order, with whatever metadata the library
  records (model settings, usage, timestamps). Crucially it preserves **every turn's
  thinking**, including the prior-turn reasoning that "Pruning re-sent reasoning"
  strips from later _requests_: the log is the debugging record, not the wire form. So
  the player accumulates the thread incrementally, turn by turn, rather than rebuilding
  it from the agent's `all_messages()` — which now reports the strip-processed history
  and would progressively erase earlier turns' thinking from the dump.
- **One file per LLM player instance.** A player's thread spans the whole match
  (see "The message thread" above), so its file holds that side's entire match
  conversation across all games, keyed by player name and side (e.g.
  `opus-mouse.json`). Only LLM players have a thread; the flag is a no-op for
  `random` / `human` players.
- **Rewritten after every model call.** The file is overwritten with the complete
  thread-so-far on each `choose_move`, so an interrupted or crashed run still
  leaves the whole conversation up to the point of failure — exactly when a
  transcript is most wanted.
- **Local, untracked.** `DIR` holds debugging artifacts, not benchmark output; it
  is git-ignored and never committed.

This is purely an observation aid: it never changes what is sent to the model, and
never affects how moves or faults are scored.

Because the thread carries each game's outcome (see "The message thread"), a dump is
also a record of how the match went, and
[`tools/tally_log.py`](tools/tally_log.py) reads it back. It inverts the exact
constants the player writes rather than matching hand-written substrings — the advice
for two fault reasons mentions a cat's game, so a reader that tests for that first
scores those faults as draws. It reports **one game fewer than the match played**,
since the last game's outcome is only enqueued and never flushed; where the exact count
matters, record the match with `--tournament-results` and use `tally-tournament`
(SPEC.md §6).

## Probing a deployment

Two properties of a self-hosted endpoint decide how the LLM player must be configured
against it, and **neither is detectable from inside the project**: both follow from the
served model's chat template and the server's build. Each has a probe that answers it
by measuring. Both take one argument — a provider name from `providers.yaml` — print a
verdict, and are strictly read-only: they play no game, write no log, and change
nothing about a run.

- **[`tools/probe_tool_termination.py`](tools/probe_tool_termination.py)** — _can this
  endpoint finish a move, and in which output mode?_ It asks for one move twice, once
  through a forced output tool and once through a JSON-schema response format, and
  reports each attempt's `finish_reason`, tool-call count and output-token count. A
  healthy path stops on its own; the failure is unmistakable — a run to the token cap
  carrying repeated identical calls. This is what selects `output_mode` (see
  "Structured output"). A refusal counts as an answer: a form the endpoint rejects here
  is a form it will reject in a game.

- **[`tools/probe_reasoning_field.py`](tools/probe_reasoning_field.py)** — _does re-sent
  reasoning cost input tokens?_ It sends the same short thread three ways — prior
  reasoning carried in `reasoning`, in `reasoning_content`, and omitted — and compares
  prompt length, reporting `RENDERED` with a chars-per-token rate, or `DROPPED`. That
  decides whether `--prune-thinking` buys anything here (see "Pruning re-sent
  reasoning"). It measures through the **live** request path rather than `/tokenize`,
  since a server can render one conversation two ways, and reports any disagreement
  between them.

Re-run both whenever the container, the served model, or the serve flags change: the
answers are properties of the deployment, not of this project, and a stale answer is
worse than none.

## Comparing two machines

A tournament may be spread over more than one machine, which is sound only if a model
plays the same on each. That is harder to establish than it looks, because **a served
model is not reproducible even on one machine**: identical requests return different
reasoning and different moves, since `temperature: 0` makes each step greedy without
making the arithmetic reproducible — with CUDA graphs, chunked prefill and a
mixture-of-experts model the reduction order varies, and tiny logit differences compound
over thousands of reasoning tokens. Measured on one endpoint, eight identical requests
gave five distinct moves and reasoning spanning 3,000 to 9,000 characters.

Two consequences. Comparing single answers between machines is meaningless, and so is
reading a difference in mean reasoning length between two runs — the variance within one
machine swallows it. **Any claim that two deployments differ has to be measured against
how much a deployment differs from itself.**

[`tools/compare_endpoints.py`](tools/compare_endpoints.py) does that: it samples both
endpoints on the same prompt and compares the resulting move distributions, taking each
endpoint's own split-sample overlap as the noise floor. Cross- and self-overlap are both
measured on half-samples, since smaller samples overlap less by construction and mixing
the two sizes biases the answer. Its prompts come from driving a real `LLMPlayer` over a
scripted game, so they are byte-identical to a match's.

It answers "is there evidence these differ", not "are these the same": a modest
difference stays invisible at any sample size a slow machine can afford. The cheaper
course is to **assign each model to one machine for a whole tournament**, which removes
the question rather than answering it.

## Deferred for now

To keep the first LLM player simple, and beyond the game-playing core above:
usage / cost / latency tracking; per-player or per-provider setting overrides;
persistence of the message thread across processes; and fully managing a thread that
still outgrows the model's context window over a long match — `--prune-thinking` can
slow that growth (see "Pruning re-sent reasoning") but does not by itself cap it.
