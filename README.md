# Snakes and Mice

Let your favorite LLMs duke it out at **Snakes and Mice** — a 5×5 tic-tac-toe
variant where each player drops two pieces a turn and a lone snake squats on the
board from the start. Point two models at each other for bragging rights. Play a
pile of games, and tally who wins, who loses, and who can't even follow the game
rules.

You can also pit an LLM against one of the two built-in opponents: **random**
and **perfect**. Any model worth the name should beat the random player, and a
good model should manage to at least not lose against the perfect player. As its
name suggests, the perfect player cannot be beaten, and a draw is the best
result available.

That goes for you too, if you take a turn yourself. You'll have better luck
against random and might sometimes beat an LLM.

Is prowess at Snakes and Mice the metric you want to use when choosing the model
to power your next agent? Probably not. The game is silly, and its worth as a
benchmark is sketchy at best. Mostly this is just for fun — but it _is_ a
surprisingly honest little test of whether a model can track a board, spot a
winning line, and resist the urge to play on top of a piece that is already
there. After all, if a model keeps attempting illegal or nonsensical moves, or
if it can't play this game at the level of a 7-year-old, maybe it isn't the best
choice for an agent.

The full game rules and system design live in [`SPEC.md`](SPEC.md), with the two
substantial player types specified alongside it — the LLM player in
[`SPEC-llm-player.md`](SPEC-llm-player.md) and the perfect player in
[`SPEC-perfect-player.md`](SPEC-perfect-player.md). The architecture allows for
multiple player types, including a scripted bot player for testing, the random
player, the perfect player, LLM players, and a human player. The design of the
perfect player is especially interesting.

## Getting it

You need [Python 3.12+](https://www.python.org/) and
[uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/rdb64-hobbies/snakes-and-mice.git
cd snakes-and-mice
uv sync
```

## Playing

The quickest game needs no setup at all — take on the random player yourself:

```sh
uv run play-match --mouse human --snake random
```

Enter each move as two cells, e.g. `C3 D4`. Add `--help` to any command below to
see all its options.

Two worth trying, in this order:

- **Play against `random`.** Ransom (or Randy, as the mouse) flails, and you
  should win quite easily. Good for getting the feel of two-pieces-a-turn.
- **Play against `perfect`.** Perseus (or Percy, as the mouse) plays perfectly.
  You will not win — but you should be able to avoid losing.

```sh
uv run play-match --mouse human --snake perfect
```

There is more about the perfect player, and why you can't beat it, in [The
solved game](#the-solved-game) below.

### Bringing in LLMs

First you need some way to reach the models themselves. That means either an API
key for one of the built-in providers — Anthropic, OpenAI, Google (Gemini), or
OpenRouter — or a local OpenAI-compatible server such as ollama or vLLM, which
needs no key at all when it runs on your own machine. A roster can mix the two
freely, so a local model can play a hosted one.

Then, to pit models against each other (or against `random`, `perfect`, or
`human`), copy the three example config files and fill them in:

```sh
cp players.example.yaml   players.yaml     # your roster of LLM players
cp providers.example.yaml providers.yaml   # only for custom endpoints (optional)
cp .env.example           .env             # your API keys
```

`players.yaml` names each player and maps it to a provider and model; `.env`
holds the API keys (never commit it); `providers.yaml` is only needed for custom
OpenAI-compatible endpoints like a local ollama. All three are git-ignored.

Then a roster name can play either side:

```sh
uv run play-match --mouse opus --snake gpt5 --games 20 --watch game
```

## Running a tournament

A tournament is just a growing file of match results, so you can build one up
match by match and score it whenever you like.

```sh
# Play many matches and append the results (default: full round-robin)
uv run play-tournament-matches

# Print the standings from the results file
uv run tally-tournament
```

## The solved game

Snakes and Mice is small enough to work out completely, and we did. Every game
that can be played, from every possible starting position, has been analysed.
The answer is always the same: **with best play on both sides, the game is a
draw.** Neither side can force a win, no matter where the snake starts.

That is what `perfect` plays. It can never be beaten, and it will never beat you
unless you give it something — so a draw is always available to you, and your
first move never costs you anything, whichever side you are. The earliest you can
go wrong is your second.

It is not passive about waiting for that, though. Because the game is drawn,
almost every move it makes is a choice between options that are all equally
optimal, and which one it plays cannot change the result against perfect defence.
So it spends that free choice on you: among equally optimal moves it prefers the
ones that leave you the most ways to go wrong, counting them exactly rather than
guessing. It is still incapable of taking a risk to do it — every move it is
choosing between is equally optimal, so nothing is gambled. Against a player
moving at random, that takes it from winning about three games in five to winning
about forty-nine in fifty.

The answers are worked out ahead of time and stored in
[`perfect-tables/`](perfect-tables/), which is why the player responds instantly
early on; later in a game, with fewer moves left to consider, it simply works
the answer out. If those files are missing it still plays perfectly, but the
opening takes hours — it says so rather than appearing to hang.

You can look through the solved game without writing any code:

```sh
uv run python tools/solver/dump_table.py perfect-tables/C3.table.gz
```

The program that did the solving lives in [`tools/solver/`](tools/solver/) and
has its own [specification](tools/solver/SPEC.md); the player itself is covered
by [`SPEC-perfect-player.md`](SPEC-perfect-player.md).

## Development

```sh
uv run pytest       # tests
uv run mypy         # type-checking
```

## License

[MIT](LICENSE).
