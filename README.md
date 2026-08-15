# Snakes and Mice

A game engine and a cast of pluggable players for **Snakes and Mice**, a 5×5
tic-tac-toe variant where each player drops two pieces a turn and a lone snake
squats on the board from the start.

It is also, ostensibly, an **LLM reasoning benchmark**: point two language models
at each other, play a pile of games, and tally who wins, who draws, and who
faults by making illegal or nonsensical moves.

Should you choose the model powering your agent because it never lost at Snakes
and Mice? Almost certainly not. The game is silly, and its worth as a benchmark
is sketchy at best. Mostly this is just for fun — but it *is* a surprisingly
honest little test of whether a model can track a board, spot a winning line, and
resist the urge to play on top of a piece that is already there.

The rules engine is cleanly separated from the players, so any kind of agent can
plug in — scripted bots, a random flailer, a human, an LLM, and (eventually)
stronger algorithmic players. The full rules and design live in
[`SPEC.md`](SPEC.md).

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

### Bringing in LLMs

To pit language models against each other (or against `random` / `human`), copy
the three example config files and fill them in:

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

## Development

```sh
uv run pytest       # tests
uv run mypy         # type-checking
```

## License

[MIT](LICENSE).
