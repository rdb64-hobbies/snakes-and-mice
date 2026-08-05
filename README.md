# Snakes and Mice

A game engine and pluggable players for **Snakes and Mice**, a 5×5 tic-tac-toe
variant, built primarily as a **benchmark for comparing LLMs as a test of
reasoning skill**.

The rules engine is cleanly separated from the players: a `Player` interface lets
any kind of agent plug in — scripted bots, strong algorithmic players, humans,
or LLMs — so the same engine can pit an LLM against another LLM or against a
strong non-LLM opponent, and score how well each reasons about the game.

> More detail — the full rules, how to run a game, and how to run the tests —
> will be added here as the project fills out. For now, see `SPEC.md` for the
> complete specification.
