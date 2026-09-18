# The Reinforcement-Learning Player

> Part of the Snakes and Mice specification. This document covers **only** the
> reinforcement-learning (RL) player; the game, the `Player` abstraction, matches,
> tournaments, and the CLI are in [`SPEC.md`](SPEC.md), which lists this player
> as planned in §3. The empirical investigation into LLM mistakes and the
> hand-crafted scoring function built from it — which this player's training
> strategy depends on but does not itself define — are specified separately, in
> [`SPEC-mistake-model.md`](SPEC-mistake-model.md).

> **Status: early, partial draft.** This document covers what has actually been
> decided so far — the goal, including the need for two non-learned baselines to
> measure the result against, `perfect` with no heuristic and `perfect` with only
> the one validated bias feature (both specified in SPEC-perfect-player.md,
> "Selecting a variant"); the overall training strategy (why training directly
> against a live LLM cannot be the primary loop, the six-step loop itself, and
> why the mistake model's shaping term is needed to reward progress toward a
> rare, high-value intermediate state before any terminal win or loss — see
> "Training strategy" below); the training algorithm and network (self-play PPO,
> a critic regressed against the solved game, a small MLP — see "The algorithm
> and network" below); and periodically training and evaluating against real
> LLMs alongside self-play (see "Periodic fine-tuning" below). Left for later:
> everything below the architecture level — training hyperparameters,
> curriculum details, and the two `perfect` variants' own implementation. The
> mistake model itself — whether LLM mistakes are exploitable at all, what they
> look like, and the scoring function's exact form — is decided and measured in
> [`SPEC-mistake-model.md`](SPEC-mistake-model.md), not here.

## Goal

The **RL player** is not an attempt to out-play the algorithmic player (§10 of
SPEC.md) — that player is already provably optimal, and the game is a solved draw
from every seed. The goal instead is narrower and specifically about the project's
actual subject, LLMs (SPEC.md §1): produce a trained player that beats an LLM
player **more often than the perfect player does**, on the premise that an LLM's
mistakes may be systematically biased rather than random — a premise now measured
and confirmed to hold ([`SPEC-mistake-model.md`](SPEC-mistake-model.md), "Result:
the premise holds") — and a player that specifically learns to exploit that bias
could out-score pure optimal play against a fallible-but-patterned opponent, even
though it cannot out-score optimal play in the game-theoretic sense.

- **Breadth is open.** Ideally one RL player exploits shared weaknesses across
  most or all of the LLM roster (SPEC.md §4). If cross-model bias turns out not to
  be shared enough for that, a player specialized to a single target LLM is an
  acceptable, and possibly necessary, fallback.
- **Robustness against the perfect player is a secondary constraint, not an equal
  priority.** The RL player should rarely if ever *lose* to `perfect` — draws are
  fine and expected, since perfect play cannot be beaten — but some of that
  robustness is negotiable in exchange for a meaningfully higher LLM win rate.
- **The actual bar is comparative, not absolute.** "Beats LLM X more often than
  the perfect player does" is the success criterion — not merely "wins some games
  against LLM X."
- **The comparison needs `perfect` with no heuristic, not `perfect` as shipped.**
  `perfect` already ranks among equally optimal moves to exploit a fallible
  opponent (SPEC.md §10, "Choosing among optimal moves"). Measuring the RL player
  against that ranked version can't distinguish "the RL player found something
  real" from "it merely matches what the existing hand-coded ranking already
  does" — the clean comparison is against ranking switched off entirely.
  SPEC-perfect-player.md, "Selecting a variant" specifies this as a planned,
  separately-named, still-unimplemented `perfect` variant.
- **The comparison also needs a non-learned upper bound that already knows the
  one validated bias feature.** Beating `perfect`-with-no-heuristic is a low
  bar — it isn't even trying to exploit anything. The RL player's real claim is
  that *learning* found something a hand-coded rule using only the double-threat
  feature ([`SPEC-mistake-model.md`](SPEC-mistake-model.md), "The mistake
  model") could not. That claim needs the hand-coded rule shipped as a player,
  not just as a shaping term internal to training — SPEC-perfect-player.md,
  "Selecting a variant" specifies this as the third `perfect` variant,
  `perfect-mistake-model`.

## Training strategy

### Why training directly against a live LLM cannot be the primary loop

RL — especially anything learning an opponent-specific policy rather than general
play — typically needs many thousands of episodes. An LLM move, even from a small
model served locally for speed, costs real wall-clock seconds; a live opponent
therefore costs orders of magnitude more time per episode than RL's episode budget
assumes. Training directly against a live LLM as the main loop is not workable at
the volumes RL needs, independent of how much serving speed is optimized — a
faster deployment changes a constant factor, not the underlying mismatch between
what RL needs and what a live LLM opponent can supply.

This is not a problem specific to LLM opponents, though: it is exactly why the game
having been fully solved (SPEC.md §10) matters here. "Play well in general" does
not need to be rediscovered by trial and error against anyone — self-play and games
against `random` and `perfect` are unlimited and effectively free, and could even be
seeded or shaped from the solver's exact values rather than learned from scratch.
The genuinely scarce resource is specifically *observations of how a given LLM
actually errs*, and the strategy below is built around spending that scarce
resource deliberately rather than burning it as ordinary training volume.

### The loop

1. **Bootstrap a base agent** purely from self-play and games against `random` and
   `perfect` — cheap, unlimited, no LLM in the loop.
2. **Play a strong opponent against the target LLM live**, for a modest number of
   games, to find the *on-policy* states worth caring about — the states a strong
   player actually reaches, not an arbitrary sample of the astronomically large
   full game tree. (`perfect` already exists and already ranks moves for
   "trappiness" among ties — SPEC.md §10, "Choosing among optimal moves" — so it is
   available for this today, without waiting on step 1's agent to exist.)
3. **Densify around those states** by constructing positions directly rather than
   only encountering them incidentally in full games — a `Player` learns the board
   solely through `start_game` / `observe_move` (SPEC.md §3), so any reachable
   position can be set up by replaying whatever move sequence reaches it, then
   asking the LLM to move from it once. This yields far more targeted data per
   unit of live-LLM time than playing full games out.
4. **Fit a mistake model that generalizes** from the collected data. It must
   generalize rather than memorize positions outright: the full reachable-position
   tree runs to billions of canonical keys (`tools/solver/SPEC.md`), so no
   plausible amount of collected data could ever cover it by lookup.
5. **Mix the mistake model into further training**, alongside self-play, `random`,
   and `perfect`, so the trained policy learns to exploit the found bias without
   losing the robustness those mechanical opponents provide.
6. **Repeat.** A retrained, exploit-seeking policy will drift toward different
   states than step 1's did, so the on-policy states worth probing shift over
   time — refresh with fresh live-LLM probes periodically rather than treating any
   one round of data collection as final.

The mistake model's form, and how it rewards progress toward a rare, high-value
intermediate state before any terminal win/loss, turned out to be one decision
rather than two — see [`SPEC-mistake-model.md`](SPEC-mistake-model.md), "The
mistake model."

### The algorithm and network, decided 2026-09-18

**Self-play actor-critic, specifically PPO — not AlphaZero-style search at either
train or inference time.** AlphaZero's search-augmented self-play earns its cost in
games with enormous branching and long horizons, where raw self-play is too
sample-inefficient without search amplifying weak value estimates into stronger
training targets. This game is at most 12 plies with a shrinking branching factor;
that problem barely exists here. Search at inference time would also make this
player qualitatively different from every other player type in the project
(SPEC.md §3) for no real benefit — `random`, the LLM player, and even `perfect`'s
own full search all respond quickly.

**The critic is regressed against `evaluate()` (SPEC.md §10), not learned from
self-play alone.** The game is solved — the exact value of any position is already
known — so training budget should not be spent rediscovering it by trial and
error. Pretraining and continued regularization of the value head against the
exact solved values lets the actual learning effort go entirely toward the
genuinely unknown part: how to trade a sliver of that optimality for exploitation,
which is what step 5 of the loop (above) and the mistake model's shaping term
([`SPEC-mistake-model.md`](SPEC-mistake-model.md), "Three roles, one function")
need to learn.

**Network: a small MLP over the board occupancy, not a CNN.** The board is 5×5,
and a convolutional architecture's core assumption — a spatial filter reused
across positions, learning something translation-invariant worth sharing — doesn't
obviously earn its cost at this size, and the board has no wraparound symmetry a
CNN's receptive field would otherwise exploit either. A small MLP over the two
occupancy planes (or one signed plane, mine minus theirs) is the starting point
until shown insufficient.

### Periodic fine-tuning and evaluation against real LLMs, decided 2026-09-18

Step 5's opponent pool (self-play, `random`, `perfect`) is not the whole story:
**periodically, a training episode's opponent is a real, live LLM instead** —
`qwen-3-8` locally (free, and patient enough to afford real volume) and, for
hosted models, whatever tens to hundreds of games their modest cost affords. This
is not "more training data" in the sense of moving the needle on bulk policy
quality — that volume is negligible next to whatever scale self-play runs at.
Three things it is for instead:

- **Ground-truth evaluation.** Self-play performance is a proxy; "does this beat
  the actual target LLM more often than `perfect` does" (the Goal, above) is the
  only number that actually answers the project's question, and only real games
  against the real target produce it.
- **Continued mistake-model validation** (step 6 of the loop, above) — already the
  plan; not a new use of these games, just named here as one of several.
- **Marginal fine-tuning.** The hand-crafted mistake model
  ([`SPEC-mistake-model.md`](SPEC-mistake-model.md)) encodes exactly the
  mechanisms validated so far and almost certainly misses others a specific
  model has. A modest amount of direct experience against the real target, layered
  on a self-play-trained base, is a way to pick up what the hand-crafted score
  can't see — matching the project's actual goal (a margin beyond optimal play),
  not standing in for bulk training it was never going to afford.

This needs no new mechanism: PPO trains on the agent's own actions and rewards
within an episode and does not otherwise care who or what the opponent was, so a
live LLM is just another environment to occasionally sample, not a special case.
