# The Reinforcement-Learning Player

> Part of the Snakes and Mice specification. This document covers **only** the
> reinforcement-learning (RL) player; the game, the `Player` abstraction, matches,
> tournaments, and the CLI are in [`SPEC.md`](SPEC.md), which lists this player
> among the implemented player types in §3. The empirical investigation into LLM mistakes and the
> hand-crafted scoring function built from it — which this player's training
> strategy depends on but does not itself define — are specified separately, in
> [`SPEC-mistake-model.md`](SPEC-mistake-model.md).

> **Status: the bootstrap agent is built (1.10); the LLM half is not.** Everything
> this document designed above the architecture level is now implemented and
> shipped — see "The implementation," below, for what exists, and "What is still
> open" for what does not. In outline: the goal, including the need for two
> non-learned baselines to measure the result against, `perfect` with no heuristic
> and `perfect` with only the one validated bias feature (both specified in
> SPEC-perfect-player.md, "Selecting a variant", and both shipped in 1.8 as the
> bare `perfect` and `perfect-mistake-model`); the overall training strategy (why
> training directly against a live LLM cannot be the primary loop, the six-step
> loop itself, and why the mistake model's shaping term is needed to reward
> progress toward a rare, high-value intermediate state before any terminal win or
> loss — see "Training strategy" below); the training algorithm, network, and
> action space (self-play PPO, a critic regressed against the solved game, a small
> MLP, a factored sequential two-cell policy head — see "The algorithm and network"
> below); and periodically training and evaluating against real LLMs alongside
> self-play (see "Periodic fine-tuning" below).
>
> **Only step 1 of "The loop" is built**, and the comparison the whole player
> exists to win has now been run and **failed**: against `qwen-3-8-rtx` over 160
> games the trained player wins 6.2% where bare `perfect` wins 8.1% and
> `perfect-trappiness` 10.0% (see "Measured against qwen-3-8," below). Steps 2–6,
> which spend live-LLM time, remain to be done and are where any remaining hope
> lies. Training hyperparameters and curriculum
> details are settled only to the extent that a working run needed them, and are
> not claimed to be tuned. The mistake model itself — whether LLM mistakes are
> exploitable at all, what they look like, and the scoring function's exact form —
> is decided and measured in
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
- **The comparison needs a perfect player with no heuristic.**
  `perfect-trappiness` ranks among equally optimal moves to exploit a fallible
  opponent (SPEC.md §10, "Choosing among optimal moves"). Measuring the RL player
  against that ranked version can't distinguish "the RL player found something
  real" from "it merely matches what the existing hand-coded ranking already
  does" — the clean comparison is against ranking switched off entirely. As of
  1.8 that is the **bare `perfect`** (SPEC-perfect-player.md, "Selecting a
  variant"), which is why this bullet no longer asks for anything special: the
  unadorned name is the unadorned player.
- **The comparison also needs a non-learned upper bound that already knows the
  one validated bias feature.** Beating `perfect`-with-no-heuristic is a low
  bar — it isn't even trying to exploit anything. The RL player's real claim is
  that *learning* found something a hand-coded rule using only the double-threat
  feature ([`SPEC-mistake-model.md`](SPEC-mistake-model.md), "The mistake
  model") could not. That claim needs the hand-coded rule shipped as a player,
  not just as a shaping term internal to training — SPEC-perfect-player.md,
  "Selecting a variant" specifies this as the third `perfect` variant,
  `perfect-mistake-model`, also shipped in 1.8.

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
   full game tree. (`perfect-trappiness` already exists and already ranks moves for
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

**The action space is factored and sequential, not a joint distribution over
cell pairs.** A move places two cells (SPEC.md §2.5), but the policy treats
that as two sequential single-cell choices rather than one draw from a joint
space that would run as large as C(25,2) early in the game and grow awkward to
mask exactly as it shrinks: a 25-way softmax over cells for the first
placement, masked to the empty ones, then a second 25-way softmax for the
second, masked to whatever remains empty after the first. This keeps the
policy head a fixed size regardless of how full the board is, and needs no
separate representation for the single-piece move that is legal only when it
ends the game (SPEC.md §2.5) — that case is simply the one where the second
placement is skipped.

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

## The implementation, shipped 1.10

Step 1 of "The loop," built. Three pieces: the network and its checkpoint
(`players/rl_net.py`), the player that runs one (`players/rl.py`, CLI name `rl`),
and the trainer that produces one (`tools/rl/`). The architecture is defined once,
in `rl_net`, and imported by both the player and the trainer — so a checkpoint
cannot be loaded into a differently-shaped network than it was trained in.

### The network, as built

Input is the two 25-cell occupancy planes, **mine then theirs**, always from the
side to move. Nothing in the input names a colour, so a position and its
colour-swap share one set of weights and every self-play game yields training data
for both seats. Trunk is two 256-wide ReLU layers; heads are a 25-way policy
logit vector and a scalar `tanh` value.

The factored sequential action space is realized with **one head applied twice**:
once to the position, then again to the position with the first piece already
placed as one of ours. The conditioning on the first placement is carried by the
board rather than by a second head, which costs nothing in expressiveness — an
autoregressive factorization over a finite action space can represent any joint
distribution over cell pairs.

The network can tell the two phases apart because `|mine| − |theirs|` is **−1 at
every real to-move state and 0 at every half-move one, for both sides** — a
linearly decodable signal (the sum of the first 25 features minus the last 25).
Parity of `|mine|` alone would *not* serve: Mouse is even to move and Snake odd,
so parity identifies the phase only to a reader who already knows the side, which
the encoding deliberately withholds. An earlier draft of this section claimed
parity; that was wrong, and the difference is the correct invariant.

The single-piece move then needs no representation at all: the second call simply
does not happen, which `needs_second_placement` decides, and both the player and
the trainer route through that one function so they cannot disagree about how many
pieces a move has.

### Decoding the factored policy at inference

A factored policy makes "play the best move" and "play the best cell, then the
best next cell" different things. The policy scores a move as
`pi(a1|s) * pi(a2|s + a1)`, so descending the two placements one at a time can
commit to a cell whose best partner is poor and miss the pair the policy actually
ranks highest. Greedy play therefore **enumerates**: one forward pass for the
position, one batched pass over the positions each legal first placement leads
to, and the maximum of the real quantity. Still two forward calls per turn.

Two details fall out of asking what a *move* is, rather than what a cell is.
Orderings are **summed**: with neither cell ending the game, `(a, b)` and
`(b, a)` leave the same board, so they are one candidate and the mass the
factorization splits between them is scored once — which also repairs the
ordering redundancy inherent to a sequential policy over an unordered pair. And a
game-ending cell is its own one-piece candidate at `pi(a|s)`; that is not a
length bias in disguise, because the sampler stops at such a cell rather than
drawing a partner, so no second factor was ever going to be charged against it.

Sampling needs none of this. Drawing `a1` then `a2` from the conditional already
samples the joint exactly; it is only the argmax the factorization complicates.

**Measured: this changed nothing.** It alters the chosen move in 1.8% of
positions, and no measurement moved — two-cell wins taken, split blocks made, and
the records against `random` and `perfect` all landed within a game or two of
where they were. It is kept because it is the correct reading of "greedy" for this
policy and costs nothing, **not** because it bought strength. Anyone hunting the
off-policy coordination gap below should look at the opponent pool, which is where
the evidence points, and not here.

### The reward, and where the shaping term is measured

Terminal reward is the result from the recorded side: `+1`, `0`, `−1`. On top sits
the mistake model's potential-based shaping term
([`SPEC-mistake-model.md`](SPEC-mistake-model.md), "Three roles, one function").

**The potential is measured on the position right after our own move** — the frame
`threat_score` is defined for, with the opponent to move and our threats standing
against it. A potential on our own *to-move* states, which is the naive reading of
`γΦ(s′) − Φ(s)` over the agent's MDP, would be worthless here: by the time we are
to move again the opponent has either answered the threat (score back to 0) or
already lost, so it would reward the opponent's blunder rather than the trap that
provoked it — and at `γ = 1` a potential over consecutive agent states telescopes
away exactly, which is the same statement.

So step *t* earns `w · (Φ(after our move t) − Φ(after our move t−1))`, with the
term before the first move taken as 0 and a closing `−w · Φ(after our last move)`
folded into the terminal step. **The shaping contributions of a completed episode
then sum to exactly zero**, so the shaped return of every episode equals its true
return and the optimal policy is unchanged — the guarantee potential-based shaping
gives, arrived at without needing the theorem. What it buys is per-move credit:
the move that *builds* the split threat is the one whose reward goes up, rather
than the signal arriving only if the opponent later falls for it. This is the
requirement [`SPEC-mistake-model.md`](SPEC-mistake-model.md), "Implication for the
training strategy" identified, discharged.

The residue is that the shaped return depends on `Φ` of the previous post-move
position, which is not in the network's input, so the critic carries a bounded
offset of at most `2w`. At the default weight (0.05) that is 0.1 on a ±1 scale —
noise, not bias. Only the *direction* of the feature is validated in any case
(ibid., "What validation means for a shaping term"), and the sum-to-zero
construction makes the optimum independent of the weight, so it is not a tuned
quantity.

### Correction: `perfect` is not an effectively free opponent

"Why training directly against a live LLM cannot be the primary loop," above,
asserts that "self-play and games against `random` and `perfect` are unlimited and
effectively free." **Half of that is wrong, measured.** A self-play game costs
~0.9 ms and a `random` game ~0.5 ms — free, as claimed. A `perfect` game costs
**~5.5 s**: it live-searches every position below 16 empty cells, which is where
the opening table stops. One `perfect` game therefore costs about 11,000 free
ones.

Nor does it amortize. A transposition table kept across 30 games — the obvious
fix, and sound in principle, since the table caches pure facts about positions —
gave **no speedup at all** (166.7 s against 164.2 s) while growing to 5.9 M entries
against 101 k. Endgame positions barely repeat across games, so there is nothing
for a shared table to hit.

The consequence is a curriculum decision this document had left open: the trainer's
`--perfect-share` **defaults to zero**, and `perfect` earns its keep as the
yardstick the agent is measured against rather than as a sparring partner. It
remains available as a training opponent for anyone willing to spend the budget;
the arithmetic above is the whole of the argument against it. This does not touch
the strategy's actual load-bearing claim, which is about *live LLM* opponents
being the scarce resource — `perfect` is merely slower than it was assumed to be,
not scarce.

### The critic, anchored to the solved game

As specified: the value head is regressed against `evaluate()` rather than learned
from self-play alone. The target is the **sign** of the negamax score, not the
score itself — at `γ = 1` a win is worth 1 whenever it arrives, and asking a `tanh`
head to also encode *when* would spend capacity on something the return never
rewards.

Labels come from a cached dataset (`tools/rl/solved.py`) of positions sampled from
every seed under a mix of random and line-building playouts, since purely random
play stays in the drawn, featureless part of the tree and would teach the head only
to output zero. The dataset is used twice: alone, to pretrain the head before any
game is played, and as an auxiliary loss resampled every minibatch thereafter —
"pretraining and continued regularization," as specified.

### What the first trained agent actually does

The shipped checkpoint, 2026-10-03: 3,100 iterations of 256 games — **794,000
games in 8 hours** on an idle Mac Studio CPU — against self-play 58.8%, `random`
39.2%, `perfect` 1.5% and `perfect-trappiness` 0.5%. The network is 85,530
parameters over a (256, 256) trunk. Greedy play, which is what a match faces.

It replaced a first agent trained the same way but with **no `perfect` games at
all** (8,993 iterations, 2.3 M games, 45 minutes). The two are equally good
against `random` and the newer one loses half as often to `perfect`; see
"Longer would not have helped", below, for the comparison.

| Opponent | Games | Win | Draw | Loss |
|---|---:|---:|---:|---:|
| `random` | 400 | 96.2% | 3.8% | 0% |
| `perfect` | 200 | 0% | 95.5% | **4.5%** |
| `perfect-trappiness` | 50 | 0% | 88% | 12% |
| `perfect-mistake-model` | 50 | 0% | 88% | 12% |

**How these are measured matters more than it looks.** Every row uses a
*balanced* schedule — each of the 25 seeds equally often, alternating seats,
which is what `train.py`'s `evaluate_against` does. A `--seed random` match does
not qualify: seed cells differ sharply in difficulty (`C3` lies on both
diagonals, `A1` on neither), and four 100-game random-seed samples of one
checkpoint gave 4, 5, 12 and 11 losses against bare `perfect` — a spread that
swamps any difference worth measuring. An earlier version of this table reported
"4–5%" from two of those lucky draws. **Do not compare two players with
`--seed random`.**

Two readings, one encouraging and one not.

**It found trappy play without being told the rule.** Against `random`, bare
`perfect` wins 62.3% and `perfect-trappiness` — which counts an opponent's losing
replies exactly — wins 98.0% (SPEC.md §11, 1.4). This agent wins **97.5%**, level
with the hand-coded trap counter, having been given neither the trap count nor any
notion of exploitability beyond the mistake model's three-level shaping term. That
is the mechanism the whole player is premised on, working: it is converting a
fallible opponent's errors at close to the rate an exact search does.

**It is not safe against perfect play.** The Goal's secondary constraint asks that
the RL player "rarely if ever *lose* to `perfect`." 4.5% over 200 balanced games
is *rarely*, not *never*, and the constraint is only partly met — though it is
twice as close as the first trained agent got, which lost 9.0% on the same
schedule before `perfect` entered its training pool. An earlier draft of this section
read a side-dependence into the loss rate — 8% as Mouse against 2% as Snake — and
that was 4 losses against 1 over 50 games each, far too few to carry the claim. A
second 100-game run split 2 and 2. **There is no measured side effect**; the
earlier reading was noise. None of the 4–5% is surprising for a policy trained
without a single `perfect` game in the pool, and that pool is the obvious thing
to attack — but with a *differently mixed* run, not a longer one.

**Play against `random` converges in about 17 minutes**, at iteration ~100 of
256 games, and is flat in a 94–96% band from 125 onward. (A finer eval interval
on the second run showed this; the first run's earliest evaluation was at
iteration 1000, which is why it looked like a plateau at five minutes rather than
at one.) Measured against `random`, then, nearly all of any longer run's compute
buys nothing.

**But it is not wasted — it buys the expensive opponent.** The run that produced
this checkpoint was ~99% past that convergence point and still halved the loss
rate to bare `perfect`, from 9.0% to 4.5% across 200 balanced games, by spending
8 hours on a 2% `perfect` share (3,100 iterations, ~15,900 `perfect` games). So
the binding constraint is *what* the agent plays against, as suspected, and the
way to spend an idle machine on it is a non-zero `--perfect-share` rather than
more games of the same. Optimizing rollout throughput remains the wrong lever,
for the reasons the trainer's module docstring records.

**The number that matters is not here.** Every row above is a mechanical opponent.
Whether the agent beats an LLM more often than `perfect` does — the Goal, the only
question the player exists to answer — is unmeasured.

### Measured against qwen-3-8, 2026-10-05: the goal is not met

The Goal above asks for a player that beats an LLM **more often than the perfect
player does**. Measured at last, 160 games each, both seats, four independent
repeats, against `qwen-3-8-rtx` served locally by vLLM:

| player | games | W | D | L | win rate |
|---|---:|---:|---:|---:|---:|
| `perfect` (bare) | 160 | **13** | 145 | **0** | **8.1%** |
| `rl` | 160 | **10** | 146 | **4** | **6.2%** |
| `perfect-trappiness` (2026-09-15, 40 games) | 40 | 4 | 35 | 0 | 10.0% |

**The criterion fails.** The trained player wins less often than bare `perfect`
*and* loses four games `perfect` never loses. The ordering runs the wrong way
across all three: trappiness 10% > perfect 8.1% > rl 6.2%. On significance, 10
against 13 wins at n = 160 is not a wide gap and per-match variance is severe
(one 20-game match gave `perfect` 7 wins, three others gave it none), so this is
not evidence that `perfect` is decisively stronger — but there is no hint of the
RL player being stronger, and the 4 losses are a real one-sided cost.

The premise is intact: qwen is demonstrably fallible, 42 mistakes across the 320
games. What fails is the exploitation. The mistake records show the trained player
erring at **14.4% of games against qwen's 13.1%** — about the same rate as the
opponent it was built to exploit. It threw a drawn position away 19 times; qwen
converted only 4 of those 19, which is the whole reason its record looks
respectable. `perfect` would have converted all 19.

**Why, mechanically.** The shaping term rewards creating a *split* threat. Per
opportunity that is the right target — such positions are ~5.3x more dangerous to
this model than single-threat ones
([`SPEC-mistake-model.md`](SPEC-mistake-model.md), "Validation against 588 real
blunders"). But single threats are 8.8x more common, so **62% of qwen's losing
blunders are unanswered single threats**, which the shaping term does not reward
at all. The signal was correctly aimed and incomplete.

**And the bar was understated.** Beating `perfect` against a fallible opponent
requires converting mistakes at least as well as exact search — which `perfect`
does by construction, finding every forced win — while *inducing* more of them.
`perfect-trappiness` already induces with exact trap counting, and beats both. So
the real target is not "beat perfect play" but "out-induce an exact trap counter,"
which this document never framed as the bar. Nothing measured suggests a learned
policy is close to it.

### Shaping on move preference instead: also a null result, 2026-10-07

The gap identified above — that the shaping term rewards split threats while 62%
of the target's losses are unanswered *single* threats — was acted on. `rollout`
gained a second potential, selected by `train.py --shaping`:

**Φ(s′) = 2 · P(the opponent fails to answer our live threats)**, using the
fitted move-preference model
([`SPEC-mistake-model.md`](SPEC-mistake-model.md), "A learned move-preference
model"). A reply counts as an answer when it kills every live threat, which is
bit arithmetic, so this costs 0.31 ms and needs no search — asking `evaluate()`
per reply would have cost ~120 search calls a step and been unaffordable.

It is far more *sensitive* than the three-level score (mean 0.96 against 0.36 at
the 588 real blunder positions) and much less *specific* (non-zero at 34% of
ordinary positions against 2%), the fitted weights coming from blunders only with
no example of the target playing well.

A controlled run — identical to the shipped checkpoint's configuration, same
label file, `--perfect-share 0.02`, same seed, 8-hour budget, 3,047 iterations
against its 3,100, **only the shaping differing** — produced
`rl-models/rl-move-pref.pt`. Measured against `qwen-3-8-rtx`:

| arm | games | W | D | L | win rate | loss rate |
|---|---:|---:|---:|---:|---:|---:|
| `rl-move-pref` | **720** | 58 | 617 | 22 | **8.06% ±1.01** | 3.06% |
| `perfect` (bare) | 160 | 13 | 145 | 0 | 8.12% ±2.16 | **0%** |
| `rl` (threat-score) | 160 | 10 | 146 | 4 | 6.25% ±1.91 | 2.50% |

**A dead heat: −0.07% ± 2.39%, or 0.03 standard errors.** The criterion is still
not met, now on 720 games rather than 160, and `perfect` remains better on losses
by having none. The nominal 1.8% gain over the threat-score term is also inside
the noise (0.8 SE) and is not claimed.

**A warning about sample size, recorded because it nearly became a false
entry here.** The first 160 of those games gave 17 W — a 10.62% rate, above
`perfect`, which read as the criterion being met. It sat 2.4 SE above where 720
games put the true rate. The gap over `perfect` was 0.8 SE at the time and was
not written up on that basis; the next 560 games returned 7.32%. **Do not
conclude anything about this comparison from a few hundred games.** The variance
is mostly ordinary binomial noise, not an artifact to be designed away: six
100-game samples of one checkpoint against `perfect` have an SD of 3.44 where
binomial noise alone predicts 2.76. `tools/paired_llm_eval.py` plays every arm
over the identical 25 seeds in both seats, which cannot hurt and costs nothing,
but measured only ~1% tighter than unpaired on its first run — it is not a
substitute for games. Separating a 6-point difference at two standard errors
takes ~250 games an arm.

Robustness did not pay for any of it. On an identical balanced schedule of 100
games each, `rl-move-pref` lost 6% to bare `perfect` and 10% to
`perfect-trappiness`, against the shipped checkpoint's 3% and 12% — differences
inside this measurement's noise either way.

`rl-models/rl.pt` therefore remains the shipped player. `rl-move-pref.pt` is kept
beside it only so the 720-game figure above stays reproducible; nothing loads it
unless `SNAKES_AND_MICE_RL_MODEL` names it.

**What three attempts now say together.** Exact trap counting
(`perfect-trappiness`), a hand-coded bias score (`perfect-mistake-model`,
`threat-score` shaping) and a fitted move-preference model all land in the same
6–10% band against this target. The target hands out roughly 8% winnable games
to any competent opponent, and *converting more of them than exact search does*
is the part none of these has moved.

What is not ruled out is the rest of the loop. Steps 2-6 have never been run: the
agent has never played an LLM in training, and its pool was self-play 58.8%,
`random` 39.2%, `perfect` 2%. A learned move-preference model fitted to the
collected blunders (ibid.) predicts qwen's chosen move at 9x the chance rate and
captures a mechanism the three-level score cannot — that this model plays offence
when it should defend. Shaping on that, rather than on split threats alone, is the
untried idea with evidence behind it.

### What is still open

- **Steps 2–6 of "The loop."** Nothing here spends live-LLM time. The mistake
  model enters training only as the hand-crafted shaping term; no probing,
  densifying, or fitting against a real model has been done.
- ~~**The comparison the player exists to win.**~~ Measured, and failed — see
  "Measured against qwen-3-8," above. What remains open is whether steps 2-6 can
  close the gap, not whether step 1 alone did.
- **Hyperparameters and curriculum.** Settled only to the extent a working run
  needed them. Nothing above claims a tuned value, and the one genuinely forced
  choice — the opponent mix — was forced by measured cost, not by learning
  outcomes.
