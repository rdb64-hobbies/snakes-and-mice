# The Reinforcement-Learning Player

> Part of the Snakes and Mice specification. This document covers **only** the
> reinforcement-learning (RL) player; the game, the `Player` abstraction, matches,
> tournaments, and the CLI are in [`SPEC.md`](SPEC.md), which lists this player
> as planned in §3.

> **Status: early, partial draft.** This document covers what has actually been
> decided or measured so far — the goal, the overall training strategy, and the
> investigation into whether that strategy's premise holds (it does; see "Result"
> below). It deliberately does **not** cover the training algorithm, any model or
> network architecture, or the exact form the "mistake model" (§"Training strategy"
> below) will take — none of that has been decided yet, and this document will grow
> to cover it once it is.

## Goal

The **RL player** is not an attempt to out-play the algorithmic player (§10 of
SPEC.md) — that player is already provably optimal, and the game is a solved draw
from every seed. The goal instead is narrower and specifically about the project's
actual subject, LLMs (SPEC.md §1): produce a trained player that beats an LLM
player **more often than the perfect player does**, on the premise that an LLM's
mistakes may be systematically biased rather than random — and a player that
specifically learns to exploit that bias could out-score pure optimal play against
a fallible-but-patterned opponent, even though it cannot out-score optimal play in
the game-theoretic sense.

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

**Explicitly undecided**, and out of scope for this document until settled: the RL
algorithm itself, any network or function-approximator architecture, and the exact
form the mistake model takes (what it's fit on, what class of model it is, how
"mixing it into training" works mechanically in step 5).

## Finding out whether the premise holds

Everything above assumes LLMs make mistakes worth exploiting at all — a legal move
that is not just imperfect but *reliably*, *detectably* worse than what was
available. That is an empirical question, not an assumption to build on faith, and
it was the go/no-go gate for the rest of this document. It has now been measured,
and it came back positive ("Result", below).

### What counts as a mistake

A mistake here is **not** a `PlayerFaultReason` fault (SPEC.md §3) — an illegal
move or a misread outcome. Faults already have their own detection and tally
(SPEC.md §3, §6) and are a different failure mode entirely; they are not the
subject of this investigation. A mistake is a **legal, accepted move that gives up
value**: because the game is fully solved, every reachable position has an exact
game-theoretic value under perfect play, so any move that lowers that value (from
the mover's own perspective) is, by definition, not the best available move —
whether or not it looks like ordinary play and whether or not it triggers any
fault at all.

This grading mechanism, and the severity split below, are general-purpose and are
specified once, in SPEC.md §5 ("Flagging mistakes") and §10 (the
`evaluate(board, side)` primitive) — this document does not repeat that
specification, only how it is being used here:

- A mistake is **hard** if it crosses a win/draw/loss boundary (a drawn position
  played into a loss, a won one played into a draw or a loss), and **soft** if it
  stays within the same class but settles for a worse depth.
- Grading reads the engine's authoritative board directly at the move boundaries
  (an `Observer`'s `on_move_start` / `on_move_end` hooks, SPEC.md §3) — not by
  parsing an LLM's logged conversation, which deliberately withholds board state
  from the model in the first place (SPEC.md §4) and so cannot be a source of
  ground truth about it.

### How the measurement is run

No special tooling: this is `play-match` with `--flag-mistakes` and
`--mistakes-file` (SPEC.md §5, §7), against `perfect` — whose trap-ranking (SPEC.md
§10) makes it the best available provoker of a fallible opponent, and the closest
stand-in for what a trained exploiter would itself be. Because a match fixes sides
for its whole duration (SPEC.md §5), covering both seats means running it twice.

That the general capability and this investigation's needs turned out to be the
same thing is not a coincidence: "which legal moves gave something away" is one
question, and it has one answer whoever is asking. A separate script existed
briefly and was removed once the flags landed; the recorded file is the interface
this document depends on, not any particular program.

### Result: the premise holds

Measured 2026-09-07 against `qwen-3-8-rtx`, 40 games (20 per seat), randomized
openings:

- **188 moves graded, 4 mistakes — all four hard**, and no soft ones at all. This
  model does not drift; it either finds the exactly optimal move or throws the game
  away in one move.
- **The four mistakes are exactly the four games it lost.** Since `perfect` never
  errs, it can only ever win a game the opponent hands it, and the grading located
  the hand-off every time — each a drawn position played straight into a loss.
- Zero faults across all 40 games, which is the point of measuring mistakes at all:
  a fault tally would have reported this model as flawless.

So mistakes worth exploiting exist, they are decisive rather than marginal, and at
roughly one per ten games they are frequent enough to collect. The strategy above
is worth pursuing. What is *not* yet established is the premise it actually rests
on — that these mistakes are **systematic**, not scattered — which needs many more
of them than four, and is what the collected file is for.
