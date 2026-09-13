# The Reinforcement-Learning Player

> Part of the Snakes and Mice specification. This document covers **only** the
> reinforcement-learning (RL) player; the game, the `Player` abstraction, matches,
> tournaments, and the CLI are in [`SPEC.md`](SPEC.md), which lists this player
> as planned in §3.

> **Status: early, partial draft.** This document covers what has actually been
> decided or measured so far — the goal, the overall training strategy, the
> investigation into whether that strategy's premise holds (it does; see "Result"
> below), what the mistakes actually look like (a side-dependent column bias and
> an apparent diagonal blind spot — see "What the mistakes look like" below), a
> follow-up finding that simultaneous threats are harder to defend than one (see
> "A second lever" below), and the mistake model's form — a hand-crafted score
> over the features those findings identified, doubling as both a reward-shaping
> potential and the ranking function for targeted data collection (see "The
> mistake model" below). It deliberately does **not** cover the training
> algorithm or any network/function-approximator architecture — neither has been
> decided yet, and this document will grow to cover them once they are.

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
algorithm itself and any network or function-approximator architecture. The
mistake model's form, and how it rewards progress toward a rare, high-value
intermediate state before any terminal win/loss, turned out to be one decision
rather than two — see "The mistake model," below.

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

## What the mistakes look like

Two more characterizations of `qwen-3-8-rtx`'s mistakes, beyond that they exist —
both bear directly on what a mistake model (§"Training strategy," step 4) would
need to represent.

### Which line the opponent wins on depends on which side is playing

Measured 2026-09-13 against 552 recorded mistakes, by which of the 12 lines the
missed threat sat on:

| Side (n) | column | row | diagonal |
|---|---:|---:|---:|
| Mouse (269) | 67% | 22% | 7% |
| Snake (283) | 45% | 37% | 11% |

Column-dominant play only holds clearly on **one** side. An earlier, smaller-sample
read (pooling both sides, ~80% column) turned out to be mostly the Mouse-side
effect showing through — Snake-side was closer to an even row/column split all
along, just under-sampled. **Side matters here for the same reason SPEC.md §6
already treats it as a first-class axis for the tournament standings**: pooling
across sides risks reporting a side-specific effect as if it were general, and a
mistake model trained on pooled data would learn the wrong thing for whichever side
it under-represents.

### Diagonals appear to not be checked as a category at all

Two pieces of evidence, not yet a fully separate measurement: a diagonal miss on an
almost-empty board, and the same diagonal ignored by two different models across
four consecutive turns of one game.

`gpt-5-6-terra`, defending as Snake with only 7 pieces on the whole board (Mouse:
`A5`, `D1`, `D2`, `E1`; Snake: `B5`, `C5`, `D5` plus the seed), played `C1 C2` while
the anti-diagonal (`A5`, `B4`, `C3`, `D2`, `E1`) already held three Mouse pieces
with both `B4` and `C3` completely open (`0 → -996`). At 7 pieces there is nothing
to lose track of — this looks less like a tracking failure than like the diagonal
never being checked at all.

A cross-model game makes the same point more sharply. In one `gemma-4-dgx`
(Mouse) vs. `qwen-3-8-rtx` (Snake) game, seed `D1`, the main diagonal (`A1`, `B2`,
`C3`, `D4`, `E5`) reached three Mouse pieces (`A1`, `C3`, `D4`, gaps `B2`/`E5`) by
turn 4 and was still exactly as open after **four consecutive turns**:

- Turn 4 (Snake, qwen): plays `C5 B5` — doesn't touch `B2`/`E5`. Fails to block.
  `0 → -996`.
- Turn 5 (Mouse, gemma): the win is sitting there; plays `A5 B4` instead. Fails to
  take it. `996 → 0`.
- Turn 6 (Snake, qwen): the diagonal is still untouched by anyone; plays `A4 E4`.
  Fails to block again. `0 → -994`.
- Turn 7 (Mouse, gemma): plays `E1 B1`. Fails to take it again. `994 → 0`.

```
  1  2  3  4  5
A 🐭 ·  ·  🐍 🐭
B 🐍 ·  ·  🐭 🐍
C ·  🐭 🐭 ·  🐍
D 🐍 🐍 🐍 🐭 ·
E 🐍 ·  ·  🐍 ·
```

`B2` and `E5` — after four moves by two different models — are still both empty.
Neither the side that would win by completing the diagonal nor the side that would
lose by ignoring it ever engaged with it. A row shares a letter across its five
cells and a column shares a digit; a diagonal shares neither, so there is no
lexical cue to catch it while reconstructing the board from a move history alone
(SPEC.md §4). If this generalizes, it would be a stronger and more model-general
target than the column bias, which requires only *uneven* attention to rows and
columns — this would mean one whole category of line getting no attention by
default.

## A second lever: simultaneous threats

The training strategy above assumes creating one three-piece live line is
something a policy could learn to reach for — win/loss reward alone should
reinforce it, since it is already what causes a mistake in "Result," above. A
natural next question: does creating **two** simultaneous three-piece threats work
better, on the theory that tracking two developing threats at once is harder than
tracking one? That is a claim about attention, not about the game tree — even
against a fallible defender, two threats each needing one more cell are still
blockable in a single two-piece move, unless there are three or more.

### Measured 2026-09-13, against `qwen-3-8-rtx`

Four matched pairs of hand-built positions (`tools/probe_multi_threats.py`) — same
defender, same total piece count, one live three-piece threat versus two crossing
at a shared cell — put the model on move as the defender and checked whether its
move touched every threatened line.

- **Single-threat: 4 of 4 fully defended. Double-threat: 2 of 4.** Perfect defense
  collapses to a coin flip the moment there is a second thing to track, on
  positions otherwise matched.
- **Both misses share one mechanism**, and it is the one already visible in
  ordinary mistakes: of 126 mistakes recorded across four LLM roster players as of
  2026-09-08, 108 (86%) placed both cells of the move in the same row or column,
  against a ~37% baseline for picking two empty cells at random at that point in
  the game. Both failures here were themselves same-line pairs — one used both
  cells to (redundantly) block the threat it happened to be aligned with while
  leaving the second threat untouched; the other's pair blocked one threat with
  one cell and spent the other on a cell that helped neither. A defender that
  treated each placed piece as answering an independent tactical question would
  naturally split across two unrelated lines; a default habit of extending along
  one line cannot.

Four positions per condition is a real result, not a settled one — but the
mechanism appearing identically in both failures is more convincing than the count
alone, and it makes a specific, checkable prediction: a model with less of the
alignment habit should also show a smaller single-vs-double gap.

### Implication for the training strategy: reward is sparse exactly where the signal is rare

This sharpens something the training strategy (above) was already exposed to but
did not name. Step 5 mixes a mistake model into training so the policy learns to
exploit found bias — but a terminal win/loss reward correlates only weakly with
having *created* the intermediate state that caused the win, and worst of all for
a state that is itself rare. A single three-piece threat is common in ordinary
play (self-play, `random`, `perfect` opponents) and would turn up often enough for
a terminal-reward signal to reinforce it. **A deliberately engineered second
threat, crossing the first at a chosen cell, is not something self-play naturally
produces at any useful rate** — so the exact lever this section just confirmed
works is also the one a sparse terminal reward is least equipped to teach a policy
to reach for.

This is not solved here — the training algorithm remains explicitly undecided
(above) — but the requirement is now concrete rather than speculative: **whatever
training approach is chosen needs an answer for rewarding progress toward a rare,
high-value intermediate state, not only the terminal win or loss.** Potential-based
reward shaping is the standard tool for this shape of problem — it can reward
something like "how many live simultaneous threats do I hold" as a running signal
without provably changing what the optimal policy is — but which mechanism is
actually used is a decision for when the training algorithm itself is chosen, not
before.

## The mistake model

Two of "Training strategy"'s undecided items — the exact form the mistake model
takes, and how to reward progress toward a rare, high-value intermediate state
(§"A second lever," above) — turn out to be one decision, not two.

### Decided: a hand-crafted, feature-based score, not a learned model

The mistake model is a **scoring function over board features**, not a trained
classifier. Every finding so far has come with a mechanism, not just a
correlation — a diagonal's cells share no lexical cue at all, a column's shared
digit is harder to notice than a row's shared letter, and a threat set gets
missed specifically when defending it can't be done with one aligned two-piece
move. A formula built directly from those mechanisms already encodes as much of
what is known as a learned model plausibly could without more data than exists,
and without risking a fit to `qwen-3-8-rtx`'s specific noise rather than whatever
of it is actually shared across models (the "Breadth is open" goal, above). It is
also automatically "generalizing" in the sense step 4 of the loop already
requires: it is a function of structure, computable at any position, not a
lookup table.

### Candidate features, ranked by how much evidence currently supports them

1. **A live diagonal threat.** The strongest, most model-general signal found so
   far (§"What the mistakes look like," above) — no lexical cue links a
   diagonal's five cells at all.
2. **A threat set that cannot be covered by one aligned two-piece move.** This is
   the feature the double-threat result actually validated (§"A second lever,"
   above) — not "two threats" as such, but specifically two threats whose
   defense requires splitting across two lines, exactly where the alignment bias
   fails.
3. **A column threat against a Mouse-side defender specifically.** The column
   bias only held clearly for one side; a column threat against Snake isn't yet
   evidenced as any more dangerous than a row.

None of the actual weights or thresholds are decided here — only that these are
the features worth starting from, in roughly this order of evidential strength.

### Two roles, one function

The same score serves both remaining open items at once:

- As a **potential function** for potential-based reward shaping — added to the
  terminal reward in the standard `γΦ(s') − Φ(s)` form, which provably leaves the
  optimal policy unchanged while giving a per-move signal that directly reinforces
  building toward the configurations already shown to be hard to defend.
- As the **ranking function for step 3** of the loop ("densify") — which
  constructed positions are worth spending a live-LLM query on, rather than
  probing uniformly.

### Must be validated, not trusted on theory

Before this is relied on for either role, the formula needs the same treatment
the double-threat claim got: construct matched positions that vary only in what
the score predicts, and check the mistake rate actually tracks it —
`tools/probe_multi_threats.py` is the existing pattern to extend for this — not
accepted because the mechanism sounds right.
