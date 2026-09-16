# The Reinforcement-Learning Player

> Part of the Snakes and Mice specification. This document covers **only** the
> reinforcement-learning (RL) player; the game, the `Player` abstraction, matches,
> tournaments, and the CLI are in [`SPEC.md`](SPEC.md), which lists this player
> as planned in §3.

> **Status: early, partial draft.** This document covers what has actually been
> decided or measured so far — the goal, the overall training strategy, the
> investigation into whether that strategy's premise holds (it does; see "Result"
> below), what the mistakes look like (an initial column/diagonal read that a
> later controlled test did not support — see "What the mistakes look like"
> below for both the finding and its retraction), a follow-up finding that
> simultaneous threats are harder to defend than one (see "A second lever"
> below), the mistake model's form — a hand-crafted score, built on the one
> feature with controlled support, doubling as both a reward-shaping potential
> and the ranking function for targeted data collection (see "The mistake
> model" below) — and where cross-model testing of that feature currently
> stands: replicated on 2 of 3 models tried, at a smaller and noisier
> magnitude than first measured, with one model showing no effect so far; and
> a second, not-yet-decisive lead (recency of placement) found while testing
> that (see "Generalizing across models" below). It deliberately does **not**
> cover the training algorithm or any network/function-approximator
> architecture — neither has been decided yet, and this document will grow to
> cover them once they are.

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

**Caveat, sharpened by the controlled result below: this is a breakdown of
failures only, not a per-opportunity rate.** It says what fraction of qwen's rare
mistakes involved a column, not what fraction of column threats get missed —
there is no denominator here for how many column, row, or diagonal threats arose
in total, most of them presumably defended without incident. A line type can
dominate this table simply by coming up as a threat more often in ordinary play,
with no per-opportunity effect at all. See "Isolated single threats are almost
never missed," below, which measures the rate this table cannot.

### Diagonals: two vivid anecdotes, not a category effect (revised 2026-09-16)

**The strong claim originally drawn here — that diagonals are not checked as a
category at all — is retracted below.** The two anecdotes are real and are kept
as recorded; the generalization drawn from them was not tested against a
controlled comparison at the time, and the controlled comparison built afterward
does not support it.

`gpt-5-6-terra`, defending as Snake with only 7 pieces on the whole board (Mouse:
`A5`, `D1`, `D2`, `E1`; Snake: `B5`, `C5`, `D5` plus the seed), played `C1 C2` while
the anti-diagonal (`A5`, `B4`, `C3`, `D2`, `E1`) already held three Mouse pieces
with both `B4` and `C3` completely open (`0 → -996`).

A cross-model game makes the same shape of failure more sharply. In one
`gemma-4-dgx` (Mouse) vs. `qwen-3-8-rtx` (Snake) game, seed `D1`, the main
diagonal (`A1`, `B2`, `C3`, `D4`, `E5`) reached three Mouse pieces (`A1`, `C3`,
`D4`, gaps `B2`/`E5`) by turn 4 and was still exactly as open after **four
consecutive turns**:

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
Neither side ever engaged with it. At the time this read as a plausible
mechanism: a row shares a letter across its cells and a column shares a digit, a
diagonal shares neither, so there is no lexical cue to catch it while
reconstructing the board from a move history alone (SPEC.md §4). **What these two
anecdotes actually demonstrate, given the controlled result below, is not that
diagonals go unchecked — it is that something else was going on in both games**
(accumulated move history, or the same multi-threat mechanism §"A second lever"
describes) that happened to land on a diagonal both times. Two instances drawn
from real games, without a matched comparison, is not enough to tell "diagonals
are structurally invisible" apart from "rare failures land on all line types, and
these two happened to be memorable."

### Isolated single threats are almost never missed, regardless of line type

Measured 2026-09-16 against `qwen-3-8-rtx`, using `tools/probe_line_types.py`: 12
matched positions, one live three-piece threat each — a row, a column, or a
diagonal, two replicates of each, for both defending sides, all at the same
total piece count within a side (9 for Mouse, 11 for Snake).

**12 of 12 correctly identified and addressed the real threat.** Every row,
every column, every diagonal, on both sides. **Correction, 2026-09-16: 3 of the
12 were graded before a bug was found and fixed (see "Recency of placement,"
below) that let a move re-occupy an already-filled cell without being caught.**
All 3 still touched the actual threat with their other, legal cell — the
"no line-type effect" conclusion below is unaffected — but they were not the
clean successes originally reported here; they belong with the recency finding,
not with the 9 unqualified ones. This is still the first time any of these
findings measured a true per-opportunity rate rather than a breakdown of
existing failures, and it still says a lone, freshly-presented threat of any
line type is generally defended reliably — consistent with the tiny overall
mistake rate from "Result," above (4 of 188 graded moves) — just not as cleanly
as "12 of 12" first suggested.

This reframes both of the findings above rather than simply adding to them. The
column-dominant failure breakdown and the two diagonal anecdotes were never shown
to be *line-type* effects as such; what actually degrades performance, on the
only controlled comparison run so far, is a second simultaneous threat that can't
be covered by one aligned move (§"A second lever" — 4 of 4 single-threat positions
defended there too, dropping to 2 of 4 once a second threat was added, using the
same method as this probe). The evidence now points at **how much else is
competing for attention when a threat appears**, not at which of the 12 lines it
happens to sit on. The mistake model (below) is revised accordingly.

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
classifier. A formula built directly from a validated mechanism already encodes
as much of what is known as a learned model plausibly could without more data
than exists, and without risking a fit to `qwen-3-8-rtx`'s specific noise rather
than whatever of it is actually shared across models (the "Breadth is open"
goal, above). It is also automatically "generalizing" in the sense step 4 of the
loop already requires: it is a function of structure, computable at any
position, not a lookup table.

### Candidate features, revised 2026-09-16

The line-type features originally listed here — a live diagonal threat, a
column threat against a Mouse-side defender — are **removed**, not merely
reordered. Both were breakdowns of rare failures with no per-opportunity
denominator, and the controlled test built to check them
(§"Isolated single threats are almost never missed") found no effect: 12 of 12
single threats defended, evenly across every line type and both sides. Reusing
either as a feature now would mean scoring positions on evidence the newer,
better-controlled measurement didn't support — see the retraction in "Diagonals:
two vivid anecdotes, not a category effect."

That leaves one feature with controlled support:

1. **A threat set that cannot be covered by one aligned two-piece move.** The
   only feature with a matched, controlled result behind it (§"A second lever"):
   4 of 4 single-threat positions fully defended, 2 of 4 once a second threat
   requiring a split response was added — using the identical method that then
   found *no* degradation from line type alone.

This is now a list of one, which is a smaller and more confident starting point
than three ranked guesses. It does not close the door on line type mattering —
only on the specific claims made here without a controlled test. If further
probing turns up a line-type or side-dependent effect that survives the same
matched-position treatment, it belongs back in this list on that basis, not on a
failure-breakdown alone.

None of the actual weights or thresholds are decided here — only that this is
the feature worth starting from. A second candidate — recency of placement —
turned up after this section was written; it is documented (§"Generalizing
across models," below) but deliberately kept out of this list, since its
evidence does not yet meet the same bar this one does.

### Two roles, one function

The same score serves both remaining open items at once:

- As a **potential function** for potential-based reward shaping — added to the
  terminal reward in the standard `γΦ(s') − Φ(s)` form, which provably leaves the
  optimal policy unchanged while giving a per-move signal that directly reinforces
  building toward the configurations already shown to be hard to defend.
- As the **ranking function for step 3** of the loop ("densify") — which
  constructed positions are worth spending a live-LLM query on, rather than
  probing uniformly.

### What validation means for a shaping term: direction, not magnitude

Real per-move mistakes are rare — about 2% ("Result," above) — far too rare for
a handful of matched positions to measure directly. Telling a 1% mistake rate
apart from a 2% one (the same 2x relative effect this feature is about) needs
on the order of thousands of trials per condition, not the dozens any probe in
this document runs. Every probe here works around that by *enriching* the
position — presenting the exact tactical situation under study, every time,
rather than sampling real games — which is why they can see anything at all
with a small n (12–29 percentage points measured below, not 1–2). But it also
means a result here answers "does this situation reliably cause trouble when
engineered," never "how often it occurs, or how much it moves the needle, in
the wild." That is the only question this method is equipped to answer, and
for this feature's actual use it is the only one that needs answering.

That is because the score is used only as a **potential-based shaping term**
("Two roles," above), which is provably policy-invariant *regardless of the
weight given it* — getting the size of the effect wrong leaves the shaping
term mis-calibrated, not harmful. What would actually undermine it is a
feature that sometimes points the wrong way — helps the defender rather than
the attacker on some fraction of positions. Nothing measured so far does that:
every matched pair, on every model tried, shows double-threat performance
equal to or worse than single-threat, never better. **That is what "validated"
should require here — the sign is right, not that the exact multiplier is
known** — and it is what the cross-model results below satisfy.

## Generalizing across models: mixed, and a new lead found along the way

### The double-threat effect replicates on 2 of 3 models, at a smaller and noisier magnitude than first measured

The initial 4-pair probe (n=3–4 per condition, after excluding illegal
responses) found degradation only on `qwen-3-8-rtx`; `gpt-5-6-terra` and
`gemini-3-8` looked clean. Given how rare real mistakes are (above), that was
as consistent with "too few trials to see a real but smaller effect" as with
"no effect" — so the scenario set was doubled to 8 pairs (16 positions, 4 new
line-type combinations added to the original 4) and run again, all three
models on the identical set. Legal responses only:

| Model | Single-threat | Double-threat |
|---|---|---|
| `qwen-3-8-rtx` | 6/6 | 7/8 |
| `gpt-5-6-terra` | 6/6 | 5/7 |
| `gemini-3-8` | 6/6 | 8/8 |

**Two of three now show real degradation.** `qwen-3-8-rtx`'s own effect is
smaller here (88%, not the original run's 50%), and the specific scenario
responsible changed — one of the two double-threat positions it missed the
first time was clean this run, a different one missed instead. That is
ordinary sampling variability in the model's own output, not evidence the
earlier result was wrong, and it is exactly the kind of noise this method was
never going to fully resolve (above).

`gemini-3-8` still shows zero degradation, now at twice the trials (8 of 8) —
a real exception, not yet explained. "Breadth is open" (the Goal) should be
read accordingly: this feature helps against most of what has been tried, not
everything.

### A bug surfaced a bigger lead: recency of placement

The runs above only came back clean because of a fix made while running them.
`tools/threat_scenarios.py`'s `run()` never checked whether a returned move
reoccupied a cell the scenario already had filled — it just graded whatever
came back against the threat lines, so a move that replayed an old cell could
still look like a clean success if its *other* cell happened to touch the
threat. Two `gpt-5-6-terra` responses did exactly that. Once caught (`Response.
legal`, added 2026-09-16), the same check was run backward over every prior
probe:

- `gpt-5-6-terra`, `mouse-defends-double-row-col`: reoccupied `A1` in **both**
  runs it was tried.
- `gpt-5-6-terra`, `snake-defends-single-row`: reoccupied `B4`.
- `qwen-3-8-rtx`, `snake-defends-single-row`: reoccupied `B4` — the *same*
  scenario and cell `gpt-5-6-terra` later failed, an unplanned two-model
  replication sitting in data already collected.
- `qwen-3-8-rtx`, `probe_line_types.py`'s `mouse-defends-row-A` and
  `snake-defends-row-D`: reoccupied `C4` and `A4` respectively (the correction
  in "Isolated single threats," above).

In every one of those five, the reoccupied cell was that side's **first** move
in the constructed sequence — the oldest fact in the history. That is a
recency effect, not a general loss of board-tracking, and it is testable
directly: build the *identical final position* via a different construction
order, so the same cell lands on the last move instead of the first, and see
if it stops being forgotten.

`tools/probe_recency.py` does this — two pairs reusing the exact positions
above (only the "late" half run; the "early" half already existed), plus two
freshly-designed pairs run both ways, since no result should rest only on
positions found by accident. Full tally across every model tested:

| Condition | Instances | Forgotten |
|---|---:|---:|
| Early (first move) | 10 | 6 |
| Late (last move) | 6 | 0 |

Six of ten early placements were forgotten; zero of six late ones were. That
is a real, substantial gap, not a coincidence of the two positions where it
was first noticed. Three qualifications keep it from being cleaner than it is:

- **Only two of the three models tested ever showed it.** `gpt-5-6-terra`
  (3 of 3 early instances forgotten) and `qwen-3-8-rtx` (3 of 5) account for
  every forgotten instance; `gemini-3-8` never has (0 of 2) — the same
  qualification "A second lever" needed for the double-threat effect applies
  here too: not yet known to be general.
  `gemini-3-8`'s two correct instances are also weaker evidence than the
  others, since it simply never played near the cell at all, rather than
  demonstrating a pull toward it and then correctly leaving it alone the way
  the fresh pairs' late halves do.
- **One clear counter-example**: `qwen-3-8-rtx`, `probe_line_types.py`'s
  `snake-defends-col-2`, reoccupied `E3` — placed on the *last* snake move, not
  the first.
- **The two freshly-designed pairs showed no effect in either condition**, on
  `qwen-3-8-rtx` — `D1` and a second, unrelated `B4` were both correctly
  tracked whether placed early or late. Recency did not reproduce on demand for
  these cells the way it did for the three found by accident.

### Status: a real lead, not yet a feature

Recency of placement is not added to "Candidate features," above. The evidence
is substantial in volume but mixed in a way the double-threat feature's evidence
was not: that feature degraded cleanly and predictably across every matched
pair tried (§"A second lever"); this one degrades on some cells and models and
not others, for reasons not yet identified — what makes `A1`, `B4`, `C4`, and
`A4` forgettable but not the two cells picked for the fresh pairs is an open
question, not a settled mechanism. Promoting it would mean scoring positions on
a pattern that is still, honestly, "happens often but not reliably, on cells
that are hard to characterize in advance." Worth continuing to probe — a
promising next step would be varying board position or line role of the target
cell rather than only its recency — but it stays a documented lead, not a
scored feature, until it can predict which cells get forgotten rather than only
explain the ones that already were.
