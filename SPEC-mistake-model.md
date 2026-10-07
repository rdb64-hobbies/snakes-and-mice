# The Mistake Model

> Part of the Snakes and Mice specification. This document covers **only** the
> mistake model: the empirical investigation into whether, and how, LLMs make
> *systematic* (not merely random) mistakes at Snakes and Mice, and the
> hand-crafted scoring function built from that investigation. It has two
> consumers, each specified in its own document, not here — the
> reinforcement-learning player, in [`SPEC-rl-player.md`](SPEC-rl-player.md)
> (a reward-shaping potential and a data-collection ranking function), and the
> algorithmic player, in [`SPEC-perfect-player.md`](SPEC-perfect-player.md) (a
> tie-break key for the `perfect-mistake-model` variant, "Selecting a
> variant"). The game, the `Player` abstraction, matches, tournaments, and the
> CLI are in [`SPEC.md`](SPEC.md).

> **Status: early, partial draft.** This document covers what has actually
> been measured or decided so far — the go/no-go finding that exploitable
> mistakes exist at all, and that they are decisive rather than marginal (see
> "Result: the premise holds" below), what the mistakes look like (an initial
> column/diagonal read that a later controlled test did not support — see
> "What the mistakes look like" below for both the finding and its
> retraction), a follow-up finding that simultaneous threats are harder to
> defend than one ("A second lever" below), the model's decided form — a
> hand-crafted score, built on the one feature with controlled support, now
> serving three roles: a reward-shaping potential, the ranking function for
> targeted data collection, and the tie-break key for a third `perfect`
> variant (see "The mistake model" below) — and where cross-model testing of
> that feature currently stands: replicated on 2 of 3 models tried, at a
> smaller and noisier magnitude than first measured, with one model showing no
> effect so far.
> **Note, 2026-10-07:** every constructed-position probe in this document ran with
> the defender unable to see its own pieces, a harness defect now fixed. It
> retracted the second lead entirely (recency of placement) and left the
> double-threat feature resting on real-game evidence instead of on the probes
> that found it — see "The replay harness withheld the defender's own pieces".
> **The score itself exists as of 1.8** — the `mistake_model` module, three
> levels over one feature (see "The score, as implemented" below) — along with
> the third of its roles, `perfect-mistake-model`; the other two wait on the RL
> player. Left for later: the model's exact weights and thresholds beyond its
> current single feature (§"The mistake model"), and how the remaining consumer
> actually implements its roles, which is specified in that consumer's own
> document.

## Finding out whether the premise holds

The whole strategy in [`SPEC-rl-player.md`](SPEC-rl-player.md), "Training
strategy" assumes LLMs make mistakes worth exploiting at all — a legal move
that is not just imperfect but *reliably*, *detectably* worse than what was
available. That is an empirical question, not an assumption to build on faith,
and it was the go/no-go gate for that strategy. It has now been measured, and
it came back positive ("Result", below).

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
`--mistakes-file` (SPEC.md §5, §7), against the trap-ranking perfect player — whose
ranking (SPEC.md §10) makes it the best available provoker of a fallible opponent, and
the closest stand-in for what a trained exploiter would itself be. It was named
`perfect` when these measurements were run and is `perfect-trappiness` as of 1.8
(SPEC-perfect-player.md, "Selecting a variant"); the bare name now selects the
unranked player, which would provoke less. Because a match fixes sides
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
roughly one per ten games they are frequent enough to collect. The training
strategy they motivate (`SPEC-rl-player.md`) is worth pursuing. What is *not* yet
established is the premise it actually rests on — that these mistakes are
**systematic**, not scattered — which needs many more of them than four, and is
what the collected file is for.

## What the mistakes look like

Two more characterizations of `qwen-3-8-rtx`'s mistakes, beyond that they exist —
both bear directly on what a mistake model (`SPEC-rl-player.md`, "The loop," step
4) would need to represent.

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

> **Re-measured 2026-10-07 and much weaker than this section reports.** The probe
> below ran with the defender unable to see its own pieces; re-run correctly it
> gives 8/8 single against 7/8 double, not 4/4 against 2/4. The feature survives
> on *real-game* evidence instead (§"Validation against 588 real blunders": 5.3x
> worse per opportunity over 588 blunders), not on what follows. Read this section
> as the history of how the feature was found, and "The replay harness withheld
> the defender's own pieces" for what it is actually worth.

The training strategy in [`SPEC-rl-player.md`](SPEC-rl-player.md) assumes
creating one three-piece live line is something a policy could learn to reach
for — win/loss reward alone should reinforce it, since it is already what
causes a mistake in "Result," above. A natural next question: does creating
**two** simultaneous three-piece threats work better, on the theory that
tracking two developing threats at once is harder than tracking one? That is a
claim about attention, not about the game tree — even against a fallible
defender, two threats each needing one more cell are still blockable in a
single two-piece move, unless there are three or more.

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

This sharpens something [`SPEC-rl-player.md`](SPEC-rl-player.md)'s training
strategy was already exposed to but did not name. Its loop mixes a mistake
model into training so the policy learns to exploit found bias — but a
terminal win/loss reward correlates only weakly with having *created* the
intermediate state that caused the win, and worst of all for a state that is
itself rare. A single three-piece threat is common in ordinary play (self-play,
`random`, `perfect` opponents) and would turn up often enough for a
terminal-reward signal to reinforce it. **A deliberately engineered second
threat, crossing the first at a chosen cell, is not something self-play
naturally produces at any useful rate** — so the exact lever this section just
confirmed works is also the one a sparse terminal reward is least equipped to
teach a policy to reach for.

This is resolved, not left open: the requirement identified here — whatever
training approach is chosen needs an answer for rewarding progress toward a
rare, high-value intermediate state, not only the terminal win or loss — is
exactly what "The mistake model," below, decides, once
[`SPEC-rl-player.md`](SPEC-rl-player.md), "The algorithm and network" settled
on an algorithm for it to plug into. Potential-based reward shaping turned out
to be the mechanism: it can reward something like "how many live simultaneous
threats do I hold" as a running signal without provably changing what the
optimal policy is (see "Three roles, one function," below).

## The mistake model

Two open items — the exact form the mistake model takes, and how to reward
progress toward a rare, high-value intermediate state (§"A second lever,"
above) — turn out to be one decision, not two.

### Decided: a hand-crafted, feature-based score, not a learned model

The mistake model is a **scoring function over board features**, not a trained
classifier. A formula built directly from a validated mechanism already encodes
as much of what is known as a learned model plausibly could without more data
than exists, and without risking a fit to `qwen-3-8-rtx`'s specific noise rather
than whatever of it is actually shared across models
([`SPEC-rl-player.md`](SPEC-rl-player.md), "Goal," "Breadth is open"). It is
also automatically "generalizing" in the sense `SPEC-rl-player.md`'s loop
already requires of step 4: it is a function of structure, computable at any
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

This list decides only which feature is worth starting from; how it is turned
into a number is "The score, as implemented," below. A second candidate —
recency of placement — turned up after this section was written; it is
documented (§"Generalizing across models," below) but deliberately kept out of
this list, since its evidence does not yet meet the same bar this one does.

### The score, as implemented (1.8)

The one feature above, made arithmetic. On a position with the **defender** to
move, a **live threat** is a line holding three or more of the attacker's
pieces and none of the defender's — one or two gaps, either of which wins next
turn, since a move places two pieces (SPEC.md §2.5). The defender answers a
threat by playing any empty cell of that line, which kills it for good
(SPEC.md §2.7). The score is then one of three levels:

| Score | Condition |
|---:|---|
| 0 | fewer than two live threats |
| 1 | two or more, and one row- or column-**aligned** move answers them all |
| 2 | two or more, and no aligned move does — the defender must **split** |

Level 0 is the measured baseline that a lone threat is defended reliably
whatever its line type (§"Isolated single threats are almost never missed"),
so it is no reason to prefer one move over another. Level 2 is the measured
feature itself: a threat set requiring two pieces in different rows *and*
different columns, against a habit of placing them in one (§"A second lever").
Alignment means a shared row or column only, not a diagonal — that is the
measured habit, and the only alignment with a lexical cue in the move history
an LLM reconstructs the board from. A threat set no *pair* of cells can cover
is a forced win rather than a defensive problem; it lands in level 2 by the
same rule rather than needing a case of its own, no aligned pair covering it
either.

Three coarse levels rather than a tuned formula is the deliberate reading of
§"What validation means for a shaping term": only the **direction** of this
feature is validated, and with a single feature in the model there is no second
term to weigh it against. A finer score would be asserting precision the
evidence does not carry. Promoting a second feature is what forces the weight
question, and it is that promotion's to answer.

### Three roles, one function

The same score serves three consumers:

- As a **potential function** for potential-based reward shaping — added to the
  terminal reward in the standard `γΦ(s') − Φ(s)` form, which provably leaves the
  optimal policy unchanged while giving a per-move signal that directly reinforces
  building toward the configurations already shown to be hard to defend. Used by
  [`SPEC-rl-player.md`](SPEC-rl-player.md), "The algorithm and network."
- As the **ranking function** for [`SPEC-rl-player.md`](SPEC-rl-player.md),
  "The loop," step 3 ("densify") — which constructed positions are worth
  spending a live-LLM query on, rather than probing uniformly.
- As the **tie-break key for `perfect-mistake-model`**, decided 2026-09-18 and
  shipped in 1.8 ([`SPEC-perfect-player.md`](SPEC-perfect-player.md),
  "Selecting a variant"): among moves already tied on the exact trap count,
  that variant prefers whichever leaves the opponent facing the higher score,
  in place of the trappiness variant's liveness key. It replaces liveness rather than joining
  it because both exist for the same reason — approximating exploitability
  where exact trap-counting is too deep to run — and this score is the
  better-evidenced approximation of the two.

Adding this third consumer is why the caveat below needs its own paragraph:
the first two never needed to know the score's magnitude, but a tie-break is a
ranking, and rankings are exactly where magnitude would seem to matter.

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

That is because, for the shaping-term and densify-ranking roles, the score is
used as a **potential-based shaping term** ("Three roles," above), which is
provably policy-invariant *regardless of the weight given it* — getting the
size of the effect wrong leaves the shaping term mis-calibrated, not harmful.
What would actually undermine it is a feature that sometimes points the wrong
way — helps the defender rather than the attacker on some fraction of
positions. Nothing measured so far does that: every matched pair, on every
model tried, shows double-threat performance equal to or worse than
single-threat, never better. **That is what "validated" should require here —
the sign is right, not that the exact multiplier is known** — and it is what
the cross-model results below satisfy.

The tie-break use ("Three roles," above) looks like it should reopen the
magnitude question — a ranking, unlike a shaping term, is not policy-invariant
to getting the weight wrong. It doesn't, but only because there is currently
exactly **one** feature: with a single term, "prefer the higher score" and
"prefer the direction that's validated" are the same statement, so no weight
is actually being chosen. That stops being true the moment a second feature is
added to the list in "Candidate features" — combining two scores into one
ranking requires deciding their relative weight, which direction-only
validation cannot supply. Whoever promotes a second feature into this model
inherits that decision; it is out of scope while the list has one entry.

## Generalizing across models: mixed, and a new lead found along the way

### The double-threat effect replicates on 2 of 3 models, at a smaller and noisier magnitude than first measured

> **Measured through the defective harness** (see "The replay harness withheld the
> defender's own pieces," 2026-10-07) and not since re-run. Treat every number
> below as unverified.

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
a real exception, not yet explained. "Breadth is open"
([`SPEC-rl-player.md`](SPEC-rl-player.md), "Goal") should be read accordingly:
this feature helps against most of what has been tried, not everything.

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

Six of ten early placements were forgotten; zero of six late ones were.

**Retracted, 2026-10-07.** This was a harness defect, not a recency effect: the
replay never relayed the defender's own moves at all, so those cells were not the
oldest fact in the history but absent from it. Re-run with the board actually
relayed, every one of them is correctly avoided — see "The replay harness withheld
the defender's own pieces," below. The qualifications that follow were written
when the effect was believed real; they are kept because the counter-examples
among them now read as the signal rather than the noise:

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

### The replay harness withheld the defender's own pieces, 2026-10-07

**Every probe above was run with the defender unable to see its own pieces.**
`tools/threat_scenarios.py` built each position by replaying the move sequence
through `start_game` / `observe_move`, which SPEC-rl-player.md's step 3 states is
all a `Player` needs. That holds for every mechanical player. It is false for the
LLM player, whose `observe_move` **returns early for its own side** — in a real
game its own moves are already in the thread as its own structured responses, so
relaying them would duplicate them. In a constructed position it never made
those moves, so they existed nowhere.

Verified directly by replaying `mouse-defends-single-row` and reading the prompt.
The defender owns A2, B4, D3 and D4; the entire board information it received was:

    A new game begins. You are playing mouse. The snake is seeded at C3.
    Your opponent (snake) played B2 C1.
    Your opponent (snake) played C2 E1.

The preamble even tells the model to track the board "from the seeded snake,
**your own moves**, and your opponent's" — and its own moves were never said.

Fixed by `Player.assume_own_move(side, move)`, a new optional hook (SPEC.md §3
allows these): the default delegates to `observe_move`, which is right for every
player that rebuilds the board from what it is told, and the LLM player overrides
it to relay "You (mouse) played A2 B4." The relay is a user message where a real
game carries the move as the model's own response — the same information, a
different form, so a probe is now *close* to real conditions rather than
identical to them.

#### What the re-runs say

All three probes re-run against `qwen-3-8-rtx` with the board fully relayed.
**Zero illegal moves across all 31 scenarios**, against five reoccupations in the
prior data.

**Recency of placement was an artifact.** Each of qwen's three recorded
reoccupations is now clean, same scenario, same cell: `mouse-defends-row-A` (was
C4), `snake-defends-row-D` (was A4), `snake-defends-single-row` (was B4).
`probe_recency.py` returns 6 of 6 cells correctly avoided, both early conditions
included, against a prior 6 of 10 early placements "forgotten". The cells were
never old-and-forgotten; they were absent. The finding is retracted below.

**The double-threat effect largely vanished from the probe.**

| | before (broken harness) | re-run |
|---|---|---|
| single threat | 4/4 defended | **8/8** |
| double threat | **2/4** defended | **7/8** |

A 50-point gap became 12.5 points — one miss in eight. The reason the fix mattered
here is not that the threats were hidden (the *attacker's* moves were always
relayed) but that covering two threats requires finding two *legal* cells, and
half the board was invisible. Failing was close to unavoidable.

**Single-threat defence**: 8/8 in `probe_multi_threats`, 10/12 in
`probe_line_types` (misses on `mouse-defends-row-E` and
`snake-defends-main-diag`), so 18 of 20 overall. Note these probes take **one
sample per scenario** at the server's default temperature, so 10/12 against the
old 12/12 is well within stochastic variation and is not evidence the fix made
defence worse.

`snake-defends-main-diag` returned a **one-piece move**, which a real game would
lose on as `WRONG_PIECE_COUNT` but which `Response.legal` scored as a plain miss,
since it only tested reoccupation. A piece-count check was added at the same time.

#### What survives

The double-threat feature does, but **on real-game evidence rather than on the
probe that motivated it**. §"Validation against 588 real blunders" measured it
from ordinary play, never touching this harness: `SCORE_SPLIT` positions are 2.0%
of strong-play positions and 33.8% of real losses, a per-opportunity failure rate
**5.3x** worse than single threats. 588 real errors outweigh eight constructed
pairs, and the shipped `mistake_model` score keeps its justification.

What does not survive is the probe evidence as *independent* support. Anything in
this document resting on a `threat_scenarios` run before 2026-10-07 — including
the cross-model replication in §"The double-threat effect replicates on 2 of 3
models" — was measured under the same defect and should be re-run before being
relied on.

### Status: retracted — it was the harness, not the model

Recency of placement is **not a finding about the model at all.** It was the
replay harness withholding the defender's own pieces, so the "forgotten" cells
were never in the history to be forgotten (see "The replay harness withheld the
defender's own pieces," below). With the board relayed, all of them are avoided.
It is not a candidate feature and should not be probed further on this basis.

The reasoning that kept it out of "Candidate features" is worth preserving,
because it was pointing at the answer. The objection recorded here was that the
effect "degrades on some cells and models and not others, for reasons not yet
identified — what makes `A1`, `B4`, `C4`, and `A4` forgettable but not the two
cells picked for the fresh pairs is an open question, not a settled mechanism."
That question had an answer: the forgettable cells were the ones a *constructed*
scenario had assigned to the defender, and the fresh pairs' cells were avoided
because the model never played near them rather than because it tracked them. A
pattern that cannot predict which cells get forgotten, only explain the ones that
already were, is the signature of an artifact — and declining to promote it on
exactly that ground is what kept it out of the shipped score.

## Validation against 588 real blunders, 2026-10-05

Every finding above rests on enriched probes of a dozen or two constructed
positions. By now `mistakes.jsonl` holds **588 `draw->loss` blunders by
`qwen-3-8-rtx`** recorded in ordinary play, each with the full board and the move
chosen — a corpus two orders of magnitude larger than any probe here, and one the
model was never fitted to. Measured with `tools/fit_move_preference.py`.

### The feature holds, and now has a per-opportunity number

Positions at `SCORE_SPLIT` are **2.0%** of the to-move positions arising in
strong play (measured over 457 positions of `rl`-vs-`perfect`) but **33.8%** of
qwen's `draw->loss` blunders. Scoring the base rate against looser play instead
(12.7%, from the random playouts in `tools/rl/solved.py`) the enrichment is 2.7x
rather than 17x; the direction is firm, the multiplier is not.

Better, the two can be divided. Single-threat positions are 8.8x more common than
double-threat ones (32.6% against 3.7%), while the blunders split only 1.6 : 1 —
so **the per-opportunity failure rate is ~5.3x worse on two threats than on one.**
That is the same claim as §"A second lever" (4 of 4 defended, falling to 2 of 4)
with n = 588 behind it instead of n = 8, and it is the number that matters for a
player deciding what to engineer.

`win->draw` blunders are 100% at score 0, as they should be: missing one's own
win has nothing to do with the opponent's threats, and the model makes no claim
about it.

### But most of the *volume* of losses is single-threat

62.4% of the blunders sit below `SCORE_SPLIT`, and of those **93.3% are positions
holding exactly one live threat the defender failed to answer** (363 of 389; 4
held none, 22 held two that an aligned move covered).

This does **not** contradict §"Isolated single threats are almost never missed."
That section measures a per-opportunity rate and is upheld by the 5.3x above; this
measures the composition of failures, and the two reconcile through the 8.8x
frequency difference. A situation can be individually safe and still be where most
losses come from.

What it does bound is headroom. A strategy built only on the double-threat
feature is aiming at the third of the losses with the highest per-chance payoff,
and ignoring the two thirds that arrive by volume. Whether that is the right trade
depends on whether a player can *engineer* double threats often enough to beat
taking the common case — see `SPEC-rl-player.md`, "Measured against qwen-3-8",
where it did not.

### A learned move-preference model, and what it says

The "hand-crafted, not learned" decision above was taken explicitly because a
learned model risked fitting noise "without more data than exists." **That premise
has changed**, so it is revisited here rather than left standing.

Each record names a chosen move from a fully specified board, and every other
legal move from that board is one that was rejected — so preference over moves
*within* a position is learnable by softmax cross-entropy (a conditional logit),
with no need for the "played this correctly" examples the file does not contain.
Only move-level features can contribute: a position-level one is constant across
candidates and cancels. 11 features, 411 train / 177 test positions, 96 candidate
moves each:

| feature | weight |
|---|---:|
| `answers_all` — kills *every* opponent threat line | **-4.81** |
| `blocks_threat` — threat lines killed | **+3.25** |
| `creates_threat` — leaves a live threat of its own | **+2.41** |
| `aligned` — both cells share a row or column | **+1.75** |
| `advances_own` — lines extended, opponent absent | +1.08 |
| `creates_split`, `center`, `corner`, `adjacent`, `on_diagonal` | <= ±0.4 |

Held-out top-1 accuracy is **13.6% against a 1.5% chance rate** — a 9x lift — and
the chosen move lands in the top 10% of candidates on average. Train and test
agree (15.1%/89.9% against 13.6%/89.5%), so eleven features over 411 positions is
not overfitting.

The reading: **this model plays offence when it should defend.** It likes blocking
threats (+3.25) but specifically shuns the move that blocks them *all* (-4.81),
while favouring moves that build its own lines (+2.41, +1.08). Partial defence,
as a move preference rather than as a property of the position — which is a
mechanism the three-level score cannot express, since that score reads the board
and not the choice.

It also **confirms the alignment habit on real data**: `aligned` at +1.75. §"The
score, as implemented" asserts a shared row or column is the measured habit, and
"the only alignment with a lexical cue in the move history an LLM reconstructs the
board from," on the strength of enriched probes. It now carries a fitted positive
weight over 588 ordinary-play errors.

`wins_now` fits to exactly 0.000, which is the right sanity check rather than a
result: no winning move can exist at a `draw->loss` position, so the feature never
varies and should be dropped.

### What this corpus cannot say

It is **positive-only**. `mistakes.jsonl` records a mistake but neither the
opponent nor the moves played well, so every rate above needing a denominator
borrows one from locally generated `rl`-vs-`perfect` games — a proxy, since qwen's
own play shapes the positions it reaches. Two changes would remove the guesswork:
record the opponent on each mistake, and optionally log every graded move rather
than only the bad ones. Both want fresh games, so neither is free.

And a model fitted to blunders alone predicts move choice *given* that something
went wrong; it will overstate how error-prone the player is in general. It is a
model of how this player errs, not of how it plays. That bias is not theoretical:
used as a shaping potential it fires at 34% of ordinary positions against the
three-level score's 2%.

### Used as a shaping term: no measured gain

The obvious consumer is `SPEC-rl-player.md`'s shaping potential, and it was
tried — see that document, "Shaping on move preference instead". Over 720 games
against `qwen-3-8-rtx` a policy trained on it wins 8.06% ±1.01 where bare
`perfect` wins 8.12% ±2.16: a dead heat, 0.03 standard errors apart. The model
predicts the target's choices well (9x chance) and that did not convert into
wins.

This does not retract anything above. The feature findings are measurements of
how the target errs and stand on their own; what fails is the inference that
predicting errors well is enough to exploit them better than exact search does.
