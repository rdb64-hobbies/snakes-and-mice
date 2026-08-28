# The Algorithmic Player (Perfect Play)

> Part of the Snakes and Mice specification. This document covers **only** the
> algorithmic (perfect) player; the game, the `Player` abstraction, matches,
> tournaments, and the CLI are in [`SPEC.md`](SPEC.md), which summarizes this
> player in §10. The offline solver that produces its opening table has its own
> document, [`tools/solver/SPEC.md`](tools/solver/SPEC.md).

The **algorithmic player** plays the game **perfectly**: it searches the whole
game to its conclusion and always makes a game-theoretically optimal move. It is
the _calibrated, fixed-strength yardstick_ of SPEC.md §1 — the strong non-LLM opponent a
model can be measured against — and, unlike the LLM player, it never faults and
never misreads an outcome. Because the board is tiny (25 cells, at most 12 moves),
"perfect" is not an aspiration reached through heuristics but a _fully solvable_
target: the player searches every relevant line of play to a terminal position and
backs the true value up to the root.

The game has in fact been solved outright, and the player is built on that result.
Searching the _opening_ live is hopeless — the tree above ~16 empty cells is far too
wide — so the player is **two-tier**: an **offline retrograde solve** precomputes the
exact value of every reachable position in the upper plies, and at run time the player
**looks the opening up** and **live-searches the endgame**, where full-depth search is
a matter of seconds. Both tiers return the same values by construction, so the seam is
invisible: the player is exactly as perfect as a pure search would be, and merely
faster.

## What "perfect" means

Concretely, on every turn the player returns a move that is **optimal under
minimax**:

- if the position is a forced win for its side, it plays a move that still forces
  the win — and, among those, one that wins **as quickly as possible**;
- if the position is at best a draw, it plays a move that **never lets the opponent
  win** (securing at least the cat's game);
- if the position is lost against perfect defence, it plays a move that **delays the
  loss as long as possible**, maximizing the opponent's opportunities to err.

**The third case never actually happens.** It is stated for completeness, but this
player is never asked to move in a lost position. Every seed is drawn (see "The
result: the game is a draw" below), so a game starts drawn; the player never leaves a
drawn position, and an opponent moving from a drawn position cannot manufacture a win
out of it — so by induction every position it is handed is drawn or won. The rule
still earns its keep at the _interior_ nodes of the search, where lost positions are
everywhere, and it costs nothing to state, since it falls straight out of the
depth-folded score (see "Exact, depth-aware scoring"). The only way to reach it at the
root would be to drop the player into a lost position it did not play itself into,
which nothing in the engine does.

So it is the **"win soonest" half** of the tie-break that acts in real games, and it
matters for the benchmark: against a fallible LLM it converts winning positions into
wins with the fewest chances for a swindle.

Perfect play accounts for the **seeded snake** automatically — the seed is just a
snake piece in the search, so whether a given opening is a forced win, loss, or draw
for a side falls out of the search rather than being assumed.

## It is a mechanical player

The algorithmic player is a **mechanical** player in the sense of SPEC.md §3: it reads the
board correctly by construction, so it makes **no self-assessment** — `choose_move`
returns a `MoveChoice` with `claimed_outcome = None`, and the engine performs no
outcome check for it. It composes a `Board` (SPEC.md §3, "Shared board helper") for its
internal state, updated through `observe_move`, is assigned a side per game like
every other player, and holds no cross-game state: its transposition table is cleared
at the start of every game. The **solved opening table** is not cross-game state — it
is immutable, public fact about the game, no more carried between games than the rules
are — so it is loaded once and shared, and gives a player no memory of what it has
seen. Its only injected dependency is a `random.Random` (see "Choosing among optimal
moves"), so a seeded instance is fully reproducible.

## Full-depth alpha–beta search

The engine of the player is **alpha–beta search (minimax with pruning)** carried
**all the way to terminal positions** — a win, a cat's game — rather than to a
fixed ply depth with a heuristic evaluation. Leaves are therefore scored _exactly_,
not estimated; the pruning only removes branches that provably cannot affect the
root value, so the result is identical to exhaustive minimax but far cheaper.

Move generation follows SPEC.md §2.5: the moves at a node are all pairs of two distinct
empty cells, plus any **single-piece move that ends the game** (completes a line, or
fills the board into a cat's game). Win detection runs after each individual piece,
so a move whose _first_ piece completes a line is an immediate win; when the player
has such a one-piece win it returns a **single-piece** `MoveChoice`, the honest
representation of that turn (SPEC.md §2.5).

## Exact, depth-aware scoring

**What `depth` means.** `depth` is a property of a _position_, not of the search that
reached it: it is read straight off the piece counts as `(occupied − 1) // 2`, the
number of moves already played. Two things follow. Values are **absolute** rather than
relative to a search root, so a value computed anywhere is comparable with one computed
anywhere else — which is what lets the offline table and the live search be mixed
within a single game (see "The opening table" below). And the `depth` in a winning score is
that of the node **from which the winning move is made**, not of the position the win
lands in: a node whose mover _has_ a win is scored without descending into it. That
distinction is a constant 1 either way and invisible inside a single search, but both
tiers must adopt the same one or their values disagree at every boundary.

**How a terminal is scored.** Scores are from the perspective of the side to move,
larger being better for that side, with the depth folded in:

```
WIN = 1000        # larger than any reachable depth (a game lasts ≤ 12 plies)

# In each case `depth` is that of the node whose mover completes the line.
#
#   win     this side has a move completing a line     →  +(WIN − depth)
#   cat's   the move fills the board, every line dead  →   0
#   loss    the opponent has such a move               →  −(WIN − depth)
```

The win and loss scores are the same magnitude with opposite signs, because a loss is
never scored in its own right — under negamax it is simply the negation of the
opponent's win. Folding `depth` in this way is what produces the tie-break behaviour
above: among wins a **smaller** depth scores higher, so the player wins as quickly as
it can; among losses `−(WIN − depth)` is less negative for a **larger** depth, so it
resists as long as it can — behaviour the search relies on internally but the player
never gets to display, since it is never handed a lost position (see the note under
_What "perfect" means_).

The search is a standard **negamax**: the value of a non-terminal node is the maximum
over its legal moves of the terminal score (if the move ends the game) or
`−value(child)` otherwise. Alpha–beta cutoffs are applied in the usual way.

## Move ordering

Alpha–beta's savings depend on trying strong moves first. The player orders
candidate moves to surface likely cutoffs early — **completing a line** (an
immediate win) first, then moves that **advance the mover's own most-filled lines**
or **poison the opponent's near-complete lines** — before the remaining quiet moves.
Ordering never changes the value computed; it only reaches it sooner.

## The symmetry group and the transposition table

The dominant cost saving comes from recognizing that the same position is reached by
enormously many move orders, and that geometrically equivalent positions share a
value. Both are captured by a **transposition table** keyed on a **canonical
position**.

**The symmetry group.** The value of a position is invariant under every
transformation that maps the 12 winning lines onto themselves. These form a group
**G of order 32** — strictly larger than the 8 rigid rotations/reflections (**D₄**)
of the square. Every element has the form `(r,c) ↦ (σr, τc)` or `(r,c) ↦ (σc, τr)`
(the latter transposing rows and columns), where `σ` is one of the 8 permutations of
`{0..4}` that commute with the reversal `ρ = (0 4)(1 3)` (equivalently: fix 2 and
preserve the pairs `{0,4}`, `{1,3}`), and `τ ∈ {σ, ρσ}`. D₄ is exactly the slice
with `σ ∈ {identity, ρ}`; the other 24 elements are "warps" — such as _swap rows B↔D
and columns 2↔4_, written in the board's own labels — that preserve every line
without being a rigid motion.

Under G the 25 possible seed cells fall into just **4 equivalence classes**, and —
crucially — **every** seed has a stabilizer of order ≥ 4, so even an off-axis opening
that D₄ leaves untouched still collapses ~4-fold at the root.

Row and column positions are **0-based** throughout, matching the implementation.
Board _labels_ (rows `A`–`E`, columns `1`–`5`) are a separate, presentational layer:
column label `2` is position `1`, and the warp above is `(1 3)` on columns.

**Canonical position.** Number the cells `0..24` (`index(r,c) = 5r + c`) and
represent a position as two 25-bit masks — mouse and snake (the seed is just a snake
bit). Each of the 32 symmetries is a precomputed index permutation, so transforming
a mask is a bit permutation. The **canonical key** is the numerically smallest of
the 32 transformed positions, with mouse and snake transformed _together_ by the
same symmetry and packed into one integer for comparison:

```python
def canonical_key(mouse: int, snake: int) -> int:
    best: int = -1
    for perm in ALL_32_PERMS:                     # the 32 symmetries as index maps
        packed: int = (permute(perm, mouse) << 25) | permute(perm, snake)
        if best < 0 or packed < best:
            best = packed
    return best
```

Because the 32 transformed positions are exactly the position's orbit under G, their
minimum is identical for every member of an orbit and distinct across orbits — a
true canonical form. The **side to move is not part of the key**: it is a pure
function of the piece counts (`snake == mouse + 1` ⇒ Mouse to move, `snake == mouse
− 1` ⇒ Snake to move), which holds because every keyed position sits between complete
two-piece moves (the only single-piece moves are terminal, and terminals are scored
directly, never stored).

**The table.** Each entry stores the position's minimax **value** and an alpha–beta
**bound flag** (exact / lower-bound / upper-bound); it does **not** store a best
move, so no inverse transform is ever needed. The player selects its actual move in
the real, un-canonicalized frame at the root, evaluating each legal move's resulting
child through canonical table lookups. A value-only table is correct because the
value is symmetry-invariant.

A self-checking construction test asserts that `ALL_32_PERMS` contains exactly 32
distinct permutations and that each maps the set of 12 line-index-sets onto itself.

## Choosing among optimal moves

Often — in this game, nearly always — several moves share the optimal value. The
player collects **all** moves whose value equals the best, and then has a free
choice: since every candidate has the same minimax value, narrowing the pool cannot
cost a draw or a win, and the player remains exactly as perfect however it picks.

Against perfect defence the choice is genuinely irrelevant. Against a **fallible**
opponent it is not, and the benchmark only ever plays fallible opponents. So the
pool is **ranked before the pick**, to maximize the opponent's opportunities to go
wrong, and the random pick is applied to whatever survives.

**The keys, strongest first.**

1. **Trap count** — of the opponent's replies to this move, how many would throw the
   position away (turn a draw into a loss for them, or let us escape a lost one).
   This is not a heuristic: it is the exact quantity a swindle-seeking tie-break
   wants, and it is _counted_, by valuing every reply. Only two-piece replies are
   counted; a single-piece move is legal only when it ends the game (SPEC.md §2.5), and
   neither of its forms can be a blunder in our favour.
2. **Liveness** — our pieces in lines the opponent has not yet touched, with the
   per-line count **squared**. A dead line can never be won, so pieces in it are
   spent; the squaring is because threat value is sharply non-linear here — a move
   places two pieces, so three of ours in a live line is already a win-next-turn
   threat, making concentration worth far more than the same pieces spread thin.
   This _is_ a heuristic, standing in for the traps that lie deeper than one reply,
   in the band where counting them exactly is too slow.
3. **Uniformly at random**, from its injected `random.Random`, over whatever is left.

**Every key stands aside where it cannot discriminate.** A key that ranks all
candidates alike leaves the pool untouched, so the fallback is always the uniform
random pick. That matters because the random pick is what keeps a match between two
perfect players — or a perfect player and a deterministic opponent — from replaying
one identical game, mirroring the variety the random player and randomized openings
(SPEC.md §5) bring, while a seeded RNG keeps any given run reproducible. Ranking is allowed
to reduce that variety only where it buys something real.

Two gates enforce that, both set by measurement:

- **Trap counting runs only at 22 and 20 empty cells, and at 14 and below.** Above
  that it is either vacuous or unaffordable. Vacuous: every position with 20 or more
  empty cells is drawn (see "The result" below), so a node whose grandchildren all sit
  there has no losing reply to find — which is exactly the Mouse's first move, the
  widest node in the game, so the gate skips the most expensive count _and_ leaves the
  opening uniformly random. Unaffordable: counting needs every reply valued, which
  costs tens of seconds per move above the table's floor, where the Mouse chooses in
  nearly every game. So the count runs where the **table** already covers the
  grandchild layer, or deep enough to search outright. The 14 threshold is the one
  gate not settled by a measured win, and is called out as such: it discriminates in
  most decisions there but what it is _worth_ is untested, since `random` leaves no
  headroom and the project has no mid-strength fallible opponent to measure against.

- **Liveness is gated to 18 empty cells and below.** It is a deterministic key, so
  letting it decide the opening would replay one game per seed. By 18 the position
  has already branched widely enough that a deterministic choice cannot funnel every
  game into the same line.

A table that cannot value _every_ reply is refused wholesale, exactly as it is for
the move choice itself (see "The opening table" below): the pool is left unranked
rather than ranked on partial data.

**What this is worth.** Against the random player — a maximally fallible opponent, so
its loss rate is a direct estimate of P(the opponent goes wrong) — ranking lifts the
perfect player from **62.3%** wins to **98.0%**, measured over 300 games per policy,
and it **never loses a game under any policy**, as it cannot. Both keys earn their
place: the exact trap count does most of the work and the liveness heuristic supplies
the rest. The caveat on the numbers is that a random opponent misses traps uniformly
whereas an LLM misses _subtle_ ones, so trap density is a proxy for that rather than a
model of it, and the gain against a model will differ.

Ranking is **not configurable**: the player always ranks. A selectable policy would make
`perfect` mean two different strengths under one name, and a results file identifies a
player by name alone (SPEC.md §6) — so the yardstick would stop being calibrated
(SPEC.md §1). The unranked baseline is instead reconstructed outside the player, by a
throwaway subclass that restores the old pick;
[`tools/bench_tie_break.py`](tools/bench_tie_break.py) does exactly that, and is how the
comparison above is re-run whenever the keys or their gates change.

## Where the table comes from

The table is produced by an **offline retrograde solve** of the whole game: a forward
pass enumerates every reachable non-terminal canonical position layer by layer, and a
backward pass values those layers bottom-up from the endgame. Because G leaves only 4
seed classes, four solves cover all 25 seeds.

That solver is a separate program with separate needs — it uses numpy and a process
pool, neither of which the player itself has any use for — and it is specified in its
own document, **[`tools/solver/SPEC.md`](tools/solver/SPEC.md)**, together with the table
file format. Nothing in this section depends on _how_ the table was produced, only on
what it contains.

## The opening table

Choosing a move at a position with `E` empty cells means valuing its children, which
have `E−2` — so the layers the table _stores_ sit two below the layers at which the
player _moves_. It stores values for **22, 20, 18 and 16** empty cells, which covers
the player's choices at **24, 22, 20 and 18**. From **16** empty cells down it
live-searches. (Layer 24 is deliberately absent: nothing ever chooses at 26.)

The threshold of 16 is set by measurement, not taste. Live search costs a median of
~3 s (easy seed class) to ~7 s (hard) at 16 empties, but tens of seconds with a long
tail at 18 — while each step shallower multiplies the table by roughly 5. Sixteen is
the point where the table is still small and the search is still quick.

A table is selected by the seed's **orbit representative**, so all 25 seeds are served
by the four files. No move ever needs mapping between frames: the canonical key is
invariant under G and the table stores only **values**, so the player generates
children on the real board, canonicalizes each, and looks its value up directly. The
seed determines only _which_ file to load.

The format is deliberately plain: it is read with `array` and searched with `bisect`,
and may be gzipped (~6× smaller) — all standard library. That keeps the player free of
any dependency the engine does not already have, which is worth something for a
benchmark opponent that has to be easy to run; it is a preference, not a prohibition,
and a large enough win would justify revisiting it. See `tools/solver/SPEC.md` for the
layout.

A **missing or unreadable table is not an error**, but it is always **reported**. The
player falls back to searching the opening — correct, merely far too slow to be
practical — so it remains usable and testable on a checkout that has not been given the
solver's output. That fallback is announced on stderr, naming the directory searched,
the filename expected, and the consequence: silence would be indistinguishable from a
hang, since searching the opening can take hours per move. A table that cannot answer
for _every_ child of the current position is likewise refused wholesale in favour of
the search, so a partial or mismatched table degrades to slow-and-right rather than
fast-and-wrong.

## The result: the game is a draw

All four seed classes have been solved to completion. **Every one is a draw under
perfect play** — so from all 25 seeds, with best play by both sides, Snakes and Mice
is drawn, and neither the Mouse's first move nor the seeded snake confers a forced
win.

The opening is more strongly drawn than that summary suggests: **every position with
20 or more empty cells is drawn**, whichever side is to move. The first forced wins
appear at **18** empty cells, and even there they are a small minority — 4,075 of the
64,440 positions in the C3 class. (Nothing can be _terminal_ before 16 empty cells
either: that is the first point at which a side holds five pieces, so the first at
which a line can be completed at all.)

Two consequences. **Every** first move by the Mouse, and every reply by the Snake, is
equally optimal — all their children are drawn, so _against perfect defence_ the
choice cannot matter, and a uniformly random pick among them is exactly as good as
any other. And the earliest a choice _can_ matter is the Mouse's second move, where a
minority of moves walk into one of those 18-empty wins for the Snake: 29,768 of
417,240 for the C3 class (7.1%), concentrated in 318 of its 2,196 positions. A win
only ever arrives as a gift, and the perfect player's opening task is purely not to
blunder.

Against a fallible opponent the second consequence reads differently, and it is what
the tie-break above exploits. Those 7.1% of losing Mouse replies are **not evenly
spread**: only 318 of the 2,196 positions contain any at all, so the Snake's first
move — value-irrelevant though it is — decides whether the Mouse is offered a real
chance to go wrong or none whatever. It is the first ply at which ranking has any
effect, and the effect is large: choosing the trappiest reply routinely leaves the
Mouse with a large majority of losing moves to avoid, against a 7.1% base rate. The
Mouse's _first_ move remains the one place where nothing can be gained, since its
grandchildren are all drawn too — which is why it is left uniformly random.

This settles, for the algorithmic player, the game-balance question SPEC.md §11 lists as
out of scope: neither side is favoured under perfect play. It says nothing about balance
between _fallible_ players, which is what the benchmark actually measures.

## Performance

There is **no move-time budget**: the player plays perfectly however long that takes.
With the table installed, opening moves are effectively instant (a few milliseconds —
they are lookups), and a whole perfect-vs-perfect game runs in **seconds**. The
expensive move is the first one below the threshold, at 16 empties, which is seconds.
Without a table the player still plays perfectly but the opening is impractical.

Deliberately deferred: faster canonicalization (short-circuited comparison,
per-symmetry caching, or incremental hashing); caching a best move per entry for
stronger move ordering; and compressing the table beyond gzip (delta-encoded keys and
packed values would roughly halve it, at the cost of a custom codec — see
`tools/solver/SPEC.md`).

## Selecting and testing it

The algorithmic player is a **built-in mechanical player named `perfect`**,
constructed and selected **exactly like `random`**: it is chosen by name wherever a
side is named — e.g. `play-match --mouse perfect --snake random` (SPEC.md §7) —
alongside `random` and `human`, and it is built with a fixed per-side display name (the way
`random` is), which is the name that appears in any results it produces. Like
`random`, `perfect` is **not** part of the `players.yaml` tournament roster (SPEC.md §4,
§6): the roster lists only LLM players. Making the mechanical baselines available as
calibrated reference opponents inside tournaments (SPEC.md §1) is deferred — and when it
comes, it is no different for `perfect` than for `random`.

Testing leans on the player's exactness: unit tests assert the 32-symmetry
self-check and canonical-form invariance (all orbit members share a key); that the
player **never loses a drawn-or-won position** and **never fails to win a won one**
against an exhaustive or random opponent; and that G-equivalent seeds yield the same
game value, so the symmetry reduction is sound.

The two tiers are tested against each other, since the whole design rests on their
agreeing: table values are re-derived by the live search and must match exactly, and a
table-driven choice must achieve the same value the search would have found. Table
tests build **synthetic** tables in a temp directory rather than depending on the
solved ones, so the suite passes on any checkout; every seed is asserted to resolve to
one of the four representatives; and a deliberately incomplete table is asserted to
fall back to the search rather than answer from partial data.
