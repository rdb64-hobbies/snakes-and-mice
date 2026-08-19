# Perfect-player solver — handoff notes

Working notes for building the offline solver that powers `PerfectPlayer` (§10).
Now running on the 16-core / 64 GB Mac Studio (M4 Max: **12 performance + 4 efficiency**
cores). The handoff is done — the WIP was committed and pulled, and step 1 is complete.

## Goal

`PerfectPlayer` must play a game-theoretically optimal move every turn, for **any**
of the 25 snake seeds, fast enough to be usable. Live full-depth search is correct
but the *opening* (24/22/20/18 empties) is far too slow. The plan: an **offline
retrograde solve** produces a compact table of exact values for the upper plies;
at runtime the player looks up the opening and live-solves the endgame (≲16 empties,
~seconds with a per-game canonical transposition table).

## Decision so far

- Pure single-core Python+numpy is **infeasible** for the low-symmetry seeds: measured
  throughput is a hard **~1.4M children/s** ceiling (numpy `canon`+`unique` bound, not
  Python overhead — batching all 300 cell-pairs into one call gave 0 improvement), and
  peak layers for the hard seeds are ~12–16 GB (OOM on a 16 GB box).
- **Chosen path: parallelized pure Python on the 64 GB / 16-core Studio.** Cores fix the
  time axis (~10–14× → all 4 seed classes in ~1–2 days total); 64 GB fits the peak
  layers. No rewrite. **Rust/C++ is the fallback** only if parallel Python still
  disappoints.

## Measured facts (trust these)

- **Symmetry group G has order 32** (not just D₄'s 8). Built + validated in
  `src/snakes_and_mice/players/symmetry.py`. `canonical_key` ~11 µs/call single-thread.
- **Seed orbits under G: exactly 4 classes** (so we solve 4 representative seeds, and at
  runtime map any of the 25 seeds to its representative via a symmetry):
  - `C3` — stabilizer **32** (the easy one).
  - `A1`, `A2`, `A3` — stabilizer **4** each → layers ~**8× larger** than C3 in both time
    and peak memory. (Engine default seed `B3` ∈ A3 class; CLI default is random seed.)
- **C3 layer sizes: now solved to completion** — see the table below. The old
  extrapolation from 14 empties (~150–200M peak, ~4 h single-core) was close on time
  (3.94 h projected) but low on size: the true peak is **267M at 8 empties**.

## Where the code stands

Committed in `d65914c` (pulled onto the Studio), except the step-1 work below, which is
**uncommitted**: `tools/solver/forward.py`, the flat-vs-bucketed cross-check (since removed), the
expand/dedup timing split in the forward pass, and this file.
- `src/snakes_and_mice/players/symmetry.py` — G (order 32), `canonical_key`, fast
  5-bit-chunk permute tables. **Done, validated at import.**
- `src/snakes_and_mice/players/perfect.py` — `PerfectPlayer`: full-depth negamax
  alpha-beta to terminals, canonical TT (value + bound flag), honest win moves, random
  tiebreak. **Correct** on constructed positions; only the opening scale is the issue.
- Wiring: `players/__init__.py`, `snakes_and_mice/__init__.py`, `cli_common.py`
  (names **Percy** = Mouse, **Perseus** = Snake; CLI kind `perfect`), `match_cli.py`.
- `pyproject.toml` — added `numpy>=2.1` to the dev group (solver-only). *Clarified
  2026-08-19: this note originally read "the runtime player must stay numpy-free", and
  that phrasing propagated into SPEC §10 as a hard requirement. It never was one. The
  player uses only the standard library because nothing in it needs more, which keeps
  it installable without the dev group — a preference worth holding cheaply, and one a
  real benefit would justify revisiting.*
- `tools/solve_forward.py` — the original flat solver with a serial parent merge.
  **Removed 2026-08-19**: it cannot run at A-class size, and it shared all its
  enumeration code with the bucketed solver, so it only ever validated the
  partitioning. Its shared primitives now live in `tools/solver/enumeration.py`.
- `tools/solver/forward.py` — **the one to run.** Same expansion, but each layer is
  stored as 128 sorted bucket files cut by sampled range splitters, dedup'd in parallel.
- the flat-vs-bucketed cross-check (since removed) — asserts a bucketed layer is the exact key set of the flat one.

## Forward pass: measured on the Studio (step 1 — DONE, 2026-08-18)

Full C3 forward solve to completion, 16 workers, `chunk=50_000`:

| empties | positions | expand s | dedup s | shard rows |
|---|---|---|---|---|
| 18 | 64,440 | 2.2 | 0.2 | 1,617,807 |
| 16 | 1,616,052 | 16.1 | 0.5 | 113,666,359 |
| 14 | 12,875,213 | 54.2 | 1.9 | 78,044,138 |
| 12 | 77,452,075 | 389.7 | 25.2 | 2,374,846,826 |
| 10 | 178,580,973 | 335.7 | 6.4 | 267,432,968 |
| **8** | **267,023,819** | 433.8 | 18.6 | 2,135,003,183 |
| 6 | 140,464,171 | 65.0 | 0.9 | 34,293,472 |
| 4 | 34,269,618 | 5.0 | 0.6 | 27,118,032 |
| 2 | 1,617,740 | 0.1 | 0.0 | 0 |

- **Peak layer 267,023,819 positions at 8 empties** (1.99 GB as uint64) — well under the
  150–200M *extrapolation*, and the layer peaks at 8, not lower.
- **Total distinct non-terminal positions: 713,966,318.**
- **Wall clock 22.6 min**, against a 3.94 h single-core projection (24.31B children at the
  measured 1.72M children/s) — **~10.4x end-to-end**. Directly measured single-layer
  speedup at 16 empties: 117.2 s → 16.6 s = **7.1x** (that layer only fills 33 shards over
  16 workers, so it understates; the big layers shard evenly).
- Peak summed RSS **48 GB**, but that counts mmap'd layer/shard pages, which are
  file-backed and evictable — **swap was never touched**. Peak transient disk ~20 GB.

### `chunk` is a memory dial, not a speed dial

At `chunk=400_000` a single worker peaks at **4.11 GB**; 16 of those is ~66 GB on a 64 GB
box. Throughput is flat across chunk sizes (400k: 22.7 s, 100k: 21.5 s, 25k: 20.5 s for the
same 400k rows), so **the big chunk buys nothing and costs 4x the RAM**. Use 50k or below.
(macOS reports `ru_maxrss` in *bytes*, not KB as on Linux — easy to misread by 1024x.)

## The serial merge, and why the fix is range splitters not hashes

The serial `np.unique` in `solve_forward` was never the *time* problem — its share fell as
layers grew (35% → 32% → 28%). It is a **memory** wall: expanding layer 12 produces
**2.37 billion** shard rows, so the parent concatenates a 19 GB array and `np.unique` sorts
a second 19 GB copy beside it. The A-class seeds are ~8x C3, which puts that past RAM.

`tools/solver/forward.py` fixes it by partitioning each layer into 128 buckets that
are dedup'd independently and in parallel. **Two false starts worth not repeating:**

1. **Range-partition on the key's top bits — fails.** Canonicalization minimizes over the 32
   symmetries, which clusters keys hard: 9 of 64 buckets non-empty, 55x imbalance.
2. **Hash-partition (the original plan in these notes) — correct but barely faster.** It
   destroys key adjacency. Shards of the *sorted* layer hold neighbouring positions that
   share most of their children, so the per-shard `unique` collapses them; hash order
   scatters that, and shard rows nearly **doubled** (2.37B → 4.55B), dragging expand up 28%
   and cancelling the dedup win. Layer 12: 560.6 s → 545.5 s, a 2.7% gain. Not worth it.
3. **Sampled range splitters (a sample sort) — what shipped.** Quantiles of a sampled key
   pool give even buckets *and* keep buckets ordered by key. Shards stay contiguous key
   ranges (shard rows back to 113.7M at layer 16 vs the reference 112.8M), and concatenating
   buckets in order is a globally sorted layer, so the backward pass can `searchsorted`
   inside a bucket. Layer 12: **560.6 s → 415.6 s (1.35x)**, dedup 160.0 s → 25.2 s.

Verified with the flat-vs-bucketed cross-check (since removed): layers 22 → 10 are **byte-identical key sets** to
the flat reference run, globally ordered, no duplicates.

### Shard duplication alternates with the side to move — this is expected

Shard rows swing wildly layer to layer (1.0x vs 13x duplication). That is not a bug. The key
is `(mouse << 25) | snake`, so sorted order is dominated by the mouse mask. **When the snake
moves the mouse mask is unchanged**, so contiguous shards produce near-disjoint children
(layers 18/14/10/6: ~1.00x). **When the mouse moves** the keys scatter and every child is
reached from many shards (layers 16/12/8: 8.8x, 13.3x, 15.2x).

## Backward value pass (step 2 — DONE, 2026-08-18)

`tools/solver/backward.py`. Values every enumerated position bottom-up, 16 workers,
`chunk=50_000`, **16.8 min** (C3 forward + backward = **39.4 min** end to end).

| empties | positions | secs | win | draw | loss |
|---|---|---|---|---|---|
| 2 | 1,617,740 | 0.7 | 875,841 | 741,899 | 0 |
| 4 | 34,269,618 | 5.4 | 14,519,529 | 19,690,022 | 60,067 |
| 6 | 140,464,171 | 68.8 | 76,836,515 | 63,529,407 | 98,249 |
| 8 | 267,023,819 | 287.9 | 103,563,417 | 161,270,959 | 2,189,443 |
| 10 | 178,580,973 | 343.3 | 84,066,867 | 94,348,422 | 165,684 |
| 12 | 77,452,075 | 234.8 | 21,154,895 | 55,794,443 | 502,737 |
| 14 | 12,875,213 | 53.2 | 4,303,381 | 8,571,832 | 0 |
| 16 | 1,616,052 | 11.3 | 150,998 | 1,464,961 | 93 |
| 18 | 64,440 | 2.1 | 4,075 | 60,365 | 0 |
| 20 | 2,196 | 0.2 | 0 | 2,196 | 0 |
| 22 | 20 | 0.0 | 0 | 20 | 0 |
| 24 | 1 | 0.0 | 0 | 1 | 0 |

### **Result: with the snake seeded at C3 the game is a DRAW under perfect play.**

Layers 24/22/20 are drawn *throughout* — neither side can force anything in the first
three plies. The first forced wins appear at 18 empties (4,075 of 64,440).

### Value convention — must match `PerfectPlayer`, and nearly didn't

The win score is **`WIN - depth(P)`, the depth of the position the winning move is made
*from***, because `PerfectPlayer._negamax` returns `_WIN - depth` as soon as the side to
move *has* a winning move, without descending. Scoring the *child's* depth (the obvious
reading of "WIN = 1000 - depth" in the old notes) shifts every value by one and would
silently break the runtime player, which mixes table lookups with live search below the
threshold. Depth is `(occupied - 1) // 2`, read off the position.

Per SPEC a one-piece move is legal **only** when it ends the game, so every *non-terminal*
child comes from a two-piece move and the rectangular `C(empties, 2)` enumeration is
complete; one- and two-piece wins score identically anyway.

### How it is verified

`tools/solver/check_values.py` samples a solved layer and re-derives each value with
`PerfectPlayer._negamax` — the exact code path a live game takes. **249 sampled positions
across layers 4, 6, 8, 10, 12, 14, 16: zero mismatches.** Both side-to-move parities are
covered several times over. 16 empties is about as high as live search stays practical,
which is the whole reason the table exists.

The pass also self-checks: every non-terminal child must be found in the layer below
(`searchsorted` hit verified by equality, not just position), and each layer is asserted
globally sorted when flattened from its buckets.

### Layout on disk

Per layer: `keys_NN.npy` (uint64, sorted — the buckets concatenated in order) and
`vals_NN.npy` (int16, positionally aligned). C3 totals **5.32 GB keys + 1.33 GB values**;
12 GB including the bucket files, which can be deleted once flattened.

## Known gaps before running the A-class seeds

- ~~**Disk will not fit.**~~ Resolved — the user freed space; **288 GB free** as of
  2026-08-18, against a ~160 GB A-class transient peak.
- **Time: ~5.3 h per A-class seed** (8x C3's 39.4 min), so ~15.8 h for A1+A2+A3.
- ~~`solve_forward.py` is kept as the correctness oracle.~~ **Removed 2026-08-19** in the
  `tools/solver/` reorganisation; its shared primitives moved to `enumeration.py`. The
  live-search check (`check_values.py`) is the stronger oracle anyway — it shares no
  code with the solver.
- Sharding is capped at `workers * 4`, and shard size is floored at `chunk`, so layers under
  ~800k rows leave workers idle. Only affects the cheap upper layers.

## A-class run (step 5 — started 2026-08-18 16:24, ~16 h)

`./tools/solver/solve_all.sh /tmp/solve-data2 16`, under `caffeinate -is`, logging to
`/tmp/solve-all.log`. A1 then A2 then A3, forward+backward each, strictly sequential —
each run saturates all 16 cores. Expected ~5.3 h per seed. Check with
`tail -40 /tmp/solve-all.log`.

### Results so far

| seed | peak layer | total distinct | forward | backward | root |
|---|---|---|---|---|---|
| C3 | 267,023,819 @ 8 | 713,966,318 | 22.6 min | 16.8 min | **draw** |
| A1 | 739,963,015 @ 8 | 2,160,276,351 | 85 min | 55 min | **draw** |

A-class came in at ~3x C3, not the 8x the orbit sizes suggested — the stabilizer ratio
bounds the *worst* case, and reachability trims it. A seed is ~2h20m, not ~5.3 h.

Pruning at scale works: **A1's residue is 3.1 GB** (layers 12-24) against ~57 GB
unpruned. A2's layer counts are near-identical to A1's but genuinely distinct
(7,126,135 vs 7,125,782 at 16 empties) — a useful check that the seeds are different
orbits and not an accidental repeat.

### Scaling fixes this needed (C3 defaults unchanged; verified byte-identical)

Three things in the C3-validated pipeline were sized for C3 and would have failed at 8x:

- **Shard count was fixed at `workers * 4`**, so a worker's accumulated children grew
  with the layer: ~264 MB each on C3's peak layer, ~2.1 GB on A-class, ~68 GB across 16
  with the concatenate. Now sized by rows (`SHARD_ROWS`, 4M parents per shard).
- **Bucket count was fixed at 128**, making an A-class bucket ~1 GB (~34 GB across the
  pool). Now scaled by `TARGET_BUCKET_ROWS` (32M), clamped 128–4096. C3 still lands on
  128; A-class uses ~400–600. Bucket filenames went to **four digits and sort
  numerically** — text order stops matching bucket order past `b999`.
- **`_flatten_layer` concatenated a whole layer in the parent** and ran `np.diff` over
  it — ~34 GB transient for a 17 GB A-class layer. Now copies bucket by bucket through
  a memmap, checking order per bucket. Constant memory.

All three are tunable from the environment (`SOLVER_SHARD_ROWS`, `SOLVER_BUCKET_ROWS`,
`SOLVER_MIN_BUCKETS`, `SOLVER_MAX_BUCKETS`) so the scaling paths can be exercised on C3.

### Space-saving mode (`keep_empties`, 5th arg to `solve_backward.py`)

Three unpruned seeds (~57 GB each) plus the last one's ~166 GB working set is ~280 GB
against 288 GB free — too tight to leave unattended. With `keep >= 12`, a layer's
buckets are dropped once flattened and its keys/values once the layer above is valued.
**C3 residue: 878 MB instead of 12 GB**, so ~7 GB per A-class seed.

What survives is layers 12–24, which is everything steps 3–4 need (the runtime player
live-solves below ~16 empties). Re-deriving a pruned deep layer costs a full ~5 h re-run.

### The out-of-memory incident (2026-08-18, first A1 attempt) — read this

The first A-class launch exhausted RAM and swap ~35 min in, on A1's layer 12. Terminal
showed 132 GB. **Cause: worker memory was bounded by the wrong quantity.**
`SHARD_ROWS` caps a shard's *input* (parent rows); peak memory is set by its *output*
(unique child keys accumulated before writing). The ratio between them is not a
constant — it is the alternating-parity duplication documented above: **~1x on
snake-to-move layers, 8-15x on mouse-to-move layers.** The constant was calibrated on
C3's layer 8, a *snake* layer, i.e. the best case. A1's layer 12 is a mouse layer:
~13.5B shard rows, which at 64 shards is ~1.2 GB per worker, tripled by
`np.unique(np.concatenate(...))`, across 16 workers at once.

**Fix:** `FLUSH_ROWS` (16M keys) caps the *output* — a worker writes a file and starts
fresh whenever it accumulates that much, so one shard emits many files and peak memory
is flat in the layer and the seed. Each file carries its first/last key so the dedup
phase skips files outside its range. Confirmed on the rerun: A1 layer 12 produced 955
files from 64 shards and peaked at **351 MB swap / 6.3 GB compressor** (summed worker
RSS reads 44 GB, but that counts file-backed page-cache pages and overstates demand).

Cost of the fix: more files means less dedup per file, so shard rows rise ~1.7x and
with them the transient disk — **A1's layer 12 held ~108 GB of shard files at once**,
freed as the dedup consumed them. Disk is now the tightest resource, not RAM.

### Two traps when killing this thing

1. **`pkill -f solve_forward_bucketed` does not kill the workers.** Pool workers run as
   `python3 -c from multiprocessing.spawn import spawn_main ...` — the parent's name
   appears nowhere in their command line. The first kill attempt left all 16 workers
   holding the memory, and pressure did not drop until they were killed by
   `pkill -f "multiprocessing.spawn"`.
2. **Do not watch "Pages free" on macOS.** It is kept low by design, with reclaimable
   memory held as inactive/speculative. A free-pages threshold in the first watchdog
   killed a *healthy* run with swap at 470 MB against a 10 GB limit. Watch **swap,
   compressor, and `kern.memorystatus_vm_pressure_level`** — during the real failure
   those read 30.5 GB / 37.3 GB / critical, against ~0.5 GB / ~1.2 GB / 1 when healthy.

`tools/solver/watchdog.sh` encodes both lessons and also guards free disk. It sacrifices
the run to keep the machine, and logs why to `/tmp/solve-watchdog.log` — **check that
log first if a run is found stopped.**

### Pre-flight (do this before any future scaling change)

C3 was re-run end to end with `SOLVER_SHARD_ROWS=1000000 SOLVER_BUCKET_ROWS=8000000` to
force the many-shard/many-bucket paths (layer 8 ran 268 shards / 394 buckets), then
valued with pruning on. Every layer size, every win/draw/loss count, and the root value
matched, and the surviving `keys_NN.npy`/`vals_NN.npy` for layers 12–24 were
**byte-identical** to the known-good C3 run.

## What the opening actually needs (measured on A1, informs step 3)

No position above 18 empties is won for anyone, and the consequences are sharp:

| ply | empties | side | moves | losing moves | positions where choice matters |
|---|---|---|---|---|---|
| 1 | 24 | mouse | 276 | **0** | 0 of 1 |
| 2 | 22 | snake | 231 | **0** | 0 of 86 |
| 3 | 20 | mouse | 190 | 96,853 (4.20%) | 949 of 12,127 |

Why the cutoff is exactly there: at 18 empties the mouse has 4 cells and the snake 3
(seed included), so **no line of 5 can exist yet**. The first layer where a side holds
5 cells is 16 empties (snake: 4 placements + seed), which is the first layer where
terminal children get filtered at all.

**Correction (2026-08-19):** these notes previously claimed that above 16 empties the
layer count depends only on the *order* of the seed's stabilizer, so all three A seeds
must have identical counts. That is wrong, and A3 falsified it: 91 / 12,219 / 374,099
at 22 / 20 / 18 empties against A1 and A2's 86 / 12,127 / 373,397. Brute force with no
solver code in the path confirms 91 — the enumeration is right, the reasoning was not.

Burnside needs each group element's **fixed-point count**, not the group's order. On
2-subsets of the 24 non-seed cells the stabilizers fix:

| seed | \|Stab\| | fix(g) per element | orbits |
|---|---|---|---|
| A1 | 4 | 276, 36, 16, 16 | 344/4 = 86 |
| A2 | 4 | 276, 36, 16, 16 | 344/4 = 86 |
| A3 | 4 | 276, 36, **36**, 16 | 364/4 = 91 |

Same order, different action. A1 and A2 agreeing was a coincidence of identical
fixed-point profiles, not a law. (At 22 empties the snake is just the seed, so a
symmetry can only identify two positions if it *fixes* the seed — which is why the
stabilizer, rather than the full group, is what counts here.)

**Design consequence: do not special-case the opening.** Choosing at layer E needs
`vals_{E-2}`, so the opening's real requirement is `vals_18` — 373,397 entries, ~3.7 MB
for A1. An avoid-list for ply 3 would be ~30 KB, genuinely 100x smaller, but it is a
second mechanism beside the lookup the player needs for every deeper ply anyway, in a
second format, and it bakes "plies 1-2 are always random" — a *result*, not a rule —
into control flow. Keep `vals_20/22/24` too (12,127 / 86 / 1 entries): free, and the
runtime logic stays uniformly "look up the children in the layer below".

**The size question is at the bottom, not the top.** Per seed: `vals_16` ~71 MB,
`vals_14` ~569 MB, `vals_12` ~2.6 GB. Where the live-search threshold lands swings the
table by orders of magnitude — measure live-search timing at 14 and 16 empties before
choosing it. (`check_values.py` at 16 empties is slow enough that 3 samples was all I
ran, which is itself a hint the threshold may need to sit below 16.)

Also noted: every ply-1 and ply-2 move is optimal, so the SPEC-mandated random tiebreak
is spending a free choice. Breaking ties toward positions where the opponent has more
losing replies would stay game-theoretically optimal and be practically stronger — a
deliberate SPEC change, not something to slip in.

## Threshold decision: T = 16 (step 3, decided 2026-08-19)

The table covers layers **>= 16 empties**; the player live-searches at 16 and below.
Measured with `tools/solver/bench_live_search.py`, which times the real `choose_move` (it
evaluates every root move full-window to collect ties, so it costs more than one
alpha-beta call) on *drawn* positions — positions with an immediate win short-circuit
before searching and measure nothing.

| threshold | table, 4 seeds | table, one seed | worst move measured |
|---|---|---|---|
| >= 18 | 11 MB | ~3 MB | **67.5 s** (A1 @ 18, TT 2.0M entries) |
| **>= 16** | **226 MB** | **~69 MB (A) / ~16 MB (C3)** | **8.9 s** (A1 @ 16, TT 330k) |
| >= 14 | 1.93 GB | ~530 MB | 0.23 s |

T=18 was tempting at 11 MB but rejected on evidence quality: three samples spread
13.6 / 26.4 / 67.5 s, a 5x range, so the tail is unknown. T=16's eight A1 samples ran
1.5-8.9 s, tightly clustered. Only one seed's table is ever loaded, so the resident
cost is ~69 MB worst case.

**Layers the table needs: 16, 18, 20, 22.** To choose at layer E you look up the
children at E-2, and the player chooses at 24/22/20/18 by lookup and live-searches at
16. `vals_24` is never consulted.

### Simplification: no seed->representative move remapping is needed

These notes previously assumed the runtime would map the real seed to its orbit
representative and map the chosen move back through that symmetry. **It does not.**
The table is keyed by `canonical_key`, which is already invariant under G, and the
player only ever looks up *values*, never moves. So the player generates children in
the real board frame, canonicalizes each one, and looks it up directly. The only thing
the seed determines is *which* representative's table to load.

## Table distribution (open question)

Tables are **gitignored** (`perfect-tables/`) for now — 38 MB gzipped for all four
seeds (C3 2.6 MB, A-class ~11 MB each), against 231 MB raw. gzip is as good as a
hand-rolled codec here: varint deltas would give 8.6 MB for layer 16's keys, and the
values take only **5 distinct scores**, so stdlib `gzip` wins on zero custom code.

A missing table is deliberately not an error — `load_for_seed` returns `None` and the
player searches instead, correct but far too slow in the opening. So a fresh checkout
produces a *working but impractical* `PerfectPlayer` until tables are built with:

```
uv run python tools/solver/build_table.py /tmp/solve-data2 perfect-tables
gzip -6 perfect-tables/*.table          # optional; the loader reads either form
```

Needs a decision: commit the 38 MB, use git-lfs / release assets, or leave it as
regenerate-only (which means an ~8 h solve for anyone who wants the fast player).

## Remaining work (in order)

1. ~~**Confirm parallel speedup + real peak memory**~~ — **DONE**, see above. Peak layer
   267,023,819 at 8 empties; 22.6 min wall; ~10.4x. Bucketed dedup implemented as sampled
   range splitters (`tools/solver/forward.py`).
2. ~~**Backward value pass**~~ — **DONE**, see above. `tools/solver/backward.py`; C3 valued
   in 16.8 min; root is a draw. Children are looked up with one `np.searchsorted` into the
   flattened (globally sorted) layer below, memory-mapped so all workers share one copy. WIN =
   1000 − depth; cat's = 0; loss via negation. Depth is position-derived
   ((mouse_count+snake_count−1)//2), so values are globally consistent.
3. ~~**Persist a compact upper-ply table**~~ — **DONE**. `tools/solver/build_table.py` packs
   layers 22/20/18/16 into one flat little-endian file per seed representative
   (`MAGIC`/version header, per-layer counts, then uint64 keys + int16 values).
   Verified byte-exact against the solver's arrays for all 4 seeds x 4 layers.
4. ~~**Wire `PerfectPlayer`**~~ — **DONE**. `players/table.py` (stdlib only: `array` +
   `bisect` + `gzip`) loads by orbit and caches; `_choose_from_table` picks a
   move by looking children up, and returns `None` — falling back to search — if the
   table cannot answer for *every* child, so a partial table degrades to slow-correct
   rather than fast-wrong. Opening moves: **~3 ms**. Full perfect-vs-perfect game:
   1.8 s (C3), 9.1 s (B3 -> A3's table). Both draw, as the solve says they must.
   New `tests/test_perfect_player.py` (8 tests) uses synthetic tables in `tmp_path`,
   so the suite passes on a checkout with no tables installed.
5. **Run all 4 seed classes** — C3 **DONE**; A1/A2/A3 **running** (started 16:24,
   2026-08-18, ~16 h, see above).
6. **Update `SPEC.md` §10** to describe the retrograde/offline-solve + live-endgame
   architecture actually built (current §10 describes the naive single-search TT that
   proved infeasible). Confirm wording with the user.

## Constraints (project rules — do not violate)

- **Strong typing everywhere**: full annotations on public APIs *and* locals/attributes;
  enums / frozen dataclasses for domain types. `mypy --strict` must pass.
- **Do not `git commit`/`push` until the user explicitly says to.** Finishing a task is not
  a commit signal.
- When told to commit: commit **directly on `main`** (no feature branch), then push.
  Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## Reproducing

```
uv run python tools/solver/forward.py      C3 /tmp/solve-data2 50000 16
uv run python tools/solver/backward.py     C3 /tmp/solve-data2 50000 16 12
uv run python tools/solver/build_table.py     /tmp/solve-data2 perfect-tables
```
