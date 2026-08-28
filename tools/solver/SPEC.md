# The offline solver

Specification for the programs in `tools/solver/` that solve Snakes and Mice and
produce the opening table `PerfectPlayer` reads. The player itself is specified in
[`../../SPEC-perfect-player.md`](../../SPEC-perfect-player.md); this document covers
only how the table is produced and what is in it.

`NOTES.md` alongside this file is the *working record* — measurements, dead
ends, and the reasoning behind the constants. This is the normative description.

## Layout

```
tools/solver/
    SPEC.md              this document
    NOTES.md             working record: measurements, dead ends, incidents
    enumeration.py       vectorized position/child generation shared by both passes
    forward.py           forward pass: enumerate every reachable position
    backward.py          backward pass: value them bottom-up
    build_table.py       pack the shipped layers into one file per seed class
    check_values.py      verify solved values against the live search
    dump_table.py        read a table back in human-readable form
    bench_live_search.py time the live search; sets the depth threshold
    solve_all.sh         run the A-class seeds end to end, sequentially
    watchdog.sh          stop a run before it exhausts the machine
```

## Why an offline solve

Full-depth search is exact and, in the endgame, fast. In the opening it is hopeless:
above ~16 empty cells the tree is far too wide to search per move. Since the game is
small enough to solve outright, it is solved once, offline, and the upper plies are
looked up at run time (`SPEC-perfect-player.md`, "The opening table").

## Scope: four seeds cover twenty-five

Position values are invariant under the order-32 symmetry group **G** of the 12 winning
lines (`SPEC-perfect-player.md`). Under G the 25 seed cells fall into **4 orbits**,
represented by **C3, A1, A2 and A3**, so four solves cover every opening. C3 has a
stabilizer of order 32; the three A-class seeds have stabilizer 4 and are
correspondingly larger to solve — about 3× C3 in practice.

Every position is identified by its **canonical key**: the numerically smallest of its
32 transformed forms, mouse and snake permuted together and packed into one `uint64`
(`SPEC-perfect-player.md`, "Canonical position").

## Constraints

- **The solver may use numpy and multiple processes; the player uses neither.**
  Nothing in `tools/solver/` is imported by `src/snakes_and_mice/players/` at run
  time — the dependency runs the other way, with the solver importing the player's
  board representation and symmetry group so the two cannot drift apart. numpy is a
  dev-group dependency, so it is not installed for someone who only wants to play;
  keeping the player clear of it is a preference worth holding cheaply, not a rule.
- **Memory is the binding constraint, not time.** A single layer's intermediate
  results run to billions of keys. Every stage must have a peak that is bounded by a
  constant, not by the size of the layer or the seed class.
- **Strong typing throughout**; `mypy --strict` must pass.

## The forward pass

`forward.py`. Enumerates every reachable **non-terminal** canonical
position, layer by layer, from the opening (24 empty cells) downward. A *layer* is all
positions sharing an empty-cell count; each move fills two cells, so layers step by two.

For each layer: generate every child of every position, discard the **terminal** ones —
a completed line, or a cat's game — and deduplicate the rest by canonical key. Terminal
positions are never stored; their value is immediate and is recomputed when needed.

Expansion is embarrassingly parallel and is sharded across a process pool. Each worker
takes a contiguous run of the layer, expands it, and writes sorted unique keys to disk;
only paths cross the process boundary.

### Partitioning: range splitters, not hashes

Deduplicating a layer in the parent does not fit in memory — one layer's shards can
concatenate to billions of keys, and sorting them needs a second copy alongside. So each
layer is **range-partitioned into buckets** and each bucket is deduplicated
independently, in parallel, with no global merge.

Bucket boundaries are **quantiles of a sampled key pool**, not fixed key ranges and not
hashes:

- **Fixed ranges fail.** Canonicalization minimizes over 32 symmetries, which clusters
  keys hard in their high bits; splitting on the top bits leaves most buckets empty.
- **Hashes fail differently.** They balance perfectly but scatter key adjacency, and
  adjacency is what makes the per-shard deduplication effective — neighbouring
  positions in a sorted layer share most of their children. Hashing nearly doubles the
  intermediate data and costs more than the parallel dedup saves.
- **Sampled quantiles** balance the buckets *and* keep them ordered by key. Shards stay
  contiguous key ranges, so the per-shard dedup keeps working, and concatenating buckets
  in order yields a globally sorted layer — which the backward pass then binary-searches.

Each shard file records its first and last key, so a bucket skips shards it cannot
overlap without opening them.

### Bounding memory

Two constants bound the peak, and they bound *outputs*, not inputs:

- **`FLUSH_ROWS`** — a worker writes its accumulated child keys to a file and starts
  fresh once it reaches this many. This is what keeps peak memory flat. Sizing a shard
  by how many *parent* rows it reads does **not** bound memory: the ratio of children
  surviving deduplication to parents read swings by an order of magnitude between
  layers (below), so a constant calibrated on one layer will exhaust memory on another.
  A shard therefore emits *one or more* files.
- **`TARGET_BUCKET_ROWS`** — buckets per layer scale with the work so that one bucket
  stays small regardless of seed class.

### Duplication alternates with the side to move

Shard duplication swings by more than 10× from layer to layer. This is expected, not a
bug. The canonical key is `(mouse << 25) | snake`, so sorted order is dominated by the
mouse mask. **When the snake moves the mouse mask is unchanged**, so contiguous shards
produce nearly disjoint children (~1× duplication). **When the mouse moves** the keys
scatter and each child is reached from many shards (8–15×). Any constant calibrated
against one parity will be badly wrong for the other.

## The backward pass

`backward.py`. Values every enumerated position, working **up** from the endgame:
a layer depends only on the layer below it, which is already final.

For a position `P`, the value is the negamax maximum over its moves of

- `WIN − depth(P)` if the move completes a line,
- `0` if the move produces a cat's game,
- `−value(C)` otherwise,

with `WIN = 1000` and `depth` read off the position as `(occupied − 1) // 2`. The
depth is that of the position the winning move is made **from** — this must match the
live search exactly, since a game mixes values from both (`SPEC-perfect-player.md`,
"Exact, depth-aware scoring").

Terminal children were dropped by the forward pass and so are re-detected here rather
than looked up. Every **non-terminal** child is guaranteed to be present in the layer
below, because the forward pass built that layer as exactly that set; a child that
cannot be found is a hard error, not a miss.

Children are generated **rectangularly**: every position in a layer has exactly
`C(empties, 2)` moves, so a batch of `n` positions expands to an `n × C(empties, 2)`
grid and the negamax maximum is one reduction along the row. Values come from a single
binary search into the layer below, which is globally sorted and memory-mapped so all
workers share one copy.

Only two-piece moves are enumerated. A one-piece move is legal only when it ends the
game (SPEC §2.5), so it never produces a non-terminal child, and a one- and a two-piece
win score identically.

### Space-saving mode

A layer's buckets may be dropped once flattened, and a layer's keys and values once the
layer above has been valued and it lies below a threshold. Deep layers exist only to
value the shallow ones; only layers ≥ 16 empties reach the shipped table. This is the
difference between a seed's output being ~3 GB and ~57 GB, which is the difference
between the A-class seeds fitting on disk and not.

## The table file

`build_table.py` packs the shipped layers into one file per seed representative, named
for that seed (`A1.table`, `C3.table`, optionally `.gz`).

The stored layers sit **two below** the layers at which the player moves: choosing at
`E` empty cells means valuing children at `E−2`. Storing **22, 20, 18, 16** therefore
covers the player's choices at **24, 22, 20, 18**, and it live-searches from 16 down.

All little-endian:

```
magic     8s   b"SNM-PERF"
version   B    1
seed_bit  B    representative seed cell index, 0..24
layers    B    number of layer sections
reserved  B
then `layers` × (empties B, reserved B, count I)
then, per layer in that order:
    count × uint64   canonical keys, ascending
    count × int16    values, positionally aligned
```

Keys ascend so the player can binary-search them; values are negamax scores for the
side to move at that layer. `vals_24` is deliberately absent — choosing at layer `E`
reads layer `E−2`, and nothing chooses at 26.

`dump_table.py` reads a table back in human-readable form — layer summaries with
value distributions, filtered listings, board diagrams, and the value of a named
position — so the solved game can be explored without writing code.

The format is chosen so it can be read **without numpy**, with `array.frombytes` and
`bisect`. gzip is roughly 6× and is stdlib; a custom codec (delta-encoded keys, packed
values — the values take only a handful of distinct scores) would roughly halve it
again and is not currently worth the code.

## Verification

The solver is checked in three independent ways, and all three are part of trusting a
result:

1. **Against the live search.** `check_values.py` re-derives sampled values with
   `PerfectPlayer._negamax` — the same code path a real game takes, and an
   implementation that shares nothing with the solver but the rules. This is what
   catches a disagreement in the scoring convention, which is otherwise invisible
   because each side is internally consistent.
2. **Against combinatorics.** Shallow layer counts can be derived independently by
   Burnside's lemma over the seed's stabilizer, and by brute force. Note that the
   stabilizer's *order* is not enough — orbit counts depend on each element's
   fixed-point count, which is why two seed classes of equal stabilizer order can
   still have different layer sizes.
3. **Structurally, on every run.** The backward pass fails loudly if any non-terminal
   child is absent from the layer below, and flattening a layer asserts its buckets are
   internally sorted and non-overlapping. Together these would catch a partitioning
   that dropped or duplicated keys, which is the failure mode the bucketing risks.

`bench_live_search.py` times the real `choose_move` on solved positions; it is what
sets the table's depth threshold. It must sample **drawn** positions — a position with
an immediate win short-circuits before searching and measures nothing.

An earlier, simpler solver deduplicated each layer with one `np.unique` in the parent,
and the partitioned layers were checked against it byte-for-byte. It has been removed:
it cannot run at the size of the low-symmetry seeds, and it shared all its enumeration
code with the current solver, so it only ever validated the *partitioning* — which
check 3 now covers on every run.

## Running it

```
uv run python tools/solver/forward.py      SEED OUT_DIR [CHUNK] [WORKERS]
uv run python tools/solver/backward.py     SEED OUT_DIR [CHUNK] [WORKERS] [KEEP]
uv run python tools/solver/build_table.py  OUT_DIR TABLE_DIR [SEED...]
gzip -6 TABLE_DIR/*.table                  # optional; the loader reads either form
```

`solve_all.sh` runs all three A-class seeds end to end, sequentially — each saturates
every core, so overlapping them finishes no sooner. `watchdog.sh` should run
alongside any unattended solve: a mis-sized constant does not slow the machine down, it
exhausts RAM within a layer, and the watchdog stops the run to save the machine.

Two operational traps, both learned the hard way:

- **Killing the parent does not kill the pool.** Workers run as
  `python3 -c from multiprocessing.spawn import spawn_main ...`; the parent's name
  appears nowhere in their command line, and they will keep holding the memory.
- **Do not watch "free memory" on macOS.** It is kept low by design. Watch swap, the
  compressor, and `kern.memorystatus_vm_pressure_level`.
