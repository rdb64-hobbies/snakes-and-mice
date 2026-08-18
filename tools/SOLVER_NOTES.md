# Perfect-player solver — handoff notes

Working notes for building the offline solver that powers `PerfectPlayer` (§10).
Pick up here in a fresh session (intended target: the 16-core / 64 GB Mac Studio).
**Nothing below is committed yet** — see "Getting this onto the Studio".

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
- **C3 layer sizes** (confirmed, reproduced exactly by the numpy enumerator):
  | empties | positions |
  |---|---|
  | 24 | 1 |
  | 22 | 20 |
  | 20 | 2,196 |
  | 18 | 64,440 |
  | 16 | 1,616,052 |
  | 14 | 12,875,213 |

  Still growing ~8×/step at 14 → hasn't peaked. Extrapolated C3 forward pass ~4 h
  single-core, peak layer ~150–200M positions (~1.5 GB). Hard seeds ~8× that.

## Where the code stands

Created / modified (all **uncommitted**):
- `src/snakes_and_mice/players/symmetry.py` — G (order 32), `canonical_key`, fast
  5-bit-chunk permute tables. **Done, validated at import.**
- `src/snakes_and_mice/players/perfect.py` — `PerfectPlayer`: full-depth negamax
  alpha-beta to terminals, canonical TT (value + bound flag), honest win moves, random
  tiebreak. **Correct** on constructed positions; only the opening scale is the issue.
- Wiring: `players/__init__.py`, `snakes_and_mice/__init__.py`, `cli_common.py`
  (names **Percy** = Mouse, **Perseus** = Snake; CLI kind `perfect`), `match_cli.py`.
- `pyproject.toml` — added `numpy>=2.1` to the dev group (solver-only; the runtime player
  must stay numpy-free).
- `tools/solve_forward.py` — numpy forward enumeration, now **parallelized** across a
  process pool. Each layer is written to `layer_XX.npy`; workers mmap it, expand a
  row-range, and write their unique child keys to a shard file (only paths cross the
  process boundary); the parent merges shard files with one `np.unique`.
  Run: `uv run python tools/solve_forward.py [SEED] [OUT_DIR] [CHUNK] [WORKERS]`.

## Open issue on the parallel forward pass (READ THIS)

First naive parallel attempt (return arrays via pool IPC) was **slower** than single-core:
the workers pickled large key arrays back, and the parent's final `np.unique` ran over a
concatenation inflated by cross-shard duplicates. Switched to **disk-backed shard outputs**
(workers `np.save` + return path) to kill the IPC cost. This still leaves **one serial
`np.unique` over the merged shards** as the tail. On the merge-heavy small layers the
serial tail caps speedup; on the big layers (which dominate total time) compute
parallelizes well, so overall should still be a big win — **needs a clean-machine
measurement to confirm** (the last runs on this 8-core box were contaminated by competing
solver processes, so their times are meaningless).

**If the serial merge dominates**, upgrade to **hash-bucketed parallel dedup**: workers
partition child keys into B buckets by a hash; a second parallel phase dedups each bucket
independently (disjoint keyspaces → no global merge). Keep each layer as B sorted
bucket-files so the *backward* pass can also stay fully parallel and never do a global sort.

## Remaining work (in order)

1. **Confirm parallel speedup + real peak memory** on the Studio with a C3 run to
   completion (16 workers). Record the true peak layer and total wall-clock.
2. **Backward value pass** (`tools/solve_backward.py`, to write): enumerate all layers to
   terminals (forward already does), then value bottom-up — a position's value = negamax
   over its children's already-computed values (look children up by canonical key in the
   layer below; use `np.searchsorted` on sorted layers, or per-bucket if bucketed). WIN =
   1000 − depth; cat's = 0; loss via negation. Depth is position-derived
   ((mouse_count+snake_count−1)//2), so values are globally consistent.
3. **Persist a compact upper-ply table** — values (and/or best-move sets) for positions
   with empties above the live-solve threshold (~16). Plain format, numpy-free to load.
4. **Wire `PerfectPlayer`** to: map the actual seed → its orbit representative, look up the
   table for the opening, live-solve below the threshold. Keep the runtime numpy-free.
   Runtime move must be mapped back through the seed→rep symmetry.
5. **Run all 4 seed classes** (C3, A1, A2, A3) on the Studio.
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

## Getting this onto the Studio

The Studio is a different machine, so the WIP must travel via git: **the user must OK a
commit+push of the uncommitted work above**, then `git pull` on the Studio. A fresh session
there should read this file first, then run step 1.
