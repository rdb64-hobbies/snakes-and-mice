# Snakes and Mice — Tournament Results

These are my results from a series of matches pitting various LLMs against
each other, and against a perfect player, in the game of Snakes and Mice.
Take them for what they are and no more — a game is a narrow and quirky test,
and nothing here is an endorsement of, or a knock against, any model in
general. After all, it's just snakes and mice.

This first section covers six locally-served open-weight models. A later
section will cover frontier (hosted-API) models.

## 1. Local LLMs

Each of the six played a fixed 40-game match against a perfect algorithmic
player, to see how cleanly it plays against an opponent that never itself
errs. The four that came through cleanest then played a round-robin against
each other (§1.3); the other two were excluded from that stage.

### 1.1 Cleanliness against perfect play, and how the six reason

| Model | Games vs. Perfect | Faults | Fault % | Reasoning / move (median) | Re-sent into next turn's context? |
|---|---:|---:|---:|---|:---:|
| qwen-3-8 | 40 | 1 | 2.5% | 1,036 tokens (~2,700 chars) | **Yes** |
| gpt-oss-120b | 40 | 4 | 10.0% | not tagged by server (~7,800 chars) | **Yes** |
| gemma-4 | 40 | 5 | 12.5% | 1,641 tokens (~4,600 chars) | No |
| nemotron-3-super | 40 | 7 | 17.5% | 2,222 tokens (~7,400 chars) | No |
| ornith-1-5 | 40 | 15 | 37.5% | 5,155 tokens (~14,900 chars) | **Yes** |
| qwen-3-6 | 40 | 18 | 45.0% | 5,186 tokens (~12,600 chars) | No |

Sorted best-to-worst by fault rate. The fault breakdown against Perfect:

- **qwen-3-8**: thinking limit exceeded ×1
- **gpt-oss-120b**: illegal move (cell occupied) ×3, misjudged its own outcome ×1
- **gemma-4**: misjudged its own outcome ×2, miscounted pieces ×2, illegal move ×1
- **nemotron-3-super**: illegal move ×3, thinking limit exceeded ×3, misjudged its own outcome ×1
- **ornith-1-5**: thinking limit exceeded ×10, unparseable output ×4, misjudged its own outcome ×1
- **qwen-3-6**: thinking limit exceeded ×15, misjudged its own outcome ×2, illegal move ×1

There's a clean split: four models keep faults at or under ~12.5%, and the
faults they do commit are spread across ordinary mistakes (illegal cells,
misjudged outcomes, miscounted pieces). The other two — **ornith-1-5** and
**qwen-3-6** — fault far more often, and overwhelmingly by running out of
their thinking budget before ever committing to a move. That's not a
coincidence — they're also the two heaviest reasoners in the table.

### 1.2 Why the reasoning column looks so uneven

All six models were meant to reason at the same effort level, to keep the
comparison fair. In practice, that setting only reaches models the harness
recognizes by name — which excludes most self-hosted models — so each of
these six is really reasoning at whatever level its own deployment happens to
default to:

- **qwen-3-8** is explicitly configured to reason less than the others — the
  lightest reasoner of the six by design, not by accident.
- **gemma-4** has thinking turned on but no effort dial, so it reasons at
  whatever its own default is.
- The rest (**gpt-oss-120b**, **nemotron-3-super**, **ornith-1-5**,
  **qwen-3-6**) have no override at all — and for **ornith-1-5** and
  **qwen-3-6**, that default turns out to be very heavy, heavy enough to
  repeatedly blow through the per-move budget (§1.1).

The "re-sent into context?" column is a second, independent axis — a property
of each model's own reasoning style, not of any setting this project
controls. **gpt-oss-120b**, **qwen-3-8**, and **ornith-1-5** feed their prior
reasoning back in and pay for it again each turn; **gemma-4**,
**nemotron-3-super**, and **qwen-3-6** drop it and reason fresh each time. So
two models "reasoning the same amount" per move can still accumulate very
different amounts of context over a match.

### 1.3 Head-to-head among the four cleanest players

The four models with the lowest fault rates against Perfect —
**qwen-3-8**, **gpt-oss-120b**, **gemma-4**, and **nemotron-3-super** — then
played each other in a round-robin (both seats, several rounds). **ornith-1-5**
and **qwen-3-6** were left out: both faulted well over a third of their games
against Perfect, mostly by exceeding their thinking budget, which would make
any head-to-head result more about the token cap than about play quality.

| Model | Played | Won | Lost | Tied | Faulted | Win % | Loss % | Fault % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **qwen-3-8** | 298 | 29 | 5 | 210 | 5 | 11.9% | 2.0% | 1.7% |
| gpt-oss-120b | 299 | 11 | 10 | 196 | 53 | 5.1% | 4.6% | 17.7% |
| nemotron-3-super | 299 | 11 | 17 | 187 | 59 | 5.1% | 7.9% | 19.7% |
| gemma-4 | 296 | 6 | 25 | 187 | 32 | 2.8% | 11.5% | 10.8% |

Most games between any two of these four end in a draw — the game is small
and heavily drawish under decent play — so the win/loss split among the
non-drawn, non-faulted games is where the separation shows up.

**Head-to-head matrix** — cell (row, column) is the row player's losses to the
column player; a row sums to that player's total losses, a column to its total
wins:

|                    | 1. qwen-3-8 | 2. nemotron-3-super | 3. gpt-oss-120b | 4. gemma-4 |
|---|---:|---:|---:|---:|
| **1. qwen-3-8**        | — | 1 | 3 | 1 |
| **2. nemotron-3-super** | 9 | — | 4 | 4 |
| **3. gpt-oss-120b**     | 5 | 4 | — | 1 |
| **4. gemma-4**          | 15 | 6 | 4 | — |

**qwen-3-8 is the clear winner of the four.** It has the best win rate, the
lowest loss rate, and the lowest fault rate by a wide margin, and it beats
every other model head-to-head — most lopsidedly against gemma-4 (15–1) and
nemotron-3-super (9–1), and still comfortably against gpt-oss-120b (5–3). That
it's also the model running the *lightest* reasoning load of the four (§1.1,
§1.2) is worth flagging rather than explaining away: more reasoning tokens did
not translate into better play here.

**gemma-4 is the clear laggard.** It has the worst win rate and by far the
worst loss rate of the four, losing especially badly to qwen-3-8 (1–15) and
also losing more than it wins against the other two. gpt-oss-120b and
nemotron-3-super land in between and are close to each other overall (both
~5.1% win rate, and dead even head-to-head at 4–4), with gpt-oss-120b's edge
coming from a lower fault rate (17.7% vs. 19.7%) and fewer overall losses —
essentially a toss-up between the two, well behind qwen-3-8 and well ahead of
gemma-4.

## 2. Frontier LLMs

_To be added._
