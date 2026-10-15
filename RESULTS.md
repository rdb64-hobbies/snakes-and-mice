# Snakes and Mice — Tournament Results

These are my results from a series of matches pitting various LLMs against
each other, and against a perfect player, in the game of Snakes and Mice.
Take them for what they are and no more — a game is a narrow and quirky test,
and nothing here is an endorsement of, or a knock against, any model in
general. After all, it's just snakes and mice.

The first section covers six locally-served open-weight models. The second
covers closed-weight frontier models, reached through their vendors' APIs, and
the third covers open-weight frontier models, reached through a hosted router.
The fourth puts the best of each group against each other.

## 1. Local LLMs

Each of the six played a fixed 40-game match against a perfect algorithmic
player, to see how cleanly it plays against an opponent that never itself
errs (the opponent throughout is `perfect-trappiness`). The four that came
through cleanest then played a round-robin against each other (§1.3); the other
two were excluded from that stage.

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

## 2. Closed-weight frontier LLMs

Five closed-weight frontier models: **opus-5-5**, **sonnet-5-5**,
**gpt-6-1-sol**, **gpt-6-luna**, and **gemini-3-8**. All five were set to
medium thinking. Each played 600 games against three fixed opponents, 200
apiece: `rl`, `perfect-trappiness`, and `perfect-mistake-model` (§2.1). They
then played each other in a round-robin (§2.2).

### 2.1 Losses and faults against RL and the two perfect players

| Model | Games | Lost | Loss % | Faulted | Fault % |
|---|---:|---:|---:|---:|---:|
| **gemini-3-8** | 600 | 0 | 0.0% | 1 | 0.2% |
| opus-5-5 | 600 | 1 | 0.2% | 8 | 1.3% |
| sonnet-5-5 | 600 | 3 | 0.5% | 0 | 0.0% |
| gpt-6-1-sol | 600 | 4 | 0.7% | 4 | 0.7% |
| gpt-6-luna | 600 | 80 | 15.2% | 72 | 12.0% |

Sorted best-to-worst by losses. Loss % is losses over games where neither side
faulted; Fault % is faults over games played. The same counts by opponent,
lost / faulted out of 200 games each:

| Model | vs. rl | vs. perfect-trappiness | vs. perfect-mistake-model |
|---|---:|---:|---:|
| gemini-3-8 | 0 / 0 | 0 / 1 | 0 / 0 |
| opus-5-5 | 0 / 4 | 0 / 0 | 1 / 4 |
| sonnet-5-5 | 2 / 0 | 0 / 0 | 1 / 0 |
| gpt-6-1-sol | 1 / 2 | 1 / 0 | 2 / 2 |
| gpt-6-luna | 9 / 25 | 34 / 20 | 37 / 27 |

The fault breakdown across all three opponents:

- **gemini-3-8**: misjudged its own outcome ×1
- **opus-5-5**: miscounted pieces ×8
- **sonnet-5-5**: none
- **gpt-6-1-sol**: misjudged its own outcome ×3, miscounted pieces ×1
- **gpt-6-luna**: miscounted pieces ×37, misjudged its own outcome ×33,
  illegal move (cell occupied) ×2

Four of the five almost never lose. Across their 2,400 games against these three
opponents they lost eight, and **gemini-3-8** has yet to lose a single one. Their faults are just as
rare, and they are bookkeeping slips (a wrong piece count or a wrong claim about
the game's outcome), not thinking-budget blowouts or unplayable moves like
those in §1.1. **gpt-6-luna** is the outlier: it loses about one game in seven
and faults about one in eight. Its losses are mostly to the two perfect
players (34 and 37 of 80), who can't be beaten, so each of those was a game it
threw away. It fares best against `rl`, which is beatable.

Wins are where the opponent matters. Against the two perfect players the four
strong models can only draw, and they nearly always do. Against `rl`, which is
strong but not perfect, they win: **opus-5-5** 49 of its 200 games, **gemini-3-8**
29, **sonnet-5-5** 23, **gpt-6-1-sol** 21, and even **gpt-6-luna** 5.

### 2.2 Head-to-head among the five

Each pair of the five played each other, both seats, 100 games per pairing.

| Model | Played | Won | Lost | Tied | Faulted | Win % | Loss % | Fault % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **opus-5-5** | 400 | 6 | 0 | 373 | 3 | 1.6% | 0.0% | 0.8% |
| gpt-6-1-sol | 400 | 4 | 0 | 372 | 4 | 1.1% | 0.0% | 1.0% |
| sonnet-5-5 | 400 | 2 | 0 | 375 | 2 | 0.5% | 0.0% | 0.5% |
| gemini-3-8 | 400 | 2 | 0 | 378 | 1 | 0.5% | 0.0% | 0.2% |
| gpt-6-luna | 400 | 0 | 14 | 318 | 68 | 0.0% | 4.2% | 17.0% |

(Played also counts games won because the opponent faulted, so a row doesn't
sum to its Played.)

**Head-to-head matrix** — cell (row, column) is the row player's losses to the
column player:

|                    | 1. opus-5-5 | 2. gpt-6-1-sol | 3. sonnet-5-5 | 4. gemini-3-8 | 5. gpt-6-luna |
|---|---:|---:|---:|---:|---:|
| **1. opus-5-5**    | — | 0 | 0 | 0 | 0 |
| **2. gpt-6-1-sol** | 0 | — | 0 | 0 | 0 |
| **3. sonnet-5-5**  | 0 | 0 | — | 0 | 0 |
| **4. gemini-3-8**  | 0 | 0 | 0 | — | 0 |
| **5. gpt-6-luna**  | 6 | 4 | 2 | 2 | — |

The four strong models never beat one another: every decided game in the
round-robin is a gpt-6-luna loss, and the other four rows of the matrix are
zeros. So the head-to-head tells us little. Those four are not separable by
this test, and the ordering in the table is just the luck of whose games luna
happened to lose.

### 2.3 Takeaway

With the exception of gpt-6-luna, the closed-weight frontier models play
extremely well. Against opponents that are either unbeatable or at least
strong, and against each other, they almost always draw, they very rarely
lose, and when they fault it is a slip in counting or in reading the outcome.
They are not perfect: opus-5-5, sonnet-5-5, and gpt-6-1-sol each lost at
least once, always to `rl` or one of the perfect players, never to each other.
But **gemini-3-8** has not lost in any of its 1,000 games.
gpt-6-luna, considerably the cheapest of the five, is the one that doesn't hold
up: its loss and fault rates are far from those of its four closed-weight
peers.

## 3. Open-weight frontier LLMs

Three open-weight frontier models, hosted rather than served locally (all via
OpenRouter): **kimi-k3**, **glm-5-3**, and **deepseek-v4**, each at medium
thinking. They were not played against `perfect-trappiness`. Each played 200
games against two opponents, 100 apiece: `rl` and `perfect-mistake-model`
(§3.1). They then played each other in a round-robin (§3.2).

### 3.1 Losses and faults against RL and the perfect mistake model

| Model | Games | Lost | Loss % | Faulted | Fault % |
|---|---:|---:|---:|---:|---:|
| **deepseek-v4** | 200 | 2 | 1.0% | 6 | 3.0% |
| glm-5-3 | 200 | 18 | 9.2% | 5 | 2.5% |
| kimi-k3 | 200 | 28 | 14.1% | 2 | 1.0% |

Sorted best-to-worst by losses. Loss % is losses over games where neither side
faulted; Fault % is faults over games played. The same counts by opponent,
lost / faulted out of 100 games each:

| Model | vs. rl | vs. perfect-mistake-model |
|---|---:|---:|
| deepseek-v4 | 0 / 1 | 2 / 5 |
| glm-5-3 | 8 / 1 | 10 / 4 |
| kimi-k3 | 17 / 1 | 11 / 1 |

The fault breakdown across both opponents:

- **deepseek-v4**: thinking limit exceeded ×6
- **glm-5-3**: misjudged its own outcome ×4, miscounted pieces ×1
- **kimi-k3**: misjudged its own outcome ×1, miscounted pieces ×1

Faults are rare for all three, so the separation is in the losses.
**deepseek-v4** is the standout: it lost twice in 200 games, and all six of its
faults were running out of thinking budget. **glm-5-3** and **kimi-k3** are far
less steady, each losing to the perfect mistake model (which can't be beaten)
10 or 11 times in 100, and to `rl` another 8 and 17. Against `rl`,
**deepseek-v4** won 24 games of 100, **kimi-k3** 13, and **glm-5-3** 4.

### 3.2 Head-to-head among the three

Each pair played 100 games, both seats.

| Model | Played | Won | Lost | Tied | Faulted | Win % | Loss % | Fault % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **deepseek-v4** | 200 | 22 | 1 | 133 | 5 | 14.1% | 0.6% | 2.5% |
| kimi-k3 | 200 | 10 | 16 | 144 | 7 | 5.9% | 9.4% | 3.5% |
| glm-5-3 | 200 | 7 | 22 | 113 | 54 | 4.9% | 15.5% | 27.0% |

(Played also counts games won because the opponent faulted, so a row doesn't
sum to its Played.) The faults:

- **deepseek-v4**: thinking limit exceeded ×5
- **kimi-k3**: misjudged its own outcome ×4, miscounted pieces ×2, illegal move
  (cell occupied) ×1
- **glm-5-3**: unparseable output ×34, misjudged its own outcome ×10,
  miscounted pieces ×9, thinking limit exceeded ×1

**Head-to-head matrix** — cell (row, column) is the row player's losses to the
column player:

|                   | 1. deepseek-v4 | 2. kimi-k3 | 3. glm-5-3 |
|---|---:|---:|---:|
| **1. deepseek-v4** | — | 1 | 0 |
| **2. kimi-k3**     | 9 | — | 7 |
| **3. glm-5-3**     | 13 | 9 | — |

**deepseek-v4** again leads, beating both others (9–1 over kimi-k3, 13–0 over
glm-5-3). The other two swap places from §3.1: **kimi-k3** beats glm-5-3 (9–7)
even though it lost more to `rl`. glm-5-3's last place owes a lot to faults,
though: over a quarter of its games ended in one, most of them unparseable
output rather than bad play.

### 3.3 Takeaway

deepseek-v4 is the strongest of the three by a wide margin, and on losses it is
the only one that plays like the closed-weight models of §2. kimi-k3 and
glm-5-3 lose often, to the perfect mistake model as well as to each other.

One caveat on deepseek-v4's faults. OpenRouter picks the backend per request,
and in a test match of deepseek-v4 an Alibaba-hosted backend ran about ten times
longer reasoning than CoreWeave and hit the output cap, while the same turns on
CoreWeave were fine. The tournament file doesn't record which backend served
each game, so some of its thinking-limit faults may be the backend's doing
rather than the model's.

## 4. Best of each group

The best model from each category above played a round-robin against the other
two: **gemini-3-8** (closed-weight frontier, §2), **deepseek-v4** (open-weight
frontier, §3), and **qwen-3-8** (local, §1; served from the RTX box), 100 games per
pairing, both seats.

| Model | Played | Won | Lost | Tied | Faulted | Win % | Loss % | Fault % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **gemini-3-8** | 399 | 8 | 0 | 360 | 0 | 2.2% | 0.0% | 0.0% |
| deepseek-v4 | 399 | 7 | 0 | 361 | 27 | 1.9% | 0.0% | 6.8% |
| qwen-3-8 | 400 | 0 | 15 | 369 | 12 | 0.0% | 3.9% | 3.0% |

(One gemini-3-8 vs. deepseek-v4 game was aborted, hence 399. Played also counts
games won because the opponent faulted, so a row doesn't sum to its Played.)

**Head-to-head matrix** — cell (row, column) is the row player's losses to the
column player:

|                    | 1. gemini-3-8 | 2. deepseek-v4 | 3. qwen-3-8 |
|---|---:|---:|---:|
| **1. gemini-3-8**  | — | 0 | 0 |
| **2. deepseek-v4** | 0 | — | 0 |
| **3. qwen-3-8**    | 8 | 7 | — |

Gemini-3-8 and deepseek-v4 never beat each other or lost to qwen-3-8; every
decided game in the round-robin is a qwen-3-8 loss (8 to gemini-3-8, 7 to
deepseek-v4). So the two leaders can't be separated by wins, and what
distinguishes them is faults: deepseek-v4 had 27, mostly unparseable output
(20) against gemini-3-8, while gemini-3-8 had none. qwen-3-8 is third, but it
lost only 15 of 400 games, and 369 were draws.
