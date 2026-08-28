# Snakes and Mice — Specification

> Status: **living draft**. This document defines the game and the project's
> roadmap, through **1.0** and beyond (see §11 for the versioning scheme).

## Which document do I need?

The specification is split so that a reader — human or agent — need only load the part
their work touches. Section numbers are stable and never reused.

| Document | Covers |
| --- | --- |
| **`SPEC.md`** (this file) | The game and rules (§2), the `Player` abstraction (§3), matches (§5), tournaments (§6), the CLI (§7), architecture (§8), tooling (§9), versioning and scope (§11). |
| [`SPEC-llm-player.md`](SPEC-llm-player.md) | The **LLM player** in full: Pydantic AI, structured output, the message thread, reasoning pruning, model/provider selection, thinking levels, message logging, endpoint probes. Summarized in §4. |
| [`SPEC-perfect-player.md`](SPEC-perfect-player.md) | The **algorithmic (perfect) player** in full: alpha–beta search, depth-aware scoring, the 32-element symmetry group and transposition table, the tie-break, the opening table, the solved result. Summarized in §10. |
| [`tools/solver/SPEC.md`](tools/solver/SPEC.md) | The **offline solver** that produces the perfect player's opening table, and the table file format. |

§4 and §10 remain in this document as summaries: each states the contract the rest of
the specification depends on and points to its full document. If you are not working on
that player, the summary is all you need.

## 1. Overview

Snakes and Mice is a two-player, turn-based board game — a variant of
tic-tac-toe played on a larger grid where each player places two pieces per
turn. This project provides a **game engine** and a framework in which
**different types of players** (humans and various types of AI or
algorithmic bots) can play against each other.

**Primary goal.** A primary purpose of this project is to **compare LLMs as a
test of reasoning skill**, using the game as the benchmark instrument. Two kinds
of matchups both matter:

- **LLM vs. LLM** — models playing head-to-head against one another, to rank them
  relative to each other.
- **LLM vs. non-LLM** — models playing a strong reference opponent (e.g. the
  algorithmic player), to measure each model against a calibrated, fixed-strength
  yardstick.

A model's reasoning skill is judged not only by wins, draws, and losses but also
by **how often it faults** — a player fault being an illegal move, a failure to
produce an interpretable move, or a misreading of the game's outcome. Fault
frequency, broken down by `PlayerFaultReason` (§3), is itself a meaningful
metric. This is why supporting a range of LLM providers and models (§4) and a
tournament structure for head-to-head comparison (§3) are central rather than
incidental features.

The design goal is a clean separation between:

- the **engine** (rules, board state, win/draw detection), and
- **players** (agents that choose moves).

The engine never knows or cares _what kind_ of player is on either side.

## 2. Game play

### 2.1 The two players

There are exactly two players, identified by the piece they play:

- **Mouse** — plays mouse pieces.
- **Snake** — plays snake pieces.

**Mouse moves first.**

### 2.2 Board and coordinates

- The board is a **5 × 5 grid**.
- **Rows** are labeled `A`, `B`, `C`, `D`, `E` (top to bottom).
- **Columns** are labeled `1`, `2`, `3`, `4`, `5` (left to right).
- A **cell** is named `<row><column>`, e.g. the center cell is `C3`.

```
      1   2   3   4   5
    +---+---+---+---+---+
  A |   |   |   |   |   |
    +---+---+---+---+---+
  B |   |   |   |   |   |
    +---+---+---+---+---+
  C |   |   |   |   |   |
    +---+---+---+---+---+
  D |   |   |   |   |   |
    +---+---+---+---+---+
  E |   |   |   |   |   |
    +---+---+---+---+---+
```

Every cell is in exactly one of three states: **empty**, **mouse**, or
**snake**.

### 2.3 Lines

A **line** is a set of 5 cells that a player must fully occupy to win.
There are **12 lines** in total:

- **5 rows:** each of `A`–`E`.
- **5 columns:** each of `1`–`5`.
- **2 diagonals:**
  - Main diagonal: `A1, B2, C3, D4, E5`
  - Anti-diagonal: `A5, B4, C3, D2, E1`

### 2.4 Setup

The board begins **empty except for a single seeded snake** on one cell. This is
a starting position, not a move by the Snake player, but it counts as a real
snake piece for all win detection.

**Which cell is seeded is a setup parameter.** It defaults to **`B3`** — which
lies on **row B** and **column 3**, and on **neither diagonal** — but a caller
may pin it to any cell or have it chosen at random each game. Randomizing the
opening is how the benchmark keeps deterministic players from replaying identical
games; see §5, "Randomized openings." A seed that lies on a diagonal (e.g. the
center `C3`) gives the Snake a head start on more lines than the off-diagonal
default, but the rules are otherwise identical wherever the snake is seeded.

Every player is **told the seeded cell at the start of each game** (§3), so a
player that tracks its own board can place the snake correctly.

### 2.5 A turn / a move

- Players alternate turns, **Mouse first**: Mouse, Snake, Mouse, Snake, ...
- On a turn, the player makes one **move**, which normally consists of **placing
  two of their own pieces** on **two distinct empty cells**.
- **Single-piece exception.** A move may instead place a **single** piece, but
  _only_ when that one piece **ends the game** — i.e. it completes a line (a win)
  or fills the board into a cat's game. This mirrors §2.6 and §2.7: if the first piece
  already ends the game, the second is never placed, so a one-piece move is the
  honest representation of that turn. Placing a single piece that leaves the game
  still in play is illegal (§2.8).
- Placed pieces are permanent; they are never moved or removed.

The board starts with 24 empty cells and 2 pieces are placed per move, so a full
game is at most 12 moves and the board fills completely with no leftover single
cell. A player therefore always has at least two empty cells available at the
start of their turn (until the game ends).

### 2.6 Winning

A player **wins the moment they occupy all 5 cells of any line** (row, column,
or diagonal) with their own pieces.

- Win detection happens **after each individual piece placement**. If a player's
  _first_ of two pieces completes a line, they win immediately and the second
  piece is not placed. A player that foresees this may submit a **single-piece
  move** (§2.5) for that winning piece alone.
- The pre-placed (seeded) snake counts toward the Snake player's lines, wherever
  it sits (§2.4).

### 2.7 Cat's game (draw)

A **line is dead** once it contains **at least one mouse and at least one
snake** — it can never be completed by either player.

The game is a **cat's game** when **all 12 lines are dead**, so no player can
possibly win. This can occur before the board is completely full.

If the board fills completely without a win, that state is necessarily a cat's
game (every line is dead) and is reported as such.

### 2.8 Illegal moves

A move can be illegal for these reasons:

- a target cell is off the board,
- a target cell is not empty,
- the two target cells are the same cell,
- the wrong number of cells is specified (a move must place one or two pieces), or
- a **single**-piece move that does **not** end the game (a single piece is legal
  only when it wins or completes a cat's game — see §2.5).

How illegal moves are detected, surfaced, and attributed to a player is part of
the player/engine machinery, described in §3.

## 3. Player abstraction

A player is a **stateful agent** that tracks its own view of the game and chooses
moves for whichever side it has been assigned. All player types share one
interface, defined as a Python **abstract base class (ABC)** — chosen over a
`Protocol` because we own every implementation, want instantiation-time
enforcement of the contract (an unimplemented method fails loudly, matching the
error-first stance of §2.8), and want to share behavior and construction across
players.

### Core types

The domain is modeled with validated value types, so illegal states are
unrepresentable (see §2.8):

- `Side` — an enum, `MOUSE` or `SNAKE`; `Side.other` gives the opponent. A cell's
  occupant is a `Side` (the side whose piece sits there) or `None` when empty.
- `Cell` — a board coordinate; a frozen dataclass validated to be **on the
  board** at construction. Parses/renders labels like `C3`.
- `Move` — a frozen dataclass validated at construction to be **one or two
  `Cell`s** (and, if two, distinct), in the order the player plays them. A
  single-cell move is structurally valid but _legal_ only when that one piece
  ends the game; the engine enforces that at apply-time (§2.8).

Constructing an invalid `Cell` or `Move` raises `IllegalMove` (§2.8).

- **Each player manages its own board state.** A player keeps its own internal
  representation of the board, updated as moves happen. Because of this,
  `choose_move` takes **no state argument** — the player already has it.
- **The engine keeps the authoritative board.** The engine maintains its own
  board independently and uses it to validate legality (§2.8) and detect
  win/cat's-game. The engine's board is the source of truth; a player's private
  board is never trusted for rules enforcement.

### Roles are assigned per game

A player's side (Mouse or Snake) is **not** fixed at construction; it is assigned
at the **start of each game**. A single player instance can therefore play many
games and switch sides between them.

### Interface

```python
class Player(ABC):
    def __init__(self, name: str | None = None) -> None:
        self.name = name or type(self).__name__

    @abstractmethod
    def start_game(self, side: Side, seed: Cell) -> None:
        """Begin a new game as `side`, with the snake seeded on `seed`.
        (Re)initialize internal board state, seeding the snake on `seed` (the
        opening may be randomized, §2.4/§5). Resets any state from a prior game."""

    @abstractmethod
    def observe_move(self, side: Side, move: Move) -> None:
        """Called after every accepted move by either player, so the player can
        update its own board. A player is notified of its own moves too."""

    @abstractmethod
    def choose_move(self) -> MoveChoice:
        """Return a `MoveChoice` — a legal move (two distinct empty cells) for this
        player's side, plus an optional self-assessment of the resulting outcome
        (see "The move choice" below). Based on the player's own managed board
        state. If the player cannot produce a well-formed move at all (e.g. an
        LLM's output fails to parse), it raises `MoveUnavailable(reason)` instead
        of returning."""

    def end_game(self, result: GameResult) -> None:
        """Optional hook (default no-op): the game is over, here is the result."""
```

`start_game`, `observe_move`, and `choose_move` are required; `end_game` is an
optional hook with a default no-op body. Further optional hooks may be added
later if a player type needs them, without breaking existing players.

### The move choice and self-assessed outcome

`choose_move` returns a `MoveChoice`: the chosen move plus an **optional**
self-assessment of the resulting game state.

```python
class TurnOutcome(Enum):
    WIN         # the mover believes this move wins the game for it
    CATS_GAME   # the mover believes this move makes the game a cat's game
    IN_PLAY     # the mover believes the game continues

@dataclass(frozen=True)
class MoveChoice:
    move: Move
    claimed_outcome: TurnOutcome | None = None   # the mover's self-assessment;
                                                 # None if it does not self-assess
```

- A player can only ever claim `WIN`, `CATS_GAME`, or `IN_PLAY` about **its own**
  move — you cannot lose on your own turn (you place only your own pieces).
- **Supplying `claimed_outcome` is optional.** Mechanical players (scripted,
  algorithmic, random) read the board correctly by construction and leave it
  `None`; the engine then performs no self-assessment check for them. The LLM
  player _does_ supply it, because whether a model correctly recognizes a
  win / draw / ongoing position is itself part of what we measure.
- **The engine checks the claim against ground truth.** After applying a legal
  move, the engine computes the true outcome; if `claimed_outcome` is present and
  disagrees, the game ends as an error — a legal move whose result the player
  misread (`WRONG_OUTCOME_CLAIM`, see "Game results and termination"). Playing on
  with a demonstrably confused player would be pointless.

### Turn flow driven by the engine

1. Call `start_game(side, seed)` on both players; each seeds a fresh board with
   the snake on `seed` — the game's opening cell (§2.4/§5).
2. Ask the player to move: `choose_move()`, which returns a `MoveChoice` (a move
   and an optional `claimed_outcome`). If instead it raises `MoveUnavailable`,
   the game ends with a `PLAYER_FAULT` result — skip to step 5.
3. Validate the move against the engine's authoritative board. If it is illegal,
   the game ends immediately with a `PLAYER_FAULT` result (§2.8, and "Game results
   and termination" below) — skip to step 5. Otherwise apply it and compute the
   true outcome (win / cat's game / in play). Then, if `claimed_outcome` was
   supplied and disagrees with the true outcome, the game ends with an
   `PLAYER_FAULT` result whose reason is `WRONG_OUTCOME_CLAIM` — skip to step 5.
4. Call `observe_move(side, move)` on **both** players. A player is notified of
   its own accepted move as well as the opponent's.
5. Repeat from step 2 until the game ends, then call `end_game(result)` on both.

**Why notify a player of its own move (step 4).** This is an ergonomics choice,
not a capability one — a player always knows its own move. Keeping it symmetric
means (a) a player mutates its internal board in exactly **one place**
(`observe_move`) for every move, avoiding the drift that comes from updating
own-moves and opponent-moves in different methods, and (b) the engine loop stays
a uniform "validate → apply → notify all," with no special case for the mover. A
player that does not care simply no-ops on its own move.

### Watching a game (observation)

Separately from the players, the engine accepts an optional **observer** — a
spectator driven in lockstep with the game so a caller can watch or log it turn
by turn without being a player. `play_game(mouse, snake, observer=None)` calls
these game- and move-level hooks, each defaulting to a no-op:

```python
class Observer:
    def on_game_start(self, names: dict[Side, str], board: Board) -> None: ...
    def on_move_start(self, side: Side, board: Board) -> None: ...
    def on_move_end(self, side: Side, move: Move, board: Board,
                    outcome: TurnOutcome) -> None: ...
    def on_game_end(self, result: GameResult) -> None: ...
```

Unlike a player, an observer never influences play; it only _receives_ the
engine's authoritative board (read-only).

A move is bracketed by **two** hooks. `on_move_start` fires at the top of every
turn, _before_ the player is asked for a move, so a watcher can show whose turn
it is the moment it begins. `on_move_end` fires once per **accepted** move —
including the terminal one — after it has been applied, carrying the move and the
resulting outcome. Splitting the two matters when producing a move is slow: an
LLM player may take seconds to respond, and a spectator wants to see "Snake is
thinking…" immediately, not only once the move lands. A turn that ends in a fault
fires `on_move_start` but **no** `on_move_end` (no move was accepted); the fault
detail arrives with `on_game_end`.

This keeps rendering out of the engine and out of the players: the board is a
**fact** the engine already owns, so a watcher reads it directly rather than
reconstructing it from observed moves.

The same `Observer` also has **match-level** hooks (`on_match_start`,
`on_match_end`) and carries an **observation level** — set when the observer is
constructed — that it uses to gate how much it reports; both are introduced with
matches (§5). The engine stays level-blind: it always fires every hook, and the
observer decides what to act on.

### Game results and termination

Every game ends in one of three ways, reported to both players via
`end_game(result)`:

```python
class Termination(Enum):
    LINE_COMPLETED   # a player completed a line — normal win
    CATS_GAME        # all 12 lines dead — draw, no winner
    PLAYER_FAULT     # a player failed to complete a valid turn — error, no winner
    ABORTED          # no-contest: an environmental failure (e.g. the model backend
                     # stayed unreachable after retries) voided the game. Not a
                     # fault, not scored to either side; the rest of the match plays on.

class PlayerFaultReason(Enum):
    # Structural (caught at Cell/Move construction). For trusted players this
    # never happens; a player building moves from untrusted output (the LLM
    # player) catches the construction error and reports the matching reason.
    OFF_BOARD          # a target cell is off the board
    DUPLICATE_CELLS    # the two target cells are the same
    WRONG_PIECE_COUNT  # wrong number of pieces played. Two provenances:
                       #   - construction-time: zero or three-plus cells (structural)
                       #   - apply-time: a single-piece move that did NOT end the
                       #     game (engine-detected, stateful; attempted_move present)
    # Engine-detected, stateful: only knowable against the current board.
    # `attempted_move` is present.
    CELL_NOT_EMPTY     # a target cell is already occupied
    # Player-reported: the player could not deliver a well-formed move at all, so
    # no `Move` ever reached the engine. `attempted_move` is None.
    UNPARSEABLE_OUTPUT # output could not be interpreted as a move — e.g. an LLM's
                       # structured output failed to validate / was garbage
    THINKING_LIMIT_EXCEEDED # the model exhausted its output-token budget before
                       # emitting a move (typically thinking up to the limit); no
                       # move was produced, distinct from garbage output
    # (The LLM sub-spec may refine or add player-reported reasons, e.g. empty
    # response or refusal. A transport timeout is NOT a fault — it is a no-contest
    # abort, see Termination.ABORTED above and §4.)
    # Engine-detected misread: the move is legal, but the player's self-assessed
    # outcome disagrees with the true outcome. `attempted_move` is present, and
    # `claimed_outcome`/`actual_outcome` below are set.
    WRONG_OUTCOME_CLAIM

@dataclass(frozen=True)
class PlayerFaultDetail:
    offender: Side                       # the side that faulted
    reason: PlayerFaultReason            # the structured cause
    attempted_move: Move | None = None   # the move played, if one could be formed
    claimed_outcome: TurnOutcome | None = None  # set for WRONG_OUTCOME_CLAIM
    actual_outcome: TurnOutcome | None = None   # set for WRONG_OUTCOME_CLAIM

@dataclass(frozen=True)
class GameResult:
    termination: Termination
    winner: Side | None            # set iff termination == LINE_COMPLETED
    fault: PlayerFaultDetail | None   # set iff termination == PLAYER_FAULT
    error: str | None = None          # a short cause description, set iff ABORTED
```

Notes:

- `winner` is `None` for a cat's game, a player fault, **and** an abort; the
  cases are distinguished by `termination`. A player fault is **not** a win for
  the opponent.
- **`ABORTED` is a no-contest, not a fault.** It is reserved for failures that
  are nobody's play — the canonical case being a player's model backend staying
  unreachable after retries (surfaced as `PlayerUnavailable`, §4). Such a game
  is charged to neither side and never appears in the fault tallies; only the
  affected game ends, and the match continues.
- **`PLAYER_FAULT` covers several kinds of the same underlying failure** — "the
  player did not complete a valid turn":
  - _Structural, construction-time_ (`OFF_BOARD`, `DUPLICATE_CELLS`, and
    `WRONG_PIECE_COUNT` for zero or three-plus cells): building the `Cell`/`Move`
    raises `IllegalMove`. Trusted players never trip this; a player parsing
    untrusted output (the LLM player) catches it and reports the reason via
    `MoveUnavailable`.
  - _Engine-detected, stateful_ (`CELL_NOT_EMPTY`, and `WRONG_PIECE_COUNT` for a
    single-piece move that leaves the game in play): the player returns a
    well-typed `Move`, but the current board makes it illegal — a target cell is
    occupied, or the lone piece did not end the game. The engine's apply-time
    check raises `IllegalMove`, filling in `reason` and `attempted_move`.
  - _Player-reported_ (`UNPARSEABLE_OUTPUT`, `THINKING_LIMIT_EXCEEDED`, …): the
    player cannot even form a `Move`, so it raises `MoveUnavailable(reason)` from
    `choose_move`. Here the **player** supplies the `reason` — the engine cannot
    know it. The LLM player distinguishes a response truncated at the output-token
    limit (`THINKING_LIMIT_EXCEEDED`) from genuinely malformed output
    (`UNPARSEABLE_OUTPUT`) so the next game's feedback can tell it to think more
    briefly.
  - _Engine-detected misread_ (`WRONG_OUTCOME_CLAIM`): the move is legal, but the
    player's `claimed_outcome` disagrees with the true outcome. The engine
    records the legal `attempted_move` plus `claimed_outcome` and
    `actual_outcome`.
- **Whoever reports, reports only the facts.** Neither the engine nor the player
  puts human-readable prose or advice in `PlayerFaultDetail`. Turning these facts
  into a message — including any "how to avoid repeating this mistake" guidance —
  is done later by the consumer (e.g. the LLM player, in `end_game`, crafts an
  explanation for its model from the structured fields).

### Feedback across games (illustrative: the LLM player)

This design supports informing an agent about a mistake and carrying that into
the next game, **without any change to the interface**, because a single player
instance persists across games (roles are assigned per game, §"Roles are
assigned per game"):

1. In `end_game(result)`, the player checks whether
   `result.termination == PLAYER_FAULT` and
   `result.fault.offender == <its own side this game>`. If so, it
   **composes and stores** feedback for itself from the structured
   `PlayerFaultDetail` (what it did wrong and how to avoid it next time).
2. At the next `start_game(side, seed)`, the player **prepends** that stored
   feedback to the first prompt/message it builds for the new game.

The feedback mechanism itself needs no new parameter: the player owns its own
prompting/messaging and therefore owns the prepend. (`start_game` does carry a
`seed` — §2.4 — but that only tells the player where the snake starts this game;
it is unrelated to feedback.)

### Shared board helper

The engine module will provide a reusable **`Board`** class (board bookkeeping:
place a piece, query cells/lines, etc.). Players **may compose** a `Board` to
manage their internal state so each player type need not reimplement it. This is
composition, not required inheritance — the `Player` ABC does not mandate it.

### Player types

The engine and interface must not need changes to add a new player type; each is
just another implementation of the player interface. The types, in
implementation order (the first four are implemented — see the milestones in §11):

1. **Scripted player** _(implemented — 0.1)_. Initialized with a predetermined
   ordered sequence of moves; returns them one at a time. Used to drive
   deterministic games for testing the rules and engine (win detection, cat's
   game, illegal-move handling, etc.).
2. **Random player** _(implemented — 0.2)_. Plays uniformly at random among legal moves: each turn it
   places two pieces on two randomly chosen empty cells (or a single piece on the
   last empty cell, which necessarily ends the game). It tracks its own board
   through `observe_move` and makes no outcome claim. A trivial baseline, a
   sparring partner for the stronger players, and a convenient driver for
   non-deterministic tests. Its randomness is drawn from an injectable
   `random.Random`, so a seeded instance produces fully reproducible games.
3. **Human player** _(implemented — 0.3)_. Reads a move interactively (input format `C3 D4` — two cells separated by a space).
   So a human is never knocked out by a slip, it **re-prompts** on any locally
   detectable mistake — an unparseable label, the wrong number of cells, an
   off-board or repeated cell, a target already occupied, or a lone piece that
   would not end the game — only returning a move once it looks legal. It tracks
   its own board (to check occupancy and test a single-piece ending) and makes no
   outcome claim. End-of-input concedes the turn (a `PLAYER_FAULT`). Its input and
   output are injectable so it can be driven deterministically in tests. The
   engine stays the source of truth; the local checks only spare avoidable faults.
4. **LLM player** _(implemented — 0.5)_. Chooses moves by querying a large language
   model via Pydantic AI, with support for a **range of LLM providers and models**
   (Anthropic, OpenAI, Google Gemini, OpenRouter, and OpenAI-compatible custom
   endpoints). Specified in full in §4.
5. **Algorithmic player** _(implemented — 1.3)_. Plays **perfectly** via full-depth alpha–beta
   (minimax with pruning) search to terminal positions, backed by a transposition
   table keyed on a symmetry-canonical board. Specified in full in §10.
6. **Reinforcement-learning player** _(planned)_. A policy trained via RL (self-play). Likely
   needs supporting tooling (training loop, model persistence) beyond the game
   engine itself.

Others may be added as the project evolves.

### Tournaments

The project supports a **tournament structure** that pits player types against
each other over many games, composed from **matches** (§5) as its building block.
A tournament is simply _any set of matches_, accumulated in a shared results file,
with match-**running** kept deliberately separate from result-**tallying**. This
is specified in full in §6, and together with the LLM player (§4) it defines the
**1.0** release (§11).

## 4. The LLM player

The **LLM player** chooses its moves by querying a large language model. It is the
player type this project exists to compare: pitting models against one another (and
against strong non-LLM players) over many games is the benchmark. Because whether a
model can _reason_ about the game is exactly what we measure, it is given as little
help as possible — it sees only the opponent's moves and must maintain everything
else itself.

What the rest of this document relies on:

- It is an ordinary `Player` (§3), selected by a **roster name** from
  `players.yaml` wherever a side is named (§7). One model per player instance,
  reached through Pydantic AI across several providers.
- It keeps **one message thread spanning the whole match**, which is what carries
  feedback from one game into the next (§3, "Feedback across games").
- Its self-assessment is **required**, not optional, so a wrong claim is a
  `WRONG_OUTCOME_CLAIM` fault (§3). An illegal or unusable move **ends the game** —
  there are no in-game retries. Unreachable backends raise `PlayerUnavailable` and
  end the game as a no-contest (`ABORTED`, §3) instead of charging the player.
- Two CLI flags are its own (§7): `--log-llm` dumps the message thread, and
  `--prune-thinking` strips prior reasoning from what is re-sent.

**Full specification: [`SPEC-llm-player.md`](SPEC-llm-player.md).** Read it if you are
working on the LLM player, the roster/provider configuration, structured output,
thinking levels, message logging, or the endpoint probes under `tools/`.

A reference elsewhere in the project of the form `§4, "Structured output"` names a
**subsection of that document**; `§4` alone means this summary.

## 5. Matches

A **match** is two fixed players playing a sequence of games. It is the unit that
makes inter-player — especially inter-LLM — comparison meaningful, and because the
two `Player` instances **persist across all the games**, it is also what lets the
LLM player carry feedback from one game into the next (§3, "Feedback across games";
§4). Matches are the building block of the tournament structure (§6), which they
compose; matches themselves are part of the road to 1.0.

### Definition

- Two players and a game count `N ≥ 1`.
- **Sides are fixed for the whole match:** one player is Mouse in every game, the
  other Snake in every game. To compare two players with each taking both sides, run
  **two** matches with the seats swapped.
- The **same two instances** are reused across all `N` games — `start_game(side)` /
  `end_game(result)` are called once per game with each player's fixed side, so a
  player's cross-game state (e.g. the LLM's running thread) accumulates over the
  match.

```python
def play_match(mouse: Player, snake: Player, num_games: int,
               observer: Observer | None = None,
               opening: Cell | random.Random | None = None) -> MatchResult: ...
```

### What a match reports

```python
@dataclass(frozen=True)
class MatchResult:
    names: dict[Side, str]     # who played each side (fixed for the match)
    num_games: int
    mouse_wins: int
    snake_wins: int
    cats_games: int
    mouse_faults: int          # games the Mouse-side player faulted
    snake_faults: int          # games the Snake-side player faulted
    faults: list[GameResult]   # the faulted games' full results (with PlayerFaultDetail)
    aborted: int               # no-contest games (ABORTED) — charged to neither side
```

The tallies summarize the match; full per-game `GameResult`s are kept **only for
games that faulted** — the interesting failures, where a player made an illegal move
or misread an outcome — each carrying its `PlayerFaultDetail`. Aborted games are
counted but not otherwise recorded: they belong to neither player. By construction
`mouse_wins + snake_wins + cats_games + mouse_faults + snake_faults + aborted ==
num_games`, and `mouse_faults + snake_faults == len(faults)`. Richer scoring and
standings belong to tournaments (§6), which aggregate these per-match tallies.

### Randomized openings

By default the snake is seeded on the same cell (the board default, §2.4) every game. That is
fine for scripted or random players, but it has a sharp edge for the benchmark:
two deterministic models sampling at temperature 0 **replay the exact same game**
every time, so a long match yields no more information than a single game.

To break that, a match can **vary the opening per game**. `play_match`'s
`opening` parameter selects the policy:

- a fixed `Cell` — seed there every game (a pinned opening);
- a `random.Random` — draw a fresh cell **uniformly at random per game**, so
  successive games open differently even between two deterministic players;
- `None` — use the board's default (§2.4).

The draw is **per game, not per match**: one RNG spans the whole match and picks a
new cell each game, guaranteeing variety within a single pairing. `play_match`
resolves the policy to a concrete cell per game and passes it to
`play_game(mouse, snake, observer=None, *, seed: Cell | None = None)`; the engine
then hands each player the seeded cell via `start_game` (§3) and exposes it to the
observer on the board it already receives — the console announces it at the start
of each game.

Two deliberate boundaries keep this tidy:

- **The default value lives in one place** (the board). Every other layer passes
  `None` to mean "the default" and reads back the resolved cell, so changing the
  default never ripples outward.
- **The library defaults to deterministic; only the CLI defaults to random.**
  `play_game` / `play_match` default `seed` / `opening` to `None`, so
  programmatic callers and the test suite get repeatable games; the CLI's `--seed`
  defaults to `random` (§7), since that is where varied openings are wanted.

Per-game seeds are **not** recorded in the results file (§6): a `MatchResult`
tallies outcomes across whatever openings were played. Recording openings — to
analyze how the seed affects results — is game-balance work deferred past 1.0
(§11).

### Observation levels

Observation generalizes from a single game to the whole match. The `Observer` (§3)
gains two match-level hooks alongside its game/move hooks:

```python
class Observer:                 # ... on_game_start / on_move_start / on_move_end /
                                #     on_game_end from §3 ...
    def on_match_start(self, names: dict[Side, str], num_games: int) -> None: ...
    def on_match_end(self, result: MatchResult) -> None: ...
```

An `ObservationLevel`, **set when the observer is constructed**, selects how much
that observer reports, coarse to fine:

```python
class ObservationLevel(IntEnum):  # ordered MATCH < GAME < MOVE
    MATCH   # only the start and end of the match
    GAME    # the above, plus the start and end of each game
    MOVE    # the above, plus every individual move
```

The **engine is level-blind**: `play_game` / `play_match` always fire every hook,
in order, and the observer gates its own output against `self.level` — so "how much
to watch" is a property of the watcher, not of the run. At `MATCH` an observer acts
only on `on_match_start` / `on_match_end`; `GAME` adds the game-boundary hooks;
`MOVE` adds the move hooks. Games are always **played** in full regardless (so the
tallies are always computed). There is **no pausing** between moves or games — the
only thing that ever waits for input is a human player taking its own turn. Because
a human must see the board to play, **if either player is human the CLI builds the
observer at `MOVE`** (a coarser request is overridden, with a message).

## 6. Tournaments

A **tournament** is simply a **set of matches** — _any_ set, with no structural
requirement on which players meet or how often. This makes the tournament the
loosest possible composition of the match building block (§5), and it lets the
project keep **running matches** entirely separate from **tallying results**,
joined only by a shared results file. Anything that appends a `MatchResult`
contributes to a tournament; anything that reads them back can score it.

### The results file

Match results accumulate in a single **append-only JSON Lines** file,
`tournament-results.jsonl` by default. **Each line is one serialized
`MatchResult`** (§5) — nothing more: no wrapper, no timestamp, no extra identity
metadata. A player is identified solely by the `name` recorded in
`MatchResult.names`.

- **Append-only, shared, no dedup.** Any process that runs matches appends its
  line(s); the file simply grows. Running the same pairing twice yields two
  counted lines — there is no de-duplication and no resume/skip logic. A
  tournament is whatever set of lines the file happens to hold.
- **Crash-safe by construction.** Because each completed match is a self-contained
  line, an interrupted run leaves a valid file of every match finished so far.
- **A player name is the only identity.** Reusing one name for different
  providers/models merges them in the tally — a deliberate choice the operator
  owns, and one that can be used on purpose.
- **Documented encoding.** The line is a stable JSON encoding of `MatchResult`,
  including its `Side`-keyed `names` and its nested `faults: list[GameResult]`
  (each with its `PlayerFaultDetail`, `Move`/`Cell`, and enum fields). This
  encoding — which round-trips a `MatchResult` — is the contract between the
  runners that write it and the tally that reads it.

### Running matches

Two commands write matches into the file; a third (below) reads them. Both runners
share player construction, roster loading, and `--watch` handling with the
single-match CLI (§7).

**`play-match`** — the existing single-match command (§7), primarily a debugging
and casual-play tool (LLM logs, playing against `random` or `human`). By default
it does **not** touch the results file. The opt-in flag `--tournament-results
[FILE]` records its `MatchResult` as one line — a convenient way to hand-craft or
top up a tournament one match (or, with `--games`, a few) at a time.

**`play-tournament-matches`** — the batch runner. It generates and plays many
matches from **two player subsets** and appends each result. Appending is
intrinsic to this command (its whole purpose); its `--tournament-results [FILE]`
only overrides the default path.

_The one generating operation._ A match is an ordered `(mouse, snake)` pair of
distinct players. Given subsets **A** and **B**, the command emits a match for
every ordered distinct pair `(x, y)` whose unordered matchup **straddles** the two
subsets:

```
emit (mouse=x, snake=y)  for all x ≠ y  where  (x∈A and y∈B) or (x∈B and y∈A)
```

This single rule expresses every intended schedule; the seat-swap (each matchup
played both ways) and the exclusion of self-play (`x ≠ y`) fall directly out of
it:

- **All pairs (round-robin).** A = `all`, B = `same` (the defaults): every player
  meets every other, both seats — `N·(N−1)` matches for `N` players.
- **All pairs within a subset.** A = an explicit subset, B = `same`: the same,
  restricted to that subset.
- **Cross of two subsets.** A and B distinct: every matchup with one player from
  each, both seats — `2·N·M` matches for disjoint subsets of size `N` and `M`,
  fewer when they overlap (`x ≠ y` drops the self-play). The motivating case is
  **introducing new players**: A = the newcomers, B = `all` plays each newcomer
  against the whole roster, both seats, _without_ making the existing players
  replay one another (a pair of two existing players fails the straddle predicate).

_Subset selectors._ Each of A (`--players`, default `all`) and B (`--against`,
default `same`) is given by exactly **one** selector:

- an explicit list of player names,
- `all` — every player in `players.yaml`,
- `same` — the same set as the other subset (B only; it is B's default),
- `above <name>` — every player listed **above** `<name>` in `players.yaml`
  (exclusive of `<name>` itself),
- `below <name>` — every player listed **below** `<name>` (exclusive).

`above`/`below` make the **order of `players.yaml` a meaningful ranking** (order
the roster by rough strength and `above gpt5` names the stronger cohort); this is
a documented contract, and roster loading preserves the file's order. The pure,
unit-testable schedule computation — subsets + roster order → the ordered list of
`(mouse, snake)` name pairs — is kept separate from argument parsing.

_Other options._ `--games N` sets a uniform game count for every match; `--seed`
sets the opening for every game — `random` (the default) or a fixed cell like
`B3` (§5); `--watch` selects the observation level (§5), defaulting to `game`
(coarser than `play-match`, since a batch can be large). **Human players are rejected** — a batch
runs unattended, with no one to see the board or type a move — so the "human
forces `move`" rule (§5) never applies here. `--prune-thinking` (§4) is available and
matters most here, a batch being where a thread grows longest. There is no
`--log-llm`: a per-conversation dump across a large batch is not useful.

### Tallying results

**`tally-tournament`** reads a results file (default `tournament-results.jsonl`)
and prints **per-player standings**. For each player — keyed by the `name` in
`MatchResult.names`, so a name reused across models is merged — it walks every
match the player appears in, notes which side the player took, and accumulates:

| Column                   | If the player was Mouse | If the player was Snake |
| ------------------------ | ----------------------- | ----------------------- |
| `played` (excl. aborted) | `num_games − aborted`   | `num_games − aborted`   |
| `won`                    | `mouse_wins`            | `snake_wins`            |
| `lost`                   | `snake_wins`            | `mouse_wins`            |
| `tied`                   | `cats_games`            | `cats_games`            |
| `faulted`                | `mouse_faults`          | `snake_faults`          |
| `opponent_faulted`       | `snake_faults`          | `mouse_faults`          |

These six categories **reconcile**: `played = won + lost + tied + faulted +
opponent_faulted`. **Aborted games are excluded** everywhere — they are charged to
neither side (§3).

Derived percentages:

- **`win%` = `won / (won + lost + tied)`** and **`loss%` = `lost / (won + lost +
tied)`** — the denominator being the games played with **neither** side faulting
  (a fault is not a win or a loss for either side, §3). When that denominator is
  `0` the percentage is undefined and shown as `—`.
- **`fault%` = `faulted / played`** — here the denominator is games played
  (excl. aborted), since a fault _is_ a played game.

`--sort win%|loss%|fault%` (default `win%`) orders the standings **best-on-top**:
`win%` descending, `loss%` and `fault%` ascending (fewest losses / fewest faults
first). Ties keep **roster order** (`players.yaml`).

The standings are deliberately **side-agnostic**: no Mouse-vs-Snake breakdown,
because whether the opening favors a side — the default or any other seed
(§2.4) — is game-balance analysis deferred until after 1.0 (§11). A head-to-head player-vs-player matrix is likewise a natural
later addition the data already supports, but beyond the per-player standings
specified here.

## 7. Interface

The project ships a **text CLI** with three commands. All share the same
presentation (the `console` layer, §8): the board rendered in ASCII (as in §2.2)
with distinct emoji — 🐭 (mouse) and 🐍 (snake) — reporting of whose turn it is,
the move played, and the outcome (`Mouse wins`, `Snake wins`, or `Cat's game`),
and an `Observer` (§3) watching at a selectable **observation level**
(`--watch match|game|move`, §5). There is no artificial pausing; only a human
player's own input paces a game.

**`play-match`** — play or watch a **single match**. `--mouse` and `--snake` each
name who plays that side: `random`, `human`, `perfect` (the perfect algorithmic
player, §10), or an **LLM roster name** from `players.yaml` (§4). `--games N` (default 1) sets the match length with sides fixed
for the match (§5); `--seed` sets where the snake is seeded each game — `random`
(the default: a fresh cell per game) or a fixed cell like `B3` (§2.4, §5);
`--watch` defaults to `move`. If either player is human the
level is forced to `move` (with a message), since a human must see the board.
`--log-llm [DIR]` (off by default) dumps each LLM player's full raw message thread
as JSON for debugging (§4, "Message logging"); a no-op for non-LLM players.
`--prune-thinking` (off by default) strips earlier turns' reasoning text from each
LLM request (§4, "Pruning re-sent reasoning"); also a no-op for non-LLM players.
`--tournament-results [FILE]` (off by default) records the resulting `MatchResult`
as a line in the results file (§6) — absent ⇒ nothing is written, bare ⇒ append to
`tournament-results.jsonl`, with a path ⇒ append there. Because this command is
mainly for debugging and casual play, it stays out of the results file unless asked.
E.g. `play-match --mouse human --snake random` plays a single game as Mouse, and
`play-match --mouse opus --snake gpt5 --games 20 --watch game` runs a 20-game match
between two LLMs, reporting per game.

**`play-tournament-matches`** — run **many** matches from two player subsets and
append each result to the file (§6). Options: `--players` / `--against` (the subset
selectors), `--games N`, `--seed` (opening for every game: `random` default or a
fixed cell, §5), `--watch` (default `game`), `--prune-thinking` (§4, off by default;
most worthwhile here, since a batch is where context growth binds), and
`--tournament-results [FILE]` (path override only — this command always appends). Human players are
rejected. E.g. `play-tournament-matches` runs a full round-robin, and
`play-tournament-matches --players new-model --against all` enters a new player
against the whole roster.

**`tally-tournament`** — read a results file and print per-player standings (§6).
`--tournament-results [FILE]` selects the file (default `tournament-results.jsonl`)
and `--sort win%|loss%|fault%` (default `win%`) orders the table.

## 8. Architecture

Rough module layout:

- `board` / `core` — board representation, move validation, applying moves, and
  win/draw detection.
- `game` — the loop that runs a single game, alternating the two players and
  firing the `Observer` hooks in lockstep.
- `observer` — the `Observer` spectator interface and the `ObservationLevel` that
  an observer uses to gate its own output; depends on neither `game` nor `match`,
  so both can drive it without a cycle.
- `match` — runs a match: a sequence of games between two fixed players (§5),
  reusing the same instances so LLM feedback carries across games, and producing a
  `MatchResult`.
- **Tournament logic** (§6) — three independent, CLI-free modules, joined only by
  the shared results file: `schedule` (the pure **schedule builder**: subsets +
  roster order → the ordered `(mouse, snake)` name pairs), `serialize`
  (reading/writing the JSON-Lines **results file** — the documented, round-tripping
  `MatchResult` encoding, kept out of `result` so those types stay pure data), and
  `tally` (aggregating results into per-player standings and ordering them).
  Together with `match` and `game` these form the "engine" that drives play.
- `players` — the player interface and its implementations (scripted, random,
  human, and — per §4 — the LLM player). Loading the LLM roster from
  `players.yaml` / `providers.yaml` / `.env` lives in the small `roster` module
  alongside it, which parses those files into typed specs and does no more:
  model/agent resolution — and with it every Pydantic AI import — stays in the
  LLM player module. YAML parsing is thus kept out of the player, and provider
  knowledge out of the roster loader.
- `console` — the shared presentation layer: board rendering, result summaries,
  and the stdout `Observer`.
- **CLI frontends** — one thin module per command (§7), sharing player
  construction, roster loading, and `--watch` handling via a common helper
  (`cli_common`): `match_cli` (`play-match`), `matches_cli`
  (`play-tournament-matches`), and `tally_cli` (`tally-tournament`).
- `tools/` — standalone programs outside the package, each specified where the thing
  it measures is: the endpoint probes (§4, "Probing a deployment"), the machine
  comparison (§4, "Comparing two machines"), the message-log reader (§4, "Message
  logging"), the tie-break benchmark (§10, "Choosing among
  optimal moves"), and the offline solver, which has its own document
  (`tools/solver/SPEC.md`). They import the package but nothing in
  the package imports them, so a tool may take dependencies the engine does not have.

## 9. Tech stack & tooling

- **Language:** Python.
- **Environment & dependencies:** `uv`.
- **Version control:** `git`.
- **Tests:** `pytest`, running deterministic games driven by scripted players.
- **LLM access:** `pydantic-ai` — a uniform interface across providers (Anthropic,
  OpenAI, Google Gemini, OpenRouter, and OpenAI-compatible custom endpoints), with
  structured output and a unified thinking/effort control. Roster and provider
  config load from YAML (`pyyaml`); API keys come from the environment
  (`python-dotenv`, a `.env` file). See §4.
- **Strong typing.** Use static typing wherever possible: complete type hints on
  all public functions, methods, and data structures; `enum`s and (frozen)
  `dataclass`es for domain types (as already used for `Side`, `Move`,
  `TurnOutcome`, `PlayerFaultReason`, `GameResult`, etc.); and a static type
  checker (e.g. `mypy` or `pyright`) run over the codebase, aiming for a clean,
  strict configuration.

## 10. The algorithmic player (perfect play)

The **algorithmic player** plays the game **perfectly**: it always makes a
game-theoretically optimal move. It is the _calibrated, fixed-strength yardstick_ of
§1 — the strong non-LLM opponent a model can be measured against.

What the rest of this document relies on:

- It is a **mechanical** player in the sense of §3: it reads the board correctly by
  construction, returns `claimed_outcome = None`, and never faults or misreads an
  outcome. Its only injected dependency is a `random.Random`, so a seeded instance is
  reproducible.
- It is a **built-in player named `perfect`**, selected exactly like `random`
  wherever a side is named (§7). Like `random`, it is **not** part of the
  `players.yaml` tournament roster (§4, §6).
- It is **two-tier**: an offline retrograde solve supplies an opening table, and the
  endgame is searched live. Both tiers return the same values, so the seam is
  invisible. A missing table is not an error — the player falls back to searching the
  opening, correct but far too slow to be practical, and says so on stderr.
- Among equally optimal moves it **ranks for trappiness** rather than picking blindly,
  which is what makes it dangerous to a fallible opponent without ever risking the
  draw. This is not configurable.
- **The game is a draw.** All four seed classes are solved; every seed is drawn under
  perfect play, which settles the game-balance question §11 lists as out of scope for
  _perfect_ players (it says nothing about fallible ones).

**Full specification: [`SPEC-perfect-player.md`](SPEC-perfect-player.md).** Read it if
you are working on the algorithmic player, the search, the symmetry group, the opening
table, or the tie-break. The offline solver that produces the table has its own
document, [`tools/solver/SPEC.md`](tools/solver/SPEC.md).

A reference elsewhere in the project of the form `§10, "The opening table"` names a
**subsection of that document**; `§10` alone means this summary.

## 11. Versioning & scope

The project follows [Semantic Versioning](https://semver.org/); the version in
`pyproject.toml` tracks capability milestones. The purpose of the project — an
LLM-reasoning benchmark — is what defines "useful," so the version reflects
progress toward that, not incidental churn.

- **0.x — pre-release.** The engine is playable and watchable, but the project
  does not yet do the job it exists for. Each minor bump marks a shipped
  capability:
  - **0.1** — the game engine and rules, the scripted player, and the text CLI.
  - **0.2** — the random player and turn-by-turn game observation.
  - **0.3** — the interactive human player.
  - **0.4** — matches (§5): two fixed players over a sequence of games with
    tallied results, the generalized `Observer` with `ObservationLevel`, and the
    `--games` / `--watch` CLI.
  - **0.5** — the **LLM player** (§4): moves chosen by a model via Pydantic AI
    across a range of providers, a single cross-game message thread with
    feedback, structured output with fault mapping, YAML roster / provider config
    (keys in `.env`), the `--mouse`/`--snake` roster names, and `--log-llm`
    message logging.
- **1.0 — first genuinely useful release.** Reached when the **LLM player** (§4,
  landed in 0.5) can be driven by the **tournament structure** (§6) — the
  `play-tournament-matches` and `tally-tournament` commands over a shared results
  file: only then can the project do what it exists to do — pit LLMs against each
  other, and against strong non-LLM players, over many games and score them. Other
  0.x milestones (e.g. a heuristic or MCTS player, §3) may ship first, but **1.0 is
  defined by the LLM-player + tournament pair**, regardless of what else arrives
  before it.
- **After 1.0**, standard SemVer applies: incompatible changes to the player API
  or CLI bump the major, backward-compatible capabilities bump the minor, and
  fixes bump the patch. Each minor so far:
  - **1.1** — richer console observation, and `--watch none`.
  - **1.2** — the snake's opening cell randomized per game (§2.4, §5).
  - **1.3** — the **algorithmic player** (§10) made practical rather than merely
    correct: the game solved offline for all four seed classes, and the opening
    table the player reads instead of searching.
  - **1.4** — the algorithmic player made _dangerous_ as well as perfect (§10,
    "Choosing among optimal moves"): among equally optimal moves it now prefers
    those giving a fallible opponent the most ways to go wrong, cutting the games it
    fails to win against the random player from 37.7% to 2.0% — while still never
    putting the draw at risk, since every candidate ranked is equally optimal.
  - **1.5** — structured output made portable across self-hosted endpoints (§4,
    "Structured output"): a custom provider declares `output_mode: native` when
    its server has no tool-call parser for the model it serves, and
    `tools/probe_tool_termination.py` measures which mode an endpoint can
    actually finish a move in. This unpins a model from the one old container
    that happened to work.
  - **1.6** — `output_mode: prompted`, the third and least constrained way to ask
    for the move (§4, "Structured output"): the schema goes in the prompt and the
    reply is parsed as text, so no grammar touches generation. Built to test whether
    constrained decoding was costing play quality against the perfect player. It is
    not: prompted and native draw alike. The mode stays, as the answer to a question
    that will otherwise be asked again. _(current)_

Each of these players — LLM, algorithmic, RL — arrives without requiring engine
changes, as the `Player` abstraction (§3) is designed to allow. The algorithmic
player is specified in full in §10.

Out of scope for now:

- Game-balance analysis between _fallible_ players (whether Snake or Mouse is
  favored in practice, and how the seeded opening cell affects that, §2.4). Under
  **perfect** play the question is settled — every seed is a draw (§10).
- Any GUI/TUI.
- Network/remote play.
