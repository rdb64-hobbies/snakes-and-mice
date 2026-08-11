# Snakes and Mice — Specification

> Status: **living draft**. This document defines the game and the roadmap
> through **1.0** (see §16 for the versioning scheme).

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
frequency, broken down by `PlayerFaultReason` (§10), is itself a meaningful
metric. This is why supporting a range of LLM providers and models (§11) and a
tournament structure for head-to-head comparison (§10) are central rather than
incidental features.

The design goal is a clean separation between:

- the **engine** (rules, board state, win/draw detection), and
- **players** (agents that choose moves).

The engine never knows or cares *what kind* of player is on either side.

## 2. The two players

There are exactly two players, identified by the piece they play:

- **Mouse** — plays mouse pieces.
- **Snake** — plays snake pieces.

**Mouse moves first.**

## 3. Board and coordinates

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

## 4. Lines

A **line** is a set of 5 cells that a player must fully occupy to win. 
There are **12 lines** in total:

- **5 rows:** each of `A`–`E`.
- **5 columns:** each of `1`–`5`.
- **2 diagonals:**
  - Main diagonal: `A1, B2, C3, D4, E5`
  - Anti-diagonal: `A5, B4, C3, D2, E1`

## 5. Setup

The board begins **empty except for a single snake at `B3`**. This is the
starting position, not a move by the Snake player. It counts as a real snake
piece for all win detection.

Note: `B3` lies on **row B** and **column 3**, and on **neither diagonal**.

## 6. A turn / a move

- Players alternate turns, **Mouse first**: Mouse, Snake, Mouse, Snake, ...
- On a turn, the player makes one **move**, which normally consists of **placing
  two of their own pieces** on **two distinct empty cells**.
- **Single-piece exception.** A move may instead place a **single** piece, but
  *only* when that one piece **ends the game** — i.e. it completes a line (a win)
  or fills the board into a cat's game. This mirrors §7: if the first piece
  already ends the game, the second is never placed, so a one-piece move is the
  honest representation of that turn. Placing a single piece while the game is
  still in play is a fault (`WRONG_PIECE_COUNT`, §9).
- Placed pieces are permanent; they are never moved or removed.

The board starts with 24 empty cells and 2 pieces are placed per move, so a full
game is at most 12 moves and the board fills completely with no leftover single
cell. A player therefore always has at least two empty cells available at the
start of their turn (until the game ends).

## 7. Winning

A player **wins the moment they occupy all 5 cells of any line** (row, column,
or diagonal) with their own pieces.

- Win detection happens **after each individual piece placement**. If a player's
  *first* of two pieces completes a line, they win immediately and the second
  piece is not placed. A player that foresees this may submit a **single-piece
  move** (§6) for that winning piece alone.
- The pre-placed snake at `B3` counts toward the Snake player's lines.

## 8. Cat's game (draw)

A **line is dead** once it contains **at least one mouse and at least one
snake** — it can never be completed by either player.

The game is a **cat's game** when **all 12 lines are dead**, so no player can
possibly win. This can occur before the board is completely full.

If the board fills completely without a win, that state is necessarily a cat's
game (every line is dead) and is reported as such.

## 9. Illegal moves

A move can be illegal for these reasons:

- a target cell is off the board,
- a target cell is not empty,
- the two target cells are the same cell,
- the wrong number of cells is specified (a move must place one or two pieces), or
- a **single**-piece move that does **not** end the game (a single piece is legal
  only when it wins or completes a cat's game — see §6).

**Strong typing makes the structural cases unrepresentable.** `Cell` and `Move`
are validated value types (§10, "Core types"): a `Cell` cannot be constructed
off-board, and a `Move` cannot be constructed unless it is **one or two** cells
(and, if two, *distinct*). Attempting either raises `IllegalMove` at
**construction time**. As a result:

- **Structural illegality** (off-board, duplicate cells, or a count that is
  neither one nor two) is caught when a `Cell`/`Move` is *built*, before it ever
  reaches the engine. A player that builds moves from trusted code (scripted,
  algorithmic) will simply never hit this. A player that builds moves from
  **untrusted external output — the LLM player — must catch these construction
  errors and report them** as faults through the move-production mechanism
  (`MoveUnavailable`, §10), because only it can associate the failure with a
  `PlayerFaultReason`.
- **Stateful illegality** cannot be a type invariant — it depends on the current
  board — so the **engine** checks it when applying a move and raises
  `IllegalMove`. There are two such cases: `CELL_NOT_EMPTY` (a target cell is
  occupied), and `WRONG_PIECE_COUNT` when a **single-piece** move is well-formed
  but leaves the game *in play* (a single piece is legal only when it ends the
  game). Structurally, `WRONG_PIECE_COUNT` therefore has **two provenances**:
  construction-time (zero or three-plus cells) and apply-time (one cell, game not
  over).

An illegal move is an **error** — never silently ignored or re-prompted. It is
surfaced by exceptions and handled at two layers:

- **Construction / move application (low level):** raises an `IllegalMove`
  exception (structural at construction; `CELL_NOT_EMPTY` at engine apply-time).
  This makes bugs fail loudly in direct unit tests.
- **Game loop:** catches `IllegalMove` (from applying a move) and the
  player-reported `MoveUnavailable` (from `choose_move`) and ends the game with a
  terminal `GameResult` (see §10) whose `termination` is `PLAYER_FAULT`. This is
  an **error termination with no winner** (`winner = None`) — distinct from a
  cat's game, and *not* a win for the opponent. The result carries the facts of
  the fault so agents such as the LLM player can be informed via `end_game`.

## 10. Player abstraction

A player is a **stateful agent** that tracks its own view of the game and chooses
moves for whichever side it has been assigned. All player types share one
interface, defined as a Python **abstract base class (ABC)** — chosen over a
`Protocol` because we own every implementation, want instantiation-time
enforcement of the contract (an unimplemented method fails loudly, matching the
error-first stance of §9), and want to share behavior and construction across
players.

### Core types

The domain is modeled with validated value types, so illegal states are
unrepresentable (see §9):

- `Side` — an enum, `MOUSE` or `SNAKE`; `Side.other` gives the opponent. A cell's
  occupant is a `Side` (the side whose piece sits there) or `None` when empty.
- `Cell` — a board coordinate; a frozen dataclass validated to be **on the
  board** at construction. Parses/renders labels like `C3`.
- `Move` — a frozen dataclass validated at construction to be **one or two
  `Cell`s** (and, if two, distinct), in the order the player plays them. A
  single-cell move is structurally valid but *legal* only when that one piece
  ends the game; the engine enforces that at apply-time (§9).

Constructing an invalid `Cell` or `Move` raises `IllegalMove` (§9).

- **Each player manages its own board state.** A player keeps its own internal
  representation of the board, updated as moves happen. Because of this,
  `choose_move` takes **no state argument** — the player already has it.
- **The engine keeps the authoritative board.** The engine maintains its own
  board independently and uses it to validate legality (§9) and detect
  win/cat's-game. The engine's board is the source of truth; a player's private
  board is never trusted for rules enforcement.

### Roles are assigned per game

A player's side (Mouse or Snake) is **not** fixed at construction; it is assigned
at the **start of each game**. A single player instance can therefore play many
games and switch sides between them — required by self-play (RL) and by the
tournament structure.

### Interface

```python
class Player(ABC):
    def __init__(self, name: str | None = None) -> None:
        self.name = name or type(self).__name__

    @abstractmethod
    def start_game(self, side: Side) -> None:
        """Begin a new game as `side`. (Re)initialize internal board state,
        including the starting snake at B3. Resets any state from a prior game."""

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
  player *does* supply it, because whether a model correctly recognizes a
  win / draw / ongoing position is itself part of what we measure.
- **The engine checks the claim against ground truth.** After applying a legal
  move, the engine computes the true outcome; if `claimed_outcome` is present and
  disagrees, the game ends as an error — a legal move whose result the player
  misread (`WRONG_OUTCOME_CLAIM`, see "Game results and termination"). Playing on
  with a demonstrably confused player would be pointless.

### Turn flow driven by the engine

1. Call `start_game(side)` on both players; each seeds a fresh board including the
   snake at `B3`.
2. Ask the player to move: `choose_move()`, which returns a `MoveChoice` (a move
   and an optional `claimed_outcome`). If instead it raises `MoveUnavailable`,
   the game ends with a `PLAYER_FAULT` result — skip to step 5.
3. Validate the move against the engine's authoritative board. If it is illegal,
   the game ends immediately with a `PLAYER_FAULT` result (§9, and "Game results
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

Unlike a player, an observer never influences play; it only *receives* the
engine's authoritative board (read-only).

A move is bracketed by **two** hooks. `on_move_start` fires at the top of every
turn, *before* the player is asked for a move, so a watcher can show whose turn
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
matches (§12). The engine stays level-blind: it always fires every hook, and the
observer decides what to act on.

### Game results and termination

Every game ends in one of three ways, reported to both players via
`end_game(result)`:

```python
class Termination(Enum):
    LINE_COMPLETED   # a player completed a line — normal win
    CATS_GAME        # all 12 lines dead — draw, no winner
    PLAYER_FAULT     # a player failed to complete a valid turn — error, no winner

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
    # response, refusal, or timeout.)
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
    winner: Side | None            # None for both CATS_GAME and PLAYER_FAULT
    fault: PlayerFaultDetail | None   # set iff termination == PLAYER_FAULT
```

Notes:

- `winner` is `None` for **both** a cat's game and a player fault; the two are
  distinguished by `termination`. A player fault is **not** a win for the
  opponent.
- **`PLAYER_FAULT` covers several kinds of the same underlying failure** — "the
  player did not complete a valid turn":
  - *Structural, construction-time* (`OFF_BOARD`, `DUPLICATE_CELLS`, and
    `WRONG_PIECE_COUNT` for zero or three-plus cells): building the `Cell`/`Move`
    raises `IllegalMove`. Trusted players never trip this; a player parsing
    untrusted output (the LLM player) catches it and reports the reason via
    `MoveUnavailable`.
  - *Engine-detected, stateful* (`CELL_NOT_EMPTY`, and `WRONG_PIECE_COUNT` for a
    single-piece move that leaves the game in play): the player returns a
    well-typed `Move`, but the current board makes it illegal — a target cell is
    occupied, or the lone piece did not end the game. The engine's apply-time
    check raises `IllegalMove`, filling in `reason` and `attempted_move`.
  - *Player-reported* (`UNPARSEABLE_OUTPUT`, `THINKING_LIMIT_EXCEEDED`, …): the
    player cannot even form a `Move`, so it raises `MoveUnavailable(reason)` from
    `choose_move`. Here the **player** supplies the `reason` — the engine cannot
    know it. The LLM player distinguishes a response truncated at the output-token
    limit (`THINKING_LIMIT_EXCEEDED`) from genuinely malformed output
    (`UNPARSEABLE_OUTPUT`) so the next game's feedback can tell it to think more
    briefly.
  - *Engine-detected misread* (`WRONG_OUTCOME_CLAIM`): the move is legal, but the
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
2. At the next `start_game(side)`, the player **prepends** that stored feedback
   to the first prompt/message it builds for the new game.

`start_game` needs no new parameter for this: it only assigns the role, while the
player owns its own prompting/messaging and therefore owns the prepend.

### Shared board helper

The engine module will provide a reusable **`Board`** class (board bookkeeping:
place a piece, query cells/lines, etc.). Players **may compose** a `Board` to
manage their internal state so each player type need not reimplement it. This is
composition, not required inheritance — the `Player` ABC does not mandate it.

### Player types

The engine and interface must not need changes to add a new player type; each is
just another implementation of the player interface. The types, in
implementation order (the first three are implemented — see the milestones in §16):

1. **Scripted player** *(implemented — 0.1)*. Initialized with a predetermined
   ordered sequence of moves; returns them one at a time. Used to drive
   deterministic games for testing the rules and engine (win detection, cat's
   game, illegal-move handling, etc.).
2. **Random player** *(implemented — 0.2)*. Plays uniformly at random among legal moves: each turn it
   places two pieces on two randomly chosen empty cells (or a single piece on the
   last empty cell, which necessarily ends the game). It tracks its own board
   through `observe_move` and makes no outcome claim. A trivial baseline, a
   sparring partner for the stronger players, and a convenient driver for
   non-deterministic tests. Its randomness is drawn from an injectable
   `random.Random`, so a seeded instance produces fully reproducible games.
3. **Human player** *(implemented — 0.3)*. Reads a move interactively (input format `C3 D4`, see §17).
   So a human is never knocked out by a slip, it **re-prompts** on any locally
   detectable mistake — an unparseable label, the wrong number of cells, an
   off-board or repeated cell, a target already occupied, or a lone piece that
   would not end the game — only returning a move once it looks legal. It tracks
   its own board (to check occupancy and test a single-piece ending) and makes no
   outcome claim. End-of-input concedes the turn (a `PLAYER_FAULT`). Its input and
   output are injectable so it can be driven deterministically in tests. The
   engine stays the source of truth; the local checks only spare avoidable faults.
4. **LLM player** *(implemented — 0.5)*. Chooses moves by querying a large language
   model via Pydantic AI, with support for a **range of LLM providers and models**
   (Anthropic, OpenAI, Google Gemini, OpenRouter, and OpenAI-compatible custom
   endpoints). Specified in full in §11.
5. **Algorithmic player** *(planned)*. Searches the game tree — e.g. alpha–beta (minimax with
   pruning) — to choose strong moves.
6. **Reinforcement-learning player** *(planned)*. A policy trained via RL (self-play). Likely
   needs supporting tooling (training loop, model persistence) beyond the game
   engine itself.

Others may be added as the project evolves. Candidate ideas: a **heuristic
player** (rule-based: win if you can, block an opponent's near-win, else a
positional choice) as a cheap, explainable mid-strength opponent and a benchmark
for the AI players; a **Monte Carlo Tree Search (MCTS) player**; and a
**remote/network player** for play across machines.

### Tournaments

The project **will support a tournament structure** that pits player types against
each other over many games, composed from **matches** (§12) as its building block.
This is a committed goal, but its details (pairing schedule, scoring/standings,
handling of the Snake/Mouse start asymmetry, reporting) warrant their own sub-spec,
to be written later. This is a **1.0** goal, not part of the current 0.x line (see
§16).

## 11. The LLM player

The **LLM player** chooses its moves by querying a large language model. It is the
player type this project exists to compare: pitting models against one another (and
against strong non-LLM players) over many games is the benchmark. Because whether a
model can *reason* about the game — track the board, find lines, recognize a win or a
dead position, avoid illegal moves — is exactly what we measure, the LLM player is
given as little help as possible: it sees only the opponent's moves and must
maintain everything else itself.

### Interaction: Pydantic AI

The player wraps a single [Pydantic AI](https://ai.pydantic.dev/) `Agent`, bound to
one model (one model per player instance — see "Model selection" below). Two library
features carry the design:

- **Structured output.** The agent is configured to return a typed object rather
  than free text, so the move and the player's self-assessment arrive already parsed
  (see "Structured output" below).
- **A single running message thread.** The player keeps one conversation that
  **spans every game it plays** — not one per game. The model thus accumulates
  context across games (including how earlier games ended), which is what lets it
  learn from a mistake with no change to the `Player` interface (§10, "Feedback
  across games"). The thread is **in-memory for the life of the player instance**; it
  is not persisted across processes.

### Structured output

Each move request returns an object with three fields:

```python
class LLMMove(BaseModel):
    move_rationale: str            # a SHORT justification; logged, never validated
    cells: list[str]               # one or two cell labels, e.g. ["C3", "D4"]
    claimed_outcome: TurnOutcome   # required self-assessment: in_play | win | cats_game
```

- **`cells`** are turned into a `Move` via `Move.from_labels(*cells)`. If that cannot
  form a well-formed move — an unparseable label, an off-board or duplicated cell, or
  the wrong number of cells — the player catches the `ValueError` / `IllegalMove` and
  raises `MoveUnavailable` with the matching reason (§10, "Game results and
  termination"), ending the game as a `PLAYER_FAULT`. Cells that are well-formed but
  *illegal against the current board* (already occupied, or a lone piece that does
  not end the game) are caught by the engine at apply-time — the same fault, detected
  one layer down.
- **`claimed_outcome` is required** of the LLM (unlike the optional
  `MoveChoice.claimed_outcome` mechanical players may omit): the model must state,
  every turn, whether it believes the move wins, draws, or leaves the game in play.
  Recognizing the outcome is part of the reasoning test, so a wrong claim is a
  `WRONG_OUTCOME_CLAIM` fault (§10). The parsed move and this claim become the
  `MoveChoice` the player returns.
- **`move_rationale`** is a brief, human-readable justification — a *summary* of why
  the model chose its move, not its full chain of thought. It is **log-only**: never
  validated, never acted on, kept for observation and later analysis. It is placed
  first in the schema so the model articulates a reason before committing to a move.

### The message thread

No network call is made except when a move is actually needed: `start_game` and
`end_game` (and the one-time rules preamble) only **enqueue** messages, flushed as
the next user turn on the following `choose_move`.

- **Opening message (once, lazily on the first `choose_move`).** Explains the rules —
  board, coordinates, the snake seeded at `B3`, two pieces per turn, the winning
  lines, cat's game, what counts as illegal — and the response protocol (return
  exactly the structured fields above; you see only your opponent's moves and must
  track the board yourself).
- **`start_game(side)`** enqueues a "you are playing {side} this game" message. Per
  §10 it also prepends any stored feedback from a game the player faulted.
- **`observe_move(side, move)`** — for the **opponent's** move, enqueues "opponent
  played {move}"; for the player's **own** move it is a no-op (that move is already
  present as the model's own prior structured response).
- **`choose_move()`** flushes the queued messages as the user turn, runs the agent
  over the accumulated history, appends the response, and returns the `MoveChoice`.
- **`end_game(result)`** composes a message describing how the game ended and
  enqueues it — deferred, sent only if there is a next game. It reports the result
  from the structured `GameResult` / `PlayerFaultDetail`, and in particular:
  - the **opponent's game-ending move(s)**, when the opponent won, drew, or faulted
    on their own turn — in that case the player never got a `choose_move` to be told
    that move, so the thread would otherwise be missing it;
  - if **the player itself faulted**, what it did wrong and how to avoid repeating it,
    composed from the fault detail (§10, "Whoever reports, reports only the facts").

### No retries

An illegal or unusable move **ends the game** — no retries, no re-prompting within a
game (contrast the human player, §10, which re-prompts a person). The consequence is
delivered to the model as feedback in the *next* game's opening messages, not
mid-game. This makes "does the model play legal, well-assessed moves" a scored
property of the benchmark rather than something the harness papers over.

(Transient *API* failures — network errors, rate limits — are a separate, operational
concern, not a player fault; the player may lean on Pydantic AI's own request
retries. Handling an ultimately-unrecoverable API failure gracefully is left to the
implementation.)

### Model selection

A player is specified by a **provider** and a **model name**, one model per player
instance. Providers supported to start:

| Provider | Kind | Key (env var) |
| --- | --- | --- |
| `anthropic` | built-in | `ANTHROPIC_API_KEY` |
| `openai` | built-in | `OPENAI_API_KEY` |
| `gemini` | built-in (Google Gemini API, key-based — not Vertex) | `GEMINI_API_KEY` |
| `openrouter` | built-in | `OPENROUTER_API_KEY` |
| *(custom name)* | OpenAI-compatible endpoint (e.g. ollama, vLLM) | declared per provider |

All four built-ins — **including OpenRouter** — are supported directly by Pydantic
AI, so none needs extra endpoint configuration; we resolve `(provider, model)` to the
appropriate Pydantic AI model. Custom providers are OpenAI-compatible endpoints
reached at a configured base URL.

Three sources of configuration, parsed **outside** the player (§14, Architecture):

- **`players.yaml`** — the roster of available players. Each entry is a free-form
  **name** (its identity in matches and standings, independent of the model — in
  common use it will just echo the model), a provider, and a model name:

  ```yaml
  players:
    - name: opus
      provider: anthropic
      model: claude-opus-4-8
    - name: gpt5
      provider: openai
      model: gpt-5
    - name: gemini-pro
      provider: gemini
      model: gemini-3-pro
    - name: llama-local
      provider: my-ollama       # a custom provider, defined in providers.yaml
      model: llama3.3
  ```

- **`providers.yaml`** — **only** for custom OpenAI-compatible endpoints; built-in
  providers are never listed. Each entry maps a provider name to a base URL and,
  optionally, the environment variable holding its key (a local ollama may need
  none):

  ```yaml
  providers:
    - name: my-ollama
      base_url: http://localhost:11434/v1
      # no api_key_env — local endpoint needs no key
    - name: my-vllm
      base_url: http://gpu-box:8000/v1
      api_key_env: VLLM_API_KEY
  ```

- **`.env`** — API keys, kept out of the YAML and out of version control, read from
  the environment:

  ```
  ANTHROPIC_API_KEY=...
  OPENAI_API_KEY=...
  GEMINI_API_KEY=...
  OPENROUTER_API_KEY=...
  ```

**None of these three files is tracked in git** — they hold local roster choices and
secrets. The repository instead ships tracked templates — `players.example.yaml`,
`providers.example.yaml`, and `.env.example` — that document the format and are
copied and edited into the real (git-ignored) files.

A **config loader** reads these three sources and produces resolved, typed model
specifications, from which `LLMPlayer` instances are constructed. The player itself
never touches YAML.

### Thinking / effort level

The one non-default setting is the model's **thinking / reasoning effort**. Pydantic
AI exposes a **unified** effort control (`ModelSettings(thinking=...)` / the
`Thinking(effort=...)` capability) with levels `minimal | low | medium | high |
xhigh`; it translates each to the provider's native mechanism (Anthropic's thinking
budget, OpenAI's reasoning effort, Gemini's thinking budget, …) and maps an
unsupported level to the nearest available one.

For now every LLM player uses the **same** level — a single global default, set to
**`high`** — so each model reasons strongly and comparisons are made on an even
footing, without the steep cost of the top (`xhigh`) tier. "Consistent" here means
*each model at the same effort level*, not a byte-identical configuration across
providers. Per-player effort levels and provider-specific setting overrides are
deliberately deferred.

### Message logging (debugging)

Because the benchmark turns on *how a model reasons*, it helps to be able to read
the exact conversation a player had with its model. A debugging option captures
that: with the CLI flag **`--log-llm [DIR]`** (off by default; `DIR` defaults to a
git-ignored `llm-logs/`), each LLM player writes the **full raw message thread** —
everything sent to and received from the model — as JSON under `DIR`.

- **The complete thread, verbatim.** The dump is Pydantic AI's own serialized
  message history (`ModelMessagesTypeAdapter` over the agent's `all_messages()`),
  so it is faithful and replayable: the opening rules preamble, each game's "you
  are {side}" and opponent-move messages, the flushed user turns, and the model's
  structured responses (`move_rationale`, `cells`, `claimed_outcome`) — plus the
  deferred `end_game` feedback — all in order, with whatever metadata the library
  records (model settings, usage, timestamps).
- **One file per LLM player instance.** A player's thread spans the whole match
  (see "The message thread" above), so its file holds that side's entire match
  conversation across all games, keyed by player name and side (e.g.
  `opus-mouse.json`). Only LLM players have a thread; the flag is a no-op for
  `random` / `human` players.
- **Rewritten after every model call.** The file is overwritten with the complete
  thread-so-far on each `choose_move`, so an interrupted or crashed run still
  leaves the whole conversation up to the point of failure — exactly when a
  transcript is most wanted.
- **Local, untracked.** `DIR` holds debugging artifacts, not benchmark output; it
  is git-ignored and never committed.

This is purely an observation aid: it never changes what is sent to the model, and
never affects how moves or faults are scored.

### Deferred for now

To keep the first LLM player simple, and beyond the game-playing core above:
usage / cost / latency tracking; per-player or per-provider setting overrides;
persistence of the message thread across processes; and managing a thread that
outgrows the model's context window over a long match.

## 12. Matches

A **match** is two fixed players playing a sequence of games. It is the unit that
makes inter-player — especially inter-LLM — comparison meaningful, and because the
two `Player` instances **persist across all the games**, it is also what lets the
LLM player carry feedback from one game into the next (§10, "Feedback across games";
§11). Matches are the building block of the eventual tournament structure (§10,
"Tournaments"), which is deferred; matches themselves are part of the road to 1.0.

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
               observer: Observer | None = None) -> MatchResult: ...
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
```

The tallies summarize the match; full per-game `GameResult`s are kept **only for
games that faulted** — the interesting failures, where a player made an illegal move
or misread an outcome — each carrying its `PlayerFaultDetail`. By construction
`mouse_wins + snake_wins + cats_games + mouse_faults + snake_faults == num_games`,
and `mouse_faults + snake_faults == len(faults)`. Richer scoring and standings belong
to tournaments and are deferred.

### Observation levels

Observation generalizes from a single game to the whole match. The `Observer` (§10)
gains two match-level hooks alongside its game/move hooks:

```python
class Observer:                 # ... on_game_start / on_move_start / on_move_end /
                                #     on_game_end from §10 ...
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

## 13. Interface

A **text CLI**:

- Prints the board (ASCII, as in §3) with mice and snakes shown as distinct
  emoji: 🐭 (mouse) and 🐍 (snake).
- Reports whose turn it is, the move played, and the outcome
  (`Mouse wins`, `Snake wins`, or `Cat's game`).
- **Watches play at a chosen level.** The CLI attaches an `Observer` (§10) whose
  **observation level** (`--watch`, default `move`) sets the detail — `match`,
  `game`, or `move` (§12). There is no artificial pausing; only a human player's
  own input paces the game.
- **Runs a match.** `--mouse` and `--snake` each name who plays that side —
  `random`, `human`, or an **LLM roster name** from `players.yaml` (§11); `--games
  N` (default 1) sets the match length, with sides fixed for the match (§12). If
  either player is human the level is forced to `move` (with a message), since a
  human must see the board. E.g. `snakes-and-mice --mouse human --snake random`
  plays a single game as Mouse, and `snakes-and-mice --mouse opus --snake gpt5
  --games 20 --watch game` runs a 20-game match between two LLMs, reporting per
  game.
- **Logs LLM conversations (debugging).** `--log-llm [DIR]` (off by default) dumps
  each LLM player's full raw message thread as JSON, for inspecting how a model
  reasoned (§11, "Message logging"). A no-op for non-LLM players.

## 14. Architecture (proposed)

Rough module layout (subject to change once we start coding):

- `board` / `core` — board representation, move validation, applying moves, and
  win/draw detection.
- `game` — the loop that runs a single game, alternating the two players and
  firing the `Observer` hooks in lockstep.
- `observer` — the `Observer` spectator interface and the `ObservationLevel` that
  an observer uses to gate its own output; depends on neither `game` nor `match`,
  so both can drive it without a cycle.
- `match` — runs a match: a sequence of games between two fixed players (§12),
  reusing the same instances so LLM feedback carries across games, and producing a
  `MatchResult`. Tournaments will compose matches in their own module later;
  together these form the "engine" that drives play.
- `players` — the player interface and its implementations (scripted, random,
  human, and — per §11 — the LLM player). Loading the LLM roster from
  `players.yaml` / `providers.yaml` / `.env` lives in a small config module
  alongside it, keeping YAML parsing out of the player itself.
- `cli` — rendering the board, reporting outcomes, and watching play via an
  `Observer` at a selectable observation level (§12).

Data types to nail down when we build: cell coordinate, piece/player enum,
board, move (a pair of cells), and game result.

## 15. Tech stack & tooling

- **Language:** Python.
- **Environment & dependencies:** `uv`.
- **Version control:** `git`.
- **Tests:** `pytest`, running deterministic games driven by scripted players.
- **LLM access:** `pydantic-ai` — a uniform interface across providers (Anthropic,
  OpenAI, Google Gemini, OpenRouter, and OpenAI-compatible custom endpoints), with
  structured output and a unified thinking/effort control. Roster and provider
  config load from YAML (`pyyaml`); API keys come from the environment
  (`python-dotenv`, a `.env` file). See §11.
- **Strong typing.** Use static typing wherever possible: complete type hints on
  all public functions, methods, and data structures; `enum`s and (frozen)
  `dataclass`es for domain types (as already used for `Side`, `Move`,
  `TurnOutcome`, `PlayerFaultReason`, `GameResult`, etc.); and a static type
  checker (e.g. `mypy` or `pyright`) run over the codebase, aiming for a clean,
  strict configuration.

## 16. Versioning & scope

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
  - **0.4** — matches (§12): two fixed players over a sequence of games with
    tallied results, the generalized `Observer` with `ObservationLevel`, and the
    `--games` / `--watch` CLI.
  - **0.5** — the **LLM player** (§11): moves chosen by a model via Pydantic AI
    across a range of providers, a single cross-game message thread with
    feedback, structured output with fault mapping, YAML roster / provider config
    (keys in `.env`), the `--mouse`/`--snake` roster names, and `--log-llm`
    message logging. *(current)*
- **1.0 — first genuinely useful release.** Reached when the **LLM player** (§11,
  landed in 0.5) can be driven by the **tournament structure** (§10): only then
  can the project do what it exists to do — pit LLMs against each other, and
  against strong non-LLM players, over many games and score them. Other 0.x
  milestones (e.g. a heuristic or MCTS player, §10) may ship first, but **1.0 is
  defined by the LLM-player + tournament pair**, regardless of what else arrives
  before it.
- **After 1.0**, standard SemVer applies: incompatible changes to the player API
  or CLI bump the major, backward-compatible capabilities bump the minor, and
  fixes bump the patch.

Each of these players — LLM, algorithmic, RL — arrives without requiring engine
changes, as the `Player` abstraction (§10) is designed to allow.

Out of scope until at least 1.0, and possibly beyond:

- Game-balance analysis (whether Snake or Mouse is favored given the `B3` start).
- Any GUI/TUI.
- Network/remote play.

## 17. Resolved decisions

- **Row orientation:** `A` is the top row, `E` is the bottom.
- **Illegal-move handling:** the engine raises an error (see §9).
- **Piece glyphs:** 🐭 mouse, 🐍 snake.
- **Move input format** (for the future human player): two cells separated by a
  space, e.g. `C3 D4`.
