# Snakes and Mice — Specification

> Status: **draft**, v0.1. This document defines the game and the v1 scope. We
> agree on this before writing code.

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
metric. This is why supporting a range of LLM providers and models (§10) and a
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
three hooks, each of which defaults to a no-op:

```python
class GameObserver:
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
  - *Player-reported* (`UNPARSEABLE_OUTPUT`, …): the player cannot even form a
    `Move`, so it raises `MoveUnavailable(reason)` from `choose_move`. Here the
    **player** supplies the `reason` — the engine cannot know it.
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
just another implementation of the player interface. Planned types, roughly in
implementation order:

1. **Scripted player** *(first to implement)*. Initialized with a predetermined
   ordered sequence of moves; returns them one at a time. Used to drive
   deterministic games for testing the rules and engine (win detection, cat's
   game, illegal-move handling, etc.).
2. **Random player.** Plays uniformly at random among legal moves: each turn it
   places two pieces on two randomly chosen empty cells (or a single piece on the
   last empty cell, which necessarily ends the game). It tracks its own board
   through `observe_move` and makes no outcome claim. A trivial baseline, a
   sparring partner for the stronger players, and a convenient driver for
   non-deterministic tests. Its randomness is drawn from an injectable
   `random.Random`, so a seeded instance produces fully reproducible games.
3. **Human player.** Reads a move interactively (input format `C3 D4`, see §15).
4. **LLM player.** Chooses moves by querying a large language model, with support
   for a **range of LLM providers and models**. This player warrants its own
   detailed sub-spec (provider abstraction, prompting, parsing/validating the
   model's move, retry/fallback on invalid output, cost/latency, etc.), to be
   written later.
5. **Algorithmic player.** Searches the game tree — e.g. alpha–beta (minimax with
   pruning) — to choose strong moves.
6. **Reinforcement-learning player.** A policy trained via RL (self-play). Likely
   needs supporting tooling (training loop, model persistence) beyond the game
   engine itself.

Others may be added as the project evolves. Candidate ideas: a **heuristic
player** (rule-based: win if you can, block an opponent's near-win, else a
positional choice) as a cheap, explainable mid-strength opponent and a benchmark
for the AI players; a **Monte Carlo Tree Search (MCTS) player**; and a
**remote/network player** for play across machines.

### Tournaments

The project **will support a tournament structure** that pits player types
against each other over many games. This is a committed goal, but its details
(match/series format, scheduling of pairings, scoring/standings, handling of the
Snake/Mouse start asymmetry, reporting) warrant their own sub-spec, to be written
later. Not part of v1 (see §14).

## 11. Interface (v1)

A **text CLI**:

- Prints the board (ASCII, as in §3) with mice and snakes shown as distinct
  emoji: 🐭 (mouse) and 🐍 (snake).
- Reports whose turn it is, the move played, and the outcome
  (`Mouse wins`, `Snake wins`, or `Cat's game`).
- **Watches a game turn by turn.** The CLI attaches a `GameObserver` (see §10,
  "Watching a game") that re-renders the board after every move, pausing for the
  user to advance when stdin is interactive and playing straight through when it
  is not. The bundled demo plays two random players against each other.
- The CLI mainly **renders** a game rather than reading interactive move input;
  interactive human input is a future addition.

## 12. Architecture (proposed)

Rough module layout (subject to change once we start coding):

- `board` / `core` — board representation, move validation, applying moves, and
  win/draw detection.
- `game` — the loop that runs a single game, alternating the two players, plus
  the optional `GameObserver` hook for watching/logging. Matches (repeated games)
  and tournaments will live in their own modules alongside it; together these
  form the "engine" that drives play.
- `players` — the player interface and its implementations (scripted and random
  bots so far).
- `cli` — rendering the board, reporting outcomes, and watching a game turn by
  turn via a `GameObserver`.

Data types to nail down when we build: cell coordinate, piece/player enum,
board, move (a pair of cells), and game result.

## 13. Tech stack & tooling

- **Language:** Python.
- **Environment & dependencies:** `uv`.
- **Version control:** `git`.
- **Tests:** `pytest`, running deterministic games driven by scripted players.
- **Strong typing.** Use static typing wherever possible: complete type hints on
  all public functions, methods, and data structures; `enum`s and (frozen)
  `dataclass`es for domain types (as already used for `Side`, `Move`,
  `TurnOutcome`, `PlayerFaultReason`, `GameResult`, etc.); and a static type
  checker (e.g. `mypy` or `pyright`) run over the codebase, aiming for a clean,
  strict configuration.

## 14. Out of scope for v1

The player types beyond the scripted player (§10) are planned but **not** part of
v1 — the human, LLM, algorithmic, RL, and any other players come later, each
without requiring engine changes. Also out of scope for v1:

- Game-balance analysis (whether Snake or Mouse is favored given the `B3` start).
- The arena / tournament runner (§10).
- Any GUI/TUI.
- Network/remote play.

## 15. Resolved decisions

- **Row orientation:** `A` is the top row, `E` is the bottom.
- **Illegal-move handling:** the engine raises an error (see §9).
- **Piece glyphs:** 🐭 mouse, 🐍 snake.
- **Move input format** (for the future human player): two cells separated by a
  space, e.g. `C3 D4`.
