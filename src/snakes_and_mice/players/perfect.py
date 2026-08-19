"""The perfect algorithmic player: full-depth alpha–beta with a canonical TT.

This player solves the game (§10). Each turn it runs **negamax alpha–beta search
all the way to terminal positions** — a completed line or a cat's game — scoring
leaves exactly and backing the true value up to the root, so it always plays a
minimax-optimal move: it wins a won position as quickly as possible, never loses a
drawn-or-won one, and drags out a lost one as long as possible.

Two things make the search cheap enough. Positions are represented as a pair of
25-bit masks (mouse, snake), and a **transposition table** keyed on the
symmetry-canonical form of that pair (see :mod:`.symmetry`) collapses the many move
orders — and the 32 geometric variants — that reach the same value. The table
stores a value and an alpha–beta bound flag, never a move, so no inverse transform
is needed; the actual move is chosen in the real board frame at the root.

Like the random player it is a **mechanical** player (§3): it reads the board
correctly by construction, makes no self-assessment (``claimed_outcome`` is always
``None``), holds no cross-game state (the table is cleared each game), and takes an
injectable :class:`random.Random` used only to break ties among equally optimal
moves — so a seeded instance is fully reproducible.
"""

from __future__ import annotations

import random
from itertools import combinations

from ..board import LINES, Board
from ..core import BOARD_SIZE, Cell, Move, MoveChoice, Side
from .base import Player
from .symmetry import CELL_COUNT, canonical_key
from .table import PerfectTable, load_for_seed

_FULL_MASK: int = (1 << CELL_COUNT) - 1
"""A mask with every cell bit set."""

_WIN: int = 1000
"""A win bound larger than the deepest possible ply (a game lasts ≤ 12 turns)."""

# Alpha–beta bound flags for a stored value: exact, a lower bound (fail-high), or
# an upper bound (fail-low).
_EXACT: int = 0
_LOWER: int = 1
_UPPER: int = 2


def _cell_index(cell: Cell) -> int:
    """The 0..24 index of a cell, row-major."""
    return cell.row * BOARD_SIZE + cell.col


_CELLS_BY_INDEX: tuple[Cell, ...] = tuple(
    Cell(i // BOARD_SIZE, i % BOARD_SIZE) for i in range(CELL_COUNT)
)
"""Cell for each 0..24 index — for turning search results back into moves."""

_LINE_MASKS: tuple[int, ...] = tuple(
    sum(1 << _cell_index(cell) for cell in line) for line in LINES
)
"""The 12 winning lines as bit-masks."""

_LINES_THROUGH: tuple[tuple[int, ...], ...] = tuple(
    tuple(li for li, line in enumerate(_LINE_MASKS) if (line >> i) & 1)
    for i in range(CELL_COUNT)
)
"""For each cell index, the indices of the lines that pass through it."""


class PerfectPlayer(Player):
    """Plays a game-theoretically optimal move every turn (§10)."""

    def __init__(
        self, name: str | None = None, rng: random.Random | None = None
    ) -> None:
        super().__init__(name)
        self._rng: random.Random = rng if rng is not None else random.Random()
        self._board: Board = Board()
        self._side: Side | None = None
        # Value + bound flag per canonical position. Cleared each game: the table
        # caches only pure facts about positions, but §10 keeps the player free of
        # cross-game state, so it does not persist.
        self._tt: dict[int, tuple[int, int]] = {}
        # Exact values for the opening plies, if the solver's output is installed.
        # None means fall back to searching the opening: correct, just slow.
        self._table: PerfectTable | None = None

    def start_game(self, side: Side, seed: Cell) -> None:
        self._side = side
        self._board = Board(seed)
        self._tt = {}
        self._table = load_for_seed(seed)

    def observe_move(self, side: Side, move: Move) -> None:
        for cell in move.cells:
            if self._board.is_empty(cell):
                self._board.place(cell, side)

    def choose_move(self) -> MoveChoice:
        assert self._side is not None, "choose_move called before start_game"
        mouse, snake = self._masks()
        mine: int = mouse if self._side is Side.MOUSE else snake
        opponents: int = snake if self._side is Side.MOUSE else mouse
        empties: list[int] = self._empty_indices(mouse | snake)
        if not empties:
            raise RuntimeError(
                f"{self.name}: asked to move with no empty cells left"
            )

        # An immediate win is optimal (nothing beats winning this turn), and the
        # honest representation is a one-piece move when a single piece wins — so
        # handle wins first, before any search.
        wins: list[Move] = self._winning_moves(mine, empties)
        if wins:
            return MoveChoice(self._rng.choice(wins))

        if len(empties) == 1:
            # Forced: the lone cell fills the board into a cat's game (it cannot
            # win, or it would have been found above).
            return MoveChoice(Move.of(_CELLS_BY_INDEX[empties[0]]))

        # The opening is exact in the table and ruinous to search, so look the
        # children up instead. Below the table's threshold the search is quick.
        from_table: MoveChoice | None = self._choose_from_table(mouse, snake, empties)
        if from_table is not None:
            return from_table

        depth: int = (mouse.bit_count() + snake.bit_count() - 1) // 2
        ordered: list[int] = self._ordered(empties, mine, opponents)

        # Evaluate every non-winning two-piece move exactly (a full window, no
        # root pruning) so we can gather *all* moves that tie for the best value
        # and pick among them at random.
        best: int = -_WIN - 1
        pool: list[Move] = []
        for a, b in combinations(ordered, 2):
            added: int = (1 << a) | (1 << b)
            child_mouse: int = mouse | added if self._side is Side.MOUSE else mouse
            child_snake: int = snake if self._side is Side.MOUSE else snake | added
            if self._is_cats(child_mouse, child_snake):
                value: int = 0
            else:
                value = -self._negamax(
                    child_mouse,
                    child_snake,
                    self._side.other,
                    depth + 1,
                    -_WIN - 1,
                    _WIN + 1,
                )
            move: Move = Move.of(_CELLS_BY_INDEX[a], _CELLS_BY_INDEX[b])
            if value > best:
                best = value
                pool = [move]
            elif value == best:
                pool.append(move)
        return MoveChoice(self._rng.choice(pool))

    def _choose_from_table(
        self, mouse: int, snake: int, empties: list[int]
    ) -> MoveChoice | None:
        """Pick a move by looking every child up, or ``None`` to search instead.

        Returns ``None`` whenever the table cannot answer for the whole move list —
        no table installed, the children's layer is not covered, or any child is
        missing — so a partial or mismatched table degrades to a slower *correct*
        answer rather than a fast wrong one.

        Callers must have handled winning moves already: this assumes no child ends
        the game as a win, which holds because ``_winning_moves`` runs first.
        """
        assert self._side is not None
        child_empties: int = len(empties) - 2
        if self._table is None or not self._table.covers(child_empties):
            return None

        best: int = -_WIN - 1
        pool: list[Move] = []
        for a, b in combinations(empties, 2):
            added: int = (1 << a) | (1 << b)
            child_mouse: int = mouse | added if self._side is Side.MOUSE else mouse
            child_snake: int = snake if self._side is Side.MOUSE else snake | added
            if self._is_cats(child_mouse, child_snake):
                value: int = 0
            else:
                stored: int | None = self._table.value(
                    child_empties, canonical_key(child_mouse, child_snake)
                )
                if stored is None:
                    return None
                value = -stored
            move: Move = Move.of(_CELLS_BY_INDEX[a], _CELLS_BY_INDEX[b])
            if value > best:
                best = value
                pool = [move]
            elif value == best:
                pool.append(move)
        return MoveChoice(self._rng.choice(pool))

    def _negamax(
        self,
        mouse: int,
        snake: int,
        to_move: Side,
        depth: int,
        alpha: int,
        beta: int,
    ) -> int:
        """The value of this in-play position for ``to_move`` (larger is better).

        ``depth`` is the absolute ply count of the position (a function of the
        piece counts), so stored values are consistent across the whole game.
        """
        key: int = canonical_key(mouse, snake)
        entry: tuple[int, int] | None = self._tt.get(key)
        if entry is not None:
            value, flag = entry
            if flag == _EXACT:
                return value
            if flag == _LOWER:
                if value > alpha:
                    alpha = value
            elif value < beta:
                beta = value
            if alpha >= beta:
                return value

        alpha_orig: int = alpha
        beta_orig: int = beta

        mine: int = mouse if to_move is Side.MOUSE else snake
        opponents: int = snake if to_move is Side.MOUSE else mouse

        if self._wins_now(mine, opponents):
            best: int = _WIN - depth
            self._tt[key] = (best, _EXACT)
            return best

        empties: list[int] = self._empty_indices(mouse | snake)
        if len(empties) == 1:
            # The lone move fills the board into a cat's game (it cannot win, or
            # ``_wins_now`` would have caught it).
            self._tt[key] = (0, _EXACT)
            return 0

        ordered: list[int] = self._ordered(empties, mine, opponents)
        best = -_WIN - 1
        for a, b in combinations(ordered, 2):
            added: int = (1 << a) | (1 << b)
            child_mouse: int = mouse | added if to_move is Side.MOUSE else mouse
            child_snake: int = snake if to_move is Side.MOUSE else snake | added
            # No move wins here (``_wins_now`` was false), so a completed two-piece
            # move is impossible; the outcome is a cat's game or continued play.
            if self._is_cats(child_mouse, child_snake):
                value = 0
            else:
                value = -self._negamax(
                    child_mouse, child_snake, to_move.other, depth + 1, -beta, -alpha
                )
            if value > best:
                best = value
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break

        if best <= alpha_orig:
            flag = _UPPER
        elif best >= beta_orig:
            flag = _LOWER
        else:
            flag = _EXACT
        self._tt[key] = (best, flag)
        return best

    def _masks(self) -> tuple[int, int]:
        """The (mouse, snake) bit-masks of the player's current board view."""
        mouse: int = 0
        snake: int = 0
        for i, cell in enumerate(_CELLS_BY_INDEX):
            occupant: Side | None = self._board.occupant(cell)
            if occupant is Side.MOUSE:
                mouse |= 1 << i
            elif occupant is Side.SNAKE:
                snake |= 1 << i
        return mouse, snake

    @staticmethod
    def _empty_indices(occupied: int) -> list[int]:
        """The indices of the empty cells, in ascending (row-major) order."""
        return [i for i in range(CELL_COUNT) if not (occupied >> i) & 1]

    @staticmethod
    def _completes(mask: int) -> bool:
        """Whether ``mask`` fully occupies some winning line."""
        return any((line & mask) == line for line in _LINE_MASKS)

    @staticmethod
    def _wins_now(mine: int, opponents: int) -> bool:
        """Whether the mover can complete a line this turn (in one or two pieces).

        True iff some line holds no opponent piece and has at most two empty
        cells — the mover already owns at least three of it and can fill the rest.
        """
        for line in _LINE_MASKS:
            if line & opponents:
                continue
            gaps: int = (line & ~mine).bit_count()
            if 1 <= gaps <= 2:
                return True
        return False

    @staticmethod
    def _is_cats(mouse: int, snake: int) -> bool:
        """Whether every line is dead (holds both a mouse and a snake) — a cat's game."""
        return all((line & mouse) and (line & snake) for line in _LINE_MASKS)

    def _winning_moves(self, mine: int, empties: list[int]) -> list[Move]:
        """All honest winning moves: one-piece completions, then true two-piece ones.

        A cell that wins on its own is returned as a single-piece move; a pair is
        included only when it completes a line and neither cell wins alone (so a
        one-piece win is never padded into two).
        """
        moves: list[Move] = []
        single: set[int] = set()
        for a in empties:
            if self._completes(mine | (1 << a)):
                moves.append(Move.of(_CELLS_BY_INDEX[a]))
                single.add(a)
        for a, b in combinations(empties, 2):
            if a in single or b in single:
                continue
            if self._completes(mine | (1 << a) | (1 << b)):
                moves.append(Move.of(_CELLS_BY_INDEX[a], _CELLS_BY_INDEX[b]))
        return moves

    @staticmethod
    def _ordered(empties: list[int], mine: int, opponents: int) -> list[int]:
        """Empty cells ordered to surface likely alpha–beta cutoffs first.

        A pure move-ordering heuristic (it never changes a computed value): weight
        each cell by how much it advances the mover's own live lines and denies the
        opponent's, and try heavier cells first.
        """
        my_counts: list[int] = [(line & mine).bit_count() for line in _LINE_MASKS]
        opp_counts: list[int] = [
            (line & opponents).bit_count() for line in _LINE_MASKS
        ]

        def weight(cell: int) -> int:
            total: int = 0
            for li in _LINES_THROUGH[cell]:
                mine_here: int = my_counts[li]
                opp_here: int = opp_counts[li]
                if opp_here and mine_here:
                    continue  # a dead line: nothing to gain or defend here
                if opp_here == 0:
                    total += 1 << (2 * mine_here)  # advance our own line
                else:
                    total += 1 << (2 * opp_here)  # block the opponent's line
            return total

        return sorted(empties, key=weight, reverse=True)
