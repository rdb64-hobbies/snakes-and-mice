"""The perfect algorithmic player: full-depth alpha–beta with a canonical TT.

This player solves the game (§10). Each turn it runs **negamax alpha–beta search
all the way to terminal positions** — a completed line or a cat's game — scoring
leaves exactly and backing the true value up to the root, so it always plays a
minimax-optimal move: it wins a won position as quickly as possible, never loses a
drawn-or-won one, and drags out a lost one as long as possible.

That last case never actually arises on this player's own turn. Every seed is drawn
(§10), so a game starts drawn; the player never leaves a drawn position, and an
opponent moving from a drawn position cannot manufacture a win out of it — so by
induction the positions it is asked to move in are always drawn or won, never lost.
The lose-latest rule earns its keep at the search's *interior* nodes, where lost
positions are everywhere, and it costs nothing to have, falling straight out of the
depth-folded score.

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

Because the game is drawn from every seed, nearly every real decision is a choice
among moves that all score zero, and which one is played cannot change the result
against perfect defence. It can change the result against a *fallible* opponent, so
the pool is ranked before the random pick (§10, "Choosing among optimal moves"):
first by how many of the opponent's replies would throw the position away, then by
how much winning potential the move keeps alive. Both keys are applied **only where
they provably discriminate**, so the choice stays uniformly random wherever ranking
would merely make play predictable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
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


_ALL_DRAWN_ABOVE_EMPTIES: int = 20
"""Every position with at least this many empty cells is drawn (§10, the solve).

So a node whose *grandchildren* all sit at or above this layer has no losing reply to
find, and counting traps there is provably vacuous — it would burn the widest search
in the game to learn nothing, and would collapse the opening to one line.
"""

_TRAP_SEARCH_MAX_EMPTIES: int = 14
"""Shallowest node at which trap counting may fall back to the live search.

The next layer up is where it stops being affordable: valuing every grandchild
costs ~18–23 s at 16 empty cells, on a move the Mouse makes in nearly every game.
§10 has the measurements, and why this threshold is the one judgement call among
these gates.
"""

_LIVENESS_MAX_EMPTIES: int = 18
"""Shallowest node at which the liveness key may narrow the pool.

It is a deterministic key, so letting it decide the opening would replay one game per
seed. By 18 empty cells the position has already branched widely enough that a
deterministic choice cannot funnel every game into the same line.
"""

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

@dataclass(frozen=True)
class _Candidate:
    """A candidate move and the position it leads to.

    The child masks are carried alongside the move because every tie-break key is a
    property of the resulting position, not of the move in isolation.
    """

    move: Move
    mouse: int
    snake: int


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
        # and pick among them.
        best: int = -_WIN - 1
        pool: list[_Candidate] = []
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
            candidate = _Candidate(move, child_mouse, child_snake)
            if value > best:
                best = value
                pool = [candidate]
            elif value == best:
                pool.append(candidate)
        return MoveChoice(self._pick(pool, best, len(empties), depth))

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
        pool: list[_Candidate] = []
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
            candidate = _Candidate(move, child_mouse, child_snake)
            if value > best:
                best = value
                pool = [candidate]
            elif value == best:
                pool.append(candidate)
        depth: int = (mouse.bit_count() + snake.bit_count() - 1) // 2
        return MoveChoice(self._pick(pool, best, len(empties), depth))

    def _pick(
        self, pool: list[_Candidate], best: int, empties: int, depth: int
    ) -> Move:
        """Choose among moves that are all exactly as good, and return one.

        Every candidate already shares the same minimax value, so no key applied here
        can cost a draw or a win — the choice is free, and against a fallible opponent
        it is worth spending. The keys run strongest-first and each is applied **only
        where it provably discriminates**: a key that ranks every candidate alike
        leaves the pool untouched, so the fallback is always the uniform random pick
        that keeps games from repeating (§10).
        """
        if len(pool) == 1:
            return pool[0].move
        pool = self._most_trapping(pool, best, empties, depth)
        if empties <= _LIVENESS_MAX_EMPTIES:
            pool = self._liveliest(pool)
        return self._rng.choice(pool).move

    def _most_trapping(
        self, pool: list[_Candidate], best: int, empties: int, depth: int
    ) -> list[_Candidate]:
        """Narrow to the candidates offering the opponent the most losing replies.

        The number of ways the opponent can go wrong is the direct measure of what a
        tie-break among drawn moves is for, and it can be counted *exactly* rather
        than guessed at, so nothing here is heuristic. Returns ``pool`` unchanged
        whenever the count cannot discriminate or cannot be afforded, which keeps the
        pick uniformly random exactly where ranking would buy nothing.
        """
        # A forced win is already played as fast as possible; no reply can improve on
        # it, so every candidate would score alike. Skip the work.
        if best > 0:
            return pool
        # Grandchildren of a node this shallow are all drawn, so no reply loses.
        if empties - 4 >= _ALL_DRAWN_ABOVE_EMPTIES:
            return pool
        from_table: bool = self._table is not None and self._table.covers(empties - 4)
        if not from_table and empties > _TRAP_SEARCH_MAX_EMPTIES:
            return pool

        # We are looking for replies that leave us better off than the position's
        # true value allows: a win when the position is drawn, or an escape to at
        # least a draw when it is lost. Values are 0 or ±(_WIN − depth), so "> −1"
        # is exactly "not a loss". (The lost case is unreachable in a real game —
        # see the module docstring — but costs nothing to handle correctly.)
        threshold: int = 0 if best == 0 else -1
        counts: list[int] = []
        for candidate in pool:
            count: int | None = self._trap_count(
                candidate.mouse, candidate.snake, threshold, depth, from_table
            )
            if count is None:
                # The table could not answer for the whole reply list. Refuse partial
                # data rather than rank on it, matching `_choose_from_table`.
                return pool
            counts.append(count)

        top: int = max(counts)
        if top == min(counts):
            return pool
        return [c for c, n in zip(pool, counts) if n == top]

    def _trap_count(
        self,
        child_mouse: int,
        child_snake: int,
        threshold: int,
        depth: int,
        from_table: bool,
    ) -> int | None:
        """How many of the opponent's replies to this child would give us more.

        ``None`` means the table could not value every reply, so the count is
        unusable. Only two-piece replies are counted: a single-piece move is legal
        only when it ends the game (§2.5), and neither of its forms can be a trap —
        a completed line is a win for the opponent, and the lone cat's-game fill is
        forced, so it is never a choice they could get wrong.
        """
        assert self._side is not None
        if self._is_cats(child_mouse, child_snake):
            return 0  # the game ended on our move; there is no reply to get wrong

        replies: list[int] = self._empty_indices(child_mouse | child_snake)
        theirs: int = child_snake if self._side is Side.MOUSE else child_mouse
        grand_empties: int = len(replies) - 2
        grand_depth: int = depth + 2
        count: int = 0
        for a, b in combinations(replies, 2):
            added: int = (1 << a) | (1 << b)
            gm: int = child_mouse if self._side is Side.MOUSE else child_mouse | added
            gs: int = child_snake | added if self._side is Side.MOUSE else child_snake
            if self._completes(theirs | added):
                continue  # the reply wins for them — the opposite of a blunder
            if self._is_cats(gm, gs):
                value: int = 0
            elif from_table:
                assert self._table is not None
                stored: int | None = self._table.value(
                    grand_empties, canonical_key(gm, gs)
                )
                if stored is None:
                    return None
                # We are to move at a grandchild, so the stored value is already
                # from our perspective and is not negated (unlike a child's).
                value = stored
            else:
                value = self._negamax(
                    gm, gs, self._side, grand_depth, -_WIN - 1, _WIN + 1
                )
            if value > threshold:
                count += 1
        return count

    @staticmethod
    def _liveness(mine: int, theirs: int) -> int:
        """How much winning potential a position still holds for us.

        Counts our pieces in lines the opponent has not touched — a dead line can
        never be won, so pieces in it are spent. The per-line count is **squared**
        because threat value is sharply non-linear here: since a move places two
        pieces, three of ours in a live line is already a win-next-turn threat, so
        concentrating in one line is worth far more than the same pieces spread
        across several. A position with more of this left is one the opponent must
        keep finding the right defence against, and every defence is a chance to err.
        """
        total: int = 0
        for line in _LINE_MASKS:
            if line & theirs:
                continue
            count: int = (line & mine).bit_count()
            total += count * count
        return total

    def _liveliest(self, pool: list[_Candidate]) -> list[_Candidate]:
        """Narrow to the candidates keeping the most winning potential alive.

        A heuristic, unlike :meth:`_most_trapping` — it stands in for the traps that
        lie deeper than one reply, in the band where counting them exactly is too
        slow. As with every key here, a tie leaves the pool untouched.
        """
        assert self._side is not None
        scores: list[int] = [
            self._liveness(*self._ours_theirs(c.mouse, c.snake)) for c in pool
        ]
        top: int = max(scores)
        if top == min(scores):
            return pool
        return [c for c, s in zip(pool, scores) if s == top]

    def _ours_theirs(self, mouse: int, snake: int) -> tuple[int, int]:
        """The (ours, theirs) masks of a position, from this player's side."""
        return (mouse, snake) if self._side is Side.MOUSE else (snake, mouse)

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
