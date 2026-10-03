"""The reinforcement-learning player: a trained policy, played greedily.

This is the *inference* half of the RL player (SPEC-rl-player.md). It loads a
checkpoint produced by ``tools/rl/train.py`` and plays, each turn, the move its
policy assigns the highest probability to — scored over whole moves rather than
one cell at a time, since the factored policy makes those different things
(:func:`most_likely_move`). A move costs two forward passes: one for the position
and one batched pass over the positions each legal first placement would lead to.

That is enumeration of the policy's own action space, not search: no position is
ever *evaluated* beyond the current one, and the spec's ban on AlphaZero-style
lookahead at train or inference time is untouched.

Like ``random`` and ``perfect`` it is a **mechanical** player in the sense of §3:
it returns ``claimed_outcome = None``, holds no cross-game state, and takes an
injectable :class:`random.Random` so a seeded instance is reproducible. Unlike
those two it is neither optimal nor aimless, so — as ``cli_common`` already
anticipated — its moves *are* worth grading for mistakes (§5).

Two ways it differs from every other built-in player are worth stating plainly:

- **It can play a losing move.** Nothing here checks for an immediate win or a
  forced loss; whether the policy sees them is exactly what training decides and
  what grading measures. Bolting a one-ply safety check on would measure the
  check, not the policy.
- **A missing checkpoint is an error, not a degradation.** ``perfect`` without its
  table is slow but still perfect; an RL player without its weights would be a
  randomly-initialized network — a plausible-looking player of no defined
  strength. :func:`~.rl_net.load_checkpoint` raises instead.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch import Tensor

from ..board import Board
from ..core import Cell, Move, MoveChoice, Side
from .base import Player
from .rl_net import (
    CELLS_BY_INDEX,
    Meta,
    PolicyValueNet,
    draw_index,
    empty_indices,
    encode,
    encode_batch,
    legal_mask,
    load_checkpoint,
    mask_logits,
    model_path,
    needs_second_placement,
    own_masks,
)

GREEDY: float = 0.0
"""The default temperature: always the highest-scoring legal cell."""


def choose_cells(
    net: PolicyValueNet,
    mine: int,
    theirs: int,
    rng: random.Random,
    temperature: float = GREEDY,
) -> tuple[int, ...]:
    """The one or two cell indices the policy plays from ``(mine, theirs)``.

    At ``temperature == GREEDY`` this plays the **most likely move**, scored
    against the joint (:func:`most_likely_move`). Above it, the two placements are
    drawn in sequence, which already samples the joint exactly and so needs no
    enumeration — it is only the *argmax* that the factorization complicates.

    Returns a single index only when that piece ends the game — a win, a cat's
    game, or the last empty cell (:func:`~.rl_net.needs_second_placement`).
    Raises :class:`ValueError` on a full board, which the turn flow never
    produces.
    """
    if temperature <= GREEDY:
        return most_likely_move(net, mine, theirs, rng)
    first: int = _sample(net, mine, theirs, rng, temperature)
    after: int = mine | (1 << first)
    if not needs_second_placement(after, theirs):
        return (first,)
    return (first, _sample(net, after, theirs, rng, temperature))


def most_likely_move(
    net: PolicyValueNet, mine: int, theirs: int, rng: random.Random
) -> tuple[int, ...]:
    """The move the policy assigns the highest probability to.

    Taking the best first cell and then the best second cell is *not* the best
    move. The policy scores a move as ``pi(a1|s) * pi(a2|s + a1)``, so the cell
    that looks strongest alone may have no good partner, and greedy descent of
    the two placements can miss the pair the policy actually likes most. This
    enumerates instead: one forward pass for ``s``, then a single batched pass
    over all of its children, and maximizes the real quantity.

    **Orderings are summed, not competed.** When neither cell alone ends the
    game, ``(a, b)`` and ``(b, a)`` leave the same board and are therefore the
    same move, so the mass the factorization splits between them belongs to one
    candidate and is scored as one. That also repairs the ordering redundancy
    inherent to a sequential policy over what is really an unordered pair.

    **A game-ending cell is its own one-piece candidate**, scored at
    ``pi(a|s)``. That is honestly its probability rather than a length bias: the
    sampler stops at such a cell instead of drawing a partner, so no second
    factor was ever going to be charged against it.

    Nothing here looks for a win or a loss — scoring is purely the policy's own
    distribution, so what this plays is what the network believes, as the module
    docstring promises.
    """
    legal: list[int] = empty_indices(mine, theirs)
    if not legal:
        raise ValueError("no empty cell to place on")
    with torch.no_grad():
        logits, _ = net(encode(mine, theirs))
    first: list[float] = torch.softmax(
        mask_logits(logits, legal_mask(mine, theirs)), dim=-1
    ).tolist()

    # Total probability per candidate move, keyed by its cell set, plus the
    # ordering to emit it in (always one whose first cell leaves the game in
    # play, so a two-piece move is never really a one-piece move in disguise).
    totals: dict[frozenset[int], float] = {}
    emit: dict[frozenset[int], tuple[int, ...]] = {}
    best_ordering: dict[frozenset[int], float] = {}

    follow_on: list[int] = []
    for cell in legal:
        if needs_second_placement(mine | (1 << cell), theirs):
            follow_on.append(cell)
        else:
            key = frozenset((cell,))
            totals[key] = first[cell]
            emit[key] = (cell,)

    if follow_on:
        positions: list[tuple[int, int]] = [
            (mine | (1 << cell), theirs) for cell in follow_on
        ]
        with torch.no_grad():
            child_logits, _ = net(encode_batch(positions))
        child_legal: Tensor = torch.stack(
            [legal_mask(own, other) for own, other in positions]
        )
        second: list[list[float]] = torch.softmax(
            mask_logits(child_logits, child_legal), dim=-1
        ).tolist()
        for row, a in enumerate(follow_on):
            for b in legal:
                if b == a:
                    continue
                joint: float = first[a] * second[row][b]
                key = frozenset((a, b))
                totals[key] = totals.get(key, 0.0) + joint
                if joint > best_ordering.get(key, -1.0):
                    best_ordering[key] = joint
                    emit[key] = (a, b)

    highest: float = max(totals.values())
    # Ties are real — a fresh or near-symmetric policy produces them — so break
    # them with the injected rng rather than by iteration order, which would make
    # the player replay one game per opening. Sorted first so the candidate list
    # does not depend on dict insertion order.
    tied: list[frozenset[int]] = sorted(
        (key for key, total in totals.items() if total == highest), key=sorted
    )
    return emit[rng.choice(tied)]


def _sample(
    net: PolicyValueNet,
    mine: int,
    theirs: int,
    rng: random.Random,
    temperature: float,
) -> int:
    """One cell index drawn from the masked policy over ``(mine, theirs)``.

    The sampling path only — ``temperature`` is positive here, the greedy case
    having gone to :func:`most_likely_move`. Sampling is driven by ``rng`` rather
    than torch's global generator, so a seeded player reproduces exactly the way
    every other player in the project does, and a match never depends on global
    torch state.
    """
    legal: Tensor = legal_mask(mine, theirs)
    if not bool(legal.any()):
        raise ValueError("no empty cell to place on")
    with torch.no_grad():
        logits, _ = net(encode(mine, theirs))
    scores: Tensor = mask_logits(logits, legal)
    # Dividing before the softmax is what temperature does: below 1 sharpens the
    # distribution toward the greedy choice, above 1 flattens it toward uniform.
    # An illegal cell's `-inf` survives the division, so it stays at probability 0.
    return draw_index(torch.softmax(scores / temperature, dim=-1).tolist(), rng)


class RLPlayer(Player):
    """Plays the move a trained policy network scores highest.

    The constructor takes the **network**, since that is what the player actually
    needs; :meth:`from_checkpoint` is the way to get one from a file, and is what
    the CLI uses. The trainer holds a live network and no file, so it constructs
    directly — the same split as
    :meth:`~snakes_and_mice.players.llm.LLMPlayer.from_roster`.
    """

    def __init__(
        self,
        net: PolicyValueNet,
        name: str | None = None,
        rng: random.Random | None = None,
        temperature: float = GREEDY,
        meta: Meta | None = None,
    ) -> None:
        super().__init__(name)
        self._net: PolicyValueNet = net
        self._meta: Meta = meta if meta is not None else {}
        self._rng: random.Random = rng if rng is not None else random.Random()
        self._temperature: float = temperature
        self._board: Board = Board()
        self._side: Side | None = None

    @classmethod
    def from_checkpoint(
        cls,
        path: Path | None = None,
        name: str | None = None,
        rng: random.Random | None = None,
        temperature: float = GREEDY,
    ) -> RLPlayer:
        """Load a trained network and wrap it in a player.

        ``path`` defaults to :func:`~.rl_net.model_path`. Raises
        :class:`~.rl_net.CheckpointError` if no usable checkpoint is there.
        """
        net, meta = load_checkpoint(path if path is not None else model_path())
        return cls(net, name=name, rng=rng, temperature=temperature, meta=meta)

    @property
    def meta(self) -> Meta:
        """The network's provenance (see :class:`~.rl_net.Meta`).

        Empty for a player built straight from a live network, which has no
        checkpoint to have recorded any.
        """
        return self._meta

    def start_game(self, side: Side, seed: Cell) -> None:
        self._side = side
        self._board = Board(seed)

    def observe_move(self, side: Side, move: Move) -> None:
        for cell in move.cells:
            if self._board.is_empty(cell):
                self._board.place(cell, side)

    def choose_move(self) -> MoveChoice:
        assert self._side is not None, "choose_move called before start_game"
        mine, theirs = own_masks(self._board, self._side)
        indices: tuple[int, ...] = choose_cells(
            self._net, mine, theirs, self._rng, self._temperature
        )
        return MoveChoice(Move.of(*(CELLS_BY_INDEX[i] for i in indices)))


__all__ = ["GREEDY", "RLPlayer", "choose_cells", "most_likely_move"]
