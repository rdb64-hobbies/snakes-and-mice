"""Playing games for PPO: the recording player, and the reward the episode earns.

Rollouts run through the **real engine** (`play_game`), not a reimplemented
bitmask environment. A game costs about a quarter of a millisecond of board
bookkeeping against a millisecond of forward passes, so a private environment
would buy nothing and could drift from the rules the player is eventually judged
by.

## The reward

The terminal reward is the game's result from the recorded side: ``+1`` win, ``0``
cat's game, ``-1`` loss. A fault is impossible for a mechanical player and is
treated as a loss if it ever happens, so a bug cannot look like a draw.

On top of that sits the mistake model's **potential-based shaping** term
(SPEC-mistake-model.md, "Three roles, one function"), which exists because a
deliberately engineered double threat is exactly the high-value intermediate state
a sparse terminal reward is least able to teach (ibid., "Implication for the
training strategy").

The potential is measured on the position **right after our own move** — the frame
`threat_score` is defined for, with the opponent to move and our threats standing
against it. A potential on our own to-move states would be worthless here: by then
the opponent has already answered the threat (score back to 0) or already lost, so
it would reward the opponent's blunder rather than the trap that provoked it.

Step ``t`` therefore earns ``w · (Phi(after our move t) − Phi(after our move
t−1))``, with the term before the first move taken as 0 and a closing
``−w · Phi(after our last move)`` folded into the terminal step. The shaping
contributions of a completed episode then sum to exactly zero, so the shaped
return of every episode equals its true return and the optimal policy is unchanged
— the same guarantee potential-based shaping gives, arrived at without needing the
theorem. What it buys is per-move credit: the move that *builds* the split threat
is the one whose reward goes up, instead of the signal arriving only if the
opponent later falls for it.

The residue is that the shaped return depends on ``Phi`` of the previous post-move
position, which is not in the network's input, so the critic carries a bounded
offset of at most ``2w``. At the default weight that is 0.1 on a ±1 scale —
noise, not bias.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
from torch import Tensor

from snakes_and_mice.board import Board
from snakes_and_mice.core import Cell, Move, MoveChoice, Side
from snakes_and_mice.mistake_model import threat_score
from snakes_and_mice.players.base import Player
from snakes_and_mice.players.rl_net import (
    CELLS_BY_INDEX,
    PolicyValueNet,
    draw_index,
    encode,
    legal_mask,
    mask_logits,
    needs_second_placement,
    own_masks,
)
from snakes_and_mice.result import GameResult, Termination

DEFAULT_SHAPING_WEIGHT: float = 0.05
"""How much one level of the mistake model's score is worth, next to a ±1 result.

Only the *direction* of the feature is validated, never its magnitude
(SPEC-mistake-model.md, "What validation means for a shaping term"), and the
episode-sum-zero construction above makes the optimal policy independent of this
number — so it is set small enough to steer exploration without drowning the
result, and is not a tuned quantity.
"""


@dataclass
class Step:
    """One recorded turn: what the policy saw, chose, and was worth.

    ``second_position`` is the position after our first piece — the input the
    second placement was chosen from — and is ``None`` for a one-piece move, where
    no second choice was ever made.
    """

    position: tuple[int, int]
    first: int
    second_position: tuple[int, int] | None
    second_cell: int | None
    log_prob: float
    value: float
    potential: int
    reward: float = 0.0
    advantage: float = 0.0
    target: float = 0.0


@dataclass
class Episode:
    """One side's whole game: its steps, and the result they earned."""

    steps: list[Step] = field(default_factory=list)
    outcome: float = 0.0


class RecordingPlayer(Player):
    """Plays from ``net`` by sampling, and records everything PPO needs.

    Sampling, not greedy selection: PPO's gradient is defined against the policy
    the actions were actually drawn from, and greedy play would collect no
    exploration at all. Inference (``RLPlayer``) is greedy; this is not the same
    player, deliberately.
    """

    def __init__(
        self,
        net: PolicyValueNet,
        rng: random.Random,
        name: str = "rl-learner",
        shaping_weight: float = DEFAULT_SHAPING_WEIGHT,
    ) -> None:
        super().__init__(name)
        self._net: PolicyValueNet = net
        self._rng: random.Random = rng
        self._weight: float = shaping_weight
        self._board: Board = Board()
        self._side: Side | None = None
        self.episode: Episode = Episode()

    def start_game(self, side: Side, seed: Cell) -> None:
        self._side = side
        self._board = Board(seed)
        self.episode = Episode()

    def observe_move(self, side: Side, move: Move) -> None:
        for cell in move.cells:
            if self._board.is_empty(cell):
                self._board.place(cell, side)

    def choose_move(self) -> MoveChoice:
        assert self._side is not None, "choose_move called before start_game"
        mine, theirs = own_masks(self._board, self._side)

        first, first_log_prob, value = self._sample(mine, theirs, want_value=True)
        after: int = mine | (1 << first)
        second_position: tuple[int, int] | None = None
        second_cell: int | None = None
        log_prob: float = first_log_prob
        if needs_second_placement(after, theirs):
            second_position = (after, theirs)
            second_cell, second_log_prob, _ = self._sample(after, theirs)
            log_prob += second_log_prob
            after |= 1 << second_cell

        self.episode.steps.append(
            Step(
                position=(mine, theirs),
                first=first,
                second_position=second_position,
                second_cell=second_cell,
                log_prob=log_prob,
                value=value,
                # Scored on the position our move leaves behind, with the
                # opponent to move — the frame `threat_score` is defined in.
                potential=threat_score(after, theirs),
            )
        )
        cells: list[Cell] = [CELLS_BY_INDEX[first]]
        if second_cell is not None:
            cells.append(CELLS_BY_INDEX[second_cell])
        return MoveChoice(Move.of(*cells))

    def _sample(
        self, mine: int, theirs: int, *, want_value: bool = False
    ) -> tuple[int, float, float]:
        """Draw one cell; return it with its log-probability and the state value.

        ``want_value`` says whether the value head's output is wanted — it is
        meaningful only at a real to-move state, never at the half-move state the
        second placement sees.
        """
        with torch.no_grad():
            logits, value = self._net(encode(mine, theirs))
        scores: Tensor = mask_logits(logits, legal_mask(mine, theirs))
        log_probabilities: Tensor = torch.log_softmax(scores, dim=-1)
        choice: int = draw_index(log_probabilities.exp().tolist(), self._rng)
        return (
            choice,
            float(log_probabilities[choice]),
            float(value) if want_value else 0.0,
        )

    def end_game(self, result: GameResult) -> None:
        assert self._side is not None
        self.episode.outcome = _outcome(result, self._side)
        _assign_rewards(self.episode, self._weight)


def _outcome(result: GameResult, side: Side) -> float:
    """The terminal reward for ``side``: ``+1`` won, ``0`` drawn, ``-1`` otherwise.

    A fault or an abort counts as a loss. Neither can happen to this player — it
    reads the board correctly by construction and has no backend to lose — so the
    point is that a bug in the encoding or the masking shows up as a collapsing
    win rate rather than hiding among the draws.
    """
    if result.termination is Termination.LINE_COMPLETED:
        return 1.0 if result.winner is side else -1.0
    if result.termination is Termination.CATS_GAME:
        return 0.0
    return -1.0


def _assign_rewards(episode: Episode, weight: float) -> None:
    """Fill in every step's reward: the result, plus the shaping difference.

    Per this module's docstring, step ``t`` gets ``weight ·
    (potential[t] − potential[t−1])`` and the last step additionally gets the
    game's outcome and ``−weight · potential[last]``, so the shaping terms of a
    complete episode cancel exactly.
    """
    previous: int = 0
    for step in episode.steps:
        step.reward = weight * (step.potential - previous)
        previous = step.potential
    if episode.steps:
        last: Step = episode.steps[-1]
        last.reward += episode.outcome - weight * last.potential


def finish(episode: Episode, gae_lambda: float) -> None:
    """Compute each step's advantage and value target, in place.

    Generalized advantage estimation at ``gamma = 1``: the game is at most 12
    turns long and a win is worth the same whenever it lands, so there is nothing
    for a discount to express. The bootstrap after the final step is 0 — every
    episode ends in a real terminal state, never a cut-off horizon.
    """
    running: float = 0.0
    next_value: float = 0.0
    for step in reversed(episode.steps):
        delta: float = step.reward + next_value - step.value
        running = delta + gae_lambda * running
        step.advantage = running
        step.target = running + step.value
        next_value = step.value
