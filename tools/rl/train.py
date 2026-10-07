"""Train the RL player by self-play PPO (SPEC-rl-player.md, "The algorithm and network").

This is step 1 of the loop: bootstrap a base agent purely from self-play and games
against `random` and `perfect` — cheap, unlimited, no LLM involved. The live-LLM
steps (2–6) are a separate piece of work; nothing here talks to a model backend.

    uv run python tools/rl/train.py [--iterations N] [--out rl-models/rl.pt]

What each iteration does:

1. **Collect.** Play ``--games-per-iteration`` games. Each game draws an opponent
   from the pool (below) and a seed cell uniformly from all 25, and is played
   through the real engine. Self-play games record *both* seats, so they yield two
   episodes; games against a mechanical opponent record one.
2. **Credit.** Turn each episode into per-step rewards (the result plus the
   mistake model's shaping term — see :mod:`rollout`) and GAE advantages at
   ``gamma = 1``.
3. **Update.** Several epochs of clipped PPO over minibatches of those steps, with
   an entropy bonus, a value loss, and the solved-value regularizer below.

**The opponent pool is chosen by cost, and the cost is lopsided.** SPEC-rl-player.md
assumes "games against `random` and `perfect` are unlimited and effectively free."
Half of that holds: a self-play game costs ~0.9 ms and a `random` game ~0.5 ms.
A `perfect` game costs **~5.5 s** — it live-searches every position below 16 empty
cells, and the search does not amortize across games (a transposition table kept
across 30 games gave no speedup at all and grew to 5.9 M entries, because endgame
positions barely repeat). One `perfect` game therefore costs about 11,000 free
ones, so any share large enough to matter as *training* costs most of the run's
volume.

That argument once put ``--perfect-share`` at 0, on the reasoning that `perfect`
earns its keep as the yardstick rather than as a sparring partner. **Measured,
that was wrong**: a 2% share over 3,100 iterations halved the trained player's
losses to bare `perfect`, from 9.0% to 4.5% across 200 balanced games, at no cost
against `random` (96.2% either way). The default is now
:data:`DEFAULT_PERFECT_SHARE`. Cost is still the reason it is *small* — the
arithmetic above is unchanged — but it is no longer a reason for it to be zero.

**The critic is pinned to ground truth.** Every update adds a regression of the
value head against exact solved values from a fixed dataset (:mod:`solved`), so
the value head is anchored to what the game actually is, and PPO's own learning
goes into the policy rather than into rediscovering position values. The same
regression runs alone for ``--pretrain-epochs`` before any game is played.

That dataset's *composition* turned out to matter far more than its size or the
network's width: see :mod:`solved`, which enriches it for positions where the
side to move is in trouble, because ordinary play produces almost none and a
value head fit without them cannot tell a lost position from a drawn one.

Evaluation is greedy — it plays the `RLPlayer` the checkpoint will produce, not
the sampling learner — so the number reported is the number a match would see.

**It runs single-threaded on purpose** (:data:`TORCH_THREADS`), and does not want
a GPU. Half an iteration is thousands of batch-1 forward passes, where MPS is
5.5x *slower* than the CPU (101 us against 18 us) because launch overhead dwarfs
a 50x256 matmul; the other half is batch-256 updates that leave any real GPU
idling well above 99%.

Vectorizing `collect` so games step in lockstep does **not** change that answer,
and an earlier version of this docstring was wrong to name it as the way to make
this GPU-shaped. Batching the rollouts turns ~4,100 batch-1 calls into ~12
batch-256 ones, and at that shape the CPU is still ahead on one thread (73 us
against MPS's 134 us); the crossover is somewhere past batch 16k, which would
need 16,000 games in flight per ply. As a pure CPU optimization it is also capped
low: only 46% of `collect` is torch at all (186 ms against 100 ms with the
network stubbed out), so perfect batching would reach ~235 ms an iteration from
319 ms, for a rewrite of the rollout loop that would *add* bookkeeping to the
Python floor it cannot touch.

None of which matters, because **throughput is not this trainer's constraint**.
Play against `random` converges by iteration ~100 — about 17 minutes — and is flat
in a 94–96% band from 125 onward. Any run longer than that is spending its time
on something other than bulk policy quality against `random`.

What it *can* still buy is exposure to opponents that are expensive per game,
which is exactly the `--perfect-share` finding above: the 8-hour run was ~99% past
convergence by this measure and still halved the loss rate to `perfect`. So the
lever is the opponent pool rather than the clock, and a GPU would not help with
either.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn, optim

from snakes_and_mice.core import Cell, Side
from snakes_and_mice.game import play_game
from snakes_and_mice.players.base import Player
from snakes_and_mice.players.perfect import PerfectPlayer, TieBreak
from snakes_and_mice.players.random import RandomPlayer
from snakes_and_mice.players.rl import RLPlayer
from snakes_and_mice.players.rl_net import (
    DEFAULT_HIDDEN,
    Meta,
    PolicyValueNet,
    encode_batch,
    legal_mask,
    mask_logits,
    save_checkpoint,
)
from snakes_and_mice.result import GameResult, Termination
from rollout import (
    DEFAULT_SHAPING_WEIGHT,
    POTENTIALS,
    Episode,
    Potential,
    RecordingPlayer,
    Step,
    finish,
)
from solved import (
    DEFAULT_PATH as SOLVED_PATH,
    DEFAULT_POSITIONS as SOLVED_POSITIONS,
    Labelled,
    load_or_build,
)

ALL_SEEDS: tuple[Cell, ...] = tuple(Cell(r, c) for r in range(5) for c in range(5))

SELF_PLAY: str = "self"
"""The pool entry meaning "the current policy, recorded on both seats"."""

TORCH_THREADS: int = 1
"""How many CPU threads torch may use. One, measured, is fastest.

This network is far too small for intra-op parallelism to pay for its own
synchronization: a forward+backward+Adam step at the PPO minibatch size of 256
costs 360 us on one thread against 977 us on twelve. The rollout and inference
shapes (batch 1 and the policy's batched pass over a position's children) are
flat across thread counts, so only the update is affected — but it is ~45% of an
iteration, and one thread takes a full iteration from 384 ms to 319 ms.

Set here and not in the package: thread count is a process-wide global, and a
library must not reconfigure its host (the same line `cli_common` draws around
HTTP log levels). ``--threads`` overrides it, since the measurement is for this
net shape on this machine and a much wider ``--hidden`` could invert it.
"""

FREE_POOL: dict[str, float] = {SELF_PLAY: 0.60, "random": 0.40}
"""The opponents that cost nothing: the current policy, and `random`.

Measured on this repo: a self-play game is ~0.9 ms and a `random` game ~0.5 ms, so
both really are the unlimited, effectively free opponents the spec assumes.
"""

DEFAULT_PERFECT_SHARE: float = 0.02
"""Fraction of training games played against perfect play.

Two percent is what produced the shipped checkpoint, and what halved its losses
to bare `perfect` (9.0% to 4.5% over 200 balanced games). It is small because a
`perfect` game costs ~11,000 free ones, not because perfect play is of little
use: at 256 games an iteration this is ~5 such games, which dominates the
iteration's wall clock at ~9 s against ~0.3 s for everything else.

The share is what was measured; the *amount* of exposure also depends on how many
iterations run, and the shipped checkpoint had 3,100 of them (~15,900 `perfect`
games) under an 8-hour ``--time-budget``. A default 400-iteration run gets ~2,050,
which is untested — raise the iteration count or the budget for a real run.
"""

PERFECT_SPLIT: dict[str, float] = {"perfect": 0.75, "perfect-trappiness": 0.25}
"""How a non-zero ``--perfect-share`` is divided between the two variants.

Mostly the bare player, since that is the one whose ranking adds nothing of its
own; a quarter goes to `perfect-trappiness` so the agent meets the opponent that
actively steers toward positions a fallible player mishandles.
"""


def build_pool(perfect_share: float) -> dict[str, float]:
    """The opponent pool: the free opponents, plus ``perfect_share`` of perfect play.

    See :func:`main`'s ``--perfect-share`` for why that share defaults to zero.
    """
    if perfect_share <= 0.0:
        return dict(FREE_POOL)
    pool: dict[str, float] = {
        kind: weight * (1.0 - perfect_share) for kind, weight in FREE_POOL.items()
    }
    pool.update(
        {kind: weight * perfect_share for kind, weight in PERFECT_SPLIT.items()}
    )
    return pool


@dataclass(frozen=True)
class Settings:
    """Every knob the run uses, so one object can be logged and reproduced.

    Hyperparameters are ordinary PPO defaults, not tuned quantities — the spec
    leaves "training hyperparameters and curriculum details" open, and nothing
    here claims to have closed them.
    """

    iterations: int = 400
    games_per_iteration: int = 256
    epochs: int = 4
    minibatch: int = 256
    learning_rate: float = 3e-4
    clip: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    solved_coefficient: float = 0.5
    solved_batch: int = 512
    pretrain_epochs: int = 20
    gae_lambda: float = 0.95
    shaping_weight: float = DEFAULT_SHAPING_WEIGHT
    shaping: str = "threat-score"
    max_grad_norm: float = 0.5


def opponent_for(kind: str, side: Side, rng: random.Random) -> Player:
    """One mechanical opponent of the named kind, seeded from ``rng``."""
    child: random.Random = random.Random(rng.getrandbits(64))
    if kind == "random":
        return RandomPlayer(name="random", rng=child)
    if kind == "perfect":
        return PerfectPlayer(name="perfect", rng=child, tie_break=TieBreak.NONE)
    if kind == "perfect-trappiness":
        return PerfectPlayer(
            name="perfect-trappiness", rng=child, tie_break=TieBreak.TRAPPINESS
        )
    if kind == "perfect-mistake-model":
        return PerfectPlayer(
            name="perfect-mistake-model", rng=child, tie_break=TieBreak.MISTAKE_MODEL
        )
    raise ValueError(f"unknown opponent kind: {kind}")


def collect(
    net: PolicyValueNet,
    settings: Settings,
    pool: dict[str, float],
    rng: random.Random,
) -> list[Episode]:
    """Play a round of games and return every recorded episode.

    The learner takes either seat with equal probability, so it never learns to
    play only first. Against a mechanical opponent one episode comes back; in a
    self-play game both seats are recorded, which is free data — the two seats see
    different positions, and the network is side-agnostic by construction.
    """
    kinds: list[str] = list(pool)
    weights: list[float] = [pool[kind] for kind in kinds]
    episodes: list[Episode] = []
    for _ in range(settings.games_per_iteration):
        kind: str = rng.choices(kinds, weights=weights)[0]
        seed: Cell = rng.choice(ALL_SEEDS)
        potential: Potential = POTENTIALS[settings.shaping]
        learner = RecordingPlayer(
            net, rng, shaping_weight=settings.shaping_weight, potential=potential
        )
        if kind == SELF_PLAY:
            other = RecordingPlayer(
                net, rng, name="rl-learner-2",
                shaping_weight=settings.shaping_weight, potential=potential,
            )
            play_game(learner, other, seed=seed)
            episodes.extend((learner.episode, other.episode))
            continue
        if rng.random() < 0.5:
            play_game(learner, opponent_for(kind, Side.SNAKE, rng), seed=seed)
        else:
            play_game(opponent_for(kind, Side.MOUSE, rng), learner, seed=seed)
        episodes.append(learner.episode)
    return episodes


@dataclass
class Batch:
    """One iteration's steps, flattened into the tensors an update needs."""

    first_features: Tensor
    first_legal: Tensor
    first_cell: Tensor
    second_features: Tensor
    second_legal: Tensor
    second_cell: Tensor
    has_second: Tensor
    old_log_prob: Tensor
    advantage: Tensor
    target: Tensor

    def __len__(self) -> int:
        return int(self.first_cell.shape[0])


def to_batch(episodes: list[Episode], gae_lambda: float) -> Batch:
    """Flatten episodes into tensors, computing advantages on the way.

    A one-piece move has no second placement, so its second-placement slot is
    filled with its own first position and masked out by ``has_second``. Padding
    rather than a ragged batch keeps the update a single pair of forward passes;
    the mask makes the padding contribute exactly nothing to the loss.
    """
    steps: list[Step] = []
    for episode in episodes:
        finish(episode, gae_lambda)
        steps.extend(episode.steps)

    first_positions: list[tuple[int, int]] = [step.position for step in steps]
    second_positions: list[tuple[int, int]] = [
        step.second_position if step.second_position is not None else step.position
        for step in steps
    ]
    advantage: Tensor = torch.tensor(
        [step.advantage for step in steps], dtype=torch.float32
    )
    return Batch(
        first_features=encode_batch(first_positions),
        first_legal=torch.stack([legal_mask(*p) for p in first_positions]),
        first_cell=torch.tensor([step.first for step in steps], dtype=torch.long),
        second_features=encode_batch(second_positions),
        second_legal=torch.stack([legal_mask(*p) for p in second_positions]),
        second_cell=torch.tensor(
            [step.second_cell if step.second_cell is not None else step.first
             for step in steps],
            dtype=torch.long,
        ),
        has_second=torch.tensor(
            [step.second_position is not None for step in steps], dtype=torch.float32
        ),
        old_log_prob=torch.tensor(
            [step.log_prob for step in steps], dtype=torch.float32
        ),
        advantage=(advantage - advantage.mean()) / (advantage.std() + 1e-8),
        target=torch.tensor([step.target for step in steps], dtype=torch.float32),
    )


def _log_prob_and_entropy(
    net: PolicyValueNet, features: Tensor, legal: Tensor, cell: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Log-probability of ``cell``, the distribution's entropy, and the value."""
    logits, value = net(features)
    scores: Tensor = mask_logits(logits, legal)
    log_probabilities: Tensor = torch.log_softmax(scores, dim=-1)
    chosen: Tensor = log_probabilities.gather(1, cell.unsqueeze(1)).squeeze(1)
    # Illegal cells carry probability 0 and log-probability -inf, whose product is
    # nan rather than the 0 the entropy sum wants — so drop those terms outright.
    probabilities: Tensor = log_probabilities.exp()
    entropy: Tensor = -(probabilities * log_probabilities.nan_to_num(neginf=0.0)).sum(-1)
    return chosen, entropy, value


def update(
    net: PolicyValueNet,
    optimizer: optim.Optimizer,
    batch: Batch,
    solved: list[Labelled],
    settings: Settings,
    rng: random.Random,
) -> dict[str, float]:
    """Run PPO's epochs over ``batch``; return the averaged loss components."""
    size: int = len(batch)
    totals: dict[str, float] = {
        "policy": 0.0, "value": 0.0, "entropy": 0.0, "solved": 0.0
    }
    updates: int = 0
    order: list[int] = list(range(size))
    for _ in range(settings.epochs):
        rng.shuffle(order)
        for start in range(0, size, settings.minibatch):
            rows: Tensor = torch.tensor(
                order[start : start + settings.minibatch], dtype=torch.long
            )
            first_log, first_entropy, value = _log_prob_and_entropy(
                net,
                batch.first_features[rows],
                batch.first_legal[rows],
                batch.first_cell[rows],
            )
            second_log, second_entropy, _ = _log_prob_and_entropy(
                net,
                batch.second_features[rows],
                batch.second_legal[rows],
                batch.second_cell[rows],
            )
            has_second: Tensor = batch.has_second[rows]
            log_prob: Tensor = first_log + second_log * has_second
            entropy: Tensor = (first_entropy + second_entropy * has_second).mean()

            ratio: Tensor = (log_prob - batch.old_log_prob[rows]).exp()
            advantage: Tensor = batch.advantage[rows]
            unclipped: Tensor = ratio * advantage
            clipped: Tensor = (
                ratio.clamp(1.0 - settings.clip, 1.0 + settings.clip) * advantage
            )
            policy_loss: Tensor = -torch.min(unclipped, clipped).mean()
            value_loss: Tensor = nn.functional.mse_loss(value, batch.target[rows])
            solved_loss: Tensor = _solved_loss(net, solved, settings, rng)

            loss: Tensor = (
                policy_loss
                + settings.value_coefficient * value_loss
                + settings.solved_coefficient * solved_loss
                - settings.entropy_coefficient * entropy
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), settings.max_grad_norm)
            optimizer.step()

            totals["policy"] += float(policy_loss.detach())
            totals["value"] += float(value_loss.detach())
            totals["entropy"] += float(entropy.detach())
            totals["solved"] += float(solved_loss.detach())
            updates += 1
    return {name: total / max(updates, 1) for name, total in totals.items()}


def _solved_loss(
    net: PolicyValueNet, solved: list[Labelled], settings: Settings, rng: random.Random
) -> Tensor:
    """Squared error of the value head against exact solved values.

    A fresh random sample each minibatch rather than a fixed slice: the dataset is
    far larger than one batch, and resampling keeps the regularizer from pulling
    the head toward whichever corner of it was drawn first.
    """
    if not solved or settings.solved_batch <= 0:
        return torch.zeros(())
    rows: list[Labelled] = rng.sample(
        solved, min(settings.solved_batch, len(solved))
    )
    features: Tensor = encode_batch([(row.mine, row.theirs) for row in rows])
    targets: Tensor = torch.tensor(
        [row.target for row in rows], dtype=torch.float32
    )
    _, value = net(features)
    return nn.functional.mse_loss(value, targets)


def pretrain(
    net: PolicyValueNet,
    solved: list[Labelled],
    settings: Settings,
    rng: random.Random,
) -> None:
    """Regress the value head against the solved values before any game is played.

    The policy head is not touched — its gradient is zero here, since nothing in
    this loss passes through it — but the shared trunk is, which is the point: the
    policy starts on top of features that already separate won positions from lost
    ones instead of on noise.
    """
    if not solved or settings.pretrain_epochs <= 0:
        return
    optimizer: optim.Optimizer = optim.Adam(
        net.parameters(), lr=settings.learning_rate
    )
    order: list[int] = list(range(len(solved)))
    for epoch in range(settings.pretrain_epochs):
        rng.shuffle(order)
        total: float = 0.0
        batches: int = 0
        for start in range(0, len(order), settings.minibatch):
            rows: list[Labelled] = [
                solved[i] for i in order[start : start + settings.minibatch]
            ]
            features: Tensor = encode_batch([(r.mine, r.theirs) for r in rows])
            targets: Tensor = torch.tensor(
                [r.target for r in rows], dtype=torch.float32
            )
            _, value = net(features)
            loss: Tensor = nn.functional.mse_loss(value, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        print(
            f"  pretrain epoch {epoch + 1}/{settings.pretrain_epochs}: "
            f"value mse {total / max(batches, 1):.4f}",
            file=sys.stderr,
        )


@dataclass(frozen=True)
class Score:
    """A greedy evaluation's record over some games, from the RL player's seat."""

    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    def __str__(self) -> str:
        total: int = max(self.games, 1)
        return (
            f"{self.wins:>4}W {self.draws:>4}D {self.losses:>4}L "
            f"({100.0 * self.wins / total:5.1f}% win, "
            f"{100.0 * self.losses / total:5.1f}% loss)"
        )


def evaluate_against(
    net: PolicyValueNet, kind: str, games: int, rng: random.Random
) -> Score:
    """Play ``games`` greedy games against ``kind``, alternating seats and seeds.

    The greedy :class:`RLPlayer` is what a real match would face, so this measures
    the shipped player rather than the sampling learner that produced it. Seats
    alternate and seeds walk all 25 cells, so neither a first-move advantage nor a
    lucky opening can carry the score.
    """
    player = RLPlayer(net, name="rl", rng=random.Random(rng.getrandbits(64)))
    wins: int = 0
    draws: int = 0
    losses: int = 0
    net.eval()
    for index in range(games):
        as_mouse: bool = index % 2 == 0
        side: Side = Side.MOUSE if as_mouse else Side.SNAKE
        opponent: Player = opponent_for(kind, side.other, rng)
        seed: Cell = ALL_SEEDS[index % len(ALL_SEEDS)]
        result: GameResult = (
            play_game(player, opponent, seed=seed)
            if as_mouse
            else play_game(opponent, player, seed=seed)
        )
        if result.termination is Termination.LINE_COMPLETED:
            if result.winner is side:
                wins += 1
            else:
                losses += 1
        elif result.termination is Termination.CATS_GAME:
            draws += 1
        else:
            losses += 1
    net.train()
    return Score(wins, draws, losses)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Train the Snakes and Mice RL player by self-play PPO.",
    )
    parser.add_argument("--out", type=Path, default=Path("rl-models/rl.pt"))
    parser.add_argument("--solved", type=Path, default=SOLVED_PATH)
    parser.add_argument(
        "--solved-positions", type=int, default=SOLVED_POSITIONS,
        help=f"exact-valued positions for the critic (default: "
             f"{SOLVED_POSITIONS}); built once and cached, ~29 min",
    )
    parser.add_argument("--iterations", type=int, default=Settings.iterations)
    parser.add_argument(
        "--games-per-iteration", type=int, default=Settings.games_per_iteration
    )
    parser.add_argument("--pretrain-epochs", type=int, default=Settings.pretrain_epochs)
    parser.add_argument("--hidden", type=int, nargs="+", default=list(DEFAULT_HIDDEN))
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-games", type=int, default=200,
                        help="games per periodic progress check, vs random only")
    parser.add_argument("--final-eval-games", type=int, default=200,
                        help="games vs random in the closing baseline report; "
                             "each perfect variant gets a quarter as many")
    parser.add_argument(
        "--perfect-share", type=float, default=DEFAULT_PERFECT_SHARE,
        metavar="FRACTION",
        help=f"fraction of training games played against a perfect opponent "
             f"(default: {DEFAULT_PERFECT_SHARE}); measured to halve losses to "
             f"`perfect`, and the reason a run takes hours rather than minutes "
             f"— pass 0 for a fast run that skips perfect play entirely",
    )
    parser.add_argument("--time-budget", type=float, default=0.0, metavar="SECONDS",
                        help="stop after this much wall clock (0: no limit)")
    parser.add_argument(
        "--shaping", choices=sorted(POTENTIALS), default=Settings.shaping,
        help=f"which shaping potential to use (default: {Settings.shaping}); "
             f"'move-preference' scores how likely the modelled opponent is to "
             f"fail to answer, covering single threats the three-level score rates 0",
    )
    parser.add_argument(
        "--threads", type=int, default=TORCH_THREADS, metavar="N",
        help=f"CPU threads for torch (default: {TORCH_THREADS}); this network is "
             f"small enough that more threads cost more than they save",
    )
    args: argparse.Namespace = parser.parse_args(argv)

    settings = Settings(
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        pretrain_epochs=args.pretrain_epochs,
        shaping=args.shaping,
    )
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    rng = random.Random(args.rng_seed)
    torch.manual_seed(args.rng_seed)

    solved: list[Labelled] = load_or_build(
        args.solved, args.solved_positions, random.Random(args.rng_seed + 1)
    )
    net = PolicyValueNet(args.hidden)
    print(f"pretraining the value head on {len(solved)} solved positions",
          file=sys.stderr)
    pretrain(net, solved, settings, rng)

    optimizer: optim.Optimizer = optim.Adam(
        net.parameters(), lr=settings.learning_rate
    )
    pool: dict[str, float] = build_pool(args.perfect_share)
    print("opponents: " + _describe(pool), file=sys.stderr)
    started: float = time.time()
    games_played: int = 0
    iteration: int = 0
    for iteration in range(1, settings.iterations + 1):
        episodes: list[Episode] = collect(net, settings, pool, rng)
        games_played += settings.games_per_iteration
        batch: Batch = to_batch(episodes, settings.gae_lambda)
        losses: dict[str, float] = update(
            net, optimizer, batch, solved, settings, rng
        )
        mean_outcome: float = sum(e.outcome for e in episodes) / max(len(episodes), 1)
        elapsed: float = time.time() - started
        print(
            f"iter {iteration:>4}/{settings.iterations}  "
            f"{elapsed:6.0f}s  steps {len(batch):>5}  "
            f"outcome {mean_outcome:+.3f}  "
            f"policy {losses['policy']:+.4f}  value {losses['value']:.4f}  "
            f"solved {losses['solved']:.4f}  entropy {losses['entropy']:.3f}",
            file=sys.stderr, flush=True,
        )
        if args.eval_every and iteration % args.eval_every == 0:
            _report_progress(net, args.eval_games, rng)
            save_checkpoint(args.out, net, _meta(iteration, games_played, pool))
        if args.time_budget and elapsed >= args.time_budget:
            print(f"time budget reached after {iteration} iterations", file=sys.stderr)
            break

    print("final evaluation:", file=sys.stderr)
    _report_baselines(net, args.final_eval_games, rng)
    save_checkpoint(args.out, net, _meta(iteration, games_played, pool))
    print(f"wrote {args.out}", file=sys.stderr)


def _meta(iterations: int, games: int, pool: dict[str, float]) -> Meta:
    return {"iterations": iterations, "games": games, "opponents": _describe(pool)}


def _describe(pool: dict[str, float]) -> str:
    return ", ".join(f"{kind} {share:.2%}" for kind, share in pool.items())


def _report_progress(net: PolicyValueNet, games: int, rng: random.Random) -> None:
    """The cheap periodic check: `random` only.

    `random` is free, so this can run as often as wanted; it is a coarse but
    honest progress signal, since a policy that is getting stronger converts more
    of a random opponent's blunders. The `perfect` baselines are deliberately not
    here — at ~5.5 s a game, running them every few hundred iterations would spend
    more of the budget on measurement than on training.
    """
    print(f"  vs {'random':<22} {evaluate_against(net, 'random', games, rng)}",
          file=sys.stderr, flush=True)


def _report_baselines(net: PolicyValueNet, games: int, rng: random.Random) -> None:
    """The full record, against `random` and all three `perfect` variants.

    All three variants appear because they are the non-learned baselines
    SPEC-rl-player.md asks the RL player to be measured against: the bare
    `perfect` with no ranking at all, and the two that already rank moves for
    exploitability. Run once at the end, where its cost is paid once.
    """
    for kind in ("random", "perfect", "perfect-trappiness", "perfect-mistake-model"):
        count: int = games if kind == "random" else max(games // 4, 10)
        print(f"  vs {kind:<22} {evaluate_against(net, kind, count, rng)}",
              file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
