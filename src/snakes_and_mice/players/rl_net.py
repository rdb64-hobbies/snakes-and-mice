"""The RL player's network: board encoding, the policy/value MLP, drawing a move
from it, and checkpoints.

One module shared by the player that *runs* a trained network and the trainer that
*produces* one (``tools/rl/``), so the architecture is defined exactly once and a
checkpoint can never be loaded into a differently-shaped net than it was trained in.

The encoding is two 25-cell occupancy planes, **mine then theirs**, always from the
perspective of the side to move (SPEC-rl-player.md, "The algorithm and network").
Nothing says which side that is: the planes make the network side-agnostic, so a
position and its colour-swap share one set of weights and every self-play game
yields training data for both seats.

The policy is a single 25-way head applied **twice** — once to the position, then
again to the position with the first piece already placed as one of ours. That is
the factored sequential action space the spec calls for, with the conditioning on
the first placement carried by the board itself rather than by a second head. The
two phases stay distinguishable because ``|mine| - |theirs|`` is ``-1`` at every
real to-move state and ``0`` at every half-move one, for *both* sides — a signal
the first linear layer gets for free, being the sum of the first 25 features minus
the sum of the last 25. (Parity of ``|mine|`` alone will not do it: Mouse is even
to move and Snake odd, so parity identifies the phase only to someone who already
knows the side, which the encoding deliberately withholds.) The single-piece move
then needs no representation at all, being simply the case where the second call
never happens.
"""

from __future__ import annotations

import os
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, TypedDict

import torch
from torch import Tensor, nn

from ..board import LINES, Board
from ..core import BOARD_SIZE, Cell, Side
from ..faults import SnakesAndMiceError

CELL_COUNT: Final[int] = BOARD_SIZE * BOARD_SIZE
"""Cells on the board, and the width in bits of a single-side occupancy mask."""

FEATURE_SIZE: Final[int] = 2 * CELL_COUNT
"""Input width: the two occupancy planes laid end to end."""

FULL_MASK: Final[int] = (1 << CELL_COUNT) - 1

CELLS_BY_INDEX: Final[tuple[Cell, ...]] = tuple(
    Cell(i // BOARD_SIZE, i % BOARD_SIZE) for i in range(CELL_COUNT)
)
"""Every cell, indexed row-major — the order the occupancy planes are laid out in."""

LINE_MASKS: Final[tuple[int, ...]] = tuple(
    sum(1 << (cell.row * BOARD_SIZE + cell.col) for cell in line) for line in LINES
)
"""The 12 winning lines as bit-masks."""

DEFAULT_HIDDEN: Final[tuple[int, ...]] = (256, 256)
"""Trunk widths when a caller names none. Small, per the spec's "small MLP"."""

CHECKPOINT_VERSION: Final[int] = 1
"""Bumped when a stored checkpoint's layout changes incompatibly."""

MODEL_PATH_ENV: Final[str] = "SNAKES_AND_MICE_RL_MODEL"
"""Environment variable naming the checkpoint the ``rl`` player loads."""

DEFAULT_MODEL_PATH: Final[Path] = (
    Path(__file__).resolve().parents[3] / "rl-models" / "rl.pt"
)
"""Where the checkpoint lives when the environment says nothing — the repo's own
directory, mirroring how ``perfect-tables/`` holds the solver's output."""


class CheckpointError(SnakesAndMiceError):
    """A trained network could not be found, read, or matched to this code."""


class Meta(TypedDict, total=False):
    """Free-form provenance recorded alongside the weights.

    Nothing reads these to decide behaviour — they exist so a checkpoint can say
    where it came from. ``iterations``, ``games`` and ``opponents`` are what the
    trainer fills in; a hand-built checkpoint may carry none of them.
    """

    iterations: int
    games: int
    opponents: str
    note: str


def cell_index(cell: Cell) -> int:
    """The 0..24 index of ``cell``, row-major."""
    return cell.row * BOARD_SIZE + cell.col


def board_masks(board: Board) -> tuple[int, int]:
    """``board``'s occupancy as ``(mouse, snake)`` bit-masks."""
    mouse: int = 0
    snake: int = 0
    for index, cell in enumerate(CELLS_BY_INDEX):
        occupant: Side | None = board.occupant(cell)
        if occupant is Side.MOUSE:
            mouse |= 1 << index
        elif occupant is Side.SNAKE:
            snake |= 1 << index
    return mouse, snake


def own_masks(board: Board, side: Side) -> tuple[int, int]:
    """``board``'s occupancy as ``(mine, theirs)`` for ``side`` — the encoder's frame."""
    mouse, snake = board_masks(board)
    return (mouse, snake) if side is Side.MOUSE else (snake, mouse)


def completes_line(mask: int) -> bool:
    """Whether ``mask`` covers all five cells of some line — i.e. wins."""
    return any(mask & line == line for line in LINE_MASKS)


def is_cats_game(mine: int, theirs: int) -> bool:
    """Whether every line holds a piece of each side, so none can be completed."""
    return all(line & mine and line & theirs for line in LINE_MASKS)


def empty_indices(mine: int, theirs: int) -> list[int]:
    """The indices of the unoccupied cells, ascending."""
    free: int = FULL_MASK & ~(mine | theirs)
    return [i for i in range(CELL_COUNT) if (free >> i) & 1]


def needs_second_placement(mine_after: int, theirs: int) -> bool:
    """Whether a second piece must follow the first one just placed.

    The single-piece move is legal exactly when that piece ends the game
    (SPEC.md §2.5), and the sequential action space has no separate
    representation for it: the second placement is simply skipped. The three
    ways it is skipped are a completed line, a cat's game, and a board with no
    cell left — the last only reachable when one cell remained to begin with,
    which fills the board and is therefore itself a cat's game.

    The player and the trainer both route through here, so they cannot disagree
    about how many pieces a move has.
    """
    if completes_line(mine_after):
        return False
    if is_cats_game(mine_after, theirs):
        return False
    return bool(FULL_MASK & ~(mine_after | theirs))


def encode(mine: int, theirs: int) -> Tensor:
    """One position as a ``FEATURE_SIZE`` float vector: our plane, then theirs."""
    return torch.tensor(
        [float((mine >> i) & 1) for i in range(CELL_COUNT)]
        + [float((theirs >> i) & 1) for i in range(CELL_COUNT)],
        dtype=torch.float32,
    )


def encode_batch(positions: Sequence[tuple[int, int]]) -> Tensor:
    """A batch of ``(mine, theirs)`` positions as an ``(N, FEATURE_SIZE)`` tensor."""
    return torch.stack([encode(mine, theirs) for mine, theirs in positions])


def legal_mask(mine: int, theirs: int) -> Tensor:
    """A ``CELL_COUNT`` bool tensor, true on the cells still empty."""
    free: int = FULL_MASK & ~(mine | theirs)
    return torch.tensor(
        [bool((free >> i) & 1) for i in range(CELL_COUNT)], dtype=torch.bool
    )


def mask_logits(logits: Tensor, legal: Tensor) -> Tensor:
    """``logits`` with every illegal cell driven to ``-inf``.

    Broadcasts over a leading batch dimension. Illegal cells become exactly zero
    probability after the softmax, so no clipping or renormalizing is needed.
    """
    return logits.masked_fill(~legal, float("-inf"))


def draw_index(probabilities: Sequence[float], rng: random.Random) -> int:
    """Sample an index from ``probabilities`` by inverse transform.

    Laid end to end the probabilities partition ``[0, 1)`` into slices and a
    uniform draw lands in exactly one, so index ``i`` comes back with probability
    ``probabilities[i]``. A zero-probability entry — every occupied cell, after
    :func:`mask_logits` — is unreachable without being checked for: it leaves the
    running total unchanged, so the test could only pass there if it had already
    passed at an earlier entry and returned.

    Driven by ``rng`` rather than :func:`torch.multinomial`, so a seeded player
    reproduces the way every other player in the project does and no game depends
    on torch's global generator.

    Raises :class:`ValueError` if nothing has positive probability.
    """
    target: float = rng.random()
    cumulative: float = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if target < cumulative:
            return index
    # The probabilities sum to 1 only up to float error, so `target` can sit a few
    # ULPs past the final total. The last non-zero slice is the one it would have
    # landed in had the sum been exact, which makes it the honest answer rather
    # than whichever legal cell happens to be likeliest.
    for index in range(len(probabilities) - 1, -1, -1):
        if probabilities[index] > 0.0:
            return index
    raise ValueError("no cell has positive probability")


class PolicyValueNet(nn.Module):
    """A small MLP with a 25-way placement head and a scalar value head.

    ``forward`` returns ``(logits, value)`` — *unmasked* logits, which the caller
    masks to the legal cells with :func:`mask_logits`, and a value in ``(-1, 1)``
    for the side to move: ``+1`` a win, ``-1`` a loss, ``0`` a draw.
    """

    def __init__(self, hidden: Sequence[int] = DEFAULT_HIDDEN) -> None:
        super().__init__()
        self.hidden: tuple[int, ...] = tuple(hidden)
        layers: list[nn.Module] = []
        width: int = FEATURE_SIZE
        for size in self.hidden:
            layers.append(nn.Linear(width, size))
            layers.append(nn.ReLU())
            width = size
        self.trunk: nn.Sequential = nn.Sequential(*layers)
        self.policy: nn.Linear = nn.Linear(width, CELL_COUNT)
        self.value: nn.Linear = nn.Linear(width, 1)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        body: Tensor = self.trunk(features)
        return self.policy(body), torch.tanh(self.value(body)).squeeze(-1)


def save_checkpoint(
    path: Path, net: PolicyValueNet, meta: Meta | None = None
) -> None:
    """Write ``net``'s weights and shape to ``path``, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "hidden": list(net.hidden),
            "state_dict": net.state_dict(),
            "meta": dict(meta or {}),
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[PolicyValueNet, Meta]:
    """Read a checkpoint into a fresh network in eval mode.

    Raises :class:`CheckpointError` for a missing file, an unreadable one, or a
    version this code does not understand — never a partially-initialized network,
    which would play plausibly and badly rather than failing.
    """
    if not path.exists():
        raise CheckpointError(
            f"no trained RL network at {path} — train one with "
            f"`uv run python tools/rl/train.py`, or point {MODEL_PATH_ENV} at a "
            f"checkpoint"
        )
    try:
        # The file is this project's own output, never untrusted input, but
        # weights_only still keeps the load to plain tensors rather than pickle.
        blob: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # torch raises a variety of types for a bad file
        raise CheckpointError(f"could not read RL checkpoint {path}: {exc}") from exc

    if not isinstance(blob, dict):
        raise CheckpointError(f"RL checkpoint {path} is not a checkpoint")
    version: int = int(blob.get("version", 0))
    if version != CHECKPOINT_VERSION:
        raise CheckpointError(
            f"RL checkpoint {path} is version {version}, this code reads "
            f"version {CHECKPOINT_VERSION}"
        )
    # One `try` for shape and weights together: a checkpoint missing either is
    # unusable in exactly the same way, and both must fail as CheckpointError
    # rather than as a raw KeyError from a dict lookup.
    try:
        net = PolicyValueNet(tuple(int(size) for size in blob["hidden"]))
        net.load_state_dict(blob["state_dict"])
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise CheckpointError(
            f"RL checkpoint {path} does not fit the current network: {exc}"
        ) from exc
    net.eval()
    meta: Meta = blob.get("meta", {})
    return net, meta


def model_path() -> Path:
    """Which checkpoint the ``rl`` player loads: the environment, then the default."""
    configured: str | None = os.environ.get(MODEL_PATH_ENV)
    return Path(configured) if configured else DEFAULT_MODEL_PATH


__all__ = [
    "CELLS_BY_INDEX",
    "CELL_COUNT",
    "CHECKPOINT_VERSION",
    "CheckpointError",
    "DEFAULT_HIDDEN",
    "DEFAULT_MODEL_PATH",
    "FEATURE_SIZE",
    "FULL_MASK",
    "LINE_MASKS",
    "MODEL_PATH_ENV",
    "Meta",
    "PolicyValueNet",
    "board_masks",
    "cell_index",
    "draw_index",
    "completes_line",
    "empty_indices",
    "encode",
    "encode_batch",
    "is_cats_game",
    "legal_mask",
    "load_checkpoint",
    "mask_logits",
    "needs_second_placement",
    "model_path",
    "own_masks",
    "save_checkpoint",
]
