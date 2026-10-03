"""Tests for the RL player, its network, and the PPO trainer's credit assignment.

Nothing here needs a *trained* checkpoint: a randomly-initialized network is
enough to test that the player always produces a legal move, and the handful of
tests that need a specific decision build a network whose policy is pinned to a
chosen preference (:func:`fixed_policy`) rather than training one.

The trainer lives in ``tools/rl``, outside the package, so the pieces of it tested
here are imported the way ``test_tally_log.py`` imports its tool.
"""

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "rl"))

from rollout import (  # noqa: E402
    DEFAULT_SHAPING_WEIGHT,
    Episode,
    RecordingPlayer,
    Step,
    finish,
)

from snakes_and_mice import Cell, Move, Side, Termination  # noqa: E402
from snakes_and_mice.board import Board  # noqa: E402
from snakes_and_mice.cli_common import (  # noqa: E402
    BUILTIN_KINDS,
    NON_GRADABLE_KINDS,
    RL_KIND,
    gradable_sides,
)
from snakes_and_mice.game import play_game  # noqa: E402
from snakes_and_mice.players import RandomPlayer, RLPlayer  # noqa: E402
from snakes_and_mice.players.rl import (  # noqa: E402
    choose_cells,
    most_likely_move,
)
from snakes_and_mice.players.rl_net import (  # noqa: E402
    CELLS_BY_INDEX,
    CELL_COUNT,
    CHECKPOINT_VERSION,
    FEATURE_SIZE,
    FULL_MASK,
    CheckpointError,
    PolicyValueNet,
    board_masks,
    cell_index,
    draw_index,
    empty_indices,
    encode,
    legal_mask,
    load_checkpoint,
    mask_logits,
    needs_second_placement,
    own_masks,
    save_checkpoint,
)

ALL_SEEDS: list[Cell] = [Cell(r, c) for r in range(5) for c in range(5)]


def fixed_policy(preferences: dict[str, float]) -> PolicyValueNet:
    """A network whose policy logits are a constant, whatever the position.

    Every trunk weight and bias is zeroed, so the body is the zero vector and the
    policy head's *bias* alone decides the logits. That turns the network into a
    scripted preference over cells, which is what a test about move *plumbing* —
    masking, the second placement, the single-piece rule — actually wants.
    """
    net = PolicyValueNet((8,))
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.zero_()
        for label, score in preferences.items():
            net.policy.bias[cell_index(Cell.from_label(label))] = score
    net.eval()
    return net


def conditional_policy(
    bias: dict[str, float], boost: dict[tuple[str, str], float]
) -> PolicyValueNet:
    """A network whose logits depend on what is already ours.

    ``bias[c]`` is ``c``'s logit on an empty board; ``boost[(held, c)]`` is added
    to ``c``'s logit once ``held`` is one of our pieces. Built by making the trunk
    the identity — its input is 0/1, so ReLU passes it through unchanged — which
    lets the policy head read the occupancy planes directly.

    :func:`fixed_policy` cannot express this, and that matters: with constant
    logits the second placement's distribution is just the first's renormalized,
    under which joint-argmax and cell-by-cell descent provably agree. Showing
    they can disagree needs a policy that reacts to its own first piece.
    """
    net = PolicyValueNet((FEATURE_SIZE,))
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.zero_()
        # Sequential.__getitem__ is typed as Module, so narrow before reaching
        # for the weight.
        trunk_input = net.trunk[0]
        assert isinstance(trunk_input, nn.Linear)
        trunk_input.weight.copy_(torch.eye(FEATURE_SIZE))
        for label, value in bias.items():
            net.policy.bias[cell_index(Cell.from_label(label))] = value
        for (held, label), value in boost.items():
            # Column `held` of the "mine" plane -> row `label` of the logits.
            net.policy.weight[
                cell_index(Cell.from_label(label)), cell_index(Cell.from_label(held))
            ] = value
    net.eval()
    return net


def board_with(mouse: list[str], snake: list[str], seed: str) -> Board:
    """A board with the named cells occupied, seeded on ``seed``."""
    board = Board(Cell.from_label(seed))
    for label in mouse:
        board.place(Cell.from_label(label), Side.MOUSE)
    for label in snake:
        if Cell.from_label(label) != board.seed:
            board.place(Cell.from_label(label), Side.SNAKE)
    return board


# --- encoding ---------------------------------------------------------------


def test_encoding_is_from_the_movers_point_of_view() -> None:
    # The same board encodes to mirror-image feature vectors for the two sides:
    # the planes are "mine, theirs", so nothing in the input names a colour.
    board: Board = board_with(["A1", "A2"], ["B1"], seed="B1")
    mouse_view = encode(*own_masks(board, Side.MOUSE))
    snake_view = encode(*own_masks(board, Side.SNAKE))
    assert torch.equal(mouse_view[:CELL_COUNT], snake_view[CELL_COUNT:])
    assert torch.equal(mouse_view[CELL_COUNT:], snake_view[:CELL_COUNT])
    assert mouse_view.shape == (FEATURE_SIZE,)


def test_board_masks_match_the_boards_own_occupancy() -> None:
    board: Board = board_with(["C3", "D4"], ["E5"], seed="B3")
    mouse, snake = board_masks(board)
    for index, cell in enumerate(CELLS_BY_INDEX):
        occupant = board.occupant(cell)
        assert bool((mouse >> index) & 1) is (occupant is Side.MOUSE)
        assert bool((snake >> index) & 1) is (occupant is Side.SNAKE)
    assert empty_indices(mouse, snake) == [
        i for i, c in enumerate(CELLS_BY_INDEX) if board.is_empty(c)
    ]


def test_masked_logits_give_occupied_cells_zero_probability() -> None:
    board: Board = board_with(["A1", "A2"], [], seed="B3")
    mine, theirs = own_masks(board, Side.MOUSE)
    logits: torch.Tensor = torch.zeros(CELL_COUNT)
    probabilities = torch.softmax(mask_logits(logits, legal_mask(mine, theirs)), -1)
    for label in ("A1", "A2", "B3"):
        assert probabilities[cell_index(Cell.from_label(label))] == 0.0
    assert probabilities.sum() == pytest.approx(1.0)


# --- drawing a cell from the policy ---


def test_draw_index_respects_the_distribution() -> None:
    # Inverse-transform sampling: index i must come back about as often as its
    # own probability says. Shared by the player's temperature branch and the
    # trainer's sampler, so it is tested once, here.
    weights: list[float] = [0.5, 0.3, 0.2]
    rng = random.Random(11)
    counts: list[int] = [0, 0, 0]
    for _ in range(6000):
        counts[draw_index(weights, rng)] += 1
    for index, weight in enumerate(weights):
        assert counts[index] / 6000 == pytest.approx(weight, abs=0.02)


def test_draw_index_never_picks_a_zero_probability_cell() -> None:
    # Occupied cells are exactly the zero-probability entries after masking, so
    # this is the property that keeps the sampled move legal.
    weights: list[float] = [0.0, 0.5, 0.0, 0.5, 0.0]
    rng = random.Random(3)
    assert {draw_index(weights, rng) for _ in range(500)} == {1, 3}


def test_draw_index_falls_back_to_the_last_live_slice() -> None:
    # The probabilities sum to 1 only up to float error, so a draw can land just
    # past the final total. It must still return a cell that had real mass.
    weights: list[float] = [0.4, 0.0, 0.4]
    picks: set[int] = {draw_index(weights, random.Random(seed)) for seed in range(200)}
    assert picks <= {0, 2}
    assert 2 in picks, "the past-the-end draw should resolve to the last live slice"


def test_draw_index_rejects_an_empty_distribution() -> None:
    with pytest.raises(ValueError, match="positive probability"):
        draw_index([0.0, 0.0], random.Random(0))


def test_the_two_placement_phases_are_distinguishable_from_the_input() -> None:
    # The second placement reuses the first placement's head, so the network can
    # only behave differently in the two phases if the input says which it is in.
    # `|mine| - |theirs|` is that signal: -1 to move, 0 mid-move, for BOTH sides.
    # Parity of `|mine|` alone is not -- it is inverted between the sides, which
    # is why rl_net's docstring names the difference instead.
    for side in (Side.MOUSE, Side.SNAKE):
        board = Board(Cell.from_label("B3"))
        mover: Side = Side.MOUSE
        for _ in range(4):
            if mover is side:
                mine, theirs = own_masks(board, side)
                assert mine.bit_count() - theirs.bit_count() == -1
                first: Cell = board.empty_cells()[0]
                half: int = mine | (1 << cell_index(first))
                assert half.bit_count() - theirs.bit_count() == 0
            for cell in board.empty_cells()[:2]:
                board.place(cell, mover)
            mover = mover.other


# --- the single-piece rule --------------------------------------------------


def test_second_placement_is_skipped_only_when_the_first_piece_ends_it() -> None:
    # Four of row C plus the fifth: the line is complete, so no second piece.
    board: Board = board_with(["C1", "C2", "C3", "C4"], [], seed="A1")
    mine, theirs = own_masks(board, Side.MOUSE)
    completed: int = mine | (1 << cell_index(Cell.from_label("C5")))
    assert not needs_second_placement(completed, theirs)
    # Any other cell leaves the game in play, so a second piece is owed.
    ordinary: int = mine | (1 << cell_index(Cell.from_label("E5")))
    assert needs_second_placement(ordinary, theirs)


def test_second_placement_is_skipped_when_no_cell_is_left() -> None:
    # A full board leaves nothing to place, whether it ended as a win or a draw.
    assert not needs_second_placement(FULL_MASK, 0)


def test_one_piece_moves_are_played_only_when_they_end_the_game() -> None:
    # The legality rule (SPEC.md §2.5) stated as a property and checked over real
    # games rather than one hand-built position: every single-piece move the
    # player emits must leave a terminal board, and every two-piece move must not
    # have ended after its first piece.
    net = PolicyValueNet((16,))
    singles: int = 0
    for index, seed in enumerate(ALL_SEEDS):
        player = RLPlayer(net, rng=random.Random(index))
        opponent = RandomPlayer(rng=random.Random(2000 + index))
        board = Board(seed)
        player.start_game(Side.MOUSE, seed)
        opponent.start_game(Side.SNAKE, seed)
        to_move: Side = Side.MOUSE
        while True:
            mover = player if to_move is Side.MOUSE else opponent
            move: Move = mover.choose_move().move
            if to_move is Side.MOUSE and len(move.cells) == 1:
                singles += 1
            ended: bool = False
            for position, cell in enumerate(move.cells):
                board.place(cell, to_move)
                ended = board.winner() is not None or board.is_cats_game()
                if ended:
                    # A second piece after the game has ended would be unplayed
                    # by the engine, so the mover owed only this one.
                    assert position == len(move.cells) - 1 or to_move is not Side.MOUSE
                    break
            if to_move is Side.MOUSE and not ended:
                assert len(move.cells) == 2
            if ended:
                break
            player.observe_move(to_move, move)
            opponent.observe_move(to_move, move)
            to_move = to_move.other
    assert singles > 0, "no single-piece move occurred — the rule went untested"


def test_a_winning_first_piece_is_played_alone() -> None:
    # The policy is pinned to prefer C5, which completes row C.
    net: PolicyValueNet = fixed_policy({"C5": 10.0, "E5": 5.0})
    board: Board = board_with(["C1", "C2", "C3", "C4"], [], seed="A1")
    cells = choose_cells(net, *own_masks(board, Side.MOUSE), random.Random(0))
    assert cells == (cell_index(Cell.from_label("C5")),)


def test_an_ordinary_move_places_two_distinct_cells() -> None:
    net: PolicyValueNet = fixed_policy({"C3": 10.0, "D4": 9.0})
    board: Board = board_with([], [], seed="A1")
    cells = choose_cells(net, *own_masks(board, Side.MOUSE), random.Random(0))
    assert cells == (
        cell_index(Cell.from_label("C3")),
        cell_index(Cell.from_label("D4")),
    )


def test_greedy_play_maximizes_the_joint_not_each_cell_alone() -> None:
    # A1 is the best cell on its own but has no good partner. B1 is worse alone,
    # and makes B2 enormously attractive once played -- so {B1, B2} is far and
    # away the likeliest *move*. Descending cell by cell takes A1 and never sees
    # it; this is the bug joint-argmax decoding fixes.
    net: PolicyValueNet = conditional_policy(
        bias={"A1": 3.0, "B1": 2.0}, boost={("B1", "B2"): 8.0}
    )
    board: Board = board_with([], [], seed="E5")
    mine, theirs = own_masks(board, Side.MOUSE)

    assert frozenset(most_likely_move(net, mine, theirs, random.Random(0))) == {
        cell_index(Cell.from_label("B1")),
        cell_index(Cell.from_label("B2")),
    }

    # Confirm the premise rather than assuming it: A1 really is the best first
    # cell, and descending from it really does land on B1, not B2.
    with torch.no_grad():
        logits, _ = net(encode(mine, theirs))
    best_alone: int = int(mask_logits(logits, legal_mask(mine, theirs)).argmax())
    assert best_alone == cell_index(Cell.from_label("A1"))
    after: int = mine | (1 << best_alone)
    with torch.no_grad():
        next_logits, _ = net(encode(after, theirs))
    descended: int = int(mask_logits(next_logits, legal_mask(after, theirs)).argmax())
    assert {best_alone, descended} != {
        cell_index(Cell.from_label("B1")),
        cell_index(Cell.from_label("B2")),
    }


def test_greedy_play_sums_a_pairs_two_orderings() -> None:
    # (a, b) and (b, a) leave the same board, so they are one move and their
    # probabilities add. With four equally-liked cells, every pair among them
    # must score identically -- which only holds if both orderings are counted.
    net: PolicyValueNet = fixed_policy({"A1": 5.0, "A2": 5.0, "A3": 5.0, "A4": 5.0})
    board: Board = board_with([], [], seed="E5")
    mine, theirs = own_masks(board, Side.MOUSE)
    picks = {frozenset(most_likely_move(net, mine, theirs, random.Random(seed)))
             for seed in range(60)}
    wanted = {cell_index(Cell.from_label(label)) for label in ("A1", "A2", "A3", "A4")}
    assert len(picks) > 1, "ties should be broken by the rng, not by iteration order"
    for pick in picks:
        assert pick <= wanted and len(pick) == 2


def test_greedy_play_prefers_a_game_ending_single_piece_when_it_is_likeliest() -> None:
    # C5 completes row C, so it is its own one-piece candidate scored at pi(C5|s).
    # Pinned high, it must win against every pair -- and be emitted alone.
    net: PolicyValueNet = fixed_policy({"C5": 8.0})
    board: Board = board_with(["C1", "C2", "C3", "C4"], [], seed="A1")
    mine, theirs = own_masks(board, Side.MOUSE)
    assert most_likely_move(net, mine, theirs, random.Random(0)) == (
        cell_index(Cell.from_label("C5")),
    )


def test_greedy_play_emits_a_pair_with_a_non_ending_cell_first() -> None:
    # If the emitted ordering led with a game-ending cell, the engine would stop
    # there and the second piece would never be played -- a two-piece move that
    # is really a one-piece move. The first cell must leave the game in play.
    net: PolicyValueNet = PolicyValueNet((16,))
    for seed_index, seed in enumerate(ALL_SEEDS):
        board = Board(seed)
        player = RLPlayer(net, rng=random.Random(seed_index))
        player.start_game(Side.MOUSE, seed)
        opponent = RandomPlayer(rng=random.Random(900 + seed_index))
        opponent.start_game(Side.SNAKE, seed)
        to_move: Side = Side.MOUSE
        while len(board.empty_cells()) >= 2 and not board.winner() \
                and not board.is_cats_game():
            mover = player if to_move is Side.MOUSE else opponent
            move: Move = mover.choose_move().move
            if to_move is Side.MOUSE and len(move.cells) == 2:
                mine, theirs = own_masks(board, Side.MOUSE)
                after = mine | (1 << cell_index(move.cells[0]))
                assert needs_second_placement(after, theirs)
            for cell in move.cells:
                board.place(cell, to_move)
                if board.winner() or board.is_cats_game():
                    break
            player.observe_move(to_move, move)
            opponent.observe_move(to_move, move)
            to_move = to_move.other


def test_greedy_play_matches_an_independent_brute_force_reference() -> None:
    # most_likely_move batches and accumulates; this spells the same definition
    # out the slow, obvious way over real positions, so an optimization bug in
    # the batched version cannot pass unnoticed.
    net = PolicyValueNet((16,))

    def distribution(mine: int, theirs: int) -> list[float]:
        with torch.no_grad():
            logits, _ = net(encode(mine, theirs))
        return torch.softmax(mask_logits(logits, legal_mask(mine, theirs)), -1).tolist()

    def reference(mine: int, theirs: int) -> frozenset[int]:
        legal: list[int] = empty_indices(mine, theirs)
        first: list[float] = distribution(mine, theirs)
        conditional: dict[int, list[float]] = {
            a: distribution(mine | (1 << a), theirs)
            for a in legal
            if needs_second_placement(mine | (1 << a), theirs)
        }
        scored: dict[frozenset[int], float] = {
            frozenset((a,)): first[a] for a in legal if a not in conditional
        }
        for a, b in itertools.combinations(legal, 2):
            total: float = 0.0
            if a in conditional:
                total += first[a] * conditional[a][b]
            if b in conditional:
                total += first[b] * conditional[b][a]
            if total:
                scored[frozenset((a, b))] = total
        return max(scored, key=lambda key: (scored[key], sorted(key)))

    rng = random.Random(2)
    checked: int = 0
    for seed in ALL_SEEDS:
        board = Board(seed)
        side: Side = Side.MOUSE
        while len(board.empty_cells()) >= 2 and board.winner() is None \
                and not board.is_cats_game():
            mine, theirs = own_masks(board, side)
            assert frozenset(
                most_likely_move(net, mine, theirs, random.Random(0))
            ) == reference(mine, theirs)
            checked += 1
            for cell in rng.sample(board.empty_cells(), 2):
                board.place(cell, side)
                if board.winner() is not None or board.is_cats_game():
                    break
            side = side.other
    assert checked > 100, "too few positions to be a meaningful check"


# --- the player -------------------------------------------------------------


def test_an_untrained_player_never_faults() -> None:
    # The policy is nonsense, but the masking and the single-piece rule must still
    # produce a legal move every turn, from every opening.
    net = PolicyValueNet((16,))
    for index, seed in enumerate(ALL_SEEDS):
        player = RLPlayer(net, name="rl", rng=random.Random(index))
        opponent = RandomPlayer(name="random", rng=random.Random(1000 + index))
        for as_mouse in (True, False):
            result = (
                play_game(player, opponent, seed=seed)
                if as_mouse
                else play_game(opponent, player, seed=seed)
            )
            assert result.termination is not Termination.PLAYER_FAULT


def test_the_player_makes_no_outcome_claim() -> None:
    net = PolicyValueNet((8,))
    player = RLPlayer(net, rng=random.Random(0))
    player.start_game(Side.MOUSE, Cell.from_label("B3"))
    assert player.choose_move().claimed_outcome is None


def test_a_seeded_player_replays_the_same_moves() -> None:
    # Greedy play still draws on the rng to break ties, so reproducibility has to
    # come from the injected generator — the same guarantee `random` and
    # `perfect` give.
    net = PolicyValueNet((16,))
    runs: list[list[str]] = []
    for _ in range(2):
        player = RLPlayer(net, rng=random.Random(4))
        player.start_game(Side.MOUSE, Cell.from_label("B3"))
        played: list[str] = []
        for _ in range(3):
            move = player.choose_move().move
            played.append(str(move))
            player.observe_move(Side.MOUSE, move)
        runs.append(played)
    assert runs[0] == runs[1]


def test_the_player_tracks_the_board_through_observe_move() -> None:
    net = PolicyValueNet((8,))
    player = RLPlayer(net, rng=random.Random(0))
    player.start_game(Side.SNAKE, Cell.from_label("B3"))
    player.observe_move(Side.MOUSE, Move.from_labels("A1", "A2"))
    mine, theirs = own_masks(player._board, Side.SNAKE)
    assert theirs == (1 << cell_index(Cell.from_label("A1"))) | (
        1 << cell_index(Cell.from_label("A2"))
    )
    assert mine == 1 << cell_index(Cell.from_label("B3"))


# --- checkpoints ------------------------------------------------------------


def test_a_checkpoint_round_trips(tmp_path: Path) -> None:
    net = PolicyValueNet((32, 16))
    path: Path = tmp_path / "rl.pt"
    save_checkpoint(path, net, {"iterations": 7, "games": 70})
    loaded, meta = load_checkpoint(path)
    assert loaded.hidden == (32, 16)
    assert meta == {"iterations": 7, "games": 70}
    features: torch.Tensor = encode(0b101, 0b010)
    with torch.no_grad():
        assert torch.equal(net(features)[0], loaded(features)[0])


def test_a_missing_checkpoint_is_an_error_not_a_fresh_network(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="no trained RL network"):
        RLPlayer.from_checkpoint(tmp_path / "absent.pt")


def test_a_checkpoint_from_another_version_is_refused(tmp_path: Path) -> None:
    path: Path = tmp_path / "old.pt"
    net = PolicyValueNet((8,))
    torch.save(
        {
            "version": CHECKPOINT_VERSION + 1,
            "hidden": [8],
            "state_dict": net.state_dict(),
            "meta": {},
        },
        path,
    )
    with pytest.raises(CheckpointError, match="version"):
        load_checkpoint(path)


# --- the CLI's view of the player -------------------------------------------


def test_rl_is_built_in_but_gradable() -> None:
    # The two sets were always expected to diverge; `rl` is what makes them.
    assert RL_KIND in BUILTIN_KINDS
    assert RL_KIND not in NON_GRADABLE_KINDS
    assert gradable_sides({Side.MOUSE: RL_KIND, Side.SNAKE: "random"}) == frozenset(
        {Side.MOUSE}
    )


# --- the trainer's credit assignment ----------------------------------------


def test_shaping_cancels_over_a_whole_episode() -> None:
    # The invariant the reward design rests on: whatever the shaping term does
    # per move, an episode's rewards sum to its true result, so the optimal
    # policy is unchanged (see rollout.py's module docstring).
    net = PolicyValueNet((16,))
    for index, seed in enumerate(ALL_SEEDS):
        learner = RecordingPlayer(net, random.Random(index))
        opponent = RandomPlayer(rng=random.Random(500 + index))
        play_game(learner, opponent, seed=seed)
        episode: Episode = learner.episode
        assert episode.steps
        total: float = sum(step.reward for step in episode.steps)
        assert total == pytest.approx(episode.outcome, abs=1e-9)


def test_shaping_rewards_the_move_that_builds_the_threat() -> None:
    # A step whose move leaves a split threat behind earns more than the same
    # step would with no threats — that is the whole point of the term.
    episode = Episode(
        steps=[
            Step(position=(0, 0), first=0, second_position=None, second_cell=None,
                 log_prob=0.0, value=0.0, potential=2),
            Step(position=(0, 0), first=1, second_position=None, second_cell=None,
                 log_prob=0.0, value=0.0, potential=0),
        ],
        outcome=0.0,
    )
    from rollout import _assign_rewards

    _assign_rewards(episode, DEFAULT_SHAPING_WEIGHT)
    assert episode.steps[0].reward == pytest.approx(2 * DEFAULT_SHAPING_WEIGHT)
    assert episode.steps[1].reward == pytest.approx(-2 * DEFAULT_SHAPING_WEIGHT)
    assert sum(step.reward for step in episode.steps) == pytest.approx(0.0)


def test_advantages_telescope_to_the_realized_return() -> None:
    # At gamma = 1 with lambda = 1, the advantage is exactly the remaining
    # reward minus the state's own value, and the target is the return itself.
    episode = Episode(
        steps=[
            Step(position=(0, 0), first=0, second_position=None, second_cell=None,
                 log_prob=0.0, value=0.25, potential=0, reward=0.0),
            Step(position=(0, 0), first=1, second_position=None, second_cell=None,
                 log_prob=0.0, value=-0.5, potential=0, reward=1.0),
        ]
    )
    finish(episode, gae_lambda=1.0)
    assert episode.steps[1].target == pytest.approx(1.0)
    assert episode.steps[0].target == pytest.approx(1.0)
    assert episode.steps[0].advantage == pytest.approx(1.0 - 0.25)


def test_a_truncated_checkpoint_is_refused(tmp_path: Path) -> None:
    # A file that loads but is missing the network's shape is unusable in the
    # same way a version mismatch is, and must fail the same way — not as a raw
    # KeyError out of a dict lookup.
    path: Path = tmp_path / "partial.pt"
    torch.save({"version": CHECKPOINT_VERSION, "meta": {}}, path)
    with pytest.raises(CheckpointError, match="does not fit"):
        load_checkpoint(path)
