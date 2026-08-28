"""Tests for the endpoint comparison tool (``tools/compare_endpoints.py``).

Two halves are testable offline. The conversations it replays: the comparison is
meaningless unless both endpoints receive exactly what a match would send, and a
prompt built from a second copy of the formatting logic would drift out of step with
the player without anything failing. And the statistics it reports, where the thing
worth guarding is that a difference is only called when it exceeds the machines' own
variability — the noise floor a one-shot comparison hides.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import pytest  # noqa: E402

import compare_endpoints as ce  # noqa: E402

from snakes_and_mice.players.prompts import RULES_PREAMBLE  # noqa: E402


def test_one_prefix_per_scripted_turn() -> None:
    prefixes = ce.reference_prefixes()

    assert len(prefixes) == len(ce.OUR_MOVES)
    # Each turn adds our answer and the opponent's reply, so prefixes grow by two.
    assert [len(p) for p in prefixes] == [1, 3, 5, 7]


def test_every_prefix_ends_on_our_turn() -> None:
    # The endpoint is asked to move, so each replay must stop with a user turn.
    for prefix in ce.reference_prefixes():
        assert prefix[-1]["role"] == "user"
        assert "Choose your move" in prefix[-1]["content"]


def test_first_prefix_is_the_players_own_opening() -> None:
    # Not a paraphrase: the rules preamble and the seed announcement come from the
    # player, so a change there changes the comparison's prompts too.
    opening: str = ce.reference_prefixes()[0][0]["content"]

    assert opening.startswith(RULES_PREAMBLE)
    assert f"The snake is seeded at {ce.SEED}." in opening


def test_roles_alternate_and_answers_are_the_scripted_moves() -> None:
    prefix = ce.reference_prefixes()[-1]

    assert [m["role"] for m in prefix] == ["user", "assistant"] * 3 + ["user"]
    answers = [json.loads(m["content"]) for m in prefix if m["role"] == "assistant"]
    assert [a["cells"] for a in answers] == [list(m) for m in ce.OUR_MOVES[:3]]


def test_opponent_moves_are_relayed_between_our_turns() -> None:
    prefix = ce.reference_prefixes()[-1]
    relays = [m["content"] for m in prefix[1:] if m["role"] == "user"]

    for said, (a, b) in zip(relays, ce.THEIR_MOVES):
        assert f"played {a} {b}" in said


def test_thinking_is_not_replayed() -> None:
    # Re-sent reasoning is carried in a non-standard field whose handling varies per
    # deployment, so both endpoints must be given the same plain conversation.
    for prefix in ce.reference_prefixes():
        for message in prefix:
            assert set(message) == {"role", "content"}


def _summary(moves: list[str], reasoning: list[int]) -> ce.Summary:
    return ce.summarize(
        [ce.Sample(move=m, reasoning_chars=r) for m, r in zip(moves, reasoning)]
    )


def test_overlap_is_one_for_identical_distributions() -> None:
    a = ce.summarize([ce.Sample("A1 A2", 100)] * 3 + [ce.Sample("B1 B2", 100)])

    assert ce.overlap(a.moves, a.moves) == pytest.approx(1.0)


def test_overlap_is_zero_when_nothing_is_shared() -> None:
    a = _summary(["A1 A2", "A1 A2"], [100, 100])
    b = _summary(["C3 D4", "C3 D4"], [100, 100])

    assert ce.overlap(a.moves, b.moves) == pytest.approx(0.0)


def test_overlap_counts_shared_mass_not_shared_labels() -> None:
    # Same two moves, very different frequencies: overlap is the shared mass (0.25 +
    # 0.25), not 1.0 for having the same support.
    a = _summary(["X", "X", "X", "Y"], [1, 1, 1, 1])
    b = _summary(["X", "Y", "Y", "Y"], [1, 1, 1, 1])

    assert ce.overlap(a.moves, b.moves) == pytest.approx(0.5)


def test_a_reasoning_gap_inside_the_noise_floor_is_not_called() -> None:
    # Means differ by 200, but each machine varies far more than that against itself.
    a = _summary(["X"] * 4, [1_000, 5_000, 9_000, 13_000])
    b = _summary(["X"] * 4, [1_200, 5_200, 9_200, 13_200])

    assert "Consistent with one distribution" in ce.verdict(a, b)


def test_a_reasoning_gap_beyond_the_noise_floor_is_called() -> None:
    # Same moves, tight spreads, means far apart: that is a real difference.
    a = _summary(["X"] * 4, [1_000, 1_010, 1_020, 1_030])
    b = _summary(["X"] * 4, [9_000, 9_010, 9_020, 9_030])

    assert "reasoning lengths differ by more than" in ce.verdict(a, b)


def test_a_noisy_endpoint_is_not_declared_different_from_itself() -> None:
    # The measured case: eight samples of one identical request gave five distinct
    # moves, so the halves of a single endpoint overlap only slightly. Comparing that
    # endpoint with a copy of itself must never read as a difference — which a fixed
    # threshold would, since the overlap is far below any sensible constant.
    noisy = ["B2 C3", "B1 C1", "A3 B2", "B2 C3", "B1 B4", "B1 B2", "B2 C3", "B1 C1"]
    a = _summary(noisy, [3_149, 4_000, 5_000, 6_000, 7_000, 8_000, 9_288, 6_000])
    b = _summary(noisy, [3_200, 4_100, 5_100, 6_100, 7_100, 8_100, 9_200, 6_100])

    assert a.self_overlap < 0.75, "the fixture must be genuinely noisy"
    assert "Consistent with one distribution" in ce.verdict(a, b)


def test_disjoint_moves_are_reported_as_unresolved_not_as_difference() -> None:
    # Below the noise floor should read as suggestive, never as a confident finding.
    a = _summary(["A1 A2"] * 4, [100, 110, 105, 108])
    b = _summary(["C3 D4"] * 4, [100, 110, 105, 108])

    said = ce.verdict(a, b)
    assert "below the" in said and "more samples" in said


def test_too_few_samples_to_split_says_so() -> None:
    a = _summary(["A1 A2", "C3 D4"], [100, 110])
    b = _summary(["A1 A2", "C3 D4"], [100, 110])

    assert "Too few samples to measure the noise floor" in ce.verdict(a, b)


def test_an_endpoint_with_no_usable_answers_is_not_compared() -> None:
    empty = ce.summarize([], failures=8)
    other = _summary(["A1 A2"], [100])

    assert "nothing to compare" in ce.verdict(empty, other)
    assert empty.n == 0


def test_summary_reports_its_own_spread() -> None:
    s = _summary(["X", "Y", "X"], [1_000, 3_000, 2_000])

    assert s.n == 3
    assert s.modal == ("X", 2)
    assert s.mean_reasoning == pytest.approx(2_000)
    assert s.reasoning_spread == pytest.approx(1_000)


def test_cross_overlap_is_measured_at_the_same_size_as_the_baseline() -> None:
    # Full-sample overlap would flatter the comparison: halves overlap less by
    # construction, so the cross measure must also be taken on halves.
    a = _summary(["X", "Y", "Z", "W"], [1, 1, 1, 1])
    b = _summary(["X", "Y", "Z", "W"], [1, 1, 1, 1])

    assert ce.overlap(a.moves, b.moves) == pytest.approx(1.0)
    assert ce.cross_overlap(a, b) < 1.0


def test_a_lucky_endpoint_does_not_set_an_unreachable_bar() -> None:
    # Both endpoints drew the same distribution — four X and four Y — but in orders
    # that make their self-overlap estimates 100% and 0%. Taking the larger as the
    # baseline would declare two identical distributions different; the mean does not.
    even = _summary(["X", "X", "Y", "Y", "X", "X", "Y", "Y"], [100] * 8)
    clumped = _summary(["X", "X", "X", "X", "Y", "Y", "Y", "Y"], [100] * 8)

    assert even.moves == clumped.moves, "the fixture must be one distribution"
    assert even.self_overlap == pytest.approx(1.0)
    assert clumped.self_overlap == pytest.approx(0.0)
    assert "below the" not in ce.verdict(even, clumped)
