"""Tests for the mistake-model score (`SPEC-mistake-model.md`).

What there is to test is the boundaries between the three levels: what counts as a
live threat, and when a threat set stops being answerable by one aligned move.
"""

from __future__ import annotations

from snakes_and_mice import Cell
from snakes_and_mice.mistake_model import (
    SCORE_ALIGNED,
    SCORE_NONE,
    SCORE_SPLIT,
    threat_score,
)


def mask(*labels: str) -> int:
    """The bit-mask of the named cells."""
    return sum(
        1 << (cell.row * 5 + cell.col)
        for cell in (Cell.from_label(label) for label in labels)
    )


def test_a_lone_threat_scores_nothing() -> None:
    assert threat_score(mask("A1", "A2", "A3"), mask("C4")) == SCORE_NONE


def test_two_pieces_in_a_line_are_not_yet_a_threat() -> None:
    # A threat wins *next turn*; two pieces leave three gaps, too many for one move.
    assert threat_score(mask("A1", "A2", "B1", "B2"), mask("C4")) == SCORE_NONE


def test_a_poisoned_line_is_not_a_threat() -> None:
    # One defender piece kills a line for good (§2.7).
    three_in_row_a = mask("A1", "A2", "A3")
    assert threat_score(three_in_row_a, mask("C4")) == SCORE_NONE  # live, but alone
    assert threat_score(three_in_row_a | mask("B1", "B2", "B3"), mask("A5", "C4")) == (
        SCORE_NONE
    )  # row B alone survives; row A is poisoned


def test_threats_crossing_on_an_empty_cell_are_answered_by_one_piece() -> None:
    # Row A (gaps A4, A5) and column 5 (gaps A5, B5) both run through A5, so one
    # piece there kills both.
    score = threat_score(mask("A1", "A2", "A3", "C5", "D5", "E5"), mask("C1"))
    assert score == SCORE_ALIGNED


def test_two_threats_in_one_column_are_answered_by_an_aligned_move() -> None:
    # Row A (gaps A4, A5) and row B (gaps B4, B5) are answered by A4+B4 or A5+B5,
    # each pair a single column.
    score = threat_score(mask("A1", "A2", "A3", "B1", "B2", "B3"), mask("D4"))
    assert score == SCORE_ALIGNED


def test_threats_needing_a_split_response_score_highest() -> None:
    # Row A (gaps A4, A5) and column 1 (gaps D1, E1). No cell covers both, and no
    # two cells sharing a row or column do either.
    attacker = mask("A1", "A2", "A3", "B1", "C1")
    assert threat_score(attacker, mask("E5")) == SCORE_SPLIT


def test_an_unanswerable_threat_set_scores_highest_too() -> None:
    # Rows A, C and E are three-strong with disjoint gaps, so no two cells cover
    # them at all. Top bucket by the same rule: no aligned pair covers it either.
    attacker = mask("A1", "A2", "A3", "C1", "C2", "C3", "E1", "E2", "E3")
    assert threat_score(attacker, mask("D4")) == SCORE_SPLIT


def test_a_four_piece_line_is_a_threat_with_a_single_gap() -> None:
    # Row A gaps at A5 only, column 1 at E1 only; the two share neither row nor
    # column, so the defender must split.
    attacker = mask("A1", "A2", "A3", "A4", "B1", "C1", "D1")
    assert threat_score(attacker, mask("C4")) == SCORE_SPLIT
