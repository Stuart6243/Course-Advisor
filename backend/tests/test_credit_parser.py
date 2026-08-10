from __future__ import annotations

import pytest

from credit_parser import CreditConstraint, parse_credit_constraint, parse_points_range


@pytest.mark.parametrize(
    "text,expected",
    [
        # English
        ("3.5 credits", [3.5, 3.5]),
        ("2-4 credits", [2.0, 4.0]),
        ("between 2 and 4 credits", [2.0, 4.0]),
        ("at least 3 credits", [3.0, None]),
        ("at most 4 credits", [None, 4.0]),
        ("3 credits or more", [3.0, None]),
        ("4 credits or less", [None, 4.0]),
        # Chinese
        ("3.5学分", [3.5, 3.5]),
        ("2到4学分", [2.0, 4.0]),
        ("至少3学分", [3.0, None]),
        ("不超过4学分", [None, 4.0]),
        ("3学分以上", [3.0, None]),
        # Spanish (including decimal comma)
        ("3,5 créditos", [3.5, 3.5]),
        ("entre 2 y 4 créditos", [2.0, 4.0]),
        ("al menos 3 créditos", [3.0, None]),
        ("como máximo 4 créditos", [None, 4.0]),
        ("4 créditos o menos", [None, 4.0]),
        # French
        ("3,5 crédits", [3.5, 3.5]),
        ("entre 2 et 4 crédits", [2.0, 4.0]),
        ("au moins 3 crédits", [3.0, None]),
        ("au plus 4 crédits", [None, 4.0]),
        ("3 crédits ou plus", [3.0, None]),
        # Reversed explicit ranges are normalized deterministically.
        ("4-2 credits", [2.0, 4.0]),
    ],
)
def test_parse_multilingual_credit_constraints(text: str, expected: list) -> None:
    assert parse_points_range(text) == expected


def test_combined_open_bounds() -> None:
    assert parse_points_range("at least 2 and at most 4 credits") == [2.0, 4.0]


@pytest.mark.parametrize(
    "text",
    ["", "COMS W4111", "Spring 2026", "a few credits", "between courses"],
)
def test_non_credit_text_does_not_create_constraint(text: str) -> None:
    assert parse_credit_constraint(text) is None


def test_credit_constraint_overlap_uses_course_range_intersection() -> None:
    exact = CreditConstraint(3.5, 3.5)
    assert exact.overlaps(1.0, 6.0)
    assert not exact.overlaps(4.0, 4.0)

    lower_bound = CreditConstraint(3.0, None)
    assert lower_bound.overlaps(2.0, 4.0)
    assert not lower_bound.overlaps(1.0, 2.0)


def test_invalid_constraint_bounds_are_rejected() -> None:
    with pytest.raises(ValueError):
        CreditConstraint(None, None)
    with pytest.raises(ValueError):
        CreditConstraint(4.0, 3.0)
