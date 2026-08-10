from __future__ import annotations

import pytest

from course_codes import (
    CURRENT_DOUBLE_LETTER_LEVELS,
    CURRENT_SINGLE_LETTER_LEVELS,
    extract_course_codes,
    is_valid_course_code,
    normalize_course_code,
    parse_course_code,
)


@pytest.mark.parametrize(
    "raw,canonical",
    [
        ("COMS W4111", "COMS W4111"),
        ("coms-w4111", "COMS W4111"),
        ("COMS_W4111", "COMS W4111"),
        ("COMSW4111", "COMS W4111"),
        ("PSAM UN3707", "PSAM UN3707"),
        ("psam-un3707", "PSAM UN3707"),
        ("PSAMUN3707", "PSAM UN3707"),
        ("BINF GU4001", "BINF GU4001"),
        ("BINFGU4001", "BINF GU4001"),
        ("EESC GR5400", "EESC GR5400"),
        ("EESC_GR5400", "EESC GR5400"),
        # Preserve compatibility with prerequisite codes outside the current
        # top-level catalog level inventory.
        ("MATH V2010", "MATH V2010"),
    ],
)
def test_normalize_course_code_variants(raw: str, canonical: str) -> None:
    assert normalize_course_code(raw) == canonical
    parsed = parse_course_code(raw)
    assert parsed is not None
    assert parsed.canonical == canonical


def test_current_level_inventory_is_explicit() -> None:
    assert CURRENT_SINGLE_LETTER_LEVELS == {"B", "C", "E", "W"}
    assert CURRENT_DOUBLE_LETTER_LEVELS == {"UN", "GU", "GR"}
    assert parse_course_code("PSAM UN3707").is_current_level
    assert parse_course_code("MATH V2010").is_current_level is False


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "A W1234",
        "ABCDE W1234",
        "COMS ZZ4111",
        "COMS W411",
        "COMS 4111",
        "COMS W41111",
        "not a course",
    ],
)
def test_invalid_course_codes(raw: str) -> None:
    assert normalize_course_code(raw) is None
    assert not is_valid_course_code(raw)


def test_extract_course_codes_preserves_order_and_supports_dual_levels() -> None:
    text = (
        "Compare psam-un3707, BINF GU4001, and EESCGR5400. "
        "PSAM UN3707 is repeated."
    )
    assert extract_course_codes(text) == [
        "PSAM UN3707",
        "BINF GU4001",
        "EESC GR5400",
    ]
    assert extract_course_codes(text, dedupe=False) == [
        "PSAM UN3707",
        "BINF GU4001",
        "EESC GR5400",
        "PSAM UN3707",
    ]
