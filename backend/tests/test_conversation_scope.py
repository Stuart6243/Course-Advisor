from __future__ import annotations

import pytest

from conversation_scope import (
    Attribute,
    Operation,
    Scope,
    parse_conversation_scope,
)


@pytest.mark.parametrize(
    "text",
    [
        "Which of these five has the fewest prerequisites?",
        "这五门里面，哪一门先修最少？",
        "De estos cinco, ¿cuál tiene menos prerrequisitos?",
        "Parmi ces cinq cours, lequel a le moins de prérequis ?",
    ],
)
def test_four_language_previous_result_argmin(text: str) -> None:
    parsed = parse_conversation_scope(text, previous_count=5)
    assert parsed.scope is Scope.PREVIOUS_RESULTS
    assert parsed.attribute is Attribute.PREREQUISITES
    assert parsed.operation is Operation.ARGMIN
    assert parsed.ordinal is None


@pytest.mark.parametrize(
    "text",
    [
        "How many credits is the second one?",
        "第二门多少学分？",
        "¿Cuántos créditos tiene el segundo?",
        "Combien de crédits vaut le deuxième ?",
    ],
)
def test_four_language_ordinal_selects_current_course(text: str) -> None:
    parsed = parse_conversation_scope(text, previous_count=5)
    assert parsed.scope is Scope.CURRENT_COURSE
    assert parsed.attribute is Attribute.CREDITS
    assert parsed.operation is Operation.DETAIL
    assert parsed.ordinal == 2
    assert parsed.ordinals == (2,)


@pytest.mark.parametrize(
    "text",
    [
        "When does it meet?",
        "它什么时候上课？",
        "¿Cuándo se reúne?",
        "Quand a-t-il lieu ?",
    ],
)
def test_four_language_schedule_uses_existing_focus(text: str) -> None:
    parsed = parse_conversation_scope(
        text, previous_count=5, has_current_focus=True
    )
    assert parsed.scope is Scope.CURRENT_COURSE
    assert parsed.attribute is Attribute.SCHEDULE
    assert parsed.operation is Operation.DETAIL
    assert parsed.uses_focus


@pytest.mark.parametrize(
    "text",
    [
        "Compare it with the first one.",
        "比较它和第一门。",
        "Compáralo con el primero.",
        "Compare-le au premier.",
    ],
)
def test_four_language_compare_focus_with_ordinal(text: str) -> None:
    parsed = parse_conversation_scope(
        text, previous_count=5, has_current_focus=True
    )
    assert parsed.scope is Scope.PREVIOUS_RESULTS
    assert parsed.operation is Operation.COMPARE
    assert parsed.ordinal == 1
    assert parsed.ordinals == (1,)
    assert parsed.uses_focus


@pytest.mark.parametrize(
    "text",
    [
        "List those courses.",
        "列出它们。",
        "Lista esos cursos.",
        "Liste ces cours.",
    ],
)
def test_plural_references_list_previous_results(text: str) -> None:
    parsed = parse_conversation_scope(text, previous_count=5)
    assert parsed.scope is Scope.PREVIOUS_RESULTS
    assert parsed.operation is Operation.LIST


def test_first_turn_never_invents_previous_scope() -> None:
    parsed = parse_conversation_scope("Which of those has fewer prerequisites?")
    assert parsed.scope is Scope.NEW_SEARCH


def test_explicit_new_course_code_overrides_old_focus() -> None:
    parsed = parse_conversation_scope(
        "What are the prerequisites for BINF GU4001?",
        previous_count=5,
        has_current_focus=True,
    )
    assert parsed.scope is Scope.NEW_SEARCH
    assert parsed.attribute is Attribute.PREREQUISITES


def test_new_search_anchor_overrides_attribute_ellipsis() -> None:
    parsed = parse_conversation_scope(
        "What is the robotics schedule?",
        previous_count=5,
        has_current_focus=True,
        new_search_anchor=True,
    )
    assert parsed.scope is Scope.NEW_SEARCH


def test_multiple_ordinals_preserve_textual_order() -> None:
    parsed = parse_conversation_scope(
        "Compare the second one with the first one.", previous_count=5
    )
    assert parsed.operation is Operation.COMPARE
    assert parsed.ordinals == (2, 1)
    assert parsed.as_dict()["ordinals"] == [2, 1]
