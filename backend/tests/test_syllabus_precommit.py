from __future__ import annotations

from pathlib import Path

import pytest

from syllabus_store import SyllabusStore


def _attachment(source: bytes) -> dict:
    return {
        "course_code": "COMS GU4111",
        "term": "Spring 2026",
        "section_id": "001/12345",
        "payload": {
            "description": source.decode(),
            "section": {
                "term": "Spring 2026",
                "section_call_number": "001/12345",
                "times": "M 10:00am - 11:00am",
                "points": "3.00",
            },
        },
        "source_bytes": source,
        "status": "published",
    }


def test_failed_runtime_precommit_validation_never_changes_restart_view(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    store = SyllabusStore(root)
    stable = store.attach_many([_attachment(b"stable")])
    old_current = (root / "CURRENT").read_text(encoding="ascii")
    old_manifest = store.manifest()

    def reject_runtime(overlays: list[dict]) -> None:
        assert any(item["payload"]["description"] == "candidate" for item in overlays)
        raise RuntimeError("injected runtime rebuild failure")

    with pytest.raises(RuntimeError, match="runtime rebuild failure"):
        store.attach_many(
            [_attachment(b"candidate")], before_commit=reject_runtime
        )

    assert (root / "CURRENT").read_text(encoding="ascii") == old_current
    assert store.manifest() == old_manifest
    restarted = SyllabusStore(root)
    assert restarted.get_effective(
        "COMS GU4111", "Spring 2026", "001/12345"
    )["version_id"] == stable[0]["version_id"]


def test_runtime_candidate_is_complete_before_first_commit(tmp_path: Path) -> None:
    root = tmp_path / "store"
    seen: list[list[dict]] = []
    store = SyllabusStore(root)

    result = store.attach_many(
        [_attachment(b"candidate")],
        before_commit=lambda overlays: seen.append(overlays),
    )

    assert result[0]["created"] is True
    assert len(seen) == 1
    assert seen[0][0]["version_id"] == result[0]["version_id"]
    assert SyllabusStore(root).effective_overlays() == seen[0]
