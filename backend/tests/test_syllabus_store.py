from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from syllabus_store import (
    StoreCapacityError,
    SyllabusStore,
    apply_published_overlays,
    identity_key,
)


def attach(
    store: SyllabusStore,
    *,
    code: str = "COMS GU4111",
    term: str = "Spring 2026",
    section: str = "001/12345",
    source: bytes = b"version one",
    status: str = "published",
):
    return store.attach_syllabus(
        course_code=code,
        term=term,
        section_id=section,
        payload={"description": source.decode(), "section": {"term": term}},
        source_bytes=source,
        status=status,
        provenance={"filename": "syllabus.html", "seed_course_uid": "seed-1"},
        evidence={"course_code": {"verified": True, "quote": code}},
        quality_score=95,
        quality_issues=[],
    )


def test_review_version_is_stored_but_not_effective(tmp_path: Path) -> None:
    store = SyllabusStore(tmp_path / "store")
    result = attach(store, status="review")

    assert result["created"] is True
    assert store.get_effective("COMS GU4111", "Spring 2026", "001/12345") is None
    administrative = store.get_identity(
        "COMS GU4111", "Spring 2026", "001/12345"
    )
    assert administrative["versions"][0]["status"] == "review"
    assert administrative["versions"][0]["provenance"]["seed_course_uid"] == "seed-1"
    assert store.manifest()["effective_published_count"] == 0


def test_promotion_and_source_hash_versions(tmp_path: Path) -> None:
    store = SyllabusStore(tmp_path / "store")
    first = attach(store, source=b"version one", status="review")
    same = attach(store, source=b"version one", status="published")
    assert same["created"] is False
    assert same["version_id"] == first["version_id"]

    promoted = store.set_status(
        course_code="COMS GU4111",
        term="Spring 2026",
        section_id="001/12345",
        version_id=first["version_id"],
        status="published",
    )
    assert promoted["changed"] is True
    assert store.get_effective(
        "COMS GU4111", "Spring 2026", "001/12345"
    )["version_id"] == first["version_id"]

    second = attach(store, source=b"version two", status="published")
    effective = store.get_effective("COMS GU4111", "Spring 2026", "001/12345")
    assert effective["version_id"] == second["version_id"]
    assert effective["payload"]["description"] == "version two"
    assert store.manifest()["version_count"] == 2


@pytest.mark.parametrize("code", ["PSAM UN3707", "BINF GU4001", "EESC GR5400"])
def test_level_designators_and_multiple_identities(tmp_path: Path, code: str) -> None:
    store = SyllabusStore(tmp_path / code.split()[1])
    attach(store, code=code, term="Fall 2025", section="001/11111")
    attach(store, code=code, term="Spring 2026", section="002/22222")
    assert len(store.effective_overlays()) == 2
    assert identity_key(code, "Fall 2025", "001/11111") != identity_key(
        code, "Spring 2026", "002/22222"
    )


def test_failure_before_current_replace_keeps_old_generation(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = SyllabusStore(root)
    first = attach(store, source=b"stable")
    old_current = (root / "CURRENT").read_text()
    generation_count = len(list((root / "generations").iterdir()))

    failing_store = SyllabusStore(root, failure_injector="before_current_replace")
    with pytest.raises(RuntimeError, match="injected failure"):
        attach(failing_store, source=b"not committed")

    assert (root / "CURRENT").read_text() == old_current
    effective = store.get_effective("COMS GU4111", "Spring 2026", "001/12345")
    assert effective["version_id"] == first["version_id"]
    assert store.manifest()["version_count"] == 1
    assert len(list((root / "generations").iterdir())) == generation_count


def test_store_rejects_noncanonical_identity_codes(tmp_path: Path) -> None:
    store = SyllabusStore(tmp_path / "store")
    with pytest.raises(ValueError, match="Invalid course_code"):
        attach(store, code="AERO 3001")
    with pytest.raises(ValueError, match="Invalid course_code"):
        attach(store, code="NOT A COURSE")
    assert not (tmp_path / "store").exists()


def test_post_current_failure_restores_previous_generation_on_restart(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    stable_store = SyllabusStore(root)
    stable = attach(stable_store, source=b"stable")
    old_current = (root / "CURRENT").read_text()
    generation_count = len(list((root / "generations").iterdir()))

    failing = SyllabusStore(root, failure_injector="after_current_replace")
    with pytest.raises(RuntimeError, match="injected failure"):
        attach(failing, source=b"must roll back")
    assert (root / "CURRENT").read_text() == old_current
    assert len(list((root / "generations").iterdir())) == generation_count
    restarted = SyllabusStore(root)
    assert restarted.get_effective(
        "COMS GU4111", "Spring 2026", "001/12345"
    )["version_id"] == stable["version_id"]


def test_failed_first_commit_leaves_no_current_or_generation(tmp_path: Path) -> None:
    root = tmp_path / "store"
    failing = SyllabusStore(root, failure_injector="after_current_replace")
    with pytest.raises(RuntimeError, match="injected failure"):
        attach(failing)
    assert not (root / "CURRENT").exists()
    assert list((root / "generations").iterdir()) == []
    assert SyllabusStore(root).effective_overlays() == []


def test_multi_section_batch_has_one_commit_point(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = SyllabusStore(root)
    items = []
    for section, source in (("001/11111", b"one"), ("002/22222", b"two")):
        items.append(
            {
                "course_code": "COMS GU4111",
                "term": "Spring 2026",
                "section_id": section,
                "payload": {"section": section},
                "source_bytes": source,
                "status": "published",
                "provenance": {"filename": "multi.html"},
                "evidence": {"verified": True},
            }
        )
    outcomes = store.attach_many(items)
    assert outcomes[0]["generation"] == outcomes[1]["generation"]
    assert store.manifest()["identity_count"] == 2

    failing = SyllabusStore(root, failure_injector="before_current_replace")
    failed_items = [
        {
            **item,
            "source_bytes": item["source_bytes"] + b" changed",
        }
        for item in items
    ]
    old_current = (root / "CURRENT").read_text()
    with pytest.raises(RuntimeError, match="injected failure"):
        failing.attach_many(failed_items)
    assert (root / "CURRENT").read_text() == old_current
    assert store.manifest()["version_count"] == 2


def test_two_store_instances_serialize_writers(tmp_path: Path) -> None:
    root = tmp_path / "store"

    def writer(number: int):
        return attach(
            SyllabusStore(root),
            section=f"{number:03d}/{10000 + number}",
            source=f"source {number}".encode(),
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(writer, range(1, 9)))
    assert SyllabusStore(root).manifest()["identity_count"] == 8


def test_store_capacity_limits_fail_before_a_new_generation_is_written(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    store = SyllabusStore(root, max_versions=1, max_generations=10)
    attach(store, source=b"first")
    generation_count = len(list((root / "generations").iterdir()))

    with pytest.raises(StoreCapacityError, match="version capacity"):
        attach(store, source=b"second")
    assert len(list((root / "generations").iterdir())) == generation_count
    assert store.manifest()["version_count"] == 1

    generation_limited = SyllabusStore(root, max_versions=10, max_generations=1)
    with pytest.raises(StoreCapacityError, match="generation capacity"):
        attach(generation_limited, section="002/22222", source=b"third")
    assert len(list((root / "generations").iterdir())) == generation_count


def test_store_byte_limit_fails_before_persistence(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = SyllabusStore(root, max_index_bytes=128, max_versions=10)
    with pytest.raises(StoreCapacityError, match="byte capacity"):
        attach(store, source=b"content too large for a 128-byte index")
    assert list((root / "generations").iterdir()) == []


def test_runtime_overlay_is_published_only_and_does_not_mutate_seed(
    tmp_path: Path,
) -> None:
    store = SyllabusStore(tmp_path / "store")
    published = attach(store, section="001/11111", status="published")
    attach(store, section="002/22222", source=b"review source", status="review")
    seed = [
        {
            "course_uid": "seed-1",
            "course_code": "COMS GU4111",
            "title": "Databases",
            "sections_summary": [],
            "all_instructors": [],
            "all_terms": [],
            "searchable_text": "coms gu4111 databases",
            "catalog_validation_status": "review",
            "catalog_validation_warnings": ["missing_description"],
        }
    ]

    runtime = apply_published_overlays(seed, store.effective_overlays())
    assert seed[0]["sections_summary"] == []
    assert len(runtime[0]["sections_summary"]) == 1
    assert runtime[0]["sections_summary"][0]["section_id"] == "001/11111"
    assert runtime[0]["syllabus_overlay_versions"] == [published["version_id"]]
    assert runtime[0]["catalog_validation_status"] == "published"
    assert runtime[0]["catalog_validation_warnings"] == []
    assert runtime[0]["catalog_review_overridden_by_published_overlay"] is True
    assert "002/22222" not in str(runtime)
