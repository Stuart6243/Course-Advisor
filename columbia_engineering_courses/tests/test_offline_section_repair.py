from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = SCRIPT_DIR / "offline_section_repair.py"
spec = importlib.util.spec_from_file_location("offline_section_repair", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_offline_manifest_exact_uid_and_no_writes(tmp_path: Path, capsys) -> None:
    run_id = "20260219_050514"
    year = "2025-2026"
    url = "https://bulletin.columbia.edu/columbia-engineering/test-courses/"
    title = "TEST E9999 INTRODUCTION TO TESTING"
    uid_payload = f"{year}|TEST E9999|INTRODUCTION TO TESTING|/columbia-engineering/test-courses/"
    uid = hashlib.sha1(uid_payload.encode()).hexdigest()

    data_dir = tmp_path / "data"
    raw_dir = tmp_path / "raw_html"
    courses_dir = data_dir / "courses_flat"
    raw_dir.mkdir()
    courses_dir.mkdir(parents=True)
    raw_name = f"{run_id}__{mod.sanitize_filename_from_url(url)}.html"
    (raw_dir / raw_name).write_text(
        f"""
        <html><head><title>Test Courses</title></head><body><main><h1>Testing</h1></main>
        <div id="sc_courseblock"><div class="courseblock">
          <p class="courseblocktitle">{title}. 3.00 points .</p>
          <div class="desc_sched"><table class="scheduletbl">
            <tr><td><strong>Spring 2026: TEST E9999</strong></td></tr>
            <tr><th>COURSE NUMBER</th><th>SECTION/CALL NUMBER</th><th>TIMES/LOCATION</th><th>INSTRUCTOR</th><th>POINTS</th><th>ENROLLMENT</th></tr>
            <tr><td>TEST 9999</td><td>001/12345</td><td></td><td>Ada Lovelace</td><td>3.00</td><td>10/30</td></tr>
          </table></div>
        </div></div></body></html>
        """,
        encoding="utf-8",
    )
    old_course = {
        "course_uid": uid,
        "course_code": "TEST E9999",
        "source_page_url": url,
        "needs_review": True,
        "parse_warnings": ["missing_description"],
        "sections": [
            {
                "term": "Spring 2026",
                "catalog_ref": "TEST E9999",
                "course_number": "TEST 9999",
                "section_call_number": "001/12345",
                "times": "Ada Lovelace",
                "location": "",
                "instructor": "3.00",
                "points": "10/30",
                "enrollment_raw": "",
                "enrollment_current": None,
                "enrollment_capacity": None,
            }
        ],
    }
    course_path = courses_dir / f"{uid}.json"
    course_path.write_text(json.dumps(old_course), encoding="utf-8")
    flat_index = data_dir / "courses_flat_index.json"
    flat_index.write_text(
        json.dumps(
            [
                {
                    "course_uid": uid,
                    "course_code": "TEST E9999",
                    "title": "INTRODUCTION TO TESTING",
                    "file_name": f"{uid}.json",
                }
            ]
        ),
        encoding="utf-8",
    )
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "bulletin_year": year,
                "pages": [{"is_course_page": True, "source_page_url": url}],
            }
        ),
        encoding="utf-8",
    )
    enriched_index = data_dir / "courses_enriched_index.json"
    enriched_index.write_text(
        json.dumps(
            [
                {
                    "course_uid": uid,
                    "course_code": "TEST E9999",
                    "title": "INTRODUCTION TO TESTING",
                    "department_prefix": "TEST",
                    "points_min": 3.0,
                    "points_max": 3.0,
                    "has_description": False,
                    "prerequisites_codes": [],
                    "sections_summary": [
                        {
                            "term": "Spring 2026",
                            "times": "Ada Lovelace",
                            "days": [],
                            "time_of_day": "",
                            "instructor": "3.00",
                            "location": "",
                            "enrollment_current": None,
                            "enrollment_capacity": None,
                        }
                    ],
                    "all_instructors": ["3.00"],
                    "all_terms": ["Spring 2026"],
                    "searchable_text": "test e9999 introduction to testing 3.00",
                }
            ]
        ),
        encoding="utf-8",
    )
    before = {
        path: path.read_bytes()
        for path in (course_path, flat_index, enriched_index, snapshot)
    }

    manifest = mod.build_manifest(
        snapshot_path=snapshot,
        raw_dir=raw_dir,
        flat_index_path=flat_index,
        courses_dir=courses_dir,
        enriched_index_path=enriched_index,
    )

    assert manifest["mode"] == "dry-run"
    assert manifest["network_requests"] == 0
    assert manifest["writes_performed"] == 0
    assert manifest["matching"]["matched_uid_count"] == 1
    assert manifest["repair"]["changed_record_count"] == 1
    assert manifest["repair"]["definite_shifted_sections_before"] == 1
    assert manifest["repair"]["definite_shifted_sections_after"] == 0
    assert manifest["enriched_rebuild"]["changed_record_count"] == 1
    assert manifest["enriched_rebuild"]["current_file_sha256"] != manifest[
        "enriched_rebuild"
    ]["proposed_file_sha256"]
    assert manifest["immutability"]["unchanged"] is True
    before_non_section_hash = manifest["formal_data"][
        "non_section_payload_sha256"
    ]
    assert {path: path.read_bytes() for path in before} == before

    output_path = tmp_path / "dry-run-output.json"
    assert mod.main(
        [
            "--snapshot",
            str(snapshot),
            "--raw-dir",
            str(raw_dir),
            "--flat-index",
            str(flat_index),
            "--courses-dir",
            str(courses_dir),
            "--enriched-index",
            str(enriched_index),
            "--output",
            str(output_path),
        ]
    ) == 0
    assert capsys.readouterr().out == ""
    assert json.loads(output_path.read_text(encoding="utf-8")) == manifest
    assert {path: path.read_bytes() for path in before} == before

    expected_manifest = tmp_path / "approved-before.json"
    expected_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    # A fault after the live directory was renamed to backup must restore the
    # complete old generation, not leave a partially updated tree.
    try:
        mod.apply_repair(
            expected_before_manifest_path=expected_manifest,
            snapshot_path=snapshot,
            raw_dir=raw_dir,
            flat_index_path=flat_index,
            courses_dir=courses_dir,
            enriched_index_path=enriched_index,
            failure_injector="after_backup_rename",
        )
        raise AssertionError("failure injection did not fire")
    except RuntimeError as exc:
        assert "injected failure" in str(exc)
    assert {path: path.read_bytes() for path in before} == before

    apply_output = tmp_path / "apply-output.json"
    assert mod.main(
        [
            "--apply",
            "--expected-before-manifest",
            str(expected_manifest),
            "--snapshot",
            str(snapshot),
            "--raw-dir",
            str(raw_dir),
            "--flat-index",
            str(flat_index),
            "--courses-dir",
            str(courses_dir),
            "--enriched-index",
            str(enriched_index),
            "--output",
            str(apply_output),
        ]
    ) == 0
    assert capsys.readouterr().out == ""
    result = json.loads(apply_output.read_text(encoding="utf-8"))
    assert result["mode"] == "apply"
    assert result["applied_changed_record_count"] == 1
    assert result["post_apply_manifest"]["repair"]["changed_record_count"] == 0
    assert result["post_apply_manifest"]["repair"][
        "definite_shifted_sections_before"
    ] == 0
    backup = Path(result["backup_path"])
    assert backup.is_dir()
    assert backup.name.endswith(".bak")
    assert (backup / "courses_flat" / f"{uid}.json").read_bytes() == before[
        course_path
    ]
    assert flat_index.read_bytes() == before[flat_index]

    repaired = json.loads(course_path.read_text())
    assert {key: value for key, value in repaired.items() if key != "sections"} == {
        key: value for key, value in old_course.items() if key != "sections"
    }
    assert repaired["sections"][0]["times"] == ""
    assert repaired["sections"][0]["instructor"] == "Ada Lovelace"
    assert repaired["sections"][0]["points"] == "3.00"
    assert repaired["sections"][0]["enrollment_raw"] == "10/30"
    enriched = json.loads(enriched_index.read_text())
    assert enriched[0]["sections_summary"][0]["section_id"] == "001/12345"
    assert enriched[0]["all_instructors"] == ["Ada Lovelace"]
    assert enriched[0]["catalog_validation_status"] == "published"
    assert enriched[0]["catalog_validation_warnings"] == ["missing_description"]
    assert result["post_apply_manifest"]["formal_data"][
        "non_section_payload_sha256"
    ] == before_non_section_hash
    assert result["before_fingerprint"]["non_section_payload_sha256"] == (
        result["after_fingerprint"]["non_section_payload_sha256"]
    )

    # A later validator-only migration is explicitly visible in a fresh
    # manifest and can be applied with zero section proposals.  This mirrors
    # the guarded post-repair catalog-status correction on formal data.
    enriched[0]["catalog_validation_status"] = "review"
    mod._write_json(enriched_index, enriched)
    derived_manifest = mod.build_manifest(
        snapshot_path=snapshot,
        raw_dir=raw_dir,
        flat_index_path=flat_index,
        courses_dir=courses_dir,
        enriched_index_path=enriched_index,
    )
    assert derived_manifest["repair"]["changed_record_count"] == 0
    assert derived_manifest["repair"]["definite_shifted_sections_before"] == 0
    assert derived_manifest["enriched_rebuild"]["changed_record_count"] == 1
    derived_approval = tmp_path / "derived-before.json"
    derived_approval.write_text(json.dumps(derived_manifest), encoding="utf-8")
    derived_result = mod.apply_repair(
        expected_before_manifest_path=derived_approval,
        snapshot_path=snapshot,
        raw_dir=raw_dir,
        flat_index_path=flat_index,
        courses_dir=courses_dir,
        enriched_index_path=enriched_index,
    )
    assert derived_result["applied_changed_record_count"] == 0
    canonical_enriched = json.loads(enriched_index.read_text(encoding="utf-8"))
    assert canonical_enriched[0]["catalog_validation_status"] == "published"
    assert derived_result["post_apply_manifest"]["enriched_rebuild"][
        "changed_record_count"
    ] == 0


def test_rebuild_enriched_derived_keeps_only_published_sections_searchable() -> None:
    course = {
        "course_uid": "seed-1",
        "course_code": "COMS GU4111",
        "title": "DATABASE SYSTEMS",
        "description": "Relational database design.",
        "source_page_url": "https://example.test/coms/",
        "sections": [
            {
                "term": "Spring 2026",
                "section_call_number": "001/11111",
                "times": "TBA",
                "location": "Savannah Hall",
                "instructor": "Savannah Smith",
                "points": "3.00",
                "enrollment_raw": "10/30",
                "enrollment_current": 10,
                "enrollment_capacity": 30,
            },
            {
                "term": "Fall 2026",
                "section_call_number": "002/22222",
                "times": "M 10:00am - 11:00am",
                "location": "Room 2",
                "instructor": "Review Instructor",
                "points": "3.00",
                "enrollment_raw": "31/30",
                "enrollment_current": 31,
                "enrollment_capacity": 30,
            },
            {
                "term": "Winter 2026",
                "section_call_number": "003/33333",
                "times": "Shifted Professor",
                "location": "Room 3",
                "instructor": "3.00",
                "points": "10/30",
                "enrollment_raw": "",
                "enrollment_current": None,
                "enrollment_capacity": None,
            },
        ],
    }
    old_entry = {
        "course_uid": "seed-1",
        "course_code": "COMS GU4111",
        "title": "DATABASE SYSTEMS",
        "department_prefix": "COMS",
        "prerequisites_codes": [],
        "sections_summary": [{"term": "stale"}],
        "review_sections_summary": [],
        "all_instructors": ["stale"],
        "all_terms": ["stale"],
        "searchable_text": "stale shifted professor review instructor",
    }

    rebuilt = mod._rebuild_enriched_derived([old_entry], {"seed-1": course})[0]

    assert [row["section_id"] for row in rebuilt["sections_summary"]] == [
        "001/11111"
    ]
    assert rebuilt["sections_summary"][0]["days"] == []
    assert rebuilt["sections_summary"][0]["validation_status"] == "published"
    assert rebuilt["sections_summary"][0]["provenance"]["source"] == "catalog_seed"
    assert [row["section_id"] for row in rebuilt["review_sections_summary"]] == [
        "002/22222",
        "003/33333",
    ]
    warning_row, invalid_row = rebuilt["review_sections_summary"]
    assert warning_row["validation_warnings"] == ["enrollment_exceeds_capacity"]
    assert "invalid_points" in invalid_row["validation_errors"]
    assert rebuilt["all_instructors"] == ["Savannah Smith"]
    assert rebuilt["all_terms"] == ["Spring 2026"]
    assert "review instructor" not in rebuilt["searchable_text"]
    assert "shifted professor" not in rebuilt["searchable_text"]


def test_apply_layout_rejects_data_and_courses_symlinks(tmp_path: Path) -> None:
    real_data = tmp_path / "real-data"
    real_courses = real_data / "courses_flat"
    real_courses.mkdir(parents=True)
    flat = real_data / "courses_flat_index.json"
    enriched = real_data / "courses_enriched_index.json"
    flat.write_text("[]", encoding="utf-8")
    enriched.write_text("[]", encoding="utf-8")

    data_link = tmp_path / "data-link"
    data_link.symlink_to(real_data, target_is_directory=True)
    try:
        mod._assert_apply_layout(
            data_link / flat.name,
            data_link / real_courses.name,
            data_link / enriched.name,
        )
        raise AssertionError("data-directory symlink was accepted")
    except ValueError as exc:
        assert "symlink" in str(exc).lower()

    other_courses = tmp_path / "other-courses"
    other_courses.mkdir()
    linked_data = tmp_path / "linked-data"
    linked_data.mkdir()
    linked_courses = linked_data / "courses_flat"
    linked_courses.symlink_to(other_courses, target_is_directory=True)
    linked_flat = linked_data / flat.name
    linked_enriched = linked_data / enriched.name
    linked_flat.write_text("[]", encoding="utf-8")
    linked_enriched.write_text("[]", encoding="utf-8")
    try:
        mod._assert_apply_layout(linked_flat, linked_courses, linked_enriched)
        raise AssertionError("courses-directory symlink was accepted")
    except ValueError as exc:
        assert "symlink" in str(exc).lower()
