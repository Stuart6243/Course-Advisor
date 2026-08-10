#!/usr/bin/env python3
"""Read-only manifest for repairing shifted schedule columns from saved HTML.

By default this command is a read-only dry run.  It reparses the selected local
raw-HTML batch, matches catalog records by exact ``course_uid``, and reports the
proposed section replacements as JSON.  The explicit ``--apply`` path requires
an exact saved before-manifest and commits a fully validated staged data-tree
generation while retaining the prior generation as a rollback backup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlparse

try:
    import fcntl
except ModuleNotFoundError:  # pragma: no cover - production is macOS/Linux
    fcntl = None  # type: ignore[assignment]

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from course_index import build_searchable_text, parse_days_from_times
from backend.section_validator import validate_catalog_record, validate_section
from scrape_columbia_courses import parse_course_page, sanitize_filename_from_url


DEFAULT_RUN_ID = "20260219_050514"
DEFAULT_SNAPSHOT = SCRIPT_DIR / "snapshots" / f"courses_{DEFAULT_RUN_ID}.json"
DEFAULT_RAW_DIR = SCRIPT_DIR / "raw_html"
DEFAULT_FLAT_INDEX = REPO_ROOT / "data" / "courses_flat_index.json"
DEFAULT_COURSES_DIR = REPO_ROOT / "data" / "courses_flat"
DEFAULT_ENRICHED_INDEX = REPO_ROOT / "data" / "courses_enriched_index.json"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str | None:
    try:
        return _sha256_bytes(path.read_bytes())
    except FileNotFoundError:
        return None


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _json_file_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")


def _non_section_payload_sha256(records: list[dict[str, Any]]) -> str:
    """Fingerprint every formal record while deliberately excluding sections."""

    payloads = [
        {
            "course_uid": str(course.get("course_uid") or ""),
            "payload": {
                field: value
                for field, value in course.items()
                if field != "sections"
            },
        }
        for course in records
    ]
    payloads.sort(
        key=lambda item: (
            item["course_uid"],
            _canonical_digest(item["payload"]),
        )
    )
    return _canonical_digest(payloads)


def _normalized_code(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().upper()


def _url_path(value: Any) -> str:
    return (urlparse(str(value or "")).path or "").rstrip("/").lower()


def _definitely_shifted(section: Any) -> bool:
    """The high-confidence signature used by the handoff baseline.

    A points cell that cannot be parsed after an old empty-cell-collapsing pass
    is the decisive signal.  This counts 2,019 saved sections; the fixed parse
    of the same raw pages counts zero.
    """

    if not isinstance(section, dict):
        return False
    return "invalid_points" in validate_section(section).errors


def _load_formal_records(flat_index_path: Path, courses_dir: Path) -> tuple[list[dict], list[str]]:
    raw_index = json.loads(flat_index_path.read_text(encoding="utf-8"))
    if not isinstance(raw_index, list):
        raise ValueError("Flat index must be a JSON array")
    records: list[dict] = []
    integrity_errors: list[str] = []
    for position, item in enumerate(raw_index):
        if not isinstance(item, dict):
            integrity_errors.append(f"flat_index_{position}:not_object")
            continue
        uid = str(item.get("course_uid") or "").strip()
        filename = str(item.get("file_name") or "").strip()
        if not re.fullmatch(r"[0-9a-f]{40}", uid):
            integrity_errors.append(f"flat_index_{position}:invalid_uid")
            continue
        if filename != f"{uid}.json":
            integrity_errors.append(f"flat_index_{position}:filename_uid_mismatch")
            continue
        course_path = courses_dir / filename
        try:
            course = json.loads(course_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            integrity_errors.append(f"{uid}:unreadable:{type(exc).__name__}")
            continue
        if not isinstance(course, dict) or course.get("course_uid") != uid:
            integrity_errors.append(f"{uid}:course_uid_mismatch")
            continue
        records.append(course)
    return records, integrity_errors


def _load_reparsed_records(
    snapshot_path: Path, raw_dir: Path, run_id: str | None
) -> tuple[dict[str, dict], list[dict], dict]:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("pages"), list):
        raise ValueError("Snapshot must contain a pages array")
    selected_run = run_id or str(snapshot.get("run_id") or "").strip()
    if not re.fullmatch(r"\d{8}_\d{6}", selected_run):
        raise ValueError("run_id must have YYYYMMDD_HHMMSS format")
    year = str(snapshot.get("bulletin_year") or "").strip()
    if not year:
        raise ValueError("Snapshot is missing bulletin_year")

    records: dict[str, dict] = {}
    raw_manifest: list[dict] = []
    duplicate_uids: list[str] = []
    course_page_count = 0
    for page in snapshot["pages"]:
        if not isinstance(page, dict) or not page.get("is_course_page"):
            continue
        url = str(page.get("source_page_url") or "").strip()
        if not url:
            raise ValueError("Course page is missing source_page_url")
        raw_path = raw_dir / f"{selected_run}__{sanitize_filename_from_url(url)}.html"
        raw_bytes = raw_path.read_bytes()
        courses, _review, metadata = parse_course_page(
            raw_bytes.decode("utf-8"), url, year
        )
        course_page_count += 1
        raw_manifest.append(
            {
                "file": raw_path.name,
                "source_page_url": url,
                "sha256": _sha256_bytes(raw_bytes),
                "parsed_course_count": len(courses),
                "reported_course_blocks": metadata.get("course_blocks", 0),
            }
        )
        for course in courses:
            uid = str(course.get("course_uid") or "").strip()
            if uid in records:
                duplicate_uids.append(uid)
            records[uid] = course
    return records, raw_manifest, {
        "run_id": selected_run,
        "bulletin_year": year,
        "course_page_count": course_page_count,
        "duplicate_uids": sorted(set(duplicate_uids)),
    }


def build_manifest(
    *,
    snapshot_path: Path,
    raw_dir: Path,
    flat_index_path: Path,
    courses_dir: Path,
    enriched_index_path: Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic repair proposal without writing any input path."""

    before_hashes = {
        "flat_index_sha256": _sha256_file(flat_index_path),
        "enriched_index_sha256": (
            _sha256_file(enriched_index_path) if enriched_index_path else None
        ),
    }
    formal, integrity_errors = _load_formal_records(flat_index_path, courses_dir)
    reparsed, raw_manifest, source = _load_reparsed_records(
        snapshot_path, raw_dir, run_id
    )

    formal_by_uid: dict[str, dict] = {}
    duplicate_formal_uids: list[str] = []
    for course in formal:
        uid = course["course_uid"]
        if uid in formal_by_uid:
            duplicate_formal_uids.append(uid)
        formal_by_uid[uid] = course

    proposals: list[dict[str, Any]] = []
    identity_mismatches: list[dict[str, str]] = []
    unmatched_formal: list[dict[str, str]] = []
    matched_uids: set[str] = set()
    old_shifted = 0
    new_shifted = 0
    formal_section_count = 0
    proposed_section_count = 0

    for uid, old in formal_by_uid.items():
        old_sections = old.get("sections") if isinstance(old.get("sections"), list) else []
        formal_section_count += len(old_sections)
        old_shifted += sum(_definitely_shifted(section) for section in old_sections)
        candidate = reparsed.get(uid)
        if candidate is None:
            unmatched_formal.append(
                {"course_uid": uid, "course_code": str(old.get("course_code") or "")}
            )
            proposed_section_count += len(old_sections)
            continue

        mismatched_fields: list[str] = []
        if _normalized_code(old.get("course_code")) != _normalized_code(
            candidate.get("course_code")
        ):
            mismatched_fields.append("course_code")
        if _url_path(old.get("source_page_url")) != _url_path(
            candidate.get("source_page_url")
        ):
            mismatched_fields.append("source_page_url")
        if mismatched_fields:
            identity_mismatches.append(
                {"course_uid": uid, "fields": ",".join(mismatched_fields)}
            )
            proposed_section_count += len(old_sections)
            continue

        matched_uids.add(uid)
        new_sections = (
            candidate.get("sections")
            if isinstance(candidate.get("sections"), list)
            else []
        )
        proposed_section_count += len(new_sections)
        new_shifted += sum(_definitely_shifted(section) for section in new_sections)
        if old_sections != new_sections:
            proposals.append(
                {
                    "course_uid": uid,
                    "course_code": str(old.get("course_code") or ""),
                    "old_sections_sha256": _canonical_digest(old_sections),
                    "new_sections_sha256": _canonical_digest(new_sections),
                    "section_count_before": len(old_sections),
                    "section_count_after": len(new_sections),
                    "definite_shifted_before": sum(
                        _definitely_shifted(section) for section in old_sections
                    ),
                    "definite_shifted_after": sum(
                        _definitely_shifted(section) for section in new_sections
                    ),
                }
            )

    extra_reparsed = sorted(set(reparsed) - set(formal_by_uid))
    code_counts = Counter(_normalized_code(course.get("course_code")) for course in formal)
    duplicate_histogram = sorted(
        (code, count) for code, count in code_counts.items() if code and count > 1
    )
    formal_tree = sorted(
        (course["course_uid"], _sha256_file(courses_dir / f"{course['course_uid']}.json"))
        for course in formal
    )
    enriched_rebuild: dict[str, Any] = {"enabled": False}
    if enriched_index_path is not None:
        current_enriched = json.loads(
            enriched_index_path.read_text(encoding="utf-8")
        )
        if not isinstance(current_enriched, list):
            raise ValueError("Enriched index must be an array")
        proposed_courses = {
            uid: dict(course) for uid, course in formal_by_uid.items()
        }
        for proposal in proposals:
            uid = proposal["course_uid"]
            proposed_courses[uid]["sections"] = reparsed[uid].get("sections") or []
        proposed_enriched = _rebuild_enriched_derived(
            current_enriched, proposed_courses
        )
        changed_enriched_uids = [
            str(old.get("course_uid") or "")
            for old, new in zip(current_enriched, proposed_enriched)
            if any(old.get(field) != new.get(field) for field in DERIVED_ENRICHED_FIELDS)
        ]
        enriched_rebuild = {
            "enabled": True,
            "fields_allowed_to_change": sorted(DERIVED_ENRICHED_FIELDS),
            "record_count_before": len(current_enriched),
            "record_count_after": len(proposed_enriched),
            "changed_record_count": len(changed_enriched_uids),
            "changed_uid_sha256": _canonical_digest(changed_enriched_uids),
            "current_file_sha256": _sha256_file(enriched_index_path),
            "proposed_file_sha256": _sha256_bytes(
                _json_file_bytes(proposed_enriched)
            ),
        }
    after_hashes = {
        "flat_index_sha256": _sha256_file(flat_index_path),
        "enriched_index_sha256": (
            _sha256_file(enriched_index_path) if enriched_index_path else None
        ),
    }

    return {
        "schema_version": 1,
        "mode": "dry-run",
        "network_requests": 0,
        "writes_performed": 0,
        "match_policy": "exact_course_uid_with_code_and_source_path_guard",
        "apply_contract": {
            "course_fields_allowed_to_change": ["sections"],
            "enriched_fields_allowed_to_change": [
                "all_instructors",
                "all_terms",
                "catalog_validation_status",
                "catalog_validation_warnings",
                "review_sections_summary",
                "searchable_text",
                "sections_summary",
            ],
            "flat_index_must_remain_byte_identical": True,
            "record_uid_section_and_duplicate_counts_must_remain_equal": True,
        },
        "source": {
            **source,
            "snapshot": str(snapshot_path.resolve()),
            "snapshot_sha256": _sha256_file(snapshot_path),
            "raw_dir": str(raw_dir.resolve()),
            "raw_file_count": len(raw_manifest),
            "raw_aggregate_sha256": _canonical_digest(raw_manifest),
            "raw_files": raw_manifest,
            "reparsed_course_count": len(reparsed),
        },
        "formal_data": {
            "flat_index": str(flat_index_path.resolve()),
            "courses_dir": str(courses_dir.resolve()),
            "course_record_count": len(formal),
            "unique_uid_count": len(formal_by_uid),
            "unique_course_code_count": len(code_counts),
            "duplicate_course_code_group_count": len(duplicate_histogram),
            "duplicate_record_excess_count": sum(
                count - 1 for _code, count in duplicate_histogram
            ),
            "duplicate_histogram_sha256": _canonical_digest(duplicate_histogram),
            "uid_set_sha256": _canonical_digest(sorted(formal_by_uid)),
            "non_section_payload_sha256": _non_section_payload_sha256(formal),
            "course_tree_sha256": _canonical_digest(formal_tree),
            "integrity_errors": integrity_errors,
            "duplicate_uids": sorted(set(duplicate_formal_uids)),
        },
        "matching": {
            "matched_uid_count": len(matched_uids),
            "unmatched_formal_count": len(unmatched_formal),
            "unmatched_formal": unmatched_formal,
            "extra_reparsed_uid_count": len(extra_reparsed),
            "extra_reparsed_uids": extra_reparsed,
            "identity_mismatch_count": len(identity_mismatches),
            "identity_mismatches": identity_mismatches,
        },
        "repair": {
            "changed_record_count": len(proposals),
            "changed_uid_sha256": _canonical_digest(
                sorted(item["course_uid"] for item in proposals)
            ),
            "formal_section_count_before": formal_section_count,
            "proposed_section_count_after": proposed_section_count,
            "definite_shifted_sections_before": old_shifted,
            "definite_shifted_sections_after": new_shifted,
            "proposals": proposals,
        },
        "enriched_rebuild": enriched_rebuild,
        "immutability": {
            "hashes_before": before_hashes,
            "hashes_after": after_hashes,
            "unchanged": before_hashes == after_hashes,
        },
    }


DERIVED_ENRICHED_FIELDS = frozenset(
    {
        "sections_summary",
        "review_sections_summary",
        "all_instructors",
        "all_terms",
        "catalog_validation_status",
        "catalog_validation_warnings",
        "searchable_text",
    }
)


def _inject_failure(
    injector: Callable[[str], None] | str | set[str] | None, phase: str
) -> None:
    if callable(injector):
        injector(phase)
    elif injector == phase or isinstance(injector, set) and phase in injector:
        raise RuntimeError(f"injected failure: {phase}")


def _assert_apply_layout(
    flat_index_path: Path, courses_dir: Path, enriched_index_path: Path
) -> Path:
    raw_data_dir = courses_dir.parent
    for path in (raw_data_dir, courses_dir, flat_index_path, enriched_index_path):
        if path.is_symlink():
            raise ValueError(f"Refusing symlink in apply layout: {path}")
    data_dir = courses_dir.resolve().parent
    if not data_dir.is_dir() or courses_dir.resolve().parent != data_dir:
        raise ValueError("courses_dir must be an existing directory")
    if flat_index_path.resolve().parent != data_dir:
        raise ValueError("flat index and courses_dir must share one data directory")
    if enriched_index_path.resolve().parent != data_dir:
        raise ValueError("enriched index and courses_dir must share one data directory")
    if courses_dir.resolve() != data_dir / courses_dir.name:
        raise ValueError("courses_dir must be a direct child of the data directory")
    for path in data_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Refusing data tree containing symlink: {path}")
    return data_dir


def _safe_remove_generation(path: Path, parent: Path, kind: str) -> None:
    """Remove only a UUID-named private staging/failed generation."""

    if not path.exists():
        return
    expected_prefix = f".{parent.name}.{kind}-"
    if (
        path.parent.resolve() != parent.parent.resolve()
        or not path.name.startswith(expected_prefix)
        or path.is_symlink()
        or not path.is_dir()
    ):
        raise ValueError(f"Refusing to remove unexpected generation path: {path}")
    shutil.rmtree(path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for directory in sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _fsync_directory(directory)
    _fsync_directory(root)


@contextmanager
def _repair_lock(data_dir: Path) -> Iterator[None]:
    lock_path = data_dir.parent / f".{data_dir.name}.section-repair.lock"
    with lock_path.open("a+b") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _without_key(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {field: item for field, item in value.items() if field != key}


def _without_enriched_derived(value: dict[str, Any]) -> dict[str, Any]:
    return {
        field: item
        for field, item in value.items()
        if field not in DERIVED_ENRICHED_FIELDS
    }


def _section_summary(
    section: dict[str, Any], course: dict[str, Any]
) -> dict[str, Any]:
    validation = validate_section(section)
    times = validation.normalized["times"]
    days, time_of_day = (
        parse_days_from_times(times) if not validation.errors else ([], "")
    )
    return {
        "section_id": validation.normalized["section_call_number"],
        "term": validation.normalized["term"],
        "times": times,
        "days": days,
        "time_of_day": time_of_day,
        "instructor": validation.normalized["instructor"],
        "location": validation.normalized["location"],
        "points": validation.normalized["points"],
        "enrollment_current": section.get("enrollment_current"),
        "enrollment_capacity": section.get("enrollment_capacity"),
        "validation_status": validation.status,
        "validation_errors": list(validation.errors),
        "validation_warnings": list(validation.warnings),
        "provenance": {
            "source": "catalog_seed",
            "course_uid": course.get("course_uid", ""),
            "source_page_url": course.get("source_page_url", ""),
        },
    }


def _rebuild_enriched_derived(
    enriched: list[dict[str, Any]], courses_by_uid: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rebuilt: list[dict[str, Any]] = []
    for position, old_entry in enumerate(enriched):
        if not isinstance(old_entry, dict):
            raise ValueError(f"enriched index row {position} is not an object")
        uid = str(old_entry.get("course_uid") or "").strip()
        course = courses_by_uid.get(uid)
        if course is None:
            raise ValueError(f"enriched course UID missing from flat data: {uid}")
        entry = dict(old_entry)
        sections = course.get("sections") if isinstance(course.get("sections"), list) else []
        all_summaries = [
            _section_summary(section, course)
            for section in sections
            if isinstance(section, dict)
        ]
        summaries = [
            summary
            for summary in all_summaries
            if summary["validation_status"] == "published"
        ]
        review_summaries = [
            summary
            for summary in all_summaries
            if summary["validation_status"] != "published"
        ]
        entry["sections_summary"] = summaries
        entry["review_sections_summary"] = review_summaries
        entry["all_instructors"] = list(
            dict.fromkeys(
                summary["instructor"]
                for summary in summaries
                if summary["instructor"]
            )
        )
        entry["all_terms"] = list(
            dict.fromkeys(summary["term"] for summary in summaries if summary["term"])
        )
        catalog_validation = validate_catalog_record(course)
        entry["catalog_validation_status"] = catalog_validation.status
        entry["catalog_validation_warnings"] = list(catalog_validation.warnings)
        entry["searchable_text"] = build_searchable_text(course, entry)
        rebuilt.append(entry)
    return rebuilt


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validated_manifest_output_path(path: Path, data_dir: Path) -> Path:
    if path.is_symlink():
        raise ValueError("Manifest output must not be a symlink")
    output = path.resolve()
    formal_root = data_dir.resolve()
    if output == formal_root or formal_root in output.parents:
        raise ValueError("Manifest output must be outside the formal data directory")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("Manifest output parent must be an existing real directory")
    return output


def _write_manifest_output(path: Path, manifest: dict[str, Any], data_dir: Path) -> None:
    """Atomically write a report outside the formal data generation."""

    output = _validated_manifest_output_path(path, data_dir)
    temp = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temp.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, output)
        _fsync_directory(output.parent)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _data_fingerprint(
    flat_index_path: Path, courses_dir: Path, enriched_index_path: Path
) -> dict[str, Any]:
    records, integrity_errors = _load_formal_records(flat_index_path, courses_dir)
    if integrity_errors:
        raise ValueError(f"Cannot fingerprint invalid formal data: {integrity_errors}")
    raw_index = json.loads(flat_index_path.read_text(encoding="utf-8"))
    tree: list[tuple[str, str | None]] = []
    for item in raw_index:
        uid = str(item.get("course_uid") or "")
        tree.append((uid, _sha256_file(courses_dir / f"{uid}.json")))
    return {
        "flat_index_sha256": _sha256_file(flat_index_path),
        "enriched_index_sha256": _sha256_file(enriched_index_path),
        "course_tree_sha256": _canonical_digest(sorted(tree)),
        "non_section_payload_sha256": _non_section_payload_sha256(records),
    }


def _prepare_staged_generation(
    *,
    data_dir: Path,
    staging_dir: Path,
    current_manifest: dict[str, Any],
    snapshot_path: Path,
    raw_dir: Path,
    run_id: str | None,
    flat_index_name: str,
    courses_dir_name: str,
    enriched_index_name: str,
) -> dict[str, Any]:
    shutil.copytree(data_dir, staging_dir, copy_function=shutil.copy2)
    staged_flat = staging_dir / flat_index_name
    staged_courses = staging_dir / courses_dir_name
    staged_enriched = staging_dir / enriched_index_name

    old_records, old_integrity = _load_formal_records(
        data_dir / flat_index_name, data_dir / courses_dir_name
    )
    if old_integrity:
        raise ValueError(f"Cannot apply with formal integrity errors: {old_integrity}")
    old_by_uid = {course["course_uid"]: course for course in old_records}
    reparsed, _raw_manifest, _source = _load_reparsed_records(
        snapshot_path, raw_dir, run_id
    )

    proposal_uids = {
        item["course_uid"] for item in current_manifest["repair"]["proposals"]
    }
    for proposal in current_manifest["repair"]["proposals"]:
        uid = proposal["course_uid"]
        candidate = reparsed.get(uid)
        if candidate is None:
            raise ValueError(f"Proposed UID missing from reparsed source: {uid}")
        staged_path = staged_courses / f"{uid}.json"
        staged_course = json.loads(staged_path.read_text(encoding="utf-8"))
        if _canonical_digest(staged_course.get("sections") or []) != proposal[
            "old_sections_sha256"
        ]:
            raise ValueError(f"Old section hash changed for {uid}")
        new_sections = candidate.get("sections") or []
        if _canonical_digest(new_sections) != proposal["new_sections_sha256"]:
            raise ValueError(f"Reparsed section hash changed for {uid}")
        staged_course["sections"] = new_sections
        _write_json(staged_path, staged_course)

    staged_records, staged_integrity = _load_formal_records(staged_flat, staged_courses)
    if staged_integrity:
        raise ValueError(f"Staged formal integrity errors: {staged_integrity}")
    staged_by_uid = {course["course_uid"]: course for course in staged_records}
    changed_uids = {
        uid
        for uid in old_by_uid
        if old_by_uid[uid].get("sections") != staged_by_uid[uid].get("sections")
    }
    if changed_uids != proposal_uids:
        raise ValueError("Staged changed-UID set does not equal approved proposal")
    if len(old_records) != len(staged_records):
        raise ValueError("Formal record count changed")
    if set(old_by_uid) != set(staged_by_uid):
        raise ValueError("Formal UID set changed")
    if sum(len(course.get("sections") or []) for course in old_records) != sum(
        len(course.get("sections") or []) for course in staged_records
    ):
        raise ValueError("Formal section count changed")
    for uid, old in old_by_uid.items():
        if _without_key(old, "sections") != _without_key(staged_by_uid[uid], "sections"):
            raise ValueError(f"Non-section course data changed for {uid}")
    if sum(
        _definitely_shifted(section)
        for course in staged_records
        for section in (course.get("sections") or [])
    ) != current_manifest["repair"]["definite_shifted_sections_after"]:
        raise ValueError("Staged shifted-section count differs from approved manifest")

    old_enriched = json.loads((data_dir / enriched_index_name).read_text(encoding="utf-8"))
    if not isinstance(old_enriched, list):
        raise ValueError("Enriched index must be an array")
    new_enriched = _rebuild_enriched_derived(old_enriched, staged_by_uid)
    if len(old_enriched) != len(new_enriched):
        raise ValueError("Enriched record count changed")
    if [entry.get("course_uid") for entry in old_enriched] != [
        entry.get("course_uid") for entry in new_enriched
    ]:
        raise ValueError("Enriched UID order changed")
    for old, new in zip(old_enriched, new_enriched):
        if _without_enriched_derived(old) != _without_enriched_derived(new):
            raise ValueError("Non-derived enriched data changed")
    _write_json(staged_enriched, new_enriched)

    if staged_flat.read_bytes() != (data_dir / flat_index_name).read_bytes():
        raise ValueError("Flat index bytes changed")
    staged_manifest = build_manifest(
        snapshot_path=snapshot_path,
        raw_dir=raw_dir,
        flat_index_path=staged_flat,
        courses_dir=staged_courses,
        enriched_index_path=staged_enriched,
        run_id=run_id,
    )
    if staged_manifest["repair"]["changed_record_count"] != 0:
        raise ValueError("Staged data still differs from approved raw HTML")
    if staged_manifest["repair"]["definite_shifted_sections_before"] != 0:
        raise ValueError("Staged data still contains definitely shifted sections")
    if _sha256_file(staged_enriched) != current_manifest["enriched_rebuild"][
        "proposed_file_sha256"
    ]:
        raise ValueError("Staged enriched index differs from approved rebuild")
    if staged_manifest["enriched_rebuild"]["changed_record_count"] != 0:
        raise ValueError("Staged enriched derived fields are not canonical")
    if staged_manifest["formal_data"]["course_record_count"] != current_manifest[
        "formal_data"
    ]["course_record_count"]:
        raise ValueError("Staged formal record count changed")
    if staged_manifest["formal_data"]["duplicate_histogram_sha256"] != current_manifest[
        "formal_data"
    ]["duplicate_histogram_sha256"]:
        raise ValueError("Duplicate-course histogram changed")
    if staged_manifest["formal_data"]["non_section_payload_sha256"] != current_manifest[
        "formal_data"
    ]["non_section_payload_sha256"]:
        raise ValueError("Non-section formal payload changed")
    _fsync_tree(staging_dir)
    return staged_manifest


def apply_repair(
    *,
    expected_before_manifest_path: Path,
    snapshot_path: Path,
    raw_dir: Path,
    flat_index_path: Path,
    courses_dir: Path,
    enriched_index_path: Path,
    run_id: str | None = None,
    failure_injector: Callable[[str], None] | str | set[str] | None = None,
) -> dict[str, Any]:
    """Apply an explicitly approved manifest with staging, backup, and rollback."""

    data_dir = _assert_apply_layout(flat_index_path, courses_dir, enriched_index_path)
    expected = json.loads(expected_before_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(expected, dict) or expected.get("mode") != "dry-run":
        raise ValueError("Expected-before manifest must be a dry-run manifest")
    if not expected.get("immutability", {}).get("unchanged"):
        raise ValueError("Expected-before manifest did not prove input immutability")

    staging_dir = data_dir.parent / (
        f".{data_dir.name}.section-repair-stage-{uuid.uuid4().hex}"
    )
    backup_dir = data_dir.parent / (
        f".{data_dir.name}.section-repair-backup-{time.time_ns()}-"
        f"{uuid.uuid4().hex[:8]}.bak"
    )
    failed_dir = data_dir.parent / (
        f".{data_dir.name}.section-repair-failed-{uuid.uuid4().hex}"
    )

    with _repair_lock(data_dir):
        current = build_manifest(
            snapshot_path=snapshot_path,
            raw_dir=raw_dir,
            flat_index_path=flat_index_path,
            courses_dir=courses_dir,
            enriched_index_path=enriched_index_path,
            run_id=run_id,
        )
        if _canonical_digest(current) != _canonical_digest(expected):
            raise ValueError("Current dry-run manifest does not match approved manifest")
        before_fingerprint = _data_fingerprint(
            flat_index_path, courses_dir, enriched_index_path
        )
        expected_fingerprint = {
            "flat_index_sha256": current["immutability"]["hashes_before"][
                "flat_index_sha256"
            ],
            "enriched_index_sha256": current["immutability"]["hashes_before"][
                "enriched_index_sha256"
            ],
            "course_tree_sha256": current["formal_data"]["course_tree_sha256"],
            "non_section_payload_sha256": current["formal_data"][
                "non_section_payload_sha256"
            ],
        }
        if before_fingerprint != expected_fingerprint:
            raise ValueError("Live data fingerprint differs from approved manifest")

        committed = False
        backed_up = False
        post_apply_manifest: dict[str, Any] | None = None
        try:
            staged_manifest = _prepare_staged_generation(
                data_dir=data_dir,
                staging_dir=staging_dir,
                current_manifest=current,
                snapshot_path=snapshot_path,
                raw_dir=raw_dir,
                run_id=run_id,
                flat_index_name=flat_index_path.name,
                courses_dir_name=courses_dir.name,
                enriched_index_name=enriched_index_path.name,
            )
            _inject_failure(failure_injector, "after_staging")
            if _data_fingerprint(
                flat_index_path, courses_dir, enriched_index_path
            ) != before_fingerprint:
                raise ValueError("Live data changed while staging")

            os.replace(data_dir, backup_dir)
            backed_up = True
            _fsync_directory(data_dir.parent)
            _inject_failure(failure_injector, "after_backup_rename")
            _inject_failure(failure_injector, "before_commit")
            os.replace(staging_dir, data_dir)
            committed = True
            _fsync_directory(data_dir.parent)
            post_apply_manifest = build_manifest(
                snapshot_path=snapshot_path,
                raw_dir=raw_dir,
                flat_index_path=data_dir / flat_index_path.name,
                courses_dir=data_dir / courses_dir.name,
                enriched_index_path=data_dir / enriched_index_path.name,
                run_id=run_id,
            )
            if post_apply_manifest["repair"]["changed_record_count"] != 0:
                raise RuntimeError("Committed generation failed post-apply verification")
            _inject_failure(failure_injector, "after_commit")
        except Exception:
            if committed:
                os.replace(data_dir, failed_dir)
                os.replace(backup_dir, data_dir)
                _fsync_directory(data_dir.parent)
                _safe_remove_generation(failed_dir, data_dir, "section-repair-failed")
            elif backed_up:
                os.replace(backup_dir, data_dir)
                _fsync_directory(data_dir.parent)
            _safe_remove_generation(staging_dir, data_dir, "section-repair-stage")
            raise

    final_flat = data_dir / flat_index_path.name
    final_courses = data_dir / courses_dir.name
    final_enriched = data_dir / enriched_index_path.name
    if post_apply_manifest is None:  # pragma: no cover - guarded by commit path
        raise RuntimeError("Missing post-apply verification manifest")
    return {
        "schema_version": 1,
        "mode": "apply",
        "network_requests": 0,
        "writes_performed": current["repair"]["changed_record_count"] + 1,
        "commit": "atomic_data_directory_generation_swap",
        "rollback": "validated_backup_retained",
        "backup_path": str(backup_dir.resolve()),
        "approved_before_manifest_sha256": _canonical_digest(expected),
        "applied_changed_record_count": current["repair"]["changed_record_count"],
        "applied_shifted_sections": current["repair"][
            "definite_shifted_sections_before"
        ],
        "before_fingerprint": before_fingerprint,
        "after_fingerprint": _data_fingerprint(
            final_flat, final_courses, final_enriched
        ),
        "post_apply_manifest": post_apply_manifest,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline section repair (read-only unless --apply is explicit)."
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--flat-index", type=Path, default=DEFAULT_FLAT_INDEX)
    parser.add_argument("--courses-dir", type=Path, default=DEFAULT_COURSES_DIR)
    parser.add_argument(
        "--enriched-index", type=Path, default=DEFAULT_ENRICHED_INDEX
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply only after verifying an exact previously saved dry-run manifest.",
    )
    parser.add_argument(
        "--expected-before-manifest",
        type=Path,
        help="Required with --apply; must exactly match a fresh dry run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Atomically write the resulting manifest outside formal data (else stdout).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Reject an unsafe report destination before an explicit apply can mutate
    # the formal data generation.
    if args.output is not None:
        _validated_manifest_output_path(args.output, args.courses_dir.parent)
    if args.apply:
        if args.expected_before_manifest is None:
            raise SystemExit("--apply requires --expected-before-manifest")
        manifest = apply_repair(
            expected_before_manifest_path=args.expected_before_manifest,
            snapshot_path=args.snapshot,
            raw_dir=args.raw_dir,
            flat_index_path=args.flat_index,
            courses_dir=args.courses_dir,
            enriched_index_path=args.enriched_index,
            run_id=args.run_id,
        )
    else:
        if args.expected_before_manifest is not None:
            raise SystemExit("--expected-before-manifest is only valid with --apply")
        manifest = build_manifest(
            snapshot_path=args.snapshot,
            raw_dir=args.raw_dir,
            flat_index_path=args.flat_index,
            courses_dir=args.courses_dir,
            enriched_index_path=args.enriched_index,
            run_id=args.run_id,
        )
    if args.output is not None:
        _write_manifest_output(args.output, manifest, args.courses_dir.parent)
    else:
        json.dump(manifest, sys.stdout, ensure_ascii=False, sort_keys=True, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
