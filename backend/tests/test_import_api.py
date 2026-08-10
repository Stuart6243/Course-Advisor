"""End-to-end import API contracts using only temporary catalog/store paths."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
import server as srv
from course_index import build_enriched_entry
from course_retriever import retrieve_courses
from syllabus_store import SyllabusStore


COURSE_UID = "a" * 40


class FakeModel:
    def __init__(self, *args, **kwargs) -> None:
        self.response = "{}"
        self.calls: list[dict] = []

    async def is_available(self, *args, **kwargs) -> bool:
        return True

    async def chat(self, messages, **kwargs) -> str:
        self.calls.append({"messages": list(messages), **kwargs})
        return self.response

    async def chat_stream(self, *args, **kwargs):
        yield "ok"


@pytest.fixture
def catalog_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    data_dir = tmp_path / "data"
    courses_dir = data_dir / "courses_flat"
    courses_dir.mkdir(parents=True)
    detail = {
        "course_uid": COURSE_UID,
        "course_code": "COMS GU4111",
        "title": "DATABASE SYSTEMS",
        "points_raw": "3.00 points",
        "points_min": 3.0,
        "points_max": 3.0,
        "description": "Seed catalog description.",
        "prerequisites_text": "COMS W3134",
        "source_page_url": "https://example.test/coms/",
        "sections": [
            {
                "term": "Fall 2025",
                "course_number": "COMS 4111",
                "section_call_number": "001/11111",
                "times": "T 2:00pm - 3:00pm",
                "location": "Seed Room",
                "instructor": "Seed Instructor",
                "points": "3.00",
                "enrollment_raw": "10/30",
                "enrollment_current": 10,
                "enrollment_capacity": 30,
            }
        ],
    }
    filename = f"{COURSE_UID}.json"
    course_path = courses_dir / filename
    course_path.write_text(json.dumps(detail, indent=2), encoding="utf-8")
    raw_entry = {
        "course_uid": COURSE_UID,
        "course_code": detail["course_code"],
        "title": detail["title"],
        "file_name": filename,
    }
    flat_path = data_dir / "courses_flat_index.json"
    flat_path.write_text(json.dumps([raw_entry], indent=2), encoding="utf-8")
    enriched_path = data_dir / "courses_enriched_index.json"
    enriched_path.write_text(
        json.dumps([build_enriched_entry(raw_entry, detail)], indent=2),
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "COURSES_DIR", courses_dir)
    monkeypatch.setattr(config, "RAW_INDEX_PATH", flat_path)
    monkeypatch.setattr(config, "ENRICHED_INDEX_PATH", enriched_path)
    monkeypatch.setattr(config, "WARMUP_ON_STARTUP", False)
    monkeypatch.setattr(config, "INFERENCE_MODE", "local")
    monkeypatch.setattr(srv, "OllamaClient", FakeModel)
    monkeypatch.setattr(srv, "GroqClient", FakeModel)

    formal_paths = (course_path, flat_path, enriched_path)
    return {
        "data_dir": data_dir,
        "courses_dir": courses_dir,
        "enriched_path": enriched_path,
        "formal_paths": formal_paths,
        "formal_before": {path: path.read_bytes() for path in formal_paths},
    }


def _intent(term: str) -> dict:
    return {
        "query_type": "detail",
        "course_codes": ["COMS GU4111"],
        "keywords": [],
        "department": None,
        "department_terms": [],
        "instructor": None,
        "time_preference": None,
        "day_preference": [],
        "points_range": None,
        "term": term,
        "comparison_targets": [],
        "original_question": "test",
    }


def _manual_payload(**overrides) -> dict:
    payload = {
        "course_code": "COMS GU4111",
        "title": "DATABASE SYSTEMS",
        "points_raw": "3.00 points",
        "term": "Spring 2026",
        "section_id": "002/22222",
        "times": "M 10:00am - 11:00am",
        "location": "Overlay Room",
        "instructor": "Overlay Instructor",
        "enrollment_raw": "10/30",
        "enrollment_current": 10,
        "enrollment_capacity": 30,
        "description": "Published overlay description for database systems.",
        "prerequisites_text": "COMS W3157",
    }
    payload.update(overrides)
    return payload


def _assert_formal_unchanged(environment: dict) -> None:
    assert {
        path: path.read_bytes() for path in environment["formal_paths"]
    } == environment["formal_before"]


def _generation_count(store_root: Path) -> int:
    generations = store_root / "generations"
    return len(list(generations.iterdir())) if generations.is_dir() else 0


def test_manual_published_refresh_review_isolation_and_restart(
    catalog_env: dict,
) -> None:
    with TestClient(srv.app) as client:
        published = client.post("/api/import/manual", json=_manual_payload())
        assert published.status_code == 200
        assert published.json()["status"] == "published"

        found = retrieve_courses(
            client.app.state.enriched_index,
            _intent("Spring 2026"),
            str(catalog_env["courses_dir"]),
        )
        assert len(found) == 1
        assert found[0]["description"].startswith("Published overlay")
        assert found[0]["matched_sections"][0]["section_call_number"] == (
            "002/22222"
        )
        assert client.app.state.seed_enriched_index[0].get(
            "published_syllabus_overlays"
        ) is None

        runtime_after_publish = copy.deepcopy(client.app.state.enriched_index)
        srv._refresh_runtime_overlays(client.app)
        srv._refresh_runtime_overlays(client.app)
        assert client.app.state.enriched_index == runtime_after_publish

        review = client.post(
            "/api/import/manual",
            json=_manual_payload(
                title="Conflicting submitted title",
                term="Summer 2026",
                section_id="003/33333",
                description="Review-only hidden description.",
            ),
        )
        assert review.status_code == 200
        assert review.json()["status"] == "review"
        assert client.app.state.enriched_index == runtime_after_publish
        assert retrieve_courses(
            client.app.state.enriched_index,
            _intent("Summer 2026"),
            str(catalog_env["courses_dir"]),
        ) == []
        runtime_before_restart = copy.deepcopy(client.app.state.enriched_index)
        _assert_formal_unchanged(catalog_env)

    with TestClient(srv.app) as restarted:
        assert restarted.app.state.enriched_index == runtime_before_restart
        assert retrieve_courses(
            restarted.app.state.enriched_index,
            _intent("Spring 2026"),
            str(catalog_env["courses_dir"]),
        )
        assert retrieve_courses(
            restarted.app.state.enriched_index,
            _intent("Summer 2026"),
            str(catalog_env["courses_dir"]),
        ) == []
        _assert_formal_unchanged(catalog_env)


@pytest.mark.parametrize(
    "payload",
    [
        _manual_payload(points_raw="", points_min=None, points_max=None),
        _manual_payload(points_raw="enrollment 10/30"),
        _manual_payload(course_code="BINF GU4001", title="BIOINFORMATICS"),
        _manual_payload(term="", section_id=""),
    ],
)
def test_manual_rejected_inputs_never_create_a_generation(
    catalog_env: dict, payload: dict
) -> None:
    store_root = catalog_env["data_dir"] / "syllabus_store"
    with TestClient(srv.app) as client:
        response = client.post("/api/import/manual", json=payload)
        assert response.status_code == 422
        assert response.json()["status"] == "rejected"
        assert _generation_count(store_root) == 0
        assert client.app.state.enriched_index == client.app.state.seed_enriched_index
    _assert_formal_unchanged(catalog_env)


def test_html_upload_published_and_docx_rejected(catalog_env: dict) -> None:
    llm_payload = {
        "course_code": "COMS GU4111",
        "title": "DATABASE SYSTEMS",
        "points_raw": "3.00 points",
        "points_min": 3.0,
        "points_max": 3.0,
        "description": "Published HTML overlay description for database systems.",
        "prerequisites_text": "COMS W3157",
        "sections": [
            {
                "term": "Spring 2027",
                "course_number": "COMS 4111",
                "section_call_number": "004/44444",
                "times": "W 9:00am - 10:00am",
                "location": "HTML Room",
                "instructor": "HTML Instructor",
                "points": "3.00",
                "enrollment_raw": "10/30",
                "enrollment_current": 10,
                "enrollment_capacity": 30,
            }
        ],
    }
    html = b"""
    <html><body><h1>COMS GU4111 DATABASE SYSTEMS</h1>
        <table><tr><th>Term</th><th>Section</th><th>Points</th><th>Time</th><th>Instructor</th></tr>
        <tr><td>Spring 2027</td><td>004/44444</td><td>3.00 points</td>
        <td>W 9:00am - 10:00am</td><td>HTML Instructor</td></tr></table>
    <p>Published HTML overlay description for database systems.</p></body></html>
    """
    with TestClient(srv.app) as client:
        client.app.state.ollama.response = json.dumps(llm_payload)
        response = client.post(
            "/api/import", files={"file": ("syllabus.html", html, "text/html")}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "published"
        assert retrieve_courses(
            client.app.state.enriched_index,
            _intent("Spring 2027"),
            str(catalog_env["courses_dir"]),
        )

        runtime = copy.deepcopy(client.app.state.enriched_index)
        rejected = client.post(
            "/api/import",
            files={
                "file": (
                    "unsupported.docx",
                    b"not a supported document",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )
        assert rejected.status_code == 400
        assert "PDF or HTML" in rejected.json()["message"]
        assert client.app.state.enriched_index == runtime
    _assert_formal_unchanged(catalog_env)


def test_oversize_timeout_and_cancel_leave_no_overlay(
    catalog_env: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_root = catalog_env["data_dir"] / "syllabus_store"
    monkeypatch.setattr(config, "MAX_IMPORT_SIZE_MB", 1)
    with TestClient(srv.app, raise_server_exceptions=False) as client:
        seed_runtime = copy.deepcopy(client.app.state.enriched_index)
        oversized = client.post(
            "/api/import",
            files={
                "file": (
                    "oversized.html",
                    b"x" * (1024 * 1024 + 1),
                    "text/html",
                )
            },
        )
        assert oversized.status_code == 413
        assert _generation_count(store_root) == 0
        assert client.app.state.enriched_index == seed_runtime

        async def slow_import(**kwargs):
            await asyncio.sleep(10)

        monkeypatch.setattr(srv, "import_file", slow_import)
        monkeypatch.setattr(srv, "IMPORT_PIPELINE_TIMEOUT_SECONDS", 0.001)
        timed_out = client.post(
            "/api/import",
            files={"file": ("slow.html", b"<html>slow</html>", "text/html")},
        )
        assert timed_out.status_code == 504
        assert _generation_count(store_root) == 0
        assert client.app.state.enriched_index == seed_runtime

        async def cancelled_import(**kwargs):
            raise asyncio.CancelledError

        monkeypatch.setattr(srv, "import_file", cancelled_import)
        cancelled = client.post(
            "/api/import",
            files={"file": ("cancel.html", b"<html>cancel</html>", "text/html")},
        )
        assert cancelled.status_code >= 499
        assert _generation_count(store_root) == 0
        assert client.app.state.enriched_index == seed_runtime
    _assert_formal_unchanged(catalog_env)


def test_store_commit_failure_returns_500_and_rolls_back_runtime(
    catalog_env: dict,
) -> None:
    store_root = catalog_env["data_dir"] / "syllabus_store"
    with TestClient(srv.app) as client:
        seed_runtime = copy.deepcopy(client.app.state.enriched_index)
        client.app.state.syllabus_store = SyllabusStore(
            store_root, failure_injector="after_current_replace"
        )
        response = client.post("/api/import/manual", json=_manual_payload())
        assert response.status_code == 500
        assert response.json()["status"] == "rejected"
        assert not (store_root / "CURRENT").exists()
        assert _generation_count(store_root) == 0
        assert client.app.state.enriched_index == seed_runtime

        client.app.state.ollama.response = json.dumps(
            {
                "course_code": "COMS GU4111",
                "title": "DATABASE SYSTEMS",
                "points_raw": "3.00 points",
                "points_min": 3.0,
                "points_max": 3.0,
                "description": "A detailed database systems syllabus description.",
                "sections": [
                    {
                        "term": "Spring 2026",
                        "section_call_number": "002/22222",
                        "times": "M 10:00am - 11:00am",
                        "instructor": "Overlay Instructor",
                        "points": "3.00",
                        "enrollment_raw": "",
                        "enrollment_current": None,
                        "enrollment_capacity": None,
                    }
                ],
            }
        )
        upload_failure = client.post(
            "/api/import",
            files={
                "file": (
                    "failure.html",
                    (
                        b"<html><body>COMS GU4111 DATABASE SYSTEMS 3.00 points "
                        b"Spring 2026 002/22222</body></html>"
                    ),
                    "text/html",
                )
            },
        )
        assert upload_failure.status_code == 500
        assert upload_failure.json()["error_code"] == "store_commit_failed"
        assert not (store_root / "CURRENT").exists()
        assert _generation_count(store_root) == 0
        assert client.app.state.enriched_index == seed_runtime
    _assert_formal_unchanged(catalog_env)


def test_runtime_candidate_failure_never_commits_or_appears_after_restart(
    catalog_env: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_root = catalog_env["data_dir"] / "syllabus_store"
    real_apply_overlays = srv.apply_published_overlays

    with TestClient(srv.app) as client:
        seed_runtime = copy.deepcopy(client.app.state.enriched_index)

        def fail_runtime_candidate(*_args, **_kwargs):
            raise RuntimeError("injected runtime candidate failure")

        monkeypatch.setattr(srv, "apply_published_overlays", fail_runtime_candidate)

        manual = client.post("/api/import/manual", json=_manual_payload())
        assert manual.status_code == 500
        assert manual.json()["status"] == "rejected"
        assert "no overlay was published" in manual.json()["message"]
        assert not (store_root / "CURRENT").exists()
        assert _generation_count(store_root) == 0
        assert client.app.state.enriched_index == seed_runtime

        description = "A detailed database systems syllabus description."
        client.app.state.ollama.response = json.dumps(
            {
                "course_code": "COMS GU4111",
                "title": "DATABASE SYSTEMS",
                "points_raw": "3.00 points",
                "points_min": 3.0,
                "points_max": 3.0,
                "description": description,
                "sections": [
                    {
                        "term": "Spring 2026",
                        "section_call_number": "002/22222",
                        "times": "M 10:00am - 11:00am",
                        "instructor": "Overlay Instructor",
                        "points": "3.00",
                        "enrollment_raw": "",
                        "enrollment_current": None,
                        "enrollment_capacity": None,
                    }
                ],
            }
        )
        html = (
            "<html><body>COMS GU4111 DATABASE SYSTEMS 3.00 points "
            f"{description} Spring 2026 002/22222 3.00 "
            "M 10:00am - 11:00am Overlay Instructor</body></html>"
        ).encode()
        uploaded = client.post(
            "/api/import",
            files={"file": ("candidate.html", html, "text/html")},
        )
        assert uploaded.status_code == 500
        assert uploaded.json()["error_code"] == "store_commit_failed"
        assert not (store_root / "CURRENT").exists()
        assert _generation_count(store_root) == 0
        assert client.app.state.enriched_index == seed_runtime

        monkeypatch.setattr(srv, "apply_published_overlays", real_apply_overlays)

    with TestClient(srv.app) as restarted:
        assert (
            restarted.app.state.enriched_index
            == restarted.app.state.seed_enriched_index
        )
        assert restarted.app.state.syllabus_store.manifest()[
            "effective_published_count"
        ] == 0
        assert not (store_root / "CURRENT").exists()
    _assert_formal_unchanged(catalog_env)
