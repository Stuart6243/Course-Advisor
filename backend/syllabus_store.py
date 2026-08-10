"""Versioned syllabus overlays with an atomic generation pointer.

The catalog JSON files are immutable seed data.  Imported syllabi live in a
separate store and are addressed by ``course_code + term + section_id``.  A
reader observes either the old complete generation or the new complete
generation because ``CURRENT`` is the only commit point.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from course_codes import normalize_course_code as canonical_course_code
from section_validator import parse_day_tokens, parse_points_value

try:  # ``fcntl`` is available on the macOS/Linux deployment targets.
    import fcntl
except ModuleNotFoundError:  # pragma: no cover - defensive Windows fallback
    fcntl = None  # type: ignore[assignment]


SCHEMA_VERSION = 1
VALID_STATUSES = frozenset({"rejected", "review", "published"})
GENERATION_RE = re.compile(r"^g-[0-9]+-[0-9a-f]{12}$")

_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.RLock] = {}


def _thread_lock(root: Path) -> threading.RLock:
    key = str(root.resolve())
    with _LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_course_code(value: Any) -> str:
    normalized = canonical_course_code(_normalized_text(value))
    if normalized is None:
        raise ValueError(f"Invalid course_code: {value!r}")
    return normalized


def normalize_term(value: Any) -> str:
    text = _normalized_text(value)
    match = re.fullmatch(r"(fall|spring|summer|winter)\s+(\d{4})", text, re.I)
    if not match:
        return text
    return f"{match.group(1).title()} {match.group(2)}"


def normalize_section_id(value: Any) -> str:
    return _normalized_text(value).upper()


def identity_key(course_code: Any, term: Any, section_id: Any) -> str:
    """Return a collision-resistant key for one section identity."""

    parts = (
        normalize_course_code(course_code),
        normalize_term(term),
        normalize_section_id(section_id),
    )
    if not all(parts):
        raise ValueError("course_code, term, and section_id are required")
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def hash_source(source_bytes: bytes) -> str:
    if not isinstance(source_bytes, bytes):
        raise TypeError("source_bytes must be bytes")
    return hashlib.sha256(source_bytes).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _time_of_day(times: str) -> str:
    match = re.search(r"\b(\d{1,2}):(\d{2})\s*(am|pm)\b", times, re.I)
    if not match:
        return ""
    hour = int(match.group(1)) % 12
    if match.group(3).lower() == "pm":
        hour += 12
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    return "evening"


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _widen_points_range(entry: dict[str, Any], payload: dict[str, Any], points: Any) -> None:
    """Aggregate all seed/published credit choices without order dependence."""

    candidates: list[float] = []
    for field in ("points_min", "points_max"):
        value = _numeric(entry.get(field))
        if value is not None:
            candidates.append(value)
        value = _numeric(payload.get(field))
        if value is not None:
            candidates.append(value)
    parsed = parse_points_value(points)
    if parsed is not None:
        candidates.extend(parsed)
    if candidates:
        entry["points_min"] = min(candidates)
        entry["points_max"] = max(candidates)


def apply_published_overlays(
    seed_index: list[dict[str, Any]], overlays: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return an in-memory index view containing published overlays only.

    The input seed list is never mutated.  Provenance seed UIDs select exact
    duplicate catalog records; code matching is only a fallback for legacy
    overlays without UID provenance.
    """

    effective = copy.deepcopy(seed_index)
    by_uid = {
        str(entry.get("course_uid") or ""): entry
        for entry in effective
        if isinstance(entry, dict) and entry.get("course_uid")
    }
    for overlay in overlays:
        if not isinstance(overlay, dict) or overlay.get("status") != "published":
            continue
        payload = overlay.get("payload")
        if not isinstance(payload, dict):
            continue
        section = payload.get("section")
        if not isinstance(section, dict):
            continue
        provenance = overlay.get("provenance")
        seed_uids = (
            provenance.get("seed_course_uids", [])
            if isinstance(provenance, dict)
            else []
        )
        targets = [by_uid[uid] for uid in seed_uids if uid in by_uid]
        if not targets:
            code = normalize_course_code(overlay.get("course_code"))
            targets = [
                entry
                for entry in effective
                if normalize_course_code(entry.get("course_code")) == code
            ]
        term = normalize_term(overlay.get("term") or section.get("term"))
        section_id = normalize_section_id(
            overlay.get("section_id")
            or section.get("section_call_number")
            or section.get("section_id")
        )
        times = _normalized_text(section.get("times"))
        instructor = _normalized_text(section.get("instructor"))
        summary = {
            "section_id": section_id,
            "term": term,
            "times": times,
            "days": parse_day_tokens(times),
            "time_of_day": _time_of_day(times),
            "instructor": instructor,
            "location": _normalized_text(section.get("location")),
            "points": _normalized_text(section.get("points")),
            "enrollment_current": section.get("enrollment_current"),
            "enrollment_capacity": section.get("enrollment_capacity"),
            "validation_status": "published",
            "validation_warnings": [],
            "provenance": {
                "source": "published_syllabus_overlay",
                "version_id": overlay.get("version_id"),
                "source_hash": overlay.get("source_hash"),
            },
        }
        for entry in targets:
            if entry.get("catalog_validation_status") == "review":
                entry["catalog_review_overridden_by_published_overlay"] = True
            entry["catalog_validation_status"] = "published"
            entry["catalog_validation_warnings"] = []
            summaries = list(entry.get("sections_summary") or [])
            replaced = False
            for position, existing in enumerate(summaries):
                if (
                    normalize_term(existing.get("term")) == term
                    and normalize_section_id(existing.get("section_id")) == section_id
                ):
                    summaries[position] = summary
                    replaced = True
                    break
            if not replaced:
                summaries.append(summary)
            entry["sections_summary"] = summaries
            entry["all_instructors"] = list(
                dict.fromkeys(
                    _normalized_text(item.get("instructor"))
                    for item in summaries
                    if _normalized_text(item.get("instructor"))
                )
            )
            entry["all_terms"] = list(
                dict.fromkeys(
                    normalize_term(item.get("term"))
                    for item in summaries
                    if normalize_term(item.get("term"))
                )
            )
            _widen_points_range(entry, payload, summary["points"])
            if _normalized_text(payload.get("description")):
                entry["has_description"] = True
            overlay_terms = [
                payload.get("title"),
                payload.get("description"),
                payload.get("prerequisites_text"),
                term,
                instructor,
                *summary["days"],
                summary["time_of_day"],
            ]
            appended_text = " ".join(
                _normalized_text(value).lower() for value in overlay_terms if value
            )
            entry["searchable_text"] = _normalized_text(
                f"{entry.get('searchable_text', '')} {appended_text}"
            ).lower()
            versions = list(entry.get("syllabus_overlay_versions") or [])
            if overlay.get("version_id") not in versions:
                versions.append(overlay.get("version_id"))
            entry["syllabus_overlay_versions"] = [value for value in versions if value]
            overlay_refs = list(entry.get("published_syllabus_overlays") or [])
            if not any(
                ref.get("version_id") == overlay.get("version_id")
                for ref in overlay_refs
                if isinstance(ref, dict)
            ):
                overlay_refs.append(copy.deepcopy(overlay))
            entry["published_syllabus_overlays"] = overlay_refs
    return effective


def _empty_index() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generation": None,
        "created_at_ns": None,
        "syllabi": {},
    }


class SyllabusStore:
    """Filesystem-backed, generation-versioned syllabus overlay store."""

    def __init__(
        self,
        root: str | Path,
        *,
        failure_injector: Callable[[str], None] | str | set[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.generations_dir = self.root / "generations"
        self.current_path = self.root / "CURRENT"
        self.lock_path = self.root / "LOCK"
        self.failure_injector = failure_injector
        self._thread_lock = _thread_lock(self.root)

    def _inject(self, phase: str) -> None:
        injector = self.failure_injector
        if callable(injector):
            injector(phase)
        elif injector == phase or isinstance(injector, set) and phase in injector:
            raise RuntimeError(f"injected failure: {phase}")

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        # Directory creation is not the data commit point; a generation is
        # visible only after the atomic CURRENT replacement below.
        self.root.mkdir(parents=True, exist_ok=True)
        self.generations_dir.mkdir(parents=True, exist_ok=True)
        with self._thread_lock:
            with self.lock_path.open("a+b") as lock_file:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _read_current_name(self) -> str | None:
        try:
            name = self.current_path.read_text(encoding="ascii").strip()
        except FileNotFoundError:
            return None
        if not GENERATION_RE.fullmatch(name):
            raise ValueError("Invalid CURRENT generation pointer")
        return name

    def _read_index(self) -> dict[str, Any]:
        generation = self._read_current_name()
        if generation is None:
            return _empty_index()
        index_path = self.generations_dir / generation / "index.json"
        try:
            parsed = json.loads(index_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid syllabus generation: {generation}") from exc
        self._validate_index(parsed, expected_generation=generation)
        return parsed

    @staticmethod
    def _validate_index(index: Any, *, expected_generation: str | None = None) -> None:
        if not isinstance(index, dict) or index.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Invalid syllabus index schema")
        if expected_generation is not None and index.get("generation") != expected_generation:
            raise ValueError("Generation pointer/index mismatch")
        syllabi = index.get("syllabi")
        if not isinstance(syllabi, dict):
            raise ValueError("Invalid syllabi mapping")
        for key, record in syllabi.items():
            if not re.fullmatch(r"[0-9a-f]{64}", key) or not isinstance(record, dict):
                raise ValueError("Invalid syllabus identity record")
            if identity_key(
                record.get("course_code"), record.get("term"), record.get("section_id")
            ) != key:
                raise ValueError("Syllabus identity key mismatch")
            versions = record.get("versions")
            if not isinstance(versions, list):
                raise ValueError("Invalid syllabus versions")
            version_ids: set[str] = set()
            for version in versions:
                if not isinstance(version, dict):
                    raise ValueError("Invalid syllabus version")
                version_id = version.get("version_id")
                source_hash = version.get("source_hash")
                if (
                    not isinstance(version_id, str)
                    or version_id in version_ids
                    or not isinstance(source_hash, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", source_hash)
                    or version.get("status") not in VALID_STATUSES
                ):
                    raise ValueError("Invalid syllabus version metadata")
                version_ids.add(version_id)
            active = record.get("active_published_version")
            if active is not None:
                matches = [v for v in versions if v.get("version_id") == active]
                if len(matches) != 1 or matches[0].get("status") != "published":
                    raise ValueError("Invalid active published version")

    @staticmethod
    def _refresh_active(record: dict[str, Any]) -> None:
        published = [
            version for version in record["versions"] if version["status"] == "published"
        ]
        record["active_published_version"] = (
            published[-1]["version_id"] if published else None
        )

    @staticmethod
    def _effective_overlays_from_index(index: dict[str, Any]) -> list[dict[str, Any]]:
        """Build the published-only snapshot from an uncommitted index.

        Import handlers use this hook to construct and validate the complete
        runtime search view *before* ``CURRENT`` is replaced.  A failed runtime
        rebuild therefore cannot leave a published generation that only becomes
        visible after restart.
        """

        effective: list[dict[str, Any]] = []
        for record in index["syllabi"].values():
            active = record.get("active_published_version")
            if not active:
                continue
            version = next(
                item for item in record["versions"] if item["version_id"] == active
            )
            effective.append(
                {
                    "course_code": record["course_code"],
                    "term": record["term"],
                    "section_id": record["section_id"],
                    **copy.deepcopy(version),
                }
            )
        return sorted(
            effective,
            key=lambda item: (item["course_code"], item["term"], item["section_id"]),
        )

    def _commit(self, mutated: dict[str, Any]) -> str:
        previous_generation = self._read_current_name()
        generation = f"g-{time.time_ns()}-{uuid.uuid4().hex[:12]}"
        committed = copy.deepcopy(mutated)
        committed["schema_version"] = SCHEMA_VERSION
        committed["generation"] = generation
        committed["created_at_ns"] = time.time_ns()
        self._validate_index(committed, expected_generation=generation)

        generation_dir = self.generations_dir / generation
        index_path = generation_dir / "index.json"
        pointer_tmp = self.root / f".CURRENT.{uuid.uuid4().hex}.tmp"
        pointer_committed = False
        try:
            generation_dir.mkdir(exist_ok=False)
            with index_path.open("x", encoding="utf-8") as handle:
                json.dump(committed, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            directory_fd = os.open(generation_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            self._inject("after_generation_write")

            # Validate the exact bytes that readers will consume before exposing it.
            on_disk = json.loads(index_path.read_text(encoding="utf-8"))
            self._validate_index(on_disk, expected_generation=generation)
            self._inject("before_current_write")

            with pointer_tmp.open("x", encoding="ascii") as handle:
                handle.write(generation + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._inject("before_current_replace")
            os.replace(pointer_tmp, self.current_path)
            pointer_committed = True
            root_fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(root_fd)
            finally:
                os.close(root_fd)
            self._inject("after_current_replace")
            return generation
        except Exception:
            if pointer_committed:
                rollback_tmp = self.root / f".CURRENT.rollback.{uuid.uuid4().hex}.tmp"
                try:
                    if previous_generation is None:
                        self.current_path.unlink()
                    else:
                        with rollback_tmp.open("x", encoding="ascii") as handle:
                            handle.write(previous_generation + "\n")
                            handle.flush()
                            os.fsync(handle.fileno())
                        os.replace(rollback_tmp, self.current_path)
                    root_fd = os.open(self.root, os.O_RDONLY)
                    try:
                        os.fsync(root_fd)
                    finally:
                        os.close(root_fd)
                    pointer_committed = False
                finally:
                    try:
                        rollback_tmp.unlink()
                    except FileNotFoundError:
                        pass
            if not pointer_committed:
                # This path was created by this exact transaction and contains
                # only its index file.  Failed generations must not accumulate
                # as apparent overlays beside the still-current generation.
                try:
                    index_path.unlink()
                except FileNotFoundError:
                    pass
                try:
                    generation_dir.rmdir()
                except FileNotFoundError:
                    pass
            raise
        finally:
            # Failure injection may leave only a private temp pointer; it must
            # never be mistaken for CURRENT.
            try:
                pointer_tmp.unlink()
            except FileNotFoundError:
                pass

    def attach_syllabus(
        self,
        *,
        course_code: str,
        term: str,
        section_id: str,
        payload: dict[str, Any],
        status: str,
        source_bytes: bytes | None = None,
        source_hash: str | None = None,
        provenance: dict[str, Any] | None = None,
        evidence: dict[str, Any] | None = None,
        quality_score: int | None = None,
        quality_issues: list[str] | tuple[str, ...] | None = None,
        before_commit: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> dict[str, Any]:
        """Attach a version to an existing catalog identity.

        Callers enforce seed-course existence; this store intentionally has no
        API that creates or mutates catalog records.
        """

        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid syllabus status: {status}")
        if not isinstance(payload, dict):
            raise TypeError("payload must be an object")
        code = normalize_course_code(course_code)
        normalized_term = normalize_term(term)
        normalized_section = normalize_section_id(section_id)
        key = identity_key(code, normalized_term, normalized_section)
        if source_bytes is not None:
            calculated_hash = hash_source(source_bytes)
            if source_hash is not None and source_hash != calculated_hash:
                raise ValueError("source_hash does not match source_bytes")
            source_hash = calculated_hash
        elif source_hash is None:
            source_hash = hashlib.sha256(_canonical_json(payload)).hexdigest()
        if not re.fullmatch(r"[0-9a-f]{64}", source_hash):
            raise ValueError("source_hash must be a lowercase SHA-256 digest")

        version_id = hashlib.sha256(
            f"{key}\x1f{source_hash}".encode("ascii")
        ).hexdigest()
        now_ns = time.time_ns()
        with self._exclusive_lock():
            index = self._read_index()
            record = index["syllabi"].setdefault(
                key,
                {
                    "course_code": code,
                    "term": normalized_term,
                    "section_id": normalized_section,
                    "versions": [],
                    "active_published_version": None,
                },
            )
            for existing in record["versions"]:
                if existing["version_id"] == version_id:
                    if before_commit is not None:
                        before_commit(self._effective_overlays_from_index(index))
                    return {
                        "identity_key": key,
                        "version_id": version_id,
                        "generation": index.get("generation"),
                        "status": existing["status"],
                        "created": False,
                    }
            version = {
                "version_id": version_id,
                "source_hash": source_hash,
                "status": status,
                "created_at_ns": now_ns,
                "revision": len(record["versions"]) + 1,
                "payload": copy.deepcopy(payload),
                "provenance": copy.deepcopy(provenance or {}),
                "evidence": copy.deepcopy(evidence or {}),
                "quality_score": quality_score,
                "quality_issues": list(quality_issues or ()),
            }
            record["versions"].append(version)
            self._refresh_active(record)
            if before_commit is not None:
                before_commit(self._effective_overlays_from_index(index))
            generation = self._commit(index)
        return {
            "identity_key": key,
            "version_id": version_id,
            "generation": generation,
            "status": status,
            "created": True,
        }

    def set_status(
        self,
        *,
        course_code: str,
        term: str,
        section_id: str,
        version_id: str,
        status: str,
    ) -> dict[str, Any]:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid syllabus status: {status}")
        key = identity_key(course_code, term, section_id)
        with self._exclusive_lock():
            index = self._read_index()
            record = index["syllabi"].get(key)
            if record is None:
                raise KeyError("Unknown syllabus identity")
            match = next(
                (v for v in record["versions"] if v["version_id"] == version_id), None
            )
            if match is None:
                raise KeyError("Unknown syllabus version")
            if match["status"] == status:
                return {
                    "identity_key": key,
                    "version_id": version_id,
                    "generation": index.get("generation"),
                    "status": status,
                    "changed": False,
                }
            match["status"] = status
            match["status_updated_at_ns"] = time.time_ns()
            self._refresh_active(record)
            generation = self._commit(index)
        return {
            "identity_key": key,
            "version_id": version_id,
            "generation": generation,
            "status": status,
            "changed": True,
        }

    def attach_many(
        self,
        items: list[dict[str, Any]],
        *,
        before_commit: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Atomically attach several section versions in one generation.

        Validation and source hashing happen before the lock.  Once locked, all
        mutations are written to a single generation and become visible via a
        single CURRENT replacement, so a multi-section import cannot be half
        visible.
        """

        if not isinstance(items, list) or not items:
            raise ValueError("items must be a non-empty list")
        prepared: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                raise TypeError("each attachment must be an object")
            status = item.get("status")
            payload = item.get("payload")
            if status not in VALID_STATUSES:
                raise ValueError(f"Invalid syllabus status: {status}")
            if not isinstance(payload, dict):
                raise TypeError("payload must be an object")
            code = normalize_course_code(item.get("course_code"))
            term = normalize_term(item.get("term"))
            section = normalize_section_id(item.get("section_id"))
            key = identity_key(code, term, section)
            source_bytes = item.get("source_bytes")
            source_hash = item.get("source_hash")
            if source_bytes is not None:
                calculated_hash = hash_source(source_bytes)
                if source_hash is not None and source_hash != calculated_hash:
                    raise ValueError("source_hash does not match source_bytes")
                source_hash = calculated_hash
            elif source_hash is None:
                source_hash = hashlib.sha256(_canonical_json(payload)).hexdigest()
            if not isinstance(source_hash, str) or not re.fullmatch(
                r"[0-9a-f]{64}", source_hash
            ):
                raise ValueError("source_hash must be a lowercase SHA-256 digest")
            prepared.append(
                {
                    "course_code": code,
                    "term": term,
                    "section_id": section,
                    "identity_key": key,
                    "version_id": hashlib.sha256(
                        f"{key}\x1f{source_hash}".encode("ascii")
                    ).hexdigest(),
                    "source_hash": source_hash,
                    "status": status,
                    "payload": copy.deepcopy(payload),
                    "provenance": copy.deepcopy(item.get("provenance") or {}),
                    "evidence": copy.deepcopy(item.get("evidence") or {}),
                    "quality_score": item.get("quality_score"),
                    "quality_issues": list(item.get("quality_issues") or ()),
                }
            )

        with self._exclusive_lock():
            index = self._read_index()
            outcomes: list[dict[str, Any]] = []
            created_any = False
            for item in prepared:
                key = item["identity_key"]
                record = index["syllabi"].setdefault(
                    key,
                    {
                        "course_code": item["course_code"],
                        "term": item["term"],
                        "section_id": item["section_id"],
                        "versions": [],
                        "active_published_version": None,
                    },
                )
                existing = next(
                    (
                        version
                        for version in record["versions"]
                        if version["version_id"] == item["version_id"]
                    ),
                    None,
                )
                if existing is not None:
                    outcomes.append(
                        {
                            "identity_key": key,
                            "version_id": item["version_id"],
                            "generation": index.get("generation"),
                            "status": existing["status"],
                            "created": False,
                        }
                    )
                    continue
                record["versions"].append(
                    {
                        "version_id": item["version_id"],
                        "source_hash": item["source_hash"],
                        "status": item["status"],
                        "created_at_ns": time.time_ns(),
                        "revision": len(record["versions"]) + 1,
                        "payload": item["payload"],
                        "provenance": item["provenance"],
                        "evidence": item["evidence"],
                        "quality_score": item["quality_score"],
                        "quality_issues": item["quality_issues"],
                    }
                )
                self._refresh_active(record)
                created_any = True
                outcomes.append(
                    {
                        "identity_key": key,
                        "version_id": item["version_id"],
                        "generation": None,
                        "status": item["status"],
                        "created": True,
                    }
                )
            if before_commit is not None:
                before_commit(self._effective_overlays_from_index(index))
            if created_any:
                generation = self._commit(index)
                for outcome in outcomes:
                    outcome["generation"] = generation
            else:
                generation = index.get("generation")
                for outcome in outcomes:
                    outcome["generation"] = generation
        return outcomes

    def get_identity(
        self, course_code: str, term: str, section_id: str
    ) -> dict[str, Any] | None:
        """Administrative view including review/rejected versions."""

        record = self._read_index()["syllabi"].get(
            identity_key(course_code, term, section_id)
        )
        return copy.deepcopy(record) if record is not None else None

    def get_effective(
        self, course_code: str, term: str, section_id: str
    ) -> dict[str, Any] | None:
        """Return only the active *published* overlay for search consumption."""

        record = self._read_index()["syllabi"].get(
            identity_key(course_code, term, section_id)
        )
        if not record or not record.get("active_published_version"):
            return None
        active = record["active_published_version"]
        version = next(v for v in record["versions"] if v["version_id"] == active)
        return {
            "course_code": record["course_code"],
            "term": record["term"],
            "section_id": record["section_id"],
            **copy.deepcopy(version),
        }

    def effective_overlays(self) -> list[dict[str, Any]]:
        """Published-only snapshot suitable for a retriever/index overlay."""

        return self._effective_overlays_from_index(self._read_index())

    def manifest(self) -> dict[str, Any]:
        index = self._read_index()
        statuses = {status: 0 for status in sorted(VALID_STATUSES)}
        version_count = 0
        for record in index["syllabi"].values():
            for version in record["versions"]:
                version_count += 1
                statuses[version["status"]] += 1
        effective_count = sum(
            bool(record.get("active_published_version"))
            for record in index["syllabi"].values()
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "generation": index.get("generation"),
            "identity_count": len(index["syllabi"]),
            "version_count": version_count,
            "effective_published_count": effective_count,
            "status_counts": statuses,
            "index_sha256": hashlib.sha256(_canonical_json(index)).hexdigest(),
        }


__all__ = [
    "SCHEMA_VERSION",
    "VALID_STATUSES",
    "SyllabusStore",
    "apply_published_overlays",
    "hash_source",
    "identity_key",
    "normalize_course_code",
    "normalize_section_id",
    "normalize_term",
]
