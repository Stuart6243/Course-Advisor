#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup, Tag

try:  # Parsing and offline repair must not require the network dependency.
    import requests
except ModuleNotFoundError:  # pragma: no cover - exercised in parser-only envs
    requests = None  # type: ignore[assignment]

try:
    from backend.section_validator import validate_section
except ModuleNotFoundError:  # Running this file directly sets sys.path to this folder.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend.section_validator import validate_section

DEFAULT_SEED = "https://bulletin.columbia.edu/columbia-engineering/about-school/"
DEFAULT_YEAR = "2025-2026"
DEFAULT_ROOT = str(Path(__file__).resolve().parent)
USER_AGENT = "columbia-engineering-course-scraper/1.0"

ALLOWED_PATH_PREFIXES = (
    "/columbia-engineering/about-school/",
    "/columbia-engineering/academic-departments-programs/",
    "/columbia-engineering/interdisciplinary-engineering-courses/",
)

NON_COURSE_EXCLUDES = (
    "/columbia-engineering/academic-departments-programs/key-course-listings/",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url)
    path = re.sub(r"/+", "/", parsed.path or "/")
    if path != "/" and not path.endswith("/"):
        path = f"{path}/"
    clean = parsed._replace(
        scheme=(parsed.scheme or "https"),
        netloc=parsed.netloc.lower(),
        path=path,
        params="",
        query=parsed.query,
        fragment="",
    )
    return urlunparse(clean)


def is_in_scope(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc.lower() != "bulletin.columbia.edu":
        return False

    path = parsed.path or "/"
    path_lower = path.lower()
    path_no_slash = path_lower.rstrip("/")

    # Skip non-HTML assets and malformed external-domain-like path segments.
    non_html_suffixes = (".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".jpg", ".jpeg", ".png", ".svg")
    if any(path_no_slash.endswith(ext) for ext in non_html_suffixes):
        return False

    last_segment = path_no_slash.split("/")[-1] if path_no_slash else ""
    if "." in last_segment and not last_segment.endswith((".html", ".htm")):
        return False

    if not path.startswith("/columbia-engineering/"):
        return False
    if any(path.startswith(x) for x in NON_COURSE_EXCLUDES):
        return False
    if any(path.startswith(prefix) for prefix in ALLOWED_PATH_PREFIXES):
        return True
    return False


def sanitize_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    base = parsed.path.strip("/") or "root"
    base = base.replace("/", "__")
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    if parsed.query:
        qhash = hashlib.sha1(parsed.query.encode("utf-8")).hexdigest()[:8]
        base = f"{base}__q_{qhash}"
    return base


def build_dedup_key(year: str, course_code: str, title: str, source_url: str) -> str:
    parsed = urlparse(source_url)
    payload = "|".join(
        [
            year.strip(),
            normalize_text(course_code).upper(),
            normalize_text(title).upper(),
            parsed.path.strip(),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def course_identity(bulletin_year: str, course_code: str, source_page_url: str) -> Tuple[str, str, str]:
    parsed = urlparse(source_page_url or "")
    return (
        normalize_text(bulletin_year),
        normalize_text(course_code).upper(),
        (parsed.path or "").rstrip("/").lower(),
    )


def parse_points_range(points_raw: str) -> Tuple[Optional[float], Optional[float]]:
    cleaned = normalize_text(points_raw.lower().replace("points", "").replace("point", ""))
    cleaned = cleaned.strip(". ")
    if not cleaned:
        return None, None
    if "-" in cleaned:
        left, right = [p.strip() for p in cleaned.split("-", 1)]
        try:
            return float(left), float(right)
        except ValueError:
            return None, None
    try:
        val = float(cleaned)
        return val, val
    except ValueError:
        return None, None


def parse_enrollment(raw: str) -> Tuple[Optional[int], Optional[int]]:
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", normalize_text(raw))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def split_time_location(text: str) -> Tuple[str, str]:
    lines = [normalize_text(x) for x in text.split("\n") if normalize_text(x)]
    if not lines:
        return "", ""
    if len(lines) == 1:
        one = lines[0]
        m = re.match(
            r"^(.*?\d{1,2}:\d{2}(?:am|pm)\s*-\s*\d{1,2}:\d{2}(?:am|pm))\s+(.*)$",
            one,
            flags=re.IGNORECASE,
        )
        if m:
            return normalize_text(m.group(1)), normalize_text(m.group(2))
        return one, ""
    return lines[0], " ".join(lines[1:])


def parse_title_line(title_text: str) -> Tuple[str, str, str, List[str]]:
    issues: List[str] = []
    raw = normalize_text(title_text)

    points_raw = ""
    points_match = re.search(
        r"(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?\s+points?)\s*\.?\s*$",        raw,
        flags=re.IGNORECASE,
    )
    if points_match:
        points_raw = normalize_text(points_match.group(1))
        head = normalize_text(raw[: points_match.start()]).rstrip(".").strip()
    else:
        head = raw
        issues.append("missing_points_in_title")

    code = ""
    title = ""
    code_match = re.match(r"^([A-Z]{2,6}\s+[A-Z]{0,3}\d{4}[A-Z]?)\s+(.*)$", head)
    if not code_match:
        code_match = re.match(r"^([A-Z]{2,6}\s+\d{4}[A-Z]?)\s+(.*)$", head)

    if code_match:
        code = normalize_text(code_match.group(1))
        title = normalize_text(code_match.group(2)).strip(" .")
    else:
        issues.append("missing_course_code_or_title")
        title = head.strip(" .")

    if not title:
        issues.append("empty_title")

    return code, title, points_raw, issues


def extract_links(soup: BeautifulSoup, base_url: str) -> List[str]:
    out: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue

        low = href.lower()
        if low.startswith("#") or low.startswith("mailto:") or low.startswith("tel:") or low.startswith("javascript:"):
            continue

        abs_url = canonicalize_url(urljoin(base_url, href))
        if is_in_scope(abs_url):
            out.append(abs_url)
    # keep order, dedup
    seen = set()
    deduped = []
    for u in out:
        if u in seen:
            continue
        seen.add(u)
        deduped.append(u)
    return deduped


def check_robots(seed_url: str) -> bool:
    parsed = urlparse(seed_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        return rp.can_fetch(USER_AGENT, seed_url)
    except Exception:
        # Fail-open for robots retrieval errors, but log warning at caller.
        return True


def fetch_url(session: requests.Session, url: str, timeout: int, retries: int) -> str:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:  # requests exceptions
            last_err = exc
            if attempt < retries:
                time.sleep(min(2.0, 0.5 * attempt))
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def parse_section_rows(block: Tag) -> Tuple[List[dict], List[str]]:
    issues: List[str] = []
    sections: List[dict] = []

    sched_tables = block.select(".desc_sched table.scheduletbl")
    for table in sched_tables:
        current_term = ""
        current_catalog_ref = ""
        header_map: Dict[str, int] = {}

        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"], recursive=False)
            if not cells:
                continue

            # Keep one text value per physical cell.  Empty TIMES/LOCATION or
            # INSTRUCTOR cells are meaningful placeholders; deleting them
            # shifts every later value into the preceding header column.
            texts = [normalize_text(c.get_text("\n", strip=True)) for c in cells]
            if not any(texts):
                continue

            if len(texts) == 1 and re.search(r"(Fall|Spring|Summer|Winter)\s+\d{4}", texts[0], re.I):
                term_blob = texts[0]
                if ":" in term_blob:
                    left, right = term_blob.split(":", 1)
                    current_term = normalize_text(left)
                    current_catalog_ref = normalize_text(right)
                else:
                    current_term = term_blob
                    current_catalog_ref = ""
                header_map = {}
                continue

            lowered = [t.lower() for t in texts]
            if any("course number" in t for t in lowered) and any("section/call" in t for t in lowered):
                header_map = {
                    re.sub(r"\s+", " ", t.strip().lower()): idx for idx, t in enumerate(texts)
                }
                continue

            def get_by_header(*names: str, fallback_index: Optional[int] = None) -> str:
                for name in names:
                    if name in header_map and header_map[name] < len(texts):
                        return texts[header_map[name]]
                if fallback_index is not None and fallback_index < len(texts):
                    return texts[fallback_index]
                return ""

            course_number = get_by_header("course number", fallback_index=0)
            section_call = get_by_header("section/call number", fallback_index=1)
            times_loc_raw = get_by_header("times/location", fallback_index=2)
            instructor = get_by_header("instructor", fallback_index=3)
            points = get_by_header("points", fallback_index=4)
            enrollment_raw = get_by_header("enrollment", fallback_index=5)

            if not course_number and not section_call and not instructor:
                continue

            times, location = split_time_location(times_loc_raw)
            enrollment_current, enrollment_capacity = parse_enrollment(enrollment_raw)
            if enrollment_raw and enrollment_current is None:
                issues.append("enrollment_parse_warning")

            section = {
                "term": current_term,
                "catalog_ref": current_catalog_ref,
                "course_number": course_number,
                "section_call_number": section_call,
                "times": times,
                "location": location,
                "instructor": instructor,
                "points": points,
                "enrollment_raw": enrollment_raw,
                "enrollment_current": enrollment_current,
                "enrollment_capacity": enrollment_capacity,
            }
            validation = validate_section(section)
            for problem in (*validation.errors, *validation.warnings):
                issues.append(f"section_validation:{section_call or 'unknown'}:{problem}")
            sections.append(section)

    # Dedup sections in-place by stable tuple.
    deduped: List[dict] = []
    seen = set()
    for s in sections:
        key = (
            s.get("term", ""),
            s.get("course_number", ""),
            s.get("section_call_number", ""),
            s.get("times", ""),
            s.get("location", ""),
            s.get("instructor", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)

    return deduped, sorted(set(issues))


def is_prereq_or_coreq_text(text: str) -> bool:
    low = normalize_text(text).lower()
    prefixes = (
        "prerequisites:",
        "prerequisite:",
        "corequisites:",
        "corequisite:",
        "recommended preparation:",
        "recommended prerequisite:",
    )
    return any(low.startswith(p) for p in prefixes)


def is_meta_text(text: str) -> bool:
    low = normalize_text(text).lower()
    prefixes = (
        "lect:",
        "lect.",
        "lab:",
        "lab.",
        "recit:",
        "recitation",
        "discussion:",
        "cc/gs:",
    )
    return any(low.startswith(p) for p in prefixes)


def dedup_text_list(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in items:
        t = normalize_text(raw)
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def split_first_sentence(text: str) -> Tuple[str, str]:
    text = normalize_text(text)
    if not text:
        return "", ""

    m = re.match(r"^(.*?[.?!])\s+(.*)$", text)
    if not m:
        return text, ""

    head = normalize_text(m.group(1))
    tail = normalize_text(m.group(2))
    return head, tail


def parse_course_block(
    block: Tag,
    year: str,
    source_url: str,
    source_page_title: str,
    department: str,
) -> dict:
    title_p = block.select_one("p.courseblocktitle")
    title_text = normalize_text(title_p.get_text(" ", strip=True)) if title_p else ""

    course_code, title, points_raw, title_issues = parse_title_line(title_text)
    points_min, points_max = parse_points_range(points_raw)

    direct_ps = block.find_all("p", recursive=False)
    paragraphs: List[dict] = []
    for p in direct_ps:
        classes = set(p.get("class") or [])
        text = normalize_text(p.get_text(" ", strip=True))
        if not text:
            continue
        if "courseblocktitle" in classes:
            continue

        source = "courseblockdesc" if "courseblockdesc" in classes else "extra_paragraph"
        paragraphs.append({"source": source, "text": text})

    prereq_items: List[Tuple[str, str]] = []
    notes_items: List[str] = []
    desc_candidates: List[Tuple[str, str]] = []

    for para in paragraphs:
        source = para["source"]
        p_text = para["text"]

        if is_prereq_or_coreq_text(p_text):
            lead, tail = split_first_sentence(p_text)
            prereq_items.append((source, lead or p_text))
            if tail and not is_prereq_or_coreq_text(tail) and not is_meta_text(tail):
                desc_candidates.append((source, tail))
            continue

        if is_meta_text(p_text):
            lead, tail = split_first_sentence(p_text)
            notes_items.append(lead or p_text)
            if tail and not is_prereq_or_coreq_text(tail) and not is_meta_text(tail):
                desc_candidates.append((source, tail))
            continue

        desc_candidates.append((source, p_text))

    description = ""
    description_source = "none"
    if desc_candidates:
        description_source, description = max(
            desc_candidates,
            key=lambda item: (len(item[1]), 1 if item[0] == "courseblockdesc" else 0),
        )

    prereq_texts = dedup_text_list([text for _, text in prereq_items])
    prerequisites = "\n".join(prereq_texts).strip()

    prereq_sources = {src for src, _ in prereq_items}
    if not prereq_sources:
        prerequisites_source = "none"
    elif len(prereq_sources) == 1:
        prerequisites_source = next(iter(prereq_sources))
    else:
        prerequisites_source = "merged"

    notes = "\n".join(dedup_text_list(notes_items)).strip()

    course_issues = list(title_issues)
    if not description:
        course_issues.append("missing_description")

    sections, section_issues = parse_section_rows(block)
    issues = sorted(set([*course_issues, *section_issues]))

    dedup_key = build_dedup_key(year, course_code, title, source_url)

    record = {
        "dedup_key": dedup_key,
        "course_uid": dedup_key,
        "bulletin_year": year,
        "source_page_url": source_url,
        "source_page_title": source_page_title,
        "department_or_group": department,
        "course_code": course_code,
        "title": title,
        "points_raw": points_raw,
        "points_min": points_min,
        "points_max": points_max,
        "description": description,
        "description_source": description_source,
        "prerequisites_text": prerequisites,
        "prerequisites_source": prerequisites_source,
        "notes_text": notes,
        "sections": sections,
        # Section-level anomalies are isolated by the shared validator when
        # building the enriched index.  They remain auditable here but do not
        # hide an otherwise sound course and its other published sections.
        "needs_review": bool(course_issues),
        "parse_warnings": issues,
        "course_review_warnings": sorted(set(course_issues)),
        "section_review_warnings": sorted(set(section_issues)),
        "raw_title_text": title_text,
    }
    return record


def merge_course_records(base: dict, new: dict) -> dict:
    if not base:
        return new

    def description_rank(desc_text: str) -> Tuple[int, int]:
        desc = normalize_text(desc_text)
        if not desc:
            return (0, 0)
        # Prefer non-prerequisite description text, then longer text.
        prereq_penalty = 0 if not is_prereq_or_coreq_text(desc) else -1
        return (1 + prereq_penalty, len(desc))

    if description_rank(new.get("description", "")) > description_rank(base.get("description", "")):
        base["description"] = new.get("description", "")
        base["description_source"] = new.get("description_source", base.get("description_source", "none"))

    def split_lines(text: str) -> List[str]:
        return [normalize_text(x) for x in (text or "").splitlines() if normalize_text(x)]

    prereq_merged = dedup_text_list(split_lines(base.get("prerequisites_text", "")) + split_lines(new.get("prerequisites_text", "")))
    base["prerequisites_text"] = "\n".join(prereq_merged).strip()

    prereq_source_set = {
        s
        for s in [base.get("prerequisites_source", "none"), new.get("prerequisites_source", "none")]
        if s and s != "none"
    }
    if not prereq_source_set:
        base["prerequisites_source"] = "none"
    elif len(prereq_source_set) == 1:
        base["prerequisites_source"] = next(iter(prereq_source_set))
    else:
        base["prerequisites_source"] = "merged"

    notes_merged = dedup_text_list(split_lines(base.get("notes_text", "")) + split_lines(new.get("notes_text", "")))
    base["notes_text"] = "\n".join(notes_merged).strip()

    merged_sections = base.get("sections", []) + new.get("sections", [])
    deduped_sections = []
    seen = set()
    for s in merged_sections:
        key = (
            s.get("term", ""),
            s.get("course_number", ""),
            s.get("section_call_number", ""),
            s.get("times", ""),
            s.get("location", ""),
            s.get("instructor", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped_sections.append(s)
    base["sections"] = deduped_sections

    def split_warnings(record: dict) -> Tuple[set, set]:
        raw = set(record.get("parse_warnings", []))
        explicit_course = record.get("course_review_warnings")
        explicit_section = record.get("section_review_warnings")
        section = (
            set(explicit_section)
            if isinstance(explicit_section, list)
            else {
                warning
                for warning in raw
                if warning == "enrollment_parse_warning"
                or warning.startswith("section_validation:")
            }
        )
        course = (
            set(explicit_course)
            if isinstance(explicit_course, list)
            else raw - section
        )
        return course, section

    base_course_warnings, base_section_warnings = split_warnings(base)
    new_course_warnings, new_section_warnings = split_warnings(new)
    course_warnings = base_course_warnings | new_course_warnings
    section_warnings = base_section_warnings | new_section_warnings
    warnings = course_warnings | section_warnings
    base["parse_warnings"] = sorted(warnings)
    base["course_review_warnings"] = sorted(course_warnings)
    base["section_review_warnings"] = sorted(section_warnings)
    base["needs_review"] = bool(course_warnings)

    return base


def parse_course_page(html: str, url: str, year: str) -> Tuple[List[dict], List[dict], dict]:
    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one("#sc_courseblock")
    page_title = normalize_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    department_h1 = soup.select_one("main h1")
    department = normalize_text(department_h1.get_text(" ", strip=True)) if department_h1 else ""

    if not root:
        page_meta = {
            "source_page_url": url,
            "source_page_title": page_title,
            "department_or_group": department,
            "is_course_page": False,
            "course_blocks": 0,
        }
        return [], [], page_meta

    courses_by_key: Dict[str, dict] = {}
    review_items: List[dict] = []

    blocks = root.select(".courseblock")
    for idx, block in enumerate(blocks):
        course = parse_course_block(
            block=block,
            year=year,
            source_url=url,
            source_page_title=page_title,
            department=department,
        )
        key = course["dedup_key"]
        if key in courses_by_key:
            courses_by_key[key] = merge_course_records(courses_by_key[key], course)
        else:
            courses_by_key[key] = course

        if course.get("parse_warnings"):
            review_items.append(
                {
                    "issue_types": course.get("parse_warnings", []),
                    "source_page_url": url,
                    "course_code": course.get("course_code", ""),
                    "title": course.get("title", ""),
                    "raw_title_text": course.get("raw_title_text", ""),
                    "block_index": idx,
                    "course_needs_review": bool(course.get("needs_review")),
                }
            )

    page_meta = {
        "source_page_url": url,
        "source_page_title": page_title,
        "department_or_group": department,
        "is_course_page": True,
        "course_blocks": len(blocks),
        "course_records_after_merge": len(courses_by_key),
    }
    return list(courses_by_key.values()), review_items, page_meta


def load_registry(path: Path) -> Dict[str, dict]:
    registry: Dict[str, dict] = {}
    if not path.exists():
        return registry
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            key = item.get("dedup_key")
            if key:
                registry[key] = item
        except json.JSONDecodeError:
            continue
    return registry


def save_registry(path: Path, registry: Dict[str, dict]) -> None:
    lines = [json.dumps(registry[k], ensure_ascii=False) for k in sorted(registry.keys())]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def save_keys(path: Path, keys: Sequence[str]) -> None:
    sorted_keys = sorted(set(keys))
    path.write_text("\n".join(sorted_keys) + ("\n" if sorted_keys else ""), encoding="utf-8")


def compact_registry_by_identity(registry: Dict[str, dict], current_courses: Dict[str, dict], latest_snapshot_file: str) -> Dict[str, dict]:
    buckets: Dict[Tuple[str, str, str], List[Tuple[str, dict]]] = {}
    for key, rec in registry.items():
        ident = course_identity(rec.get("bulletin_year", ""), rec.get("course_code", ""), rec.get("source_page_url", ""))
        buckets.setdefault(ident, []).append((key, rec))

    current_by_identity: Dict[Tuple[str, str, str], Tuple[str, dict]] = {}
    for key, course in current_courses.items():
        ident = course_identity(course.get("bulletin_year", ""), course.get("course_code", ""), course.get("source_page_url", ""))
        current_by_identity[ident] = (key, course)

    collapsed: Dict[str, dict] = {}
    for ident, items in buckets.items():
        preferred = current_by_identity.get(ident)
        if preferred:
            preferred_key, current_course = preferred
        else:
            preferred_key, _ = max(
                items,
                key=lambda x: (
                    x[1].get("last_seen_at", ""),
                    int(x[1].get("seen_count", 0) or 0),
                ),
            )
            current_course = None

        first_seen = [rec.get("first_seen_at") for _, rec in items if rec.get("first_seen_at")]
        last_seen = [rec.get("last_seen_at") for _, rec in items if rec.get("last_seen_at")]

        seen_total = 0
        for _, rec in items:
            try:
                seen_total += int(rec.get("seen_count", 0) or 0)
            except Exception:
                pass

        base = next((rec for key, rec in items if key == preferred_key), items[0][1])
        out = dict(base)
        out["dedup_key"] = preferred_key
        out["seen_count"] = max(1, seen_total)
        out["first_seen_at"] = min(first_seen) if first_seen else out.get("first_seen_at", "")
        out["last_seen_at"] = max(last_seen) if last_seen else out.get("last_seen_at", "")
        out["latest_snapshot_file"] = latest_snapshot_file

        if current_course:
            out["bulletin_year"] = current_course.get("bulletin_year", out.get("bulletin_year", ""))
            out["course_code"] = current_course.get("course_code", out.get("course_code", ""))
            out["title"] = current_course.get("title", out.get("title", ""))
            out["source_page_url"] = current_course.get("source_page_url", out.get("source_page_url", ""))

        collapsed[preferred_key] = out

    return collapsed


def ensure_dirs(root: Path) -> Dict[str, Path]:
    dirs = {
        "root": root,
        "raw_html": root / "raw_html",
        "snapshots": root / "snapshots",
        "index": root / "index",
        "reports": root / "reports",
        "logs": root / "logs",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def write_default_config_if_missing(root: Path, seed_url: str, year: str) -> None:
    config_path = root / "config.json"
    if config_path.exists():
        return
    payload = {
        "seed_url": seed_url,
        "bulletin_year": year,
        "allowed_path_prefixes": list(ALLOWED_PATH_PREFIXES),
        "exclude_path_prefixes": list(NON_COURSE_EXCLUDES),
    }
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def make_run_report(
    report_path: Path,
    run_id: str,
    year: str,
    seed_url: str,
    pages: List[dict],
    courses: List[dict],
    new_courses: List[dict],
    review_queue: List[dict],
    failed_pages: List[dict],
) -> None:
    course_pages = [p for p in pages if p.get("is_course_page")]
    description_missing = sum(1 for c in courses if not c.get("description"))
    sections_empty = sum(1 for c in courses if not c.get("sections"))
    warning_count = sum(1 for c in courses if c.get("parse_warnings"))
    required_ok = sum(
        1
        for c in courses
        if c.get("course_code")
        and c.get("title")
        and isinstance(c.get("description"), str)
        and isinstance(c.get("sections"), list)
    )
    success_rate = (required_ok / len(courses) * 100.0) if courses else 0.0

    lines = []
    lines.append(f"# Columbia Engineering Scrape Report ({run_id})")
    lines.append("")
    lines.append(f"- Generated at (UTC): {utc_now_iso()}")
    lines.append(f"- Bulletin year: {year}")
    lines.append(f"- Seed URL: {seed_url}")
    lines.append(f"- Pages visited: {len(pages)}")
    lines.append(f"- Course pages: {len(course_pages)}")
    lines.append(f"- Failed pages: {len(failed_pages)}")
    lines.append(f"- Course records: {len(courses)}")
    lines.append(f"- New courses this run: {len(new_courses)}")
    lines.append(f"- Existing courses this run: {len(courses) - len(new_courses)}")
    lines.append(f"- Missing description: {description_missing}")
    lines.append(f"- Empty sections: {sections_empty}")
    lines.append(f"- Parse warnings: {warning_count}")
    lines.append(f"- Required-field success rate: {success_rate:.2f}%")
    lines.append("")

    if failed_pages:
        lines.append("## Failed Pages")
        for item in failed_pages:
            lines.append(f"- {item.get('url')}: {item.get('error')}")
        lines.append("")

    if review_queue:
        lines.append("## Review Queue Summary")
        lines.append(f"- Review items: {len(review_queue)}")
        lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def crawl_and_scrape(
    seed_url: str,
    year: str,
    root: Path,
    timeout: int,
    retries: int,
    min_delay: float,
    max_delay: float,
    max_pages: int,
) -> dict:
    if requests is None:
        raise RuntimeError(
            "requests is required for network crawling; parser/offline modes do not need it"
        )
    dirs = ensure_dirs(root)
    write_default_config_if_missing(root, seed_url, year)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    log_path = dirs["logs"] / f"run_{run_id}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )

    logger = logging.getLogger("scraper")

    if not check_robots(seed_url):
        raise RuntimeError(f"Robots.txt disallows crawling seed URL: {seed_url}")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    queue: deque[str] = deque([canonicalize_url(seed_url)])
    seen = set()

    pages_meta: List[dict] = []
    failed_pages: List[dict] = []
    all_courses: Dict[str, dict] = {}
    review_queue: List[dict] = []

    while queue:
        if 0 < max_pages <= len(seen):
            logger.info("Reached max_pages=%s, stopping crawl.", max_pages)
            break

        url = queue.popleft()
        if url in seen:
            continue
        seen.add(url)

        logger.info("Fetching: %s", url)
        try:
            html = fetch_url(session, url, timeout=timeout, retries=retries)
        except Exception as exc:
            err = str(exc)
            failed_pages.append({"url": url, "error": err})
            pages_meta.append({"source_page_url": url, "is_course_page": False, "error": err})
            logger.error("Fetch failed: %s", err)
            continue

        raw_name = f"{run_id}__{sanitize_filename_from_url(url)}.html"
        (dirs["raw_html"] / raw_name).write_text(html, encoding="utf-8")

        soup = BeautifulSoup(html, "html.parser")
        for link in extract_links(soup, url):
            if link not in seen:
                queue.append(link)

        page_courses, page_reviews, page_meta = parse_course_page(html=html, url=url, year=year)
        pages_meta.append(page_meta)

        for c in page_courses:
            key = c["dedup_key"]
            if key in all_courses:
                all_courses[key] = merge_course_records(all_courses[key], c)
            else:
                all_courses[key] = c

        review_queue.extend(page_reviews)

        if max_delay > 0:
            sleep_s = random.uniform(min_delay, max_delay)
            time.sleep(max(0.0, sleep_s))

    # Load index
    registry_path = dirs["index"] / "course_registry.jsonl"
    keys_path = dirs["index"] / "course_keys.txt"
    registry = load_registry(registry_path)

    existing_identities = {
        course_identity(
            rec.get("bulletin_year", ""),
            rec.get("course_code", ""),
            rec.get("source_page_url", ""),
        )
        for rec in registry.values()
    }

    generated_at = utc_now_iso()
    snapshot_file = f"courses_{run_id}.json"

    new_courses = []
    for key, course in all_courses.items():
        ident = course_identity(
            course.get("bulletin_year", ""),
            course.get("course_code", ""),
            course.get("source_page_url", ""),
        )

        if key in registry:
            registry[key]["last_seen_at"] = generated_at
            registry[key]["seen_count"] = int(registry[key].get("seen_count", 0)) + 1
            registry[key]["latest_snapshot_file"] = snapshot_file
        else:
            registry[key] = {
                "dedup_key": key,
                "bulletin_year": year,
                "course_code": course.get("course_code", ""),
                "title": course.get("title", ""),
                "source_page_url": course.get("source_page_url", ""),
                "first_seen_at": generated_at,
                "last_seen_at": generated_at,
                "seen_count": 1,
                "latest_snapshot_file": snapshot_file,
            }
            if ident not in existing_identities:
                new_courses.append(course)

        existing_identities.add(ident)

    registry = compact_registry_by_identity(registry, all_courses, snapshot_file)
    save_registry(registry_path, registry)
    save_keys(keys_path, registry.keys())

    snapshot_payload = {
        "bulletin_year": year,
        "generated_at": generated_at,
        "seed_url": seed_url,
        "run_id": run_id,
        "summary": {
            "pages_visited": len(pages_meta),
            "failed_pages": len(failed_pages),
            "course_pages": sum(1 for p in pages_meta if p.get("is_course_page")),
            "courses_total": len(all_courses),
            "new_courses": len(new_courses),
            "existing_courses": len(all_courses) - len(new_courses),
            "review_items": len(review_queue),
        },
        "pages": pages_meta,
        "courses": sorted(all_courses.values(), key=lambda x: (x.get("course_code", ""), x.get("title", ""))),
    }

    snapshot_path = dirs["snapshots"] / snapshot_file
    snapshot_path.write_text(json.dumps(snapshot_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    new_courses_path = dirs["index"] / f"new_courses_{run_id}.json"
    new_courses_path.write_text(
        json.dumps(sorted(new_courses, key=lambda x: (x.get("course_code", ""), x.get("title", ""))), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    review_path = dirs["reports"] / f"review_queue_{run_id}.json"
    review_path.write_text(json.dumps(review_queue, ensure_ascii=False, indent=2), encoding="utf-8")

    run_report_path = dirs["reports"] / f"run_report_{run_id}.md"
    make_run_report(
        report_path=run_report_path,
        run_id=run_id,
        year=year,
        seed_url=seed_url,
        pages=pages_meta,
        courses=snapshot_payload["courses"],
        new_courses=new_courses,
        review_queue=review_queue,
        failed_pages=failed_pages,
    )

    log_json_path = dirs["logs"] / f"run_{run_id}.json"
    log_json_payload = {
        "run_id": run_id,
        "generated_at": generated_at,
        "snapshot": str(snapshot_path),
        "new_courses": str(new_courses_path),
        "review_queue": str(review_path),
        "run_report": str(run_report_path),
        "failed_pages": failed_pages,
    }
    log_json_path.write_text(json.dumps(log_json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Finished. Courses=%s, New=%s, Failed pages=%s", len(all_courses), len(new_courses), len(failed_pages))

    return {
        "run_id": run_id,
        "snapshot_path": str(snapshot_path),
        "new_courses_path": str(new_courses_path),
        "review_path": str(review_path),
        "run_report_path": str(run_report_path),
        "log_json_path": str(log_json_path),
        "courses_total": len(all_courses),
        "new_courses_total": len(new_courses),
        "failed_pages": len(failed_pages),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Columbia Engineering course pages with dedup registry.")
    parser.add_argument("--seed", default=DEFAULT_SEED, help="Seed page URL")
    parser.add_argument("--year", default=DEFAULT_YEAR, help="Bulletin year label")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Output root folder")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    parser.add_argument("--retries", type=int, default=3, help="Retry count for HTTP fetches")
    parser.add_argument("--min-delay", type=float, default=0.4, help="Min delay between page requests")
    parser.add_argument("--max-delay", type=float, default=0.9, help="Max delay between page requests")
    parser.add_argument("--max-pages", type=int, default=0, help="Max pages to crawl (0 means unlimited)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    result = crawl_and_scrape(
        seed_url=canonicalize_url(args.seed),
        year=args.year,
        root=root,
        timeout=args.timeout,
        retries=args.retries,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
        max_pages=args.max_pages,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
