# Columbia Engineering course-data utilities

[Project README](../README.md) · [中文说明](../README.zh-CN.md)

This directory contains the Columbia Engineering Bulletin crawler and an offline section-repair utility. See the project README for the advisor's scope, architecture, and application setup.

## Setup and offline tests

Run these commands from this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest tests
```

The tests use local fixtures and temporary files only; they do not request Bulletin pages. A network crawl was not run as part of the repository-cleanup verification.

## Crawler

`scrape_columbia_courses.py` makes live HTTP requests to the Columbia Bulletin. Run it only when a deliberate local capture is required:

```bash
python scrape_columbia_courses.py \
  --seed "https://bulletin.columbia.edu/columbia-engineering/about-school/" \
  --year "2025-2026"
```

Unless `--root` is provided, the output root is this scraper directory, independent of the current working directory. The crawler writes local artifacts under:

- `raw_html/` — fetched pages
- `snapshots/` — parsed crawl snapshots
- `logs/` and `reports/` — run diagnostics and review output
- `index/` — generated registry, keys, and new-course records

These generated state/history files are ignored by Git. A crawl does not automatically replace the application's formal catalog under `../data/`.

## Offline section repair

`offline_section_repair.py` never fetches the network. It requires callers to select both a saved snapshot and its raw-HTML directory explicitly; there are no bundled default inputs. Without `--apply`, it performs a dry run and reports proposed exact-UID section replacements without changing formal data:

```bash
python offline_section_repair.py \
  --snapshot snapshots/courses_<run-id>.json \
  --raw-dir raw_html
```

To save a reviewable dry-run manifest, first create an ignored report directory and use `--output`:

```bash
mkdir -p reports
python offline_section_repair.py \
  --snapshot snapshots/courses_<run-id>.json \
  --raw-dir raw_html \
  --output reports/repair-before.json
```

Applying a repair is intentionally separate. `--apply` requires the exact previously reviewed dry-run manifest via `--expected-before-manifest`; the command recomputes and verifies it before committing a validated staged data generation:

```bash
python offline_section_repair.py \
  --snapshot snapshots/courses_<run-id>.json \
  --raw-dir raw_html \
  --apply \
  --expected-before-manifest reports/repair-before.json \
  --output reports/repair-after.json
```
