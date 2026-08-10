# Columbia Engineering Course Scraper

## Run

```bash
python columbia_engineering_courses/scrape_columbia_courses.py \
  --seed "https://bulletin.columbia.edu/columbia-engineering/about-school/" \
  --year "2025-2026" \
  --root "columbia_engineering_courses"
```

## Output

- `columbia_engineering_courses/raw_html/`
- `columbia_engineering_courses/snapshots/`
- `columbia_engineering_courses/index/`
- `columbia_engineering_courses/reports/`
- `columbia_engineering_courses/logs/`

## Notes

- Each course record includes `description` and `sections`.
- Dedup is course-level via `dedup_key`.
- Historical snapshots are preserved.
