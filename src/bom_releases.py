"""
Stage 2: Box Office Mojo release pages -> table bom_release_details

Reads every wide release in bom_wide, fetches (and caches) its release page,
and extracts MPAA rating, runtime, genres, release date, budget, opening
gross, IMDb ID and a domestic gross guess. EVERY label/value pair found in the
page's summary section is also saved as JSON in `summary_json`, so a field we
did not plan for can be recovered later without re-requesting anything.

Built to run unattended:
  - resumable: releases already in bom_release_details are skipped
  - retries failed requests with a growing pause
  - one bad page is logged to bom_release_failures and skipped
  - stops itself if 25 pages in a row fail (probably blocked or offline)
  - saves progress every 50 pages

Usage (from repo root):
    python src/bom_releases.py --limit 20     # test batch first
    python src/bom_releases.py                # everything not yet parsed

"""
import re
import json
import time
import argparse
from collections import Counter

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, inspect

from bom_import import BASE, HEADERS, DB_URL, DELAY, RAW_RELEASES, money_to_int

# Plain-text fields: site label (lowercase) -> column name.
# Labels confirmed from a real page's summary_json in the 20-page test.
LABELS = {
    "mpaa": "mpaa",
    "running time": "runtime",
    "genres": "genres",
    "distributor": "distributor",
    "release date": "release_date_raw",
}
MAX_CONSECUTIVE_FAILURES = 25
FLUSH_EVERY = 50


def fetch_release(path, retries=3):
    rid = path.strip("/").split("/")[-1]
    cache = RAW_RELEASES / f"{rid}.html"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    for attempt in range(retries):
        try:
            r = requests.get(BASE + path, headers=HEADERS, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.RequestException(f"status {r.status_code}")
            r.raise_for_status()
            cache.write_text(r.text, encoding="utf-8")
            time.sleep(DELAY)
            return r.text
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(30 * (attempt + 1))


def find_imdb_id(html):
    """The film's own IMDb ID.

    The first 'tt' number in the HTML is NOT reliable: synopses link to related
    films (a sequel's page links to its predecessor). The film's own ID is the one
    the pro.imdb.com links (/cast, /filmmakers, /boxoffice, ...) point to.
    """
    ids = Counter(re.findall(r"pro\.imdb\.com/title/(tt\d{6,9})", html))
    if ids:
        return ids.most_common(1)[0][0]
    m = re.search(r"/title/(tt\d{6,9})/\?ref_=bo_rl_ti", html)  # the page's own title link
    if m:
        return m.group(1)
    m = re.search(r'<option value="/title/(tt\d{6,9})/"\s+selected', html)
    return m.group(1) if m else None


def clean_genres(s):
    """'Horror\\r\\n  \u00b7\\r\\n  Mystery' -> 'Horror|Mystery'"""
    if not s:
        return None
    parts = [g.strip() for g in re.split("\u00b7", s) if g.strip()]
    return "|".join(parts) or None


def clean_distributor(s):
    if not s:
        return None
    return s.replace("See full company information", "").strip() or None


def first_money(s):
    """First $ amount in a string, e.g. '$29,651,193 3,575 theaters' -> 29651193."""
    m = re.search(r"\$([\d,]+)", s or "")
    return money_to_int(m.group(1)) if m else None


def first_int(s):
    m = re.search(r"\d[\d,]*", s or "")
    return int(m.group(0).replace(",", "")) if m else None


def runtime_minutes(s):
    """'1 hr 40 min' -> 100"""
    if not s:
        return None
    h = re.search(r"(\d+)\s*hr", s)
    m = re.search(r"(\d+)\s*min", s)
    if not (h or m):
        return None
    return (int(h.group(1)) if h else 0) * 60 + (int(m.group(1)) if m else 0)


def parse_date_range(s):
    """'Nov 10, 2017 - Feb 15, 2018' -> ('2017-11-10', 'Feb 15, 2018').
    If the first date has no year (same-year runs), borrow it from the end date."""
    if not s:
        return None, None
    parts = re.split(r"\s+[-\u2013\u2014]\s+", s.strip())
    start = parts[0]
    end = parts[1] if len(parts) > 1 else None
    if not re.search(r"\d{4}", start) and end:
        y = re.search(r"\d{4}", end)
        if y:
            start = f"{start}, {y.group(0)}"
    d = pd.to_datetime(start, errors="coerce")
    return (d.strftime("%Y-%m-%d") if pd.notna(d) else None), end


def parse_release(html):
    soup = BeautifulSoup(html, "lxml")
    pairs = {}
    for div in soup.find_all("div"):
        spans = div.find_all("span", recursive=False)
        if len(spans) == 2:
            label = spans[0].get_text(" ", strip=True).lower()
            value = spans[1].get_text(" ", strip=True)
            if label and value and len(label) < 40 and label not in pairs:
                pairs[label] = value
    info = {col: pairs.get(label) for label, col in LABELS.items()}

    info["imdb_id"] = find_imdb_id(html)
    info["genres"] = clean_genres(info.get("genres"))
    info["distributor"] = clean_distributor(info.get("distributor"))

    # The domestic label carries the share inside it, e.g. "domestic ( 57.6% )"
    dom = next((v for k, v in pairs.items() if k.startswith("domestic")), None)
    info["domestic_gross"] = first_money(dom)
    info["budget"] = first_money(pairs.get("budget"))
    # opening looks like "$29,651,193 3,575 theaters"
    opening = pairs.get("opening")
    info["opening_gross"] = first_money(opening)
    t = re.search(r"([\d,]+)\s*theaters", opening or "")
    info["opening_theaters"] = int(t.group(1).replace(",", "")) if t else None
    info["widest_release_theaters"] = first_int(pairs.get("widest release"))
    info["runtime_min"] = runtime_minutes(info.get("runtime"))
    info["release_date"], info["close_date"] = parse_date_range(info.get("release_date_raw"))

    info["summary_json"] = json.dumps(pairs)
    return info


def main(limit=None, reset=False):
    engine = create_engine(DB_URL)
    if reset:  # reparse from the cached HTML after a parser change (no new requests)
        with engine.begin() as conn:
            conn.exec_driver_sql("DROP TABLE IF EXISTS bom_release_details")
            conn.exec_driver_sql("DROP TABLE IF EXISTS bom_release_failures")
        print("dropped bom_release_details and bom_release_failures")
    wide = pd.read_sql("SELECT release_path FROM bom_wide WHERE release_path IS NOT NULL", engine)
    paths = list(dict.fromkeys(wide["release_path"]))

    done = set()
    if inspect(engine).has_table("bom_release_details"):
        done = set(pd.read_sql("SELECT release_path FROM bom_release_details", engine)["release_path"])
    todo = [p for p in paths if p not in done]
    if limit:
        todo = todo[:limit]
    print(f"{len(paths)} wide releases, {len(done)} already parsed, fetching {len(todo)}")

    rows, failures, streak = [], [], 0

    def flush():
        if rows:
            pd.DataFrame(rows).to_sql("bom_release_details", engine, if_exists="append", index=False)
            rows.clear()

    try:
        for i, path in enumerate(todo, 1):
            try:
                info = parse_release(fetch_release(path))
                info["release_path"] = path
                rows.append(info)
                streak = 0
            except Exception as e:  # one bad page must not kill a multi-hour run
                failures.append({"release_path": path, "error": repr(e)[:300]})
                streak += 1
                if streak >= MAX_CONSECUTIVE_FAILURES:
                    print(f"{streak} failures in a row - stopping. Last error: {e!r}")
                    break
            if i % FLUSH_EVERY == 0:
                flush()
                print(f"{i}/{len(todo)} done, {len(failures)} failed so far")
    finally:
        flush()
        if failures:
            pd.DataFrame(failures).to_sql("bom_release_failures", engine, if_exists="append", index=False)

    details = pd.read_sql("SELECT * FROM bom_release_details", engine)
    print(f"\nbom_release_details: {len(details)} rows | failures this run: {len(failures)}")
    cols = ["mpaa", "runtime_min", "genres", "release_date", "budget", "imdb_id",
            "domestic_gross", "opening_gross", "opening_theaters", "widest_release_theaters"]
    print("share of rows with a value:")
    print(details[cols].notna().mean().round(3).to_string())
    if failures:
        print("Rerun the same command to retry failed pages (failed pages are not cached).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only fetch this many pages (for testing)")
    ap.add_argument("--reset", action="store_true",
                    help="drop the detail tables first and reparse (uses cached pages)")
    args = ap.parse_args()
    main(args.limit, args.reset)