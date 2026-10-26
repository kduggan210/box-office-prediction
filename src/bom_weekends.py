"""
Box Office Mojo weekend charts -> complete wide-release list

For each year, read the weekend index (/weekend/by-year/YEAR/), collect the
link to every weekend page, fetch + cache each one, and parse the table.
Then collapse to one row per release and keep releases that opened or played
in 600+ theaters at any point.

Writes to data/processed/movies.db:
  bom_weekends  one row per (weekend, release)
  bom_wide      one row per wide release (this replaces bom_years as the
                input to stage 2 in bom_import.py)

Usage (run from repo root):
    python src/bom_weekends.py 2006        # test a single year first
    python src/bom_weekends.py             # all years 2000-2025

NOT TESTED against the live site. Check the printed weekend counts (expect
about 52-53 per year) and the column list before running every year.
"""
import re
import sys
import pathlib
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup
from sqlalchemy import create_engine

from bom_import import BASE, DB_URL, WIDE_THEATERS, cached_get, money_to_int, parse_year_page

RAW_INDEX = pathlib.Path("data/raw/bom_weekend_index")
RAW_WEEKENDS = pathlib.Path("data/raw/bom_weekends")
for d in (RAW_INDEX, RAW_WEEKENDS):
    d.mkdir(parents=True, exist_ok=True)

def weekend_pages(index_html):
    """One (week_number, href) per week of the year.

    Long holiday weekends appear several times on the index (3-day, 4-day,
    5-day) with the same Week number. The extended pages list far fewer
    releases (e.g. 50 vs 118), so keep the page with the most releases.
    """
    soup = BeautifulSoup(index_html, "lxml")
    table = soup.find("table")
    headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
    best = {}
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        link = tr.find("a", href=re.compile(r"/weekend/\d{4}W\d+"))
        if not tds or link is None:
            continue
        row = dict(zip(headers, [td.get_text(strip=True) for td in tds]))
        try:
            week = int(row["week"])
            n = int(row["releases"].replace(",", ""))
        except (KeyError, ValueError):
            continue
        if week not in best or n > best[week][0]:
            best[week] = (n, link["href"])
    return [(week, href) for week, (n, href) in sorted(best.items())]


def stage_weekends(years):
    engine = create_engine(DB_URL)
    frames = []
    for year in years:
        idx = cached_get(f"{BASE}/weekend/by-year/{year}/", RAW_INDEX / f"{year}.html")
        pages = weekend_pages(idx)
        print(f"{year}: {len(pages)} weekends (weeks {pages[0][0]}-{pages[-1][0]})" if pages
              else f"{year}: no weekends parsed - check the index table headers")
        for week, href in pages:
            wid = f"{year}W{week:02d}"
            html = cached_get(urljoin(BASE, href), RAW_WEEKENDS / f"{wid}.html")
            df = parse_year_page(html, year)
            df["weekend_id"] = wid
            frames.append(df)

    wk = pd.concat(frames, ignore_index=True)
    print("columns:", list(wk.columns), "| rows:", len(wk))
    if "max_theaters" not in wk.columns:
        print("No theaters column found - fix COLUMN_MAP in bom_import.py, then rerun")
        return

    # parse_year_page maps the weekend 'gross' column to year_gross; rename for clarity
    wk = wk.rename(columns={"year_gross": "weekend_gross"})
    for col in ("weekend_gross", "total_gross"):
        if col in wk.columns:
            wk[col] = wk[col].map(money_to_int)
    wk["max_theaters"] = pd.to_numeric(
        wk["max_theaters"].astype(str).str.replace(",", ""), errors="coerce"
    )
    wk.to_sql("bom_weekends", engine, if_exists="replace", index=False)

    wk = wk[wk["release_path"].notna()]
    agg = {"title": ("title", "first"),
           "first_weekend": ("weekend_id", "min"),
           "max_theaters": ("max_theaters", "max")}
    if "total_gross" in wk.columns:
        agg["total_gross_seen"] = ("total_gross", "max")  # a lower bound, see notes
    wide = wk.groupby("release_path").agg(**agg).reset_index()
    wide = wide[wide["max_theaters"] >= WIDE_THEATERS]
    wide.to_sql("bom_wide", engine, if_exists="replace", index=False)
    print(f"bom_wide: {len(wide)} unique wide releases")

    # Cross-check against the 200-row yearly lists
    try:
        yearly = pd.read_sql("SELECT release_path, max_theaters, year FROM bom_years", engine)
        is_wide = yearly["max_theaters"] >= WIDE_THEATERS
        yw_all = set(yearly[is_wide]["release_path"].dropna())
        yw_run = set(yearly[is_wide & yearly["year"].isin(list(years))]["release_path"].dropna())
        w = set(wide["release_path"])
        print("wide releases the yearly lists missed (cut off by the 200-row cap):", len(w - yw_all))
        print("in yearly lists for these years but not in weekend list:", len(yw_run - w),
              "(carryover from the prior December is expected)")
    except Exception as e:
        print("skipped yearly cross-check:", e)


if __name__ == "__main__":
    yrs = [int(a) for a in sys.argv[1:]] or range(2000, 2026)
    stage_weekends(yrs)