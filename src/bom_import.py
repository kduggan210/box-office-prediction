"""
Box Office Mojo importer -> SQLite

Stage 1: fetch + cache yearly pages (2000-2025), parse rows incl. release links
Stage 2: fetch + cache each wide-release page for MPAA rating, runtime, IMDb ID
Results land in data/processed/movies.db (tables: bom_years, bom_releases)

Usage:
    python src/bom_import.py years      # stage 1
    python src/bom_import.py releases   # stage 2 (slow: one request per movie)

IMPORTANT: I could not test this against the live site. Run stage 1 for one
year first, inspect the printed columns, and adjust COLUMN_MAP / the release
page parser if the page layout differs from what I assumed.
Check Box Office Mojo's terms of use before scraping.
"""
import re
import sys
import time
import pathlib

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine

BASE = "https://www.boxofficemojo.com"
RAW_YEARS = pathlib.Path("data/raw/bom_years")
RAW_RELEASES = pathlib.Path("data/raw/bom_releases")
DB_URL = "sqlite:///data/processed/movies.db"
HEADERS = {"User-Agent": "CSC4850 class project (your_email@student.gsu.edu)"}
DELAY = 2.0  # seconds between uncached requests
WIDE_THEATERS = 600

# Map site column names (lowercased) -> our column names. Adjust after inspecting.
COLUMN_MAP = {
    "release": "title",
    "gross": "year_gross",
    "total gross": "total_gross",
    "max theaters": "max_theaters",
    "opening": "opening_gross",
    "open": "opening_gross",
    "opening theaters": "opening_theaters",
    "release date": "release_date",
    "distributor": "distributor",
    "theaters": "max_theaters"
}

for d in (RAW_YEARS, RAW_RELEASES, pathlib.Path("data/processed")):
    d.mkdir(parents=True, exist_ok=True)


def cached_get(url, path):
    """Fetch url once; afterwards read from disk."""
    if path.exists():
        return path.read_text(encoding="utf-8")
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    path.write_text(r.text, encoding="utf-8")
    time.sleep(DELAY)
    return r.text


def money_to_int(s):
    s = re.sub(r"[^\d]", "", str(s))
    return int(s) if s else None


def parse_year_page(html, year):
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if table is None:
        raise ValueError(f"No table found for {year}")
    headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
    rows = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        link = tr.find("a", href=re.compile(r"/release/rl\d+"))
        row = dict(zip(headers, [td.get_text(strip=True) for td in tds]))
        row["release_path"] = link["href"].split("?")[0] if link else None
        row["year"] = year
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})


def stage_years(years=range(2000, 2026)):
    engine = create_engine(DB_URL)
    frames = []
    for year in years:
        html = cached_get(f"{BASE}/year/{year}/", RAW_YEARS / f"{year}.html")
        df = parse_year_page(html, year)
        print(year, len(df), "rows; columns:", list(df.columns))
        frames.append(df)
    bom = pd.concat(frames, ignore_index=True)

    for col in ("year_gross", "total_gross", "opening_gross", "budget"):
        if col in bom.columns:
            bom[col] = bom[col].map(money_to_int)
    for col in ("max_theaters", "opening_theaters"):
        if col in bom.columns:
            bom[col] = pd.to_numeric(
                bom[col].astype(str).str.replace(",", ""), errors="coerce"
            )

    bom.to_sql("bom_years", engine, if_exists="replace", index=False)
    if "max_theaters" in bom.columns:
        wide = bom[bom["max_theaters"] >= WIDE_THEATERS]
        print(f"Saved {len(bom)} rows; {len(wide)} wide releases (>= {WIDE_THEATERS} theaters)")
    else:
        print("max_theaters column not found - fix COLUMN_MAP before filtering")


def parse_release_page(html):
    soup = BeautifulSoup(html, "lxml")
    info = {"imdb_id": None, "mpaa": None, "runtime": None, "genres": None}
    m = re.search(r"tt\d{6,9}", html)
    if m:
        info["imdb_id"] = m.group(0)
    # Summary fields are label/value span pairs, e.g. <span>MPAA</span><span>PG-13</span>
    wanted = {"mpaa": "mpaa", "running time": "runtime", "genres": "genres"}
    for div in soup.find_all("div"):
        spans = div.find_all("span", recursive=False)
        if len(spans) == 2:
            label = spans[0].get_text(strip=True).lower()
            if label in wanted:
                info[wanted[label]] = spans[1].get_text(" ", strip=True)
    return info


def stage_releases():
    engine = create_engine(DB_URL)
    bom = pd.read_sql("SELECT * FROM bom_years", engine)
    wide = bom[(bom["max_theaters"] >= WIDE_THEATERS) & bom["release_path"].notna()]
    wide = wide.drop_duplicates("release_path")
    print(f"Fetching {len(wide)} release pages (~{len(wide) * DELAY / 60:.0f} min uncached)")

    out = []
    for i, path in enumerate(wide["release_path"], 1):
        rid = path.strip("/").split("/")[-1]
        try:
            html = cached_get(BASE + path, RAW_RELEASES / f"{rid}.html")
            info = parse_release_page(html)
        except requests.RequestException as e:
            print("failed", path, e)
            continue
        info["release_path"] = path
        out.append(info)
        if i % 100 == 0:
            print(f"{i}/{len(wide)}")
            pd.DataFrame(out).to_sql("bom_releases", engine, if_exists="replace", index=False)

    pd.DataFrame(out).to_sql("bom_releases", engine, if_exists="replace", index=False)
    print("Done. Check how many rows have null imdb_id / mpaa.")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "years"
    if stage == "years":
        stage_years()
    elif stage == "releases":
        stage_releases()
    else:
        print(__doc__)
