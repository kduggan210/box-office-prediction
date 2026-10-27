"""
TMDB importer: budget, runtime, franchise, studios, director(s) and cast for every
unique IMDb ID in the Box Office Mojo data.

For each IMDb ID:   /find/{imdb_id}  ->  TMDB movie id  ->  /movie/{id}?append_to_response=credits
Raw JSON is cached under data/raw/tmdb_find and data/raw/tmdb_movie.

Tables written to data/processed/movies.db:
  tmdb_movies      one row per IMDb ID (budget, collection, studios, directors, lead actor)
  tmdb_people      one row per (film, person): directors plus the top 10 billed cast
  tmdb_failures    IDs that could not be fetched or parsed (rerun to retry)
  bom_tmdb_match   one row per Box Office Mojo film with title and year checks (see below)

Match checks, because Box Office Mojo IDs were wrong for some films once already:
  title_suspect  TMDB's title and ours look unrelated -> possibly the wrong film
  year_suspect   our release year is 2+ years after TMDB's -> usually a re-release of an
                 older film (keep the original run only), occasionally a wrong match

Setup (PowerShell):   $env:TMDB_API_KEY = "your_key"
  Accepts either the v3 API key or the v4 read access token (starts with "eyJ").

Usage (from repo root):
    python src/tmdb_import.py --limit 20     # test batch first
    python src/tmdb_import.py                # everything not yet fetched
    python src/tmdb_import.py --reset        # drop the tmdb tables, rebuild from cached JSON

Works on the machine that has movies.db; if the bom tables are missing it falls back to
data/processed/bom_movies.csv, so it also runs from a fresh clone.

NOT TESTED against the live API. Check the 20-film test before the full run.
"""
import os
import re
import sys
import json
import time
import difflib
import argparse
import pathlib

import pandas as pd
import requests
from sqlalchemy import create_engine, inspect

DB_URL = "sqlite:///data/processed/movies.db"
API = "https://api.themoviedb.org/3"
RAW_FIND = pathlib.Path("data/raw/tmdb_find")
RAW_MOVIE = pathlib.Path("data/raw/tmdb_movie")
DELAY = 0.1                      # seconds between uncached requests (TMDB allows far more)
MAX_CONSECUTIVE_FAILURES = 25
FLUSH_EVERY = 100
AUTH = {"headers": {}, "params": {}, "key": ""}

for d in (RAW_FIND, RAW_MOVIE):
    d.mkdir(parents=True, exist_ok=True)


def setup_auth():
    key = os.environ.get("TMDB_API_KEY", "").strip()
    if not key:
        sys.exit("Set TMDB_API_KEY first, e.g.  $env:TMDB_API_KEY = 'your_key'  (PowerShell)")
    AUTH["key"] = key
    if key.startswith("eyJ"):    # v4 read access token
        AUTH["headers"] = {"Authorization": f"Bearer {key}"}
    else:                        # v3 API key
        AUTH["params"] = {"api_key": key}


def scrub(text):
    """Never let the API key reach the console or the failures table."""
    return text.replace(AUTH["key"], "***") if AUTH["key"] else text


def get_json(path, cache, params=None, retries=4):
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    for attempt in range(retries):
        r = requests.get(f"{API}{path}", headers=AUTH["headers"],
                         params={**AUTH["params"], **(params or {})}, timeout=30)
        if r.status_code == 401:
            sys.exit("TMDB rejected the API key (401). Check TMDB_API_KEY.")
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", 5)) + 1)
            continue
        if r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        data = r.json()
        cache.write_text(json.dumps(data), encoding="utf-8")
        time.sleep(DELAY)
        return data
    raise RuntimeError(f"gave up on {path} after {retries} tries")


def fetch_movie(imdb_id):
    found = get_json(f"/find/{imdb_id}", RAW_FIND / f"{imdb_id}.json",
                     {"external_source": "imdb_id"})
    results = found.get("movie_results") or []
    if not results:
        raise LookupError("no TMDB movie for this IMDb ID")
    tmdb_id = results[0]["id"]
    return get_json(f"/movie/{tmdb_id}", RAW_MOVIE / f"{tmdb_id}.json",
                    {"append_to_response": "credits"})


def parse_movie(m):
    credits = m.get("credits") or {}
    cast = sorted(credits.get("cast") or [], key=lambda c: c.get("order", 999))
    directors = [c for c in (credits.get("crew") or []) if c.get("job") == "Director"]
    coll = m.get("belongs_to_collection") or {}
    companies = m.get("production_companies") or []
    row = {
        "tmdb_id": m["id"],
        "tmdb_title": m.get("title"),
        "tmdb_original_title": m.get("original_title"),
        "tmdb_release_date": m.get("release_date") or None,
        "tmdb_budget": m.get("budget") or None,       # TMDB uses 0 for "unknown"
        "tmdb_runtime": m.get("runtime") or None,
        "original_language": m.get("original_language"),
        "collection_id": coll.get("id"),
        "collection_name": coll.get("name"),
        "genres": "|".join(g["name"] for g in m.get("genres") or []),
        "production_companies": "|".join(c["name"] for c in companies),
        "production_company_ids": "|".join(str(c["id"]) for c in companies),
        "directors": "|".join(c["name"] for c in directors),
        "director_ids": "|".join(str(c["id"]) for c in directors),
        "lead_actor": cast[0]["name"] if cast else None,
        "lead_actor_id": cast[0]["id"] if cast else None,
    }
    people = [{"tmdb_id": m["id"], "person_id": c["id"], "name": c["name"],
               "role": "director", "billing_order": None} for c in directors]
    people += [{"tmdb_id": m["id"], "person_id": c["id"], "name": c["name"],
                "role": "cast", "billing_order": c.get("order")} for c in cast[:10]]
    return row, people


def norm_title(t):
    if not isinstance(t, str):      # NaN for films with no TMDB match
        return ""
    t = re.sub(r"[^a-z0-9 ]", "", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def build_match(bom, tm):
    """One row per BOM film: does the TMDB film we matched look like the same movie?"""
    m = bom.merge(tm[["imdb_id", "tmdb_id", "tmdb_title", "tmdb_release_date", "tmdb_budget"]],
                  on="imdb_id", how="left")
    bom_year = pd.to_numeric(m["release_date"].astype(str).str[:4], errors="coerce")
    fw_year = pd.to_numeric(m["first_weekend"].astype(str).str[:4], errors="coerce")
    m["bom_year"] = bom_year.fillna(fw_year)
    m["tmdb_year"] = pd.to_numeric(m["tmdb_release_date"].astype(str).str[:4], errors="coerce")
    m["year_gap"] = m["bom_year"] - m["tmdb_year"]
    a = m["title"].map(norm_title)
    b = m["tmdb_title"].map(norm_title)
    m["title_ratio"] = [difflib.SequenceMatcher(None, x, y).ratio() if y else None
                        for x, y in zip(a, b)]
    m["title_contains"] = [bool(x and y) and (y in x or x in y) for x, y in zip(a, b)]
    m["title_suspect"] = m["tmdb_id"].notna() & ~m["title_contains"] & (m["title_ratio"] < 0.6)
    m["year_suspect"] = m["year_gap"].abs() >= 2
    return m


def load_bom(engine):
    if inspect(engine).has_table("bom_release_details"):
        return pd.read_sql("""
            SELECT w.title, w.release_path, w.first_weekend, r.imdb_id,
                   r.release_date, r.budget AS bom_budget
            FROM bom_wide w JOIN bom_release_details r USING (release_path)
        """, engine)
    print("no bom tables in movies.db - using data/processed/bom_movies.csv")
    cols = ["title", "release_path", "first_weekend", "imdb_id", "release_date", "budget"]
    return pd.read_csv("data/processed/bom_movies.csv", usecols=cols).rename(
        columns={"budget": "bom_budget"})


def main(limit=None, reset=False):
    setup_auth()
    engine = create_engine(DB_URL)
    if reset:
        with engine.begin() as conn:
            for t in ("tmdb_movies", "tmdb_people", "tmdb_failures", "bom_tmdb_match"):
                conn.exec_driver_sql(f"DROP TABLE IF EXISTS {t}")
        print("dropped the tmdb tables")

    bom = load_bom(engine)
    ids = list(dict.fromkeys(bom["imdb_id"].dropna()))
    done = set()
    if inspect(engine).has_table("tmdb_movies"):
        done = set(pd.read_sql("SELECT imdb_id FROM tmdb_movies", engine)["imdb_id"])
    todo = [i for i in ids if i not in done]
    if limit:
        todo = todo[:limit]
    print(f"{len(ids)} unique IMDb IDs, {len(done)} already fetched, fetching {len(todo)}")

    rows, people, failures, streak = [], [], [], 0

    def flush():
        if rows:
            pd.DataFrame(rows).to_sql("tmdb_movies", engine, if_exists="append", index=False)
            rows.clear()
        if people:
            pd.DataFrame(people).to_sql("tmdb_people", engine, if_exists="append", index=False)
            people.clear()

    try:
        for i, iid in enumerate(todo, 1):
            try:
                row, ppl = parse_movie(fetch_movie(iid))
                row["imdb_id"] = iid
                for p in ppl:
                    p["imdb_id"] = iid
                rows.append(row)
                people.extend(ppl)
                streak = 0
            except Exception as e:     # one bad film must not kill the run
                failures.append({"imdb_id": iid, "error": scrub(repr(e))[:300]})
                streak += 1
                if streak >= MAX_CONSECUTIVE_FAILURES:
                    print(f"{streak} failures in a row - stopping. Last error: {scrub(repr(e))}")
                    break
            if i % FLUSH_EVERY == 0:
                flush()
                print(f"{i}/{len(todo)} done, {len(failures)} failed so far")
    finally:
        flush()
        if failures:
            pd.DataFrame(failures).to_sql("tmdb_failures", engine, if_exists="append", index=False)

    tm = pd.read_sql("SELECT * FROM tmdb_movies", engine)
    print(f"\ntmdb_movies: {len(tm)} rows | failures this run: {len(failures)}")
    shares = tm[["tmdb_budget", "tmdb_runtime", "collection_id", "directors",
                 "lead_actor", "production_companies"]].replace("", None).notna().mean().round(3)
    print("share of TMDB rows with a value:")
    print(shares.to_string())

    m = build_match(bom, tm)
    m[["release_path", "imdb_id", "title", "tmdb_id", "tmdb_title", "title_ratio",
       "title_contains", "bom_year", "tmdb_year", "year_gap",
       "title_suspect", "year_suspect"]].to_sql("bom_tmdb_match", engine,
                                                if_exists="replace", index=False)
    fetched = m["tmdb_id"].notna()
    print(f"\nBOM films matched to a TMDB movie: {int(fetched.sum())} of {len(m)}")
    print("title_suspect:", int(m["title_suspect"].sum()),
          "| year_suspect (gap of 2+ years):", int(m["year_suspect"].sum()))
    has_budget = (m["bom_budget"].notna() | m["tmdb_budget"].notna()).mean()
    print(f"films with a budget from BOM or TMDB: {has_budget:.1%}")
    if m["title_suspect"].any():
        print("\nTitle mismatches to review:")
        print(m.loc[m["title_suspect"], ["title", "tmdb_title", "imdb_id", "title_ratio"]]
              .head(25).to_string(index=False))
    if failures:
        print("\nRerun the same command to retry failed IDs (failed lookups are not cached).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only fetch this many IDs (for testing)")
    ap.add_argument("--reset", action="store_true",
                    help="drop the tmdb tables first and rebuild (uses cached JSON)")
    args = ap.parse_args()
    main(args.limit, args.reset)
