"""
TMDB collections: the title and release date of EVERY film in each franchise
("collection") that appears in the cleaned data.

Why: "belongs to a collection" partly encodes success, because the first film of a
series only joins a collection if sequels were made. The safe feature is "is a sequel":
the film has a collection partner that was released EARLIER. That needs every film in
the collection, including ones that are not in this dataset (pre-2000, not wide).

Reads   data/processed/movies_clean.csv   (column collection_id)
Writes  data/processed/tmdb_collections.csv
        one row per (collection, film): collection_id, collection_name, part_tmdb_id,
        part_title, part_release_date
Raw JSON is cached in data/raw/tmdb_collection/, so reruns are instant.

Needs only the committed CSV and a TMDB key, so it can run on any machine.

Setup (PowerShell):   $env:TMDB_API_KEY = "your_key"
Usage (repo root):    python src/tmdb_collections.py --limit 20     # test first
                      python src/tmdb_collections.py

NOT TESTED against the live API (only against mock JSON).
"""
import argparse
import pathlib

import pandas as pd

from tmdb_import import setup_auth, get_json, scrub

RAW = pathlib.Path("data/raw/tmdb_collection")
RAW.mkdir(parents=True, exist_ok=True)
OUT = pathlib.Path("data/processed/tmdb_collections.csv")
MAX_CONSECUTIVE_FAILURES = 25


def main(limit=None):
    setup_auth()
    clean = pd.read_csv("data/processed/movies_clean.csv", usecols=["collection_id"])
    ids = sorted(clean["collection_id"].dropna().astype(int).unique())
    if limit:
        ids = ids[:limit]
    print(f"{len(ids)} collections to fetch")

    rows, failures, streak = [], [], 0
    for i, cid in enumerate(ids, 1):
        try:
            data = get_json(f"/collection/{cid}", RAW / f"{cid}.json")
            for p in data.get("parts") or []:
                rows.append({"collection_id": cid, "collection_name": data.get("name"),
                             "part_tmdb_id": p["id"], "part_title": p.get("title"),
                             "part_release_date": p.get("release_date") or None})
            streak = 0
        except Exception as e:
            failures.append((cid, scrub(repr(e))[:200]))
            streak += 1
            if streak >= MAX_CONSECUTIVE_FAILURES:
                print(f"{streak} failures in a row - stopping. Last error: {scrub(repr(e))}")
                break
        if i % 100 == 0:
            print(f"{i}/{len(ids)} done, {len(failures)} failed so far")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    n_ok = len(ids) - len(failures)
    print(f"\nwrote {OUT}: {len(out)} rows from {n_ok} collections | failures: {len(failures)}")
    if len(out):
        sizes = out.groupby("collection_id").size()
        print("films per collection:", sizes.describe().round(1).to_dict())
        print("parts with no release date:", int(out["part_release_date"].isna().sum()))
        films = clean["collection_id"].dropna().astype(int)
        print(f"cleaned films in a collection of 2+ films: {films.map(sizes).ge(2).sum()} of {len(films)}")
    for cid, err in failures[:10]:
        print("failed:", cid, err)
    if failures:
        print("Rerun the same command to retry failed collections.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only fetch this many collections (testing)")
    main(ap.parse_args().limit)
