"""
Compare IMDb IDs before the parser fix (old data/processed/bom_movies.csv)
with IDs after the reparse (bom_release_details in movies.db).

Run AFTER `python src/bom_releases.py --reset` and BEFORE `python src/export_csv.py`,
because the export overwrites the old CSV.
"""
import pandas as pd
from sqlalchemy import create_engine

e = create_engine("sqlite:///data/processed/movies.db")
new = pd.read_sql("""
    SELECT w.title, w.release_path, r.imdb_id AS new_id
    FROM bom_wide w JOIN bom_release_details r USING (release_path)
""", e)
old = pd.read_csv("data/processed/bom_movies.csv", usecols=["release_path", "imdb_id"])
old = old.rename(columns={"imdb_id": "old_id"})

m = new.merge(old, on="release_path")
changed = m[m["new_id"] != m["old_id"]]
print(f"IDs changed: {len(changed)} of {len(m)}")
print(changed.to_string(index=False))

dup = new[new["new_id"].duplicated(keep=False)]
print(f"\nrows still sharing an ID: {len(dup)} across {dup['new_id'].nunique()} IDs")
print("missing IDs:", int(new["new_id"].isna().sum()))

for t in ["Avengers: Endgame", "Ocean's Twelve", "Epic Movie", "Meet the Spartans"]:
    r = new[new["title"] == t]
    print(t, "->", r["new_id"].iloc[0] if len(r) else "not found")
