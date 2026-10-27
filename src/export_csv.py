"""
Export the joined Box Office Mojo tables to a small CSV that can be committed
to git (data/processed/bom_movies.csv). The raw cache and movies.db stay local.

Usage (from repo root):
    python src/export_csv.py
"""
import pandas as pd
from sqlalchemy import create_engine, inspect

e = create_engine("sqlite:///data/processed/movies.db")

df = pd.read_sql("""
    SELECT w.title, w.release_path, w.first_weekend,
           w.max_theaters AS weekend_max_theaters, w.total_gross_seen,
           r.mpaa, r.runtime_min, r.genres, r.distributor,
           r.release_date_raw, r.release_date, r.close_date,
           r.imdb_id, r.domestic_gross, r.budget,
           r.opening_gross, r.opening_theaters, r.widest_release_theaters
    FROM bom_wide w
    LEFT JOIN bom_release_details r USING (release_path)
""", e)

df.to_csv("data/processed/bom_movies.csv", index=False)
print(f"wrote data/processed/bom_movies.csv: {len(df)} rows")
print("rows with details:", int(df["imdb_id"].notna().sum()))
print("rows without details (run bom_releases.py again if this is large):",
      int(df["imdb_id"].isna().sum()))

# TMDB tables (present after src/tmdb_import.py has run)
existing = inspect(e).get_table_names()
for table in ("tmdb_movies", "tmdb_people", "bom_tmdb_match"):
    if table in existing:
        out = pd.read_sql(f"SELECT * FROM {table}", e)
        out.to_csv(f"data/processed/{table}.csv", index=False)
        print(f"wrote data/processed/{table}.csv: {len(out)} rows")

# First weekend each film played in 600+ theaters (its wide-release weekend)
if "bom_weekends" in existing:
    wk = pd.read_sql("SELECT release_path, weekend_id, max_theaters FROM bom_weekends", e)
    wide_wk = (wk[wk["max_theaters"] >= 600].groupby("release_path")["weekend_id"].min()
               .rename("wide_weekend_id").reset_index())
    wide_wk.to_csv("data/processed/bom_wide_weekend.csv", index=False)
    print(f"wrote data/processed/bom_wide_weekend.csv: {len(wide_wk)} rows")
