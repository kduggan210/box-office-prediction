import pandas as pd
from sqlalchemy import create_engine
e = create_engine("sqlite:///data/processed/movies.db")
f = pd.read_sql("SELECT * FROM tmdb_failures", e)
bom = pd.read_sql("SELECT w.title, r.imdb_id FROM bom_wide w JOIN bom_release_details r USING (release_path)", e)
print(f.merge(bom, on="imdb_id").to_string(index=False))