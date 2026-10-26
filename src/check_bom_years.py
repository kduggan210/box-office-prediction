import pandas as pd
from sqlalchemy import create_engine

bom = pd.read_sql("SELECT * FROM bom_years", create_engine("sqlite:///data/processed/movies.db"))
check = bom.groupby("year").agg(
    rows=("title", "count"),
    wide=("max_theaters", lambda s: (s >= 600).sum()),
    min_theaters=("max_theaters", "min"),
)

bom["rank"] = pd.to_numeric(bom["rank"], errors="coerce")
counts = bom["release_path"].map(bom["release_path"].value_counts())
tail = bom[bom["rank"] >= 181]
solo_wide_tail = tail[(tail["max_theaters"] >= 600) & (counts[tail.index] == 1)]
print(solo_wide_tail.groupby("year").size())

#print(tail.groupby("year")["max_theaters"].agg(
#    wide_in_tail=lambda s: (s >= 600).sum(),
#    median_theaters="median",
#))


#print(check)