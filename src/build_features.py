"""
Build the model features from the adjusted table.

Reads   data/processed/movies_adjusted.csv    (adjust_cpi.py)
        data/processed/tmdb_collections.csv   (tmdb_collections.py)
        data/processed/tmdb_people.csv        (tmdb_import.py)
Writes  data/processed/features.csv           identifiers, target, and numeric features
        data/processed/feature_groups.csv     feature -> group, for add/remove-a-group experiments

Feature groups: budget, content, timing, studio, franchise, director, actor

Target: log_gross_adj = ln(domestic gross in constant 2025 dollars).

NO LEAKAGE. Nothing here uses the film's own gross, and nothing from after its release:
  - post_* columns (opening gross, theaters) are not used at all.
  - Track-record and franchise features use only films whose wide release was at least
    MIN_GAP_DAYS before this film's (default 8 weeks), so their gross is essentially known.
    The script asserts this on every row it builds.
  - "is a sequel" means an EARLIER film exists in the collection. "Belongs to a collection"
    would partly encode success, because a first film only joins a collection if sequels follow.
  - Histories only count films in this data (wide releases from 2000). A director's earlier
    career, or a film that was never wide, is invisible, so histories are thin early on
    and for newcomers. Missing means "no history in the data"; counts say how thin.

Usage (from repo root):   python src/build_features.py
"""
import re
import sys
import pathlib
import datetime as dt

import numpy as np
import pandas as pd

DIR = pathlib.Path("data/processed")
MIN_GAP_DAYS = 56            # a prior film must have opened at least this long before
GENRE_MIN = 100              # a genre needs this many films to get its own column
ACTOR_TOP_BILLED = 3         # actor record = prior films where the lead was billed in the top N
ERAS = [(2000, 2009, "2000-2009"), (2010, 2019, "2010-2019"),
        (2020, 2021, "2020-2021"), (2022, 2025, "2022-2025")]

# First matching rule wins; anything else is "Other". Subsidiaries roll up to their parent.
STUDIO_RULES = [
    ("Disney", r"Disney|Buena Vista|Touchstone|Pixar|Hollywood Pictures"),
    ("Warner", r"Warner|New Line"),
    ("Universal", r"Universal|Focus Features"),
    ("Sony", r"Sony|Columbia|TriStar|Screen Gems"),
    ("Paramount", r"Paramount"),
    ("Fox", r"20th Century|Twentieth Century|\bFox (?:Searchlight|Atomic|2000)|\bSearchlight"),
    ("Lionsgate", r"Lionsgate|Summit Entertainment"),
    ("MGM", r"Metro-Goldwyn|MGM|United Artists"),
]
BIG_SIX = {"Disney", "Warner", "Universal", "Sony", "Paramount", "Fox"}

GROUP_PREFIXES = [("log_budget", "budget"), ("has_budget", "budget"),
                  ("runtime", "content"), ("mpaa_", "content"), ("genre_", "content"),
                  ("release_month", "timing"), ("month_", "timing"), ("holiday_", "timing"),
                  ("thanksgiving", "timing"), ("christmas", "timing"), ("july4", "timing"),
                  ("studio_", "studio"), ("is_big_six", "studio"),
                  ("n_prior_films", "franchise"), ("is_sequel", "franchise"),
                  ("n_prior_in_data", "franchise"), ("prior_film_", "franchise"), ("franchise_", "franchise"),
                  ("director_", "director"), ("actor_", "actor")]


# ---------- calendar helpers ----------
def _nth_weekday(year, month, weekday, n):
    d = dt.date(year, month, 1)
    d += dt.timedelta(days=(weekday - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


def _last_weekday(year, month, weekday):
    d = dt.date(year + (month == 12), month % 12 + 1, 1) - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def holiday_flags(friday):
    """Flags for the Friday that starts a Box Office Mojo weekend."""
    y = friday.year
    monday = friday + dt.timedelta(days=3)
    monday_holidays = {_nth_weekday(y, 1, 0, 3), _nth_weekday(y, 2, 0, 3),     # MLK, Presidents' Day
                       _last_weekday(y, 5, 0), _nth_weekday(y, 9, 0, 1)}       # Memorial, Labor Day
    thanksgiving = _nth_weekday(y, 11, 3, 4)
    return {"holiday_monday_weekend": int(monday in monday_holidays),
            "thanksgiving_weekend": int(friday == thanksgiving + dt.timedelta(days=1)),
            "christmas_week": int(friday.month == 12 and 20 <= friday.day <= 27),
            "july4_week": int(friday - dt.timedelta(days=3) <= dt.date(y, 7, 4) <= friday + dt.timedelta(days=2))}


# ---------- feature groups ----------
def simple_features(m):
    f = pd.DataFrame(index=m.index)
    f["log_budget_adj"] = np.log(m["budget_adj"])
    f["has_budget"] = m["budget_adj"].notna().astype(int)
    f["runtime_min"] = m["runtime_min"]
    for r in ["G", "PG", "PG-13", "R", "NR"]:
        f[f"mpaa_{r.replace('-', '')}"] = (m["mpaa"] == r).astype(int)
    genres = m["genres"].fillna("").str.split("|")
    counts = genres.explode().value_counts()
    for name in [g for g, n in counts.items() if g and n >= GENRE_MIN]:
        f[f"genre_{name.replace('-', '_')}"] = genres.map(lambda L, name=name: int(name in L))
    f["genre_count"] = genres.map(lambda L: len([g for g in L if g]))

    f["release_month"] = m["wide_date"].dt.month
    f["month_sin"] = np.sin(2 * np.pi * f["release_month"] / 12)
    f["month_cos"] = np.cos(2 * np.pi * f["release_month"] / 12)
    days = m["wide_date"].dt.date
    flags = pd.DataFrame([holiday_flags(d) for d in days], index=m.index)
    f = pd.concat([f, flags], axis=1)

    def studio(distributor):
        for name, pattern in STUDIO_RULES:
            if pd.notna(distributor) and re.search(pattern, distributor, flags=re.IGNORECASE):
                return name
        return "Other"
    studios = m["distributor"].map(studio)
    for name in [n for n, _ in STUDIO_RULES] + ["Other"]:
        f[f"studio_{name}"] = (studios == name).astype(int)
    f["is_big_six"] = studios.isin(BIG_SIX).astype(int)
    return f


def franchise_features(m, col):
    parts = col.dropna(subset=["part_release_date"])[["collection_id", "part_tmdb_id", "part_release_date"]]
    own = (parts.rename(columns={"part_tmdb_id": "tmdb_id", "part_release_date": "own_date"})
                [["tmdb_id", "own_date"]].drop_duplicates("tmdb_id"))
    base = m[["tmdb_id", "collection_id", "wide_date"]].merge(own, on="tmdb_id", how="left")
    base["own_date"] = base["own_date"].fillna(base["wide_date"])      # TMDB date, else wide date

    pairs = base.dropna(subset=["collection_id"]).merge(parts, on="collection_id")
    prior = pairs[pairs["part_release_date"] < pairs["own_date"]]        # earlier films, in data or not
    n_prior = prior.groupby("tmdb_id").size()

    films = m[["tmdb_id", "wide_date", "domestic_gross_adj"]].rename(
        columns={"tmdb_id": "part_tmdb_id", "wide_date": "part_wide", "domestic_gross_adj": "part_gross"})
    pr = prior.merge(films, on="part_tmdb_id")                            # earlier films that are in our data
    pr = pr[pr["part_wide"] <= pr["wide_date"] - pd.Timedelta(days=MIN_GAP_DAYS)]
    assert (pr["part_wide"] < pr["wide_date"]).all(), "franchise leakage: a prior film opened later"
    pr = pr.assign(part_log=np.log(pr["part_gross"]))
    last = pr.sort_values("part_wide").groupby("tmdb_id").tail(1).set_index("tmdb_id")["part_log"]

    f = pd.DataFrame(index=m.index)
    # NOTE: "belongs to a collection" is deliberately NOT a feature. A first film only joins a collection if
    # sequels were made later, so for first films it leaks hindsight about success. is_sequel (an EARLIER
    # film exists) is knowable before release and carries the legitimate franchise signal.
    f["n_prior_films"] = m["tmdb_id"].map(n_prior).fillna(0).astype(int)
    f["is_sequel"] = (f["n_prior_films"] >= 1).astype(int)
    f["n_prior_in_data"] = m["tmdb_id"].map(pr.groupby("tmdb_id").size()).fillna(0).astype(int)
    f["prior_film_log_gross"] = m["tmdb_id"].map(last)
    f["franchise_mean_log_gross"] = m["tmdb_id"].map(pr.groupby("tmdb_id")["part_log"].mean())
    return f


def track_record(m, ppl, who):
    films = m[["tmdb_id", "wide_date", "domestic_gross_adj"]].copy()
    films["log_gross"] = np.log(films["domestic_gross_adj"])
    in_data = set(films["tmdb_id"])
    if who == "director":
        credits = ppl[ppl["role"] == "director"][["tmdb_id", "person_id"]]
        targets = credits
    else:
        cast = ppl[ppl["role"] == "cast"]
        targets = cast[cast["billing_order"] == 0][["tmdb_id", "person_id"]]
        credits = cast[cast["billing_order"] < ACTOR_TOP_BILLED][["tmdb_id", "person_id"]]
    credits = credits[credits["tmdb_id"].isin(in_data)]
    targets = targets[targets["tmdb_id"].isin(in_data)]

    t = targets.merge(films[["tmdb_id", "wide_date"]].rename(columns={"wide_date": "t_date"}), on="tmdb_id")
    c = credits.rename(columns={"tmdb_id": "f_id"}).merge(
        films.rename(columns={"tmdb_id": "f_id", "wide_date": "f_date"}), on="f_id")
    pairs = t.merge(c, on="person_id")
    pairs = pairs[(pairs["f_id"] != pairs["tmdb_id"])
                  & (pairs["f_date"] <= pairs["t_date"] - pd.Timedelta(days=MIN_GAP_DAYS))]
    pairs = pairs.drop_duplicates(["tmdb_id", "f_id"])      # a film counts once even with several directors
    assert (pairs["f_date"] < pairs["t_date"]).all(), f"{who} leakage: a prior film opened later"
    assert (pairs["f_id"] != pairs["tmdb_id"]).all(), f"{who} leakage: a film counted itself"

    g = pairs.groupby("tmdb_id")
    out = pd.DataFrame({f"{who}_n_prior": g.size(),
                        f"{who}_mean_log_gross": g["log_gross"].mean(),
                        f"{who}_max_log_gross": g["log_gross"].max()})
    out = out.reindex(m["tmdb_id"].values)
    out[f"{who}_n_prior"] = out[f"{who}_n_prior"].fillna(0).astype(int)
    out.index = m.index
    return out


def main():
    m = pd.read_csv(DIR / "movies_adjusted.csv", parse_dates=["wide_date"])
    col = pd.read_csv(DIR / "tmdb_collections.csv", parse_dates=["part_release_date"])
    ppl = pd.read_csv(DIR / "tmdb_people.csv")
    assert m["tmdb_id"].is_unique, "tmdb_id must be unique in movies_adjusted.csv"

    feats = pd.concat([simple_features(m), franchise_features(m, col),
                       track_record(m, ppl, "director"), track_record(m, ppl, "actor")], axis=1)
    banned = [c for c in feats.columns if c.startswith("post_") or c in ("domestic_gross", "domestic_gross_adj")]
    assert not banned, f"leaky columns in features: {banned}"

    meta = m[["release_path", "imdb_id", "tmdb_id", "title", "wide_date", "release_year",
              "domestic_gross_adj", "delayed_release", "label_release", "event_review", "budget_disagree"]].copy()
    meta["log_gross_adj"] = np.log(m["domestic_gross_adj"])
    out = pd.concat([meta, feats], axis=1)
    out.to_csv(DIR / "features.csv", index=False)

    groups = []
    for c in feats.columns:
        grp = next((g for prefix, g in GROUP_PREFIXES if c.startswith(prefix)), None)
        if grp is None:
            sys.exit(f"feature {c} has no group - add a prefix to GROUP_PREFIXES")
        groups.append({"feature": c, "group": grp})
    gdf = pd.DataFrame(groups)
    gdf.to_csv(DIR / "feature_groups.csv", index=False)

    # ---------- report ----------
    print(f"wrote {DIR / 'features.csv'}: {len(out)} rows, {feats.shape[1]} features")
    print("\nfeatures per group:", gdf["group"].value_counts().to_dict())
    print(f"\nprior films must open >= {MIN_GAP_DAYS} days earlier (asserted on every row).")
    era = pd.cut(out["release_year"], [e[0] - 1 for e in ERAS] + [ERAS[-1][1]], labels=[e[2] for e in ERAS])
    cov = pd.DataFrame({
        "films": out.groupby(era, observed=True).size(),
        "has_budget": out.groupby(era, observed=True)["has_budget"].mean(),
        "is_sequel": out.groupby(era, observed=True)["is_sequel"].mean(),
        "prior_film_known": out["prior_film_log_gross"].notna().groupby(era, observed=True).mean(),
        "director_history": (out["director_n_prior"] > 0).groupby(era, observed=True).mean(),
        "actor_history": (out["actor_n_prior"] > 0).groupby(era, observed=True).mean(),
        "big_six": out.groupby(era, observed=True)["is_big_six"].mean()})
    print("\ncoverage by era (share of films):")
    print(cov.round(3).to_string())
    print("\nshare of films by studio:",
          feats[[c for c in feats if c.startswith("studio_")]].mean().round(3).to_dict())
    print("holiday flags:", feats[["holiday_monday_weekend", "thanksgiving_weekend",
                                   "christmas_week", "july4_week"]].sum().to_dict())
    print("\nmissing values (features with any):")
    nulls = feats.isna().mean()
    print(nulls[nulls > 0].round(3).to_string())


if __name__ == "__main__":
    main()
