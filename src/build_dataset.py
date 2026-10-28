"""
Build the modeling table from the exported CSVs (no database needed).

Reads (data/processed/):  bom_movies.csv, bom_wide_weekend.csv, bom_tmdb_match.csv, tmdb_movies.csv
Writes (data/processed/):
  movies_clean.csv               one row per film kept
  removed_rows.csv               every film removed, with the reason
  review_event_candidates.csv    kept films that MIGHT be event cinema (you decide)
  review_year_gap.csv            kept films with a re-release label or a 2-9 year gap
  manual_exclude.csv             (optional, you create it) a column `release_path` of films to drop too

Rows are removed for the FIRST matching reason, in this order:
  no_tmdb_match   TMDB has no movie record: opera broadcasts, TV episodes, live events
  before_2000     released before 2000 (carryover from the previous year's chart)
  event_cinema    unambiguous live-event wording or a performing-arts company
  rerelease       a later run of a film already in the data, a re-release label with a
                  2+ year gap to TMDB's original date, or a 10+ year gap with no label
  manual          listed in manual_exclude.csv

Columns starting with post_ are only known AFTER release. Keep them for analysis,
never use them as model inputs.

Usage (from repo root):   python src/build_dataset.py
"""
import re
import pathlib
import datetime as dt

import pandas as pd

DIR = pathlib.Path("data/processed")

# Labels Box Office Mojo glues onto re-release titles ("Toy Story2009 Re-release")
RERELEASE_LABEL = (r"re-?release|anniversary|3D Release|Restoration|Double Bill|Fathom|IMAX|Remaster|"
                   r"Live Viewing|Recut|Special Edition|Extended Cut|R-Rated|PG-13 Version|Fest \d{4}")
# Unambiguous live events / TV shown in theaters
EVENT_TITLE = (r"Live Viewing|Live Broadcast|Live on Stage|Live at the|Sing-?Along|\bQ&A\b|Season \d|"
               r"\bEpisodes? \d|\bS\d Episodes|(?i:the chosen)|(?i:\blive$)")
PERFORMING_ARTS = (r"Metropolitan Opera|Royal Opera House|Royal Ballet|Bolshoi|National Theatre|"
                   r"Op[eé]ra national|Teatro alla Scala")
# Only used to build the review list, never to remove rows on their own
EVENT_DISTRIBUTOR = r"Fathom|Trafalgar|Iconic Events|Cinedigm|Eagle Rock|Abramorama"
EVENT_TITLE_SOFT = r"\bLive\b|Concert|\bTour\b|Encore|Marathon"

POST_RELEASE = {"opening_gross": "post_opening_gross", "opening_theaters": "post_opening_theaters",
                "widest_release_theaters": "post_widest_theaters"}


def friday_of(weekend_id):
    """Box Office Mojo week N = the Nth Friday of the year (checked: within 3 days of the
    page release date for 99% of films that opened wide)."""
    year, week = int(weekend_id[:4]), int(weekend_id[-2:])
    first = dt.date(year, 1, 1)
    first += dt.timedelta(days=(4 - first.weekday()) % 7)
    return first + dt.timedelta(weeks=week - 1)


def load():
    b = pd.read_csv(DIR / "bom_movies.csv")
    w = pd.read_csv(DIR / "bom_wide_weekend.csv")
    m = pd.read_csv(DIR / "bom_tmdb_match.csv", usecols=["release_path", "tmdb_id", "tmdb_title", "year_gap"])
    t = (pd.read_csv(DIR / "tmdb_movies.csv")
         .rename(columns={"genres": "tmdb_genres"}).drop(columns=["tmdb_id", "tmdb_title"]))
    df = (b.merge(w, on="release_path", how="left")
           .merge(m, on="release_path", how="left")
           .merge(t, on="imdb_id", how="left"))
    assert len(df) == len(b), "a merge multiplied rows - check for duplicate keys"
    return df


def main():
    df = load()
    df = df.sort_values(["wide_weekend_id", "first_weekend", "release_path"]).reset_index(drop=True)
    start = len(df)

    df["wide_date"] = pd.to_datetime(df["wide_weekend_id"].map(friday_of))
    df["bom_release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["labeled"] = df["title"].str.contains(RERELEASE_LABEL, case=False, na=False)
    df["is_earliest_run"] = ~df.duplicated("imdb_id", keep="first")   # sorted by wide weekend
    gap = df["year_gap"].abs()

    pre2000 = (df["bom_release_date"] < "2000-01-01") | (
        df["bom_release_date"].isna() & (df["wide_weekend_id"] <= "2000W06")
        & (pd.to_datetime(df["tmdb_release_date"], errors="coerce") < "2000-01-01"))
    event_strong = (df["title"].str.contains(EVENT_TITLE, na=False)
                    | df["tmdb_title"].fillna("").str.contains(PERFORMING_ARTS, case=False)
                    | df["production_companies"].fillna("").str.contains(PERFORMING_ARTS, case=False))
    rerelease = ~df["is_earliest_run"] | (df["labeled"] & (gap >= 2)) | (gap >= 10)

    manual = pd.Series(False, index=df.index)
    mpath = DIR / "manual_exclude.csv"
    if mpath.exists():
        manual = df["release_path"].isin(pd.read_csv(mpath)["release_path"])

    removed, kept_mask = [], pd.Series(True, index=df.index)
    steps = [("no_tmdb_match", df["tmdb_id"].isna()), ("before_2000", pre2000),
             ("event_cinema", event_strong), ("rerelease", rerelease), ("manual", manual)]
    funnel = [("start", start)]
    for reason, mask in steps:
        hit = kept_mask & mask
        removed.append(df[hit].assign(removed_reason=reason))
        kept_mask &= ~mask
        funnel.append((f"- {reason}", int(hit.sum())))
    df = df[kept_mask].copy()
    funnel.append(("kept", len(df)))

    # ---- flags on kept rows (reviewed, not removed) ----
    df["label_release"] = df["labeled"]                       # labeled, but its only wide run
    df["delayed_release"] = (~df["labeled"]) & gap.loc[df.index].between(2, 9)
    reasons = pd.DataFrame(index=df.index)
    reasons["event_distributor"] = df["distributor"].fillna("").str.contains(EVENT_DISTRIBUTOR, case=False)
    reasons["music_documentary"] = df["tmdb_genres"].fillna("").str.contains("Music") & \
        df["tmdb_genres"].fillna("").str.contains("Documentary")
    reasons["runtime_outlier"] = (df["runtime_min"] < 70) | (df["runtime_min"] > 220)
    reasons["live_or_concert_title"] = df["title"].str.contains(EVENT_TITLE_SOFT, na=False)
    df["event_review"] = reasons.any(axis=1)

    # ---- combine fields ----
    tmdb_budget = df["tmdb_budget"].where(df["tmdb_budget"] >= 1000)    # TMDB placeholders (e.g. $12)
    df["budget_bom"], df["budget_tmdb"] = df["budget"], tmdb_budget
    df["budget"] = tmdb_budget.fillna(df["budget_bom"])
    df["budget_source"] = None
    df.loc[df["budget_bom"].notna(), "budget_source"] = "bom"
    df.loc[tmdb_budget.notna(), "budget_source"] = "tmdb"
    ratio = tmdb_budget / df["budget_bom"]
    df["budget_disagree"] = (ratio < 0.5) | (ratio > 2)
    df["runtime_min"] = df["runtime_min"].fillna(df["tmdb_runtime"])
    df["genres"] = df["genres"].fillna(df["tmdb_genres"])
    df["mpaa"] = df["mpaa"].fillna("NR")
    df["release_year"] = df["wide_date"].dt.year
    df["release_month"] = df["wide_date"].dt.month
    df = df.rename(columns=POST_RELEASE)

    keep_cols = ["release_path", "imdb_id", "tmdb_id", "title", "tmdb_title",
                 "wide_date", "wide_weekend_id", "release_year", "release_month", "bom_release_date",
                 "domestic_gross", "budget", "budget_source", "budget_bom", "budget_tmdb", "budget_disagree",
                 "runtime_min", "mpaa", "genres", "distributor", "original_language",
                 "production_companies", "production_company_ids", "collection_id", "collection_name",
                 "directors", "director_ids", "lead_actor", "lead_actor_id",
                 "label_release", "delayed_release", "event_review"] + list(POST_RELEASE.values())
    out = df[keep_cols]
    out.to_csv(DIR / "movies_clean.csv", index=False)

    gone = pd.concat(removed)[["removed_reason", "title", "imdb_id", "release_path", "wide_date",
                               "domestic_gross", "distributor", "year_gap"]]
    gone.to_csv(DIR / "removed_rows.csv", index=False)

    ev = df[df["event_review"]].copy()
    ev["why"] = reasons.loc[ev.index].apply(lambda r: ";".join(r.index[r]), axis=1)
    ev[["title", "distributor", "tmdb_genres", "runtime_min", "domestic_gross", "wide_date", "why",
        "release_path"]].sort_values("domestic_gross", ascending=False) \
        .to_csv(DIR / "review_event_candidates.csv", index=False)
    yg = df[df["label_release"] | df["delayed_release"]]
    yg[["title", "tmdb_title", "bom_release_date", "tmdb_release_date", "year_gap", "label_release",
        "delayed_release", "domestic_gross", "release_path"]].sort_values("year_gap", ascending=False) \
        .to_csv(DIR / "review_year_gap.csv", index=False)

    # ---- report ----
    print("FUNNEL")
    for name, n in funnel:
        print(f"  {name:<20}{n:>6}")
    ok = df[df["bom_release_date"].notna() & (df["post_opening_theaters"] >= 600)]
    near = ((ok["wide_date"] - ok["bom_release_date"]).dt.days.abs() <= 3).mean()
    print(f"\nwide_date within 3 days of the page release date (films that opened wide): {near:.1%}")
    print("\nshare of kept rows with a value:")
    cols = ["budget", "runtime_min", "mpaa", "genres", "directors", "lead_actor", "production_companies",
            "collection_id"]
    print(out[cols].replace({"NR": None}).notna().mean().round(3).to_string())
    print(f"\nflagged but kept: label_release={int(df.label_release.sum())}, "
          f"delayed_release={int(df.delayed_release.sum())}, event_review={int(df.event_review.sum())}, "
          f"budget_disagree={int(df.budget_disagree.sum())}")
    print("\nkept films per year:")
    print(out["release_year"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
