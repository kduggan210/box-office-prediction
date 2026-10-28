# Predicting Domestic Box Office Revenue

Regression models that predict a wide-release movie's total domestic (North
America) gross from pre-release features: budget, genre, runtime, MPAA rating,
studio, release timing, franchise status, and director / lead actor track
records. We compare Linear Regression (baseline), Ridge Regression, Random
Forest, and XGBoost, and study which feature groups matter most.

## Status

| Step | Script | Status |
|------|--------|--------|
| Box Office Mojo yearly lists (cross-check data) | `src/bom_import.py years` | done |
| Weekend charts -> complete wide-release list | `src/bom_weekends.py` | done |
| BOM release pages (gross, budget, MPAA, runtime, dates, IMDb ID) | `src/bom_releases.py` | done |
| TMDB join (budget, franchise, studios, cast, crew) | `src/tmdb_import.py` | done |
| Export joined data to CSV | `src/export_csv.py` | done |
| Cleaning (re-releases, wide-release date, genres) | `src/build_dataset.py` | done |
| CPI inflation adjustment | `src/adjust_cpi.py` | done |
| Franchise release dates (for the sequel flag) | `src/tmdb_collections.py` | done |
| Feature engineering (56 features, 7 groups) | `src/build_features.py` | done |
| Exploratory data analysis | `src/eda.py` | done |
| Linear regression and Ridge | `src/model_linear.py`, `src/model_ridge.py` | done (mid-project) |
| Random forest and XGBoost | `src/model_random_forest.py`, `src/model_xgboost.py` | planned (full report) |
| Model comparison table and figure | `src/compare_models.py` | done |

## Setup

Python 3.10+ recommended.

```
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
python -m pip install -r requirements.txt
```

`requirements.txt` includes pandas, scikit-learn, xgboost and matplotlib.

Get a TMDB API key and set it as an environment variable (never commit it).
Only the TMDB steps need it (`tmdb_import.py`, `tmdb_collections.py`):

```
$env:TMDB_API_KEY = "your_key_here"   # PowerShell, current session only
```

**Always run scripts from the repository root**, because paths like `data/raw/...`
are relative to the working directory.

## Running the pipeline

Each download step caches raw HTML/JSON under `data/raw/`, so reruns are fast and
polite to the source sites (2-second delay between uncached requests to Box Office
Mojo).

```
python src/bom_import.py years        # yearly lists (cross-check data)
python src/bom_weekends.py            # all weekends 2000-2025 -> table bom_wide
python src/bom_releases.py            # per-film details for every wide release
python src/tmdb_import.py             # budget, franchise, studios, cast and crew from TMDB
python src/export_csv.py              # small CSVs of the joined data for git
python src/build_dataset.py           # cleaned modeling table -> movies_clean.csv
python src/adjust_cpi.py              # constant-dollar columns -> movies_adjusted.csv
python src/tmdb_collections.py        # release dates of every film in each franchise
python src/build_features.py          # features.csv, feature_groups.csv
```

Then the models (each one is a standalone script):

```
python src/model_linear.py
python src/model_ridge.py
python src/model_random_forest.py     # slowest: a few minutes
python src/model_xgboost.py
python src/compare_models.py          # merges whichever model results exist
```

Everything from `build_dataset.py` onward reads only the committed CSVs, so it can
run without the database or the raw cache.

Notes:

- `bom_releases.py` takes about 3 hours for ~4,100 films. It is resumable
  (films already saved are skipped), retries failed requests, logs bad pages
  instead of crashing, and stops itself after 25 failures in a row. Rerun the
  same command to continue or to retry failed pages.
- Test first with `python src/bom_releases.py --limit 20`.
- After changing the parser, `python src/bom_releases.py --reset` drops the
  detail tables and reparses from the cached HTML (no new requests). Do not use
  `--reset` to resume a run; it deletes saved progress.
- Do not run `bom_weekends.py` with a single year after a full run: it rebuilds
  `bom_wide` from only the years given.
- `tmdb_import.py` takes roughly 15-40 minutes. It is resumable and caches raw
  JSON; test first with `--limit 20`. Each IMDb ID is fetched once, and
  `bom_tmdb_match` compares TMDB's title and year against Box Office Mojo's to
  catch wrong matches and re-releases.
- `tmdb_collections.py` reads `movies_clean.csv` and makes about 590 requests
  (one per franchise), cached under `data/raw/tmdb_collection/`.
- `adjust_cpi.py` needs `data/external/CPIAUCNS.csv` (see Data sources).
- The model scripts take a few minutes at most; the random forest is the slowest.

All parsed tables are stored in `data/processed/movies.db` (SQLite):

| Table | Contents |
|-------|----------|
| `bom_years` | Yearly top-200 lists (Box Office Mojo) |
| `bom_weekends` | One row per (weekend, release) |
| `bom_wide` | One row per wide release; input to the release-page step |
| `bom_release_details` | Per-film details from release pages (gross, budget, MPAA, runtime, dates, IMDb ID, plus `summary_json` with every label found on the page) |
| `bom_release_failures` | Pages that failed to fetch or parse, for retrying |
| `tmdb_movies` | One row per IMDb ID: budget, collection (franchise), studios, directors, lead actor |
| `tmdb_people` | Directors and the top 10 billed cast per film |
| `tmdb_failures` | IMDb IDs TMDB could not match |
| `bom_tmdb_match` | Title and year checks between each Box Office Mojo film and its TMDB match |

## Data decisions

- **Wide release** = played in 600 or more theaters on a weekend chart.
- **Why weekend charts:** yearly lists stop at rank 200 by gross, which drops
  low-grossing wide releases (flops) and biases the dataset toward hits. The
  weekend data found 314 wide releases (of 4,141 total) that the yearly lists
  missed. Weekend pages list 100+ releases each and are not truncated.
- **Yearly vs. weekend theater counts:** the yearly lists report peak theaters
  on any day, so one-night events and special screenings can look "wide" there
  but never reach 600 theaters on a weekend. The weekend definition excludes
  them.
- **Long holiday weekends** (3-, 4-, and 5-day versions share a week number):
  we keep the page with the most releases for each week.
- **Target:** final domestic gross (`domestic_gross`) from each release page,
  converted to constant 2025 dollars and modeled on the log scale.
  `total_gross_seen` in `bom_wide` is only a lower bound taken from weekend
  charts and is used as a sanity check.
- **Budget** is missing on Box Office Mojo for 41% of films. TMDB fills most of
  the gaps, and 92% of the cleaned films have a budget from one source or the
  other.

## Data cleaning

`src/build_dataset.py` turns the exported CSVs into `data/processed/movies_clean.csv`
(3,911 of 4,141 films). Each removed film is logged with its reason in
`removed_rows.csv`.

**Removed (first matching reason, in this order):**

- **No TMDB record (19):** opera broadcasts, TV episodes and live events that
  TMDB does not list as movies.
- **Before 2000 (24):** late-1999 films still in theaters in January 2000.
- **Live events (17):** unambiguous cases such as Met Opera, concert broadcasts,
  stage recordings and TV episode screenings.
- **Re-releases (154):** a film is dropped if it is (1) a later run of an IMDb ID
  already in the data (only the earliest run is kept), (2) labeled as a
  re-release and 2+ years after TMDB's original release year, or (3) 10+ years
  after TMDB's original release year with no label. The label alone is not
  enough: some films labeled "Re-release" are the only wide run of that film
  (e.g., *Splice*, *Conviction*), so they are kept and flagged (`label_release`).
- **Manual exclusions (16):** films whose theatrical run was not their primary
  release (e.g., Disney+-first Pixar films with a small 2024 theatrical run),
  plus concert films, filmed stage shows and TV specials. They are listed in
  `manual_exclude.csv` (a column of `release_path`).

**Kept but flagged for review:** `delayed_release` (foreign or festival films
with a 2-9 year gap to TMDB's date), `event_review` (possible event cinema, such
as major-studio concert films), and `budget_disagree` (TMDB and Box Office Mojo
budgets differ by more than 2x). The `review_*.csv` files list them.

**Fields:**

- **Release date:** the page's release date is missing for about 10% of films
  and can be a limited-release date (e.g., *Defiance* opened in 2 theaters). The
  wide-release date is instead the first weekend with 600+ theaters. Week N is
  the Nth Friday of the year, and the result is within 3 days of the page date
  for 99.7% of films that opened wide.
- **Budget:** TMDB first, Box Office Mojo as a fallback; 92% of films have one.
  TMDB values under $1,000 are treated as missing.
- **Genres** are stored pipe-separated (`Action|Drama`).
- **Inflation:** `adjust_cpi.py` converts gross and budget to constant 2025
  dollars using the CPI of the wide-release month. The 2025 level is the average
  of its twelve months; October 2025 (not collected by BLS) is interpolated.
- **No leakage:** columns starting with `post_` (opening gross, opening
  theaters, widest-release theaters) are only known after release and are never
  used as features.

## Features

`src/build_features.py` builds 56 numeric features in seven groups (listed in
`feature_groups.csv`, which the model scripts use for the group experiments):

| Group | Features |
|-------|----------|
| budget (2) | log inflation-adjusted budget, has-budget flag |
| content (26) | runtime, MPAA rating, genre indicators, genre count |
| timing (7) | release month (plus sine/cosine), holiday-weekend flags |
| studio (10) | distributor group (Disney, Warner, Universal, Sony, Paramount, Fox, Lionsgate, MGM, Other), big-six flag |
| franchise (5) | is a sequel, number of earlier films in the series, log gross of the previous film and of earlier films in the data |
| director (3) | number of earlier films, mean and max log gross of earlier films |
| actor (3) | the same, for the lead actor (billed in the top 3) |

Rules that keep the features honest:

- Track-record and franchise features only use films that opened **at least 8
  weeks before** the film being predicted, so their gross is essentially known.
  The script asserts this on every row.
- **"Belongs to a collection" is deliberately not a feature.** A series' first
  film only joins a collection if sequels were made later, so for first films it
  leaks hindsight about success (44% of collection films are first films). The
  sequel flag (an *earlier* film exists) carries the legitimate franchise signal.
  Removing the collection flag lowered test R² by 0.03-0.07, which shows how much
  of the apparent franchise effect it was.
- Histories only count wide releases from 2000 onward, so early films and
  newcomers have thin or empty histories. Missing values stay missing; the
  count features say how thin a history is.

## Modeling approach

- **Target:** ln(domestic gross in 2025 dollars).
- **Time-based split** (nothing from the future predicts the past):
  tune on 2000-2015 and validate on 2016-2019; refit on 2000-2019; test on
  2022-2025. The pandemic years (2020-2021) are scored separately.
- **Two analyses:** films with a recorded budget (main, 3,600 films) and all
  films (secondary, 3,911). Budget-missing films are mostly small releases, so
  including them also rewards the model for recognizing "no recorded budget".
- **Metrics:** R², RMSE and MAE on the log scale, median percentage error in
  dollars, share of films within a factor of 2, and a 95% bootstrap interval for
  R² over the test films.
- **Per-model analysis:** each script also removes one feature group at a time
  and refits, adds groups cumulatively, and measures permutation importance of
  each group.
- Missing values are filled with the training median for linear, Ridge and the
  forest; XGBoost handles them natively.

Each model script writes its results to `reports/results/` (`<model>_metrics.csv`,
`<model>_ablation.csv`, `<model>_group_importance.csv`) and its figures to
`reports/figures/` (predicted vs actual, ablation, and a loss or validation
curve where one exists).

## Results so far (mid-project)

Test R² on 2022-2025 films, with 95% bootstrap intervals:

| Model | Films with a budget (n=505) | All films (n=656) |
|-------|-----------------------------|-------------------|
| Mean baseline | -0.32 (-0.43, -0.23) | -0.59 (-0.74, -0.47) |
| Linear, log budget only | 0.21 (0.11, 0.29) | 0.30 (0.21, 0.36) |
| Linear, all features | 0.28 (0.19, 0.36) | 0.37 (0.30, 0.44) |
| Ridge | 0.28 (0.19, 0.36) | 0.37 (0.30, 0.43) |

- Budget carries most of the signal; adding the other features raises R² by
  about 0.07-0.08. Studio adds the most beyond budget; franchise, timing,
  director and lead-actor track records add little.
- Removing the **content** group (genres, MPAA, runtime) *improves* the linear
  models, so its 2000-2019 patterns do not carry into recent years.
- Ridge is no better than plain linear regression (its validation curve is flat
  until the penalty is large).
- Every model **over-predicts** recent films: error grows from about 0.83 on the
  training years to 0.93 on 2016-2019 validation and 1.36 on 2022-2025 test, and
  the typical 2022-2025 film is predicted about 2x too high.

Random forest and XGBoost results will be added for the final report.

## Data sources

- Box Office Mojo (boxofficemojo.com) — domestic gross, theaters, MPAA rating
- TMDB (themoviedb.org) — metadata, cast, crew, franchise collections
- FRED CPIAUCNS (fred.stlouisfed.org/series/cpiaucns) — Consumer Price Index for
  All Urban Consumers, all items, not seasonally adjusted. Download the CSV and
  save it as `data/external/CPIAUCNS.csv` (committed; the file is small).

Scraping is rate-limited and cached. Review each site's terms of use before
reusing this code.

## Known limitations

- One test window (2022-2025); results may differ for other periods.
- A clear era shift: models trained on 2000-2019 over-predict recent films.
- Roughly 23% of 2020s films have no recorded budget, and the all-films scores
  partly reflect that missingness.
- Director and actor histories are thin early in the data (the data starts in
  2000) and for newcomers.
- A day-and-date-streaming flag for 2020-2021 Warner Bros. and Disney releases
  is not yet included.

## Data not included in the repo

`data/raw/` (cached HTML and JSON) and the SQLite database are excluded via
`.gitignore`. To rebuild from scratch, run the pipeline above (several hours of
requests). The small CSV exports in `data/processed/` are committed
(`bom_movies.csv`, `tmdb_movies.csv`, `tmdb_people.csv`, `bom_tmdb_match.csv`,
`bom_wide_weekend.csv`, `movies_clean.csv`, `movies_adjusted.csv`,
`tmdb_collections.csv`, `features.csv`, `feature_groups.csv`, `removed_rows.csv`,
the `review_*.csv` files and `manual_exclude.csv`), so cleaning, feature
engineering and modeling can be done without the raw cache or the database.

## Repository layout

```
src/              pipeline, analysis and model scripts
data/raw/         cached HTML / API JSON (not committed)
data/external/    CPIAUCNS.csv (inflation index from FRED)
data/processed/   SQLite DB (not committed) and small CSV exports
reports/figures/  EDA, model and diagram figures
reports/results/  per-model metrics, ablation and importance tables
```