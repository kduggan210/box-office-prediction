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
| TMDB join (budget, cast, crew, studio, franchise) | `src/tmdb_import.py` | done |
| Export joined data to CSV | `src/export_csv.py` | done |
| Cleaning (re-releases, wide-release date, genres) | `src/build_dataset.py` | done |
| CPI inflation adjustment | `src/adjust_cpi.py` | done |
| Feature engineering | TODO | TODO |
| Modeling and evaluation | TODO | TODO |

## Setup

Python 3.10+ recommended.

```
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt
```

Get a TMDB API key and set it as an environment variable (never commit it):

```
$env:TMDB_API_KEY = "your_key_here"   # PowerShell, current session only
```

**Always run scripts from the repository root**, because paths like `data/raw/...`
are relative to the working directory.

## Running the pipeline

Each step caches downloaded HTML/JSON under `data/raw/`, so reruns are fast and
polite to the source sites (2-second delay between uncached requests).

```
python src/bom_import.py years        # yearly lists (cross-check data)
python src/bom_weekends.py            # all weekends 2000-2025 -> table bom_wide
python src/bom_releases.py            # per-film details for every wide release
python src/tmdb_import.py             # budget, franchise, studios, cast and crew from TMDB
python src/export_csv.py              # small CSVs of the joined data for git
python src/build_dataset.py           # cleaned modeling table -> movies_clean.csv
python src/adjust_cpi.py              # constant-dollar columns -> movies_adjusted.csv
python src/tmdb_collections.py        # release dates of every film in each franchise
```

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
- `tmdb_import.py` needs a TMDB API key (see Setup) and takes roughly 15-40
  minutes. It is resumable and caches raw JSON; test first with `--limit 20`.
  Each IMDb ID is fetched once, and `bom_tmdb_match` compares TMDB's title and
  year against Box Office Mojo's to catch wrong matches and re-releases.

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
- **Target:** final domestic gross (`domestic_gross`) from each release page.
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
  already in the data (only the earliest run is kept), (2) labeled as a re-release 
  and 2+ years after TMDB's original release year, or (3) 10+ years after TMDB's 
  original release year with no label. The label alone is not enough: some films
  labeled "Re-release" are the only wide run of that film (e.g., *Splice*, 
  *Conviction*), so they are kept and flagged (`label_release`).
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
- **Genres** are stored pipe-separated (`Action|Drama`) and need splitting
  before encoding.
- **No leakage:** columns starting with `post_` (opening gross, opening
  theaters, widest-release theaters) are only known after release and are not
  used as features. Director and lead actor track records will use only films
  released before the one being predicted.

**Modeling choices still to make:** time-based train/test split, log-transformed
target, CPI-adjusted dollars, a day-and-date-streaming flag for 2020-2021
Warner Bros. and Disney releases, and how to treat 2020-2021 (only 58 films in
2020). The median gross-to-budget ratio was 1.54 in 2019 and about 0.8-0.9 from
2022 to 2025, so results may differ before and after the pandemic.

## Data sources

- Box Office Mojo (boxofficemojo.com) — domestic gross, theaters, MPAA rating
- TMDB (themoviedb.org) — metadata, cast, crew
- FRED CPIAUCNS (fred.stlouisfed.org/series/cpiaucns) — Consumer Price Index for All Urban Consumers, all items, not seasonally adjusted

Scraping is rate-limited and cached. Review each site's terms of use before
reusing this code.

## Data not included in the repo

`data/raw/` (cached HTML and JSON) and the SQLite database are excluded via
`.gitignore`. To rebuild from scratch, run the pipeline above (several hours of
requests). The small CSV exports in `data/processed/` are committed
(`bom_movies.csv`, `tmdb_movies.csv`, `tmdb_people.csv`, `bom_tmdb_match.csv`,
`bom_wide_weekend.csv`, `movies_clean.csv`, `removed_rows.csv`, the `review_*.csv`
files and `manual_exclude.csv`), so cleaning, feature engineering and modeling
can be done without the raw cache or the database.

## Repository layout

```
src/              pipeline and check scripts
data/raw/         cached HTML / API JSON (not committed)
data/processed/   SQLite DB (not committed) and small CSV exports
```
