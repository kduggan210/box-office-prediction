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
| BOM release pages (gross, budget, MPAA, runtime, dates, IMDb ID) | `src/bom_releases.py` | running |
| Export joined data to CSV | `src/export_csv.py` | done |
| TMDB join (budget, cast, crew, studio, franchise) | `src/tmdb_import.py` | TODO |
| Cleaning (re-releases, wide-release date, genres) | TODO | TODO |
| CPI inflation adjustment | TODO | TODO |
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
python src/export_csv.py              # small CSV of the joined data for git
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

All parsed tables are stored in `data/processed/movies.db` (SQLite):

| Table | Contents |
|-------|----------|
| `bom_years` | Yearly top-200 lists (Box Office Mojo) |
| `bom_weekends` | One row per (weekend, release) |
| `bom_wide` | One row per wide release; input to the release-page step |
| `bom_release_details` | Per-film details from release pages (gross, budget, MPAA, runtime, dates, IMDb ID, plus `summary_json` with every label found on the page) |
| `bom_release_failures` | Pages that failed to fetch or parse, for retrying |

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
- **Budget** is missing on Box Office Mojo for roughly 40% of films in a test
  sample; TMDB is the second source.

## Cleaning plan

- **Re-releases and anniversary runs** appear in the data (118 titles contain
  "re-release", and some, like "E.T. ... 20th Anniversary", do not). They share
  the original film's IMDb ID, so we flag them by (1) title text, (2) duplicate
  IMDb IDs, and (3) TMDB original release date far earlier than the Box Office
  Mojo date, and keep only a film's original run.
- **Release date:** the release-page date is sometimes missing and can be a
  limited-release date for platform releases (e.g., *Defiance* opened in 2
  theaters). The wide-release date will be derived from the first weekend with
  600+ theaters in `bom_weekends`.
- **Carryover films:** late-December films appear in the next year's early
  weekends, and 1999 films appear in 2000. Filter on the true release date.
- **No leakage:** opening gross, opening theaters, and widest-release theaters
  are only known after release and are not used as features. Director and lead
  actor track records use only films released before the one being predicted.
- **Genres** are stored with `·` separators and extra whitespace and need
  splitting into a list before encoding.
- **Planned modeling choices:** time-based train/test split, log-transformed
  target, CPI-adjusted dollars, and a decision on how to treat 2020-2021.

## Data sources

- Box Office Mojo (boxofficemojo.com) — domestic gross, theaters, MPAA rating
- TMDB (themoviedb.org) — metadata, cast, crew
- BLS Consumer Price Index (bls.gov/cpi)

Scraping is rate-limited and cached. Review each site's terms of use before
reusing this code.

## Data not included in the repo

`data/raw/` (cached HTML) and the SQLite database are excluded via `.gitignore`.
To rebuild from scratch, run the pipeline above (several hours of requests).
The joined export `data/processed/bom_movies.csv` is small and is committed, so
the cleaning, TMDB, and modeling steps can be developed without the raw cache.

## Repository layout

```
src/              pipeline and check scripts
data/raw/         cached HTML / API JSON (not committed)
data/processed/   SQLite DB (not committed) and small CSV exports
```
