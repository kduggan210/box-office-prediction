# Predicting Domestic Box Office Revenue

Regression models that predict a wide-release movie's total domestic (North
America) gross from pre-release features: budget, genre, runtime, MPAA rating,
studio, release timing, franchise status, and director / lead actor track
records. We compare Linear Regression (baseline), Ridge Regression, Random
Forest, and XGBoost, and study which feature groups matter most.

## Status

| Step | Script | Status |
|------|--------|--------|
| Box Office Mojo yearly lists | `src/bom_import.py years` | done |
| Weekend charts -> complete wide-release list | `src/bom_weekends.py` | done / running |
| BOM release pages (MPAA, runtime, IMDb ID, release date, final gross) | `src/bom_import.py releases` | in progress |
| TMDB join (budget, cast, crew, studio, franchise) | `src/tmdb_import.py` | TODO |
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
python src/bom_import.py releases     # per-film details for every wide release
```

Do not run `bom_weekends.py` with a single year after a full run: it rebuilds
`bom_wide` from only the years given.

All parsed tables are stored in `data/processed/movies.db` (SQLite):

| Table | Contents |
|-------|----------|
| `bom_years` | Yearly top-200 lists (Box Office Mojo) |
| `bom_weekends` | One row per (weekend, release) |
| `bom_wide` | One row per wide release; input to the release-page step |
| `bom_releases` | Per-film details from release pages |

## Data decisions

- **Wide release** = played in 600 or more theaters at any point.
- **Why weekend charts:** yearly lists stop at rank 200 by gross, which drops
  low-grossing wide releases (flops) and biases the dataset toward hits. In 2006
  alone, 7 wide releases were missing from the yearly list. Weekend pages list
  100+ releases each and are not truncated.
- **Long holiday weekends** (3-, 4-, and 5-day versions share a week number):
  we keep the page with the most releases for each week.
- **`total_gross_seen`** in `bom_wide` is only a lower bound. Final domestic
  gross (the prediction target) comes from the release pages.
- **Carryover films:** films released in late December appear in the next year's
  early weekends. Filter on the true release date, not first weekend seen.
- **Planned modeling choices:** time-based train/test split, log-transformed
  target, CPI-adjusted dollars, and track-record features computed only from
  films released before the one being predicted (no leakage).

## Data sources

- Box Office Mojo (boxofficemojo.com) — domestic gross, theaters, MPAA rating
- TMDB (themoviedb.org) — metadata, cast, crew
- BLS Consumer Price Index (bls.gov/cpi)

Scraping is rate-limited and cached. Review each site's terms of use before
reusing this code.

## Data not included in the repo

`data/raw/` (cached HTML) and the SQLite database are excluded via `.gitignore`.
To rebuild from scratch, run the pipeline above (several hours of requests).
Small exported CSVs of processed data may be committed under `data/processed/`.

## Repository layout

```
src/              pipeline and check scripts
notebooks/        exploration and modeling notebooks
data/raw/         cached HTML / API JSON (not committed)
data/processed/   SQLite DB (not committed) and small CSV exports
```
