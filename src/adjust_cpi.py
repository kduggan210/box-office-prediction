"""
Convert dollar columns to constant (BASE_YEAR) dollars with the CPI-U.

Reads   data/processed/movies_clean.csv       (from build_dataset.py)
        data/external/CPIAUCNS.csv            (FRED series CPIAUCNS: CPI-U, all items,
                                               not seasonally adjusted; columns: date, value)
Writes  data/processed/movies_adjusted.csv    movies_clean plus:
          cpi_release_month   CPI in the film's wide-release month
          cpi_factor          BASE_YEAR average CPI / cpi_release_month
          domestic_gross_adj, budget_adj, post_opening_gross_adj   (nominal x cpi_factor)
The original nominal columns are left untouched.

Choices (state them in the report):
  - Both gross and budget use the CPI of the wide-release month. Gross accrues over the
    weeks after release, so this is a close approximation; budgets were spent earlier, so it
    slightly understates their inflation adjustment. A constant-dollar budget is a modeling
    input, so keep it consistent with the gross.
  - The BASE_YEAR level is the average of its twelve monthly values.
  - BLS did not collect October 2025 data (federal shutdown), so FRED leaves it blank. It is
    filled by linear interpolation between September and November. Any gap longer than two
    consecutive months stops the script instead of being guessed.

Usage (from repo root):   python src/adjust_cpi.py
"""
import sys
import pathlib

import pandas as pd

DIR = pathlib.Path("data/processed")
CPI_FILE = pathlib.Path("data/external/CPIAUCNS.csv")
BASE_YEAR = 2025
ADJUST = ["domestic_gross", "budget", "post_opening_gross"]


def load_cpi():
    raw = pd.read_csv(CPI_FILE)
    raw.columns = ["date", "cpi"]                       # FRED: observation_date, CPIAUCNS
    raw["date"] = pd.to_datetime(raw["date"]).dt.to_period("M")
    raw["cpi"] = pd.to_numeric(raw["cpi"], errors="coerce")
    s = raw.set_index("date")["cpi"]
    s = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))
    missing = list(s[s.isna()].index)
    s = s.interpolate(limit=2)
    if s.isna().any():
        sys.exit(f"CPI gap longer than 2 months near {list(s[s.isna()].index[:3])} - fix the file")
    return s, missing


def main():
    cpi, filled = load_cpi()
    base_months = cpi[cpi.index.year == BASE_YEAR]
    if len(base_months) != 12:
        sys.exit(f"{BASE_YEAR} has {len(base_months)} monthly values in the CPI file; need all 12")
    base_cpi = base_months.mean()

    df = pd.read_csv(DIR / "movies_clean.csv", parse_dates=["wide_date"])
    months = df["wide_date"].dt.to_period("M")
    df["cpi_release_month"] = cpi.reindex(months.values).to_numpy()
    if df["cpi_release_month"].isna().any():
        sys.exit("some release months are outside the CPI file's range - download a longer series")
    df["cpi_factor"] = base_cpi / df["cpi_release_month"]
    for col in ADJUST:
        df[f"{col}_adj"] = (df[col] * df["cpi_factor"]).round(2)
    df.to_csv(DIR / "movies_adjusted.csv", index=False)

    print(f"base: {BASE_YEAR} average CPI = {base_cpi:.3f}")
    print("filled by interpolation:", [str(m) for m in filled] or "none")
    print(f"wrote {DIR / 'movies_adjusted.csv'}: {len(df)} rows")

    by_year = df.groupby("release_year").agg(
        factor=("cpi_factor", "mean"),
        median_gross=("domestic_gross", "median"),
        median_gross_adj=("domestic_gross_adj", "median"))
    print("\nby release year (factor = multiply nominal dollars by this):")
    print(by_year.round(2).to_string())

    df["rank_nominal"] = df["domestic_gross"].rank(ascending=False, method="min").astype(int)
    df["rank_adjusted"] = df["domestic_gross_adj"].rank(ascending=False, method="min").astype(int)
    top = df.sort_values("domestic_gross_adj", ascending=False).head(10)
    print("\ntop 10 by adjusted gross:")
    print(top[["title", "release_year", "domestic_gross", "domestic_gross_adj",
               "rank_nominal", "rank_adjusted"]].to_string(index=False))
    same = (df["budget"].isna() == df["budget_adj"].isna()).all()
    print("\nmissing budgets stay missing:", same)


if __name__ == "__main__":
    main()
