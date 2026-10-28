"""
Exploratory data analysis: report-ready figures and a text summary.

Reads   data/processed/features.csv         (build_features.py)
        data/processed/movies_adjusted.csv  (adjust_cpi.py)
Writes  reports/figures/*.png               one file per figure
        reports/eda_summary.txt             the tables printed below, for copying into the report

All dollar amounts are constant 2025 dollars. Needs matplotlib.

Usage (from repo root):   python src/eda.py
"""
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = pathlib.Path("data/processed")
OUT = pathlib.Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)

ERAS = ["2000-2009", "2010-2019", "2020-2021", "2022-2025"]
ERA_COLORS = dict(zip(ERAS, ["#3b6ea5", "#e08a1e", "#b23a48", "#4c9a6a"]))
MONEY_TICKS = [1e5, 1e6, 1e7, 1e8, 1e9]
MONEY_LABELS = ["$0.1M", "$1M", "$10M", "$100M", "$1B"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 100})
LINES = []


def say(text=""):
    print(text)
    LINES.append(str(text))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    say(f"  saved {OUT / name}")


def log_axis(ax, which="y"):
    ticks = np.log(MONEY_TICKS)
    (ax.set_yticks if which == "y" else ax.set_xticks)(ticks)
    (ax.set_yticklabels if which == "y" else ax.set_xticklabels)(MONEY_LABELS)


def load():
    m = pd.read_csv(DIR / "movies_adjusted.csv", parse_dates=["wide_date"])
    f = pd.read_csv(DIR / "features.csv", parse_dates=["wide_date"])
    extra = m[["release_path", "mpaa", "genres", "budget_adj"]]
    df = f.merge(extra, on="release_path", how="left")
    assert len(df) == len(f) == len(m)
    df["era"] = pd.cut(df["release_year"], [1999, 2009, 2019, 2021, 2025], labels=ERAS)
    df["gross_m"] = df["domestic_gross_adj"] / 1e6
    studio_cols = [c for c in df.columns if c.startswith("studio_")]
    df["studio"] = df[studio_cols].idxmax(axis=1).str.replace("studio_", "", regex=False)
    return df


# ---------------------------------------------------------------- figures
def fig_distribution(df):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(df["gross_m"], bins=60, color="#3b6ea5")
    ax[0].axvline(df["gross_m"].median(), color="k", ls="--", lw=1)
    ax[0].set(xlabel="Domestic gross (millions of 2025 dollars)", ylabel="Films",
              title=f"Raw gross: heavily right-skewed (skew {df['gross_m'].skew():.1f})")
    ax[0].text(df["gross_m"].median() * 1.4, ax[0].get_ylim()[1] * 0.9,
               f"median ${df['gross_m'].median():.0f}M", fontsize=9)
    ax[1].hist(df["log_gross_adj"], bins=40, color="#4c9a6a")
    ax[1].axvline(df["log_gross_adj"].median(), color="k", ls="--", lw=1)
    ax[1].set(xlabel="Domestic gross (2025 dollars, log scale)", ylabel="Films",
              title=f"Log gross: far less skewed (skew {df['log_gross_adj'].skew():.1f})")
    log_axis(ax[1], "x")
    save(fig, "01_gross_distribution.png")


def fig_films_per_year(df):
    counts = df.groupby(["release_year", "has_budget"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(counts.index, counts[1], color="#3b6ea5", label="budget recorded")
    ax.bar(counts.index, counts[0], bottom=counts[1], color="#c9c9c9", label="no budget recorded")
    ax.set(xlabel="Wide-release year", ylabel="Films kept", title="Films per year (2020 is the pandemic low)")
    ax.legend(frameon=False)
    save(fig, "02_films_per_year.png")


def fig_budget_scatter(df):
    b = df.dropna(subset=["log_budget_adj"])
    early, late = b[b["release_year"] <= 2019], b[b["release_year"] >= 2020]
    k0, c0 = np.polyfit(early["log_budget_adj"], early["log_gross_adj"], 1)
    xs = np.linspace(b["log_budget_adj"].quantile(0.005), b["log_budget_adj"].max(), 50)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)
    for a, sub, label, color in ((ax[0], early, "2000-2019", "#3b6ea5"), (ax[1], late, "2020-2025", "#b23a48")):
        a.scatter(sub["log_budget_adj"], sub["log_gross_adj"], s=9, alpha=0.4, color=color)
        k, c = np.polyfit(sub["log_budget_adj"], sub["log_gross_adj"], 1)
        a.plot(xs, k * xs + c, color="k", lw=1.3, label="this period's fit")
        r = sub["log_budget_adj"].corr(sub["log_gross_adj"])
        a.set(xlabel="Budget (2025 dollars)", title=f"{label}: r = {r:.2f}, n = {len(sub):,}")
        log_axis(a, "x"); log_axis(a, "y")
    ax[1].plot(xs, k0 * xs + c0, color="#3b6ea5", lw=1.8, ls="--", label="2000-2019 fit")
    ax[1].legend(frameon=False, loc="upper left")
    ax[0].set_ylabel("Domestic gross (2025 dollars)")
    fig.suptitle("Budget vs gross on log scales: the same budget earns less since 2020", y=1.0)
    save(fig, "03_budget_vs_gross.png")


def fig_ratio_by_year(df):
    d = df.dropna(subset=["budget_adj"]).copy()
    d["ratio"] = d["domestic_gross_adj"] / d["budget_adj"]
    med = d.groupby("release_year")["ratio"].median()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(med.index, med.values, marker="o", color="#3b6ea5")
    ax.axhline(1, color="k", ls="--", lw=1)
    ax.set(xlabel="Wide-release year", ylabel="Median domestic gross / budget",
           title="Domestic gross relative to budget fell after 2019 and has only partly recovered")
    save(fig, "04_gross_to_budget_by_year.png")
    return med


def fig_gross_by_month(df):
    groups = [df.loc[df["release_month"] == k, "log_gross_adj"] for k in range(1, 13)]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.boxplot(groups, tick_labels=[f"{n}\n(n={len(g)})" for n, g in zip(MONTHS, groups)],
               flierprops={"markersize": 3, "alpha": 0.4})
    ax.set(ylabel="Domestic gross (2025 dollars)", title="Gross by release month (wide-release Friday)")
    log_axis(ax)
    save(fig, "05_gross_by_month.png")


def fig_gross_by_category(df):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    order = ["G", "PG", "PG-13", "R", "NR"]
    g1 = [df.loc[df["mpaa"] == r, "log_gross_adj"] for r in order]
    ax[0].boxplot(g1, tick_labels=[f"{r}\n(n={len(g)})" for r, g in zip(order, g1)],
                  flierprops={"markersize": 3, "alpha": 0.4})
    ax[0].set(title="Gross by MPAA rating", ylabel="Domestic gross (2025 dollars)")
    log_axis(ax[0])
    studios = df.groupby("studio")["log_gross_adj"].median().sort_values(ascending=False).index
    g2 = [df.loc[df["studio"] == s, "log_gross_adj"] for s in studios]
    ax[1].boxplot(g2, tick_labels=[f"{s}\n(n={len(g)})" for s, g in zip(studios, g2)],
                  flierprops={"markersize": 3, "alpha": 0.4})
    ax[1].set(title="Gross by distributor group")
    ax[1].tick_params(axis="x", labelsize=8)
    log_axis(ax[1])
    save(fig, "06_gross_by_rating_and_studio.png")


def fig_genre(df):
    g = df[["gross_m", "genres"]].assign(genre=df["genres"].fillna("").str.split("|")).explode("genre")
    stats = g[g["genre"] != ""].groupby("genre")["gross_m"].agg(["median", "size"])
    stats = stats[stats["size"] >= 100].sort_values("median")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(stats.index, stats["median"], color="#3b6ea5")
    for i, (med, n) in enumerate(zip(stats["median"], stats["size"])):
        ax.text(med + 0.5, i, f"n={n}", va="center", fontsize=8)
    ax.set(xlabel="Median domestic gross (millions of 2025 dollars)",
           title="Median gross by genre (a film counts once per genre)")
    ax.grid(axis="y", alpha=0)
    save(fig, "07_gross_by_genre.png")


def fig_gross_by_era(df):
    groups = [df.loc[df["era"] == e, "log_gross_adj"] for e in ERAS]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(groups, tick_labels=[f"{e}\n(n={len(g)})" for e, g in zip(ERAS, groups)],
                    patch_artist=True, flierprops={"markersize": 3, "alpha": 0.4})
    for patch, e in zip(bp["boxes"], ERAS):
        patch.set_facecolor(ERA_COLORS[e]); patch.set_alpha(0.6)
    ax.set(ylabel="Domestic gross (2025 dollars)", title="The typical wide release grosses less in recent years")
    log_axis(ax)
    save(fig, "08_gross_by_era.png")


# ---------------------------------------------------------------- tables
def tables(df, ratio_med):
    say("\n" + "=" * 70 + "\nDATASET OVERVIEW\n" + "=" * 70)
    say(f"films: {len(df):,} | wide-release years {df['release_year'].min()}-{df['release_year'].max()}"
        f" | duplicate release_path: {int(df['release_path'].duplicated().sum())}"
        f" | duplicate imdb_id: {int(df['imdb_id'].duplicated().sum())}")

    say("\n" + "=" * 70 + "\nTARGET DISTRIBUTION (2025 dollars)\n" + "=" * 70)
    s = df["domestic_gross_adj"]
    say(f"mean ${s.mean() / 1e6:.1f}M | median ${s.median() / 1e6:.1f}M | std ${s.std() / 1e6:.1f}M | "
        f"min ${s.min() / 1e6:.2f}M | max ${s.max() / 1e6:.0f}M")
    say(f"skewness: raw {s.skew():.2f} -> log {df['log_gross_adj'].skew():.2f}")
    q1, q3 = df["log_gross_adj"].quantile([0.25, 0.75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    below, above = df[df["log_gross_adj"] < lo], df[df["log_gross_adj"] > hi]
    say(f"outliers on the log scale (1.5 x IQR): {len(below)} low (below ${np.exp(lo) / 1e6:.2f}M), "
        f"{len(above)} high (above ${np.exp(hi) / 1e6:.0f}M)")
    say("\nhighest grossing:")
    say(df.nlargest(8, "domestic_gross_adj")[["title", "release_year", "gross_m"]].round(0).to_string(index=False))
    say("\nlowest grossing:")
    say(df.nsmallest(6, "domestic_gross_adj")[["title", "release_year", "gross_m"]].round(2).to_string(index=False))

    say("\n" + "=" * 70 + "\nBY ERA\n" + "=" * 70)
    d = df.dropna(subset=["budget_adj"]).assign(ratio=lambda x: x["domestic_gross_adj"] / x["budget_adj"])
    by = pd.DataFrame({
        "films": df.groupby("era", observed=True).size(),
        "median_gross_$M": df.groupby("era", observed=True)["gross_m"].median(),
        "median_budget_$M": (df.groupby("era", observed=True)["budget_adj"].median() / 1e6),
        "median_gross/budget": d.groupby("era", observed=True)["ratio"].median(),
        "share_sequel": df.groupby("era", observed=True)["is_sequel"].mean(),
        "share_big_six": df.groupby("era", observed=True)["is_big_six"].mean()})
    say(by.round(2).to_string())

    say("\n" + "=" * 70 + "\nMISSING VALUES\n" + "=" * 70)
    say("share of films with no value, by feature (all years):")
    feats = [c for c in df.columns if c in ("log_budget_adj", "runtime_min", "prior_film_log_gross",
                                            "franchise_mean_log_gross", "director_mean_log_gross",
                                            "actor_mean_log_gross")]
    say(df[feats].isna().mean().round(3).to_string())
    say(f"MPAA 'NR' (no rating recorded): {(df['mpaa'] == 'NR').mean():.3f}")
    miss = pd.DataFrame({
        "no budget": 1 - df["has_budget"],
        "no director history": (df["director_n_prior"] == 0).astype(int),
        "no lead-actor history": (df["actor_n_prior"] == 0).astype(int),
        "no prior franchise film in data": df["prior_film_log_gross"].isna().astype(int)}
    ).groupby(df["era"], observed=True).mean()
    say("\nby era (share of films):")
    say(miss.round(3).to_string())
    say("Budget gaps are real missing data. The history and franchise columns are empty by "
        "construction (no earlier film in this data), not by error.")

    say("\n" + "=" * 70 + "\nCORRELATION WITH LOG GROSS (pairwise)\n" + "=" * 70)
    corr_cols = ["log_gross_adj", "log_budget_adj", "runtime_min", "genre_count", "release_month", "is_big_six",
                 "is_sequel", "n_prior_films", "prior_film_log_gross", "director_n_prior",
                 "director_mean_log_gross", "actor_n_prior", "actor_mean_log_gross"]
    corr = df[corr_cols].corr()                      # pairwise: each pair uses the films that have both values
    c = corr["log_gross_adj"].drop("log_gross_adj").sort_values(key=abs, ascending=False)
    n = {k: int(df[[k, "log_gross_adj"]].dropna().shape[0]) for k in c.index}
    say(pd.DataFrame({"r": c.round(3), "n_films": pd.Series(n)}).to_string())
    say("\nGross/budget ratio by year (median):")
    say(ratio_med.round(2).to_string())


def main():
    df = load()
    say("FIGURES")
    fig_distribution(df)
    fig_films_per_year(df)
    fig_budget_scatter(df)
    ratio_med = fig_ratio_by_year(df)
    fig_gross_by_month(df)
    fig_gross_by_category(df)
    fig_genre(df)
    fig_gross_by_era(df)
    tables(df, ratio_med)
    (OUT.parent / "eda_summary.txt").write_text("\n".join(LINES), encoding="utf-8")
    print(f"\nwrote {OUT.parent / 'eda_summary.txt'}")


if __name__ == "__main__":
    main()