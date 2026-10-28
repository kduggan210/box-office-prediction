import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

DIR = pathlib.Path("data/processed")
RES = pathlib.Path("reports/results")
FIG = pathlib.Path("reports/figures")
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

TARGET = "log_gross_adj"
GROUP_ORDER = ["budget", "content", "timing", "studio", "franchise", "director", "actor"]
TITLES = {"budget_known": "Films with a recorded budget (main)", "all_films": "All films (secondary)"}
COLORS = {"budget_known": "#3b6ea5", "all_films": "#e08a1e"}
MONEY = [1e5, 1e6, 1e7, 1e8, 1e9]
MONEY_LABELS = ["$0.1M", "$1M", "$10M", "$100M", "$1B"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})

# ---------------------------------------------------------------- data
def load():
    f = pd.read_csv(DIR / "features.csv", parse_dates=["wide_date"])
    feats = list(pd.read_csv(DIR / "feature_groups.csv")["feature"])
    return f, feats

def load_groups():
    return pd.read_csv(DIR / "feature_groups.csv").set_index("feature")["group"]

def datasets(f):
    return {"budget_known": f[f["has_budget"] == 1].reset_index(drop=True),
            "all_films": f.copy()}

def splits(df):
    year = df["release_year"]
    return {"tune_train": year <= 2015,
            "validate": year.between(2016, 2019),
            "train": year <= 2019,
            "test": year >= 2022,
            "disrupted": year.isin([2020, 2021])}

# ---------------------------------------------------------------- scoring
def scores(y, pred):
    err = pred - y
    return {"r2": r2_score(y, pred),
            "rmse_log": float(np.sqrt(np.mean(err ** 2))),
            "mae_log": float(np.mean(np.abs(err))),
            "bias_log": float(np.mean(err)),                          # positive = over-prediction
            "median_pct_error": float(np.median(np.abs(np.exp(pred) - np.exp(y)) / np.exp(y))),
            "within_2x": float(np.mean(np.abs(err) <= np.log(2)))}    # share of films within a factor of 2

def bootstrap_interval(y, pred, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    r2s = [r2_score(y[i], pred[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return np.percentile(r2s, [2.5, 97.5])

def evaluate(rows, dataset, model_name, model, X, y, s, cols=None):
    cols = cols or list(X.columns)
    test_pred = test_row = None
    for split in ("test", "disrupted"):
        mask = s[split]
        pred = model.predict(X.loc[mask, cols])
        lo, hi = bootstrap_interval(y[mask], pred)
        row = {"dataset": dataset, "model": model_name, "split": split, "n": int(mask.sum()),
               **scores(y[mask], pred), "r2_ci_low": lo, "r2_ci_high": hi}
        rows.append(row)
        if split == "test":
            test_pred, test_row = pred, row
    return test_pred, test_row

def finish(slug, rows):
    """Print the results table and save it as reports/results/<slug>_metrics.csv."""
    table = pd.DataFrame(rows)
    table.to_csv(RES / f"{slug}_metrics.csv", index=False)
    print("\n" + table.round(3).to_string(index=False))
    print(f"\nsaved {RES / (slug + '_metrics.csv')}")
    return table

# ---------------------------------------------------------------- understanding the results
def explain(dataset, fit_predict, model, X, y, s, feats, groups, repeats=10):
    test = s["test"]
    yt = y[test]
    full = r2_score(yt, fit_predict(feats))
    abl = [{"dataset": dataset, "experiment": "all features", "group": "-", "r2": full, "delta_vs_full": 0.0}]
    for g in GROUP_ORDER:                                            # remove one group, refit
        r2 = r2_score(yt, fit_predict([c for c in feats if groups[c] != g]))
        abl.append({"dataset": dataset, "experiment": "drop one group", "group": g,
                    "r2": r2, "delta_vs_full": r2 - full})
    used = []
    for g in GROUP_ORDER:                                            # build up group by group
        used.append(g)
        r2 = r2_score(yt, fit_predict([c for c in feats if groups[c] in used]))
        abl.append({"dataset": dataset, "experiment": "add cumulatively", "group": "+" + g,
                    "r2": r2, "delta_vs_full": r2 - full})

    Xt, rng = X.loc[test, feats], np.random.default_rng(0)           # shuffle one group, no refit
    base = r2_score(yt, model.predict(Xt))
    imp = []
    for g in GROUP_ORDER:
        cols = [c for c in feats if groups[c] == g]
        drops = []
        for _ in range(repeats):
            Xp = Xt.copy()
            Xp[cols] = Xt[cols].values[rng.permutation(len(Xt))]     # the group's columns move together
            drops.append(base - r2_score(yt, model.predict(Xp)))
        imp.append({"dataset": dataset, "group": g, "r2_drop": float(np.mean(drops))})

    abl, imp = pd.DataFrame(abl), pd.DataFrame(imp).sort_values("r2_drop", ascending=False)
    print(f"\nfeature-group ablation ({TITLES[dataset]}), test R2:")
    print(abl[["experiment", "group", "r2", "delta_vs_full"]].round(3).to_string(index=False))
    print("\ngroup importance (drop in test R2 when the group is shuffled):")
    print(imp[["group", "r2_drop"]].round(3).to_string(index=False))
    return abl, imp


def save_tables(slug, abl_list, imp_list):
    pd.concat(abl_list).to_csv(RES / f"{slug}_ablation.csv", index=False)
    pd.concat(imp_list).to_csv(RES / f"{slug}_group_importance.csv", index=False)
    print(f"saved {RES / (slug + '_ablation.csv')} and {RES / (slug + '_group_importance.csv')}")


# ---------------------------------------------------------------- plots
def plot_pred_vs_actual(slug, label, results):
    """results: {dataset: (actual ln gross, predicted ln gross)} for the 2022-2025 films."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.4))
    for a, (name, (y, p)) in zip(ax, results.items()):
        y, p = np.exp(np.asarray(y)), np.exp(np.asarray(p))
        a.scatter(p, y, s=10, alpha=0.5, color=COLORS[name])
        lim = [min(p.min(), y.min()) * 0.8, max(p.max(), y.max()) * 1.2]
        a.plot(lim, lim, "k--", lw=1)
        a.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="Predicted gross (2025 dollars)",
              title=f"{TITLES[name]}, n = {len(y)}")
        a.set_xticks(MONEY); a.set_xticklabels(MONEY_LABELS)
        a.set_yticks(MONEY); a.set_yticklabels(MONEY_LABELS)
    ax[0].set_ylabel("Actual gross (2025 dollars)")
    fig.suptitle(f"{label}: predicted vs actual, 2022-2025 films")
    fig.tight_layout()
    fig.savefig(FIG / f"{slug}_pred_vs_actual.png", dpi=150)
    plt.close(fig)
    print(f"saved {FIG / (slug + '_pred_vs_actual.png')}")


def plot_ablation(slug, label, abl):
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    w = 0.38
    for i, name in enumerate(TITLES):
        d = abl[(abl["dataset"] == name) & (abl["experiment"] == "drop one group")].set_index("group").loc[GROUP_ORDER]
        ax[0].bar(np.arange(len(GROUP_ORDER)) + (i - 0.5) * w, d["delta_vs_full"], w, color=COLORS[name],
                  label=TITLES[name])
        c = abl[(abl["dataset"] == name) & (abl["experiment"] == "add cumulatively")]
        ax[1].plot(range(len(c)), c["r2"], marker="o", color=COLORS[name], label=TITLES[name])
    ax[0].set_xticks(range(len(GROUP_ORDER))); ax[0].set_xticklabels(GROUP_ORDER, rotation=30)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set(ylabel="Change in test R$^2$ when the group is removed", title="Removing one feature group")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].set_xticks(range(len(GROUP_ORDER))); ax[1].set_xticklabels(["+" + g for g in GROUP_ORDER], rotation=30)
    ax[1].set(ylabel="Test R$^2$", title="Adding groups one at a time")
    fig.suptitle(f"{label}: feature-group ablation (single run; gaps under ~0.02 are noise)")
    fig.tight_layout()
    fig.savefig(FIG / f"{slug}_ablation.png", dpi=150)
    plt.close(fig)
    print(f"saved {FIG / (slug + '_ablation.png')}")
