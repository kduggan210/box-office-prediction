import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = pathlib.Path("reports/figures/basic")
OUT.mkdir(parents=True, exist_ok=True)
TICKS = np.log([1e6, 1e7, 1e8, 1e9])
LABELS = ["$1M", "$10M", "$100M", "$1B"]

# 1. Data, split by time
films = pd.read_csv("data/processed/features.csv")
films = films[films["has_budget"] == 1]
features = list(pd.read_csv("data/processed/feature_groups.csv")["feature"])

train = films[films["release_year"] <= 2019]             # learn from 2000-2019
test = films[films["release_year"] >= 2022]              # predict 2022-2025
fit_part = train[train["release_year"] <= 2015]          # for choosing alpha: learn from 2000-2015 ...
valid = train[train["release_year"] >= 2016]             # ... and check on 2016-2019
y_train, y_test = train["log_gross_adj"], test["log_gross_adj"]
print(f"{len(train)} training films, {len(test)} test films, {len(features)} features")


def score(y_true, y_pred):
    miss = np.abs(y_pred - y_true)
    return {"R2": round(r2_score(y_true, y_pred), 3),
            "typical miss (x)": round(float(np.exp(np.median(miss))), 2),
            "within 2x": round(float(np.mean(miss <= np.log(2))), 3)}


def ridge(alpha):
    # fill gaps with the median, rescale features to the same scale, then fit ridge
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))


def rmse(model, data):
    return float(np.sqrt(np.mean((model.predict(data[features]) - data["log_gross_adj"]) ** 2)))

# 2. Choosing alpha
alphas = [0.1, 1, 10, 100, 300, 1000, 3000, 10000]
rows = []
for alpha in alphas:
    m = ridge(alpha).fit(fit_part[features], fit_part["log_gross_adj"])
    rows.append({"alpha": alpha, "error on 2000-2015": rmse(m, fit_part), "error on 2016-2019": rmse(m, valid)})
curve = pd.DataFrame(rows)
print("\nError (RMSE of ln gross) for each alpha:")
print(curve.round(3).to_string(index=False))

# The curve is flat for small alpha, so its single lowest point would be arbitrary. Take the
# STRONGEST penalty whose validation error is within 0.2% of the best (simpler model, same accuracy).
best = curve.loc[curve["error on 2016-2019"] <= curve["error on 2016-2019"].min() * 1.002, "alpha"].max()
print(f"chosen alpha = {best}")

# 3 + 4 Final model, compared with plain linear regression
final = ridge(best).fit(train[features], y_train)
plain = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression()).fit(train[features], y_train)
ridge_pred = final.predict(test[features])

results = pd.DataFrame({"always guess the average": score(y_test, np.full(len(test), y_train.mean())),
                        "plain linear regression": score(y_test, plain.predict(test[features])),
                        f"ridge (alpha = {best})": score(y_test, ridge_pred)}).T
print("\nScores on the 2022-2025 films:")
print(results.to_string())

coefs = pd.DataFrame({"plain linear": plain[-1].coef_, "ridge": final[-1].coef_}, index=features)
top = coefs.reindex(coefs["plain linear"].abs().sort_values(ascending=False).index).head(8)
print("\nBiggest coefficients, plain vs ridge (ridge pulls them slightly toward zero):")
print(top.round(3).to_string())
print("\nWith ~3,000 films and a few dozen features there is little for ridge to fix, so the two"
      "\nmodels score almost the same. That is a normal result, not a bug.")

# 5. Visualization
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(curve["alpha"], curve["error on 2000-2015"], marker="o", color="#3b6ea5", label="films it learned from (2000-2015)")
ax.plot(curve["alpha"], curve["error on 2016-2019"], marker="o", color="#b23a48", label="films it did not see (2016-2019)")
ax.axvline(best, color="k", ls="--", lw=1)
ax.set(xscale="log", xlabel="Penalty strength (alpha)", ylabel="Error (RMSE of ln gross)",
       title="Ridge: error by penalty strength")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "ridge_alpha_curve.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5.5))
ax.scatter(np.exp(ridge_pred), np.exp(y_test), s=10, alpha=0.5, color="#3b6ea5")
lim = [1e5, 1.2e9]
ax.plot(lim, lim, "k--", lw=1)
ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="Predicted gross (2025 dollars)",
       ylabel="Actual gross (2025 dollars)", title="Ridge regression: 2022-2025 films")
ax.set_xticks(np.exp(TICKS)); ax.set_xticklabels(LABELS); ax.set_yticks(np.exp(TICKS)); ax.set_yticklabels(LABELS)
fig.tight_layout()
fig.savefig(OUT / "ridge_pred_vs_actual.png", dpi=150)
plt.close(fig)
print(f"\nsaved figures in {OUT}/")
