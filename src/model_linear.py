import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = pathlib.Path("reports/figures/basic")
OUT.mkdir(parents=True, exist_ok=True)
TICKS = np.log([1e6, 1e7, 1e8, 1e9])
LABELS = ["$1M", "$10M", "$100M", "$1B"]

# 1. Data, split by time
films = pd.read_csv("data/processed/features.csv")
films = films[films["has_budget"] == 1]                  # films with a recorded budget
features = list(pd.read_csv("data/processed/feature_groups.csv")["feature"])

train = films[films["release_year"] <= 2019]             # learn from the past
test = films[films["release_year"] >= 2022]              # predict newer films
y_train, y_test = train["log_gross_adj"], test["log_gross_adj"]
print(f"{len(train)} training films (2000-2019), {len(test)} test films (2022-2025), {len(features)} features")


def score(y_true, y_pred):
    miss = np.abs(y_pred - y_true)                       # error in log units
    return {"R2": round(r2_score(y_true, y_pred), 3),
            "typical miss (x)": round(float(np.exp(np.median(miss))), 2),   # 2.0 = off by a factor of 2
            "within 2x": round(float(np.mean(miss <= np.log(2))), 3)}       # share of films within a factor of 2


# 2. Baseline
baseline = np.full(len(test), y_train.mean())

# 3. ONE feature
line = LinearRegression().fit(train[["log_budget_adj"]], y_train)
slope, intercept = line.coef_[0], line.intercept_
print(f"\nOne feature: ln(gross) = {intercept:.2f} + {slope:.2f} * ln(budget)")
print(f"  -> a 10% bigger budget goes with a {(1.1 ** slope - 1) * 100:.1f}% bigger gross; "
      f"doubling the budget multiplies gross by {2 ** slope:.2f}")
one_pred = line.predict(test[["log_budget_adj"]])

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(train["log_budget_adj"], y_train, s=8, alpha=0.3, color="#3b6ea5", label="training films")
ax.scatter(test["log_budget_adj"], y_test, s=8, alpha=0.5, color="#e08a1e", label="test films")
xs = np.linspace(train["log_budget_adj"].min(), train["log_budget_adj"].max(), 50)
ax.plot(xs, intercept + slope * xs, color="k", lw=2, label="the fitted line")
ax.set_xticks(TICKS); ax.set_xticklabels(LABELS); ax.set_yticks(TICKS); ax.set_yticklabels(LABELS)
ax.set(xlabel="Budget (2025 dollars)", ylabel="Domestic gross (2025 dollars)",
       title="Linear regression with one feature")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "linear_one_feature.png", dpi=150)
plt.close(fig)

# 4. ALL features
# Two preparation steps, then the regression:
#   - some films have no director/actor history, so fill those gaps with the training median
#   - rescale every feature to "standard deviations from the average" so coefficients are comparable
model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())
model.fit(train[features], y_train)
all_pred = model.predict(test[features])

results = pd.DataFrame({"always guess the average": score(y_test, baseline),
                        "linear: budget only": score(y_test, one_pred),
                        "linear: all features": score(y_test, all_pred)}).T
print("\nScores on the 2022-2025 films (R2: 1 = perfect, 0 = no better than the average):")
print(results.to_string())

# 5. The biggest coefficients (per 1 standard deviation of the feature)
coefs = pd.Series(model[-1].coef_, index=features)
print("\nBiggest coefficients (+ pushes the prediction up, - pushes it down):")
print(coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(8).round(3).to_string())
print("Careful: features that overlap (e.g. a film's previous gross and the average of earlier"
      "\nfilms) share credit, so read single coefficients loosely.")

# 6. Visualization: predicted vs actual 
fig, ax = plt.subplots(figsize=(6, 5.5))
ax.scatter(np.exp(all_pred), np.exp(y_test), s=10, alpha=0.5, color="#3b6ea5")
lim = [1e5, 1.2e9]
ax.plot(lim, lim, "k--", lw=1)
ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="Predicted gross (2025 dollars)",
       ylabel="Actual gross (2025 dollars)", title="Linear regression: 2022-2025 films")
ax.set_xticks(np.exp(TICKS)); ax.set_xticklabels(LABELS); ax.set_yticks(np.exp(TICKS)); ax.set_yticklabels(LABELS)
fig.tight_layout()
fig.savefig(OUT / "linear_pred_vs_actual.png", dpi=150)
plt.close(fig)
print(f"\nsaved figures in {OUT}/")
