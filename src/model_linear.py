import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Step 1: the data, split by time
films = pd.read_csv("data/processed/features.csv")
films = films[films["has_budget"] == 1]              # keep films with a recorded budget

train = films[films["release_year"] <= 2019]         # learn from the past (2000-2019)
test = films[films["release_year"] >= 2022]          # then predict newer films (2022-2025)
print(f"{len(train)} training films, {len(test)} test films")

# The thing we predict: ln(gross). Working in logs means a model's error is a percentage miss, not a dollar miss, which suits numbers that range from $0.1M to $1B.
y_train, y_test = train["log_gross_adj"], test["log_gross_adj"]

# Step 2: ONE feature (log budget) -> the best straight line
line = LinearRegression().fit(train[["log_budget_adj"]], y_train)
slope, intercept = line.coef_[0], line.intercept_
print("\n--- Step 2: one feature ---")
print(f"ln(gross) = {intercept:.2f} + {slope:.2f} * ln(budget)")
print(f"A 10% bigger budget goes with a {(1.1 ** slope - 1) * 100:.1f}% bigger gross "
      f"(doubling the budget multiplies gross by {2 ** slope:.2f}).")
pred = line.predict(test[["log_budget_adj"]])
print(f"R2 on the test films: {r2_score(y_test, pred):.3f}   (1 = perfect, 0 = no better than the average)")

# a few predictions in plain dollars
sample = test.sample(5, random_state=1)
shown = pd.DataFrame({"film": sample["title"],
                      "predicted $M": np.exp(line.predict(sample[["log_budget_adj"]])) / 1e6,
                      "actual $M": np.exp(sample["log_gross_adj"]) / 1e6}).round(1)
print("\nfive test films:")
print(shown.to_string(index=False))

# visualization: every film is a dot, the model is the line
ticks = np.log([1e6, 1e7, 1e8, 1e9])
labels = ["$1M", "$10M", "$100M", "$1B"]
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(train["log_budget_adj"], y_train, s=8, alpha=0.3, label="training films (2000-2019)", color="#3b6ea5")
ax.scatter(test["log_budget_adj"], y_test, s=8, alpha=0.5, label="test films (2022-2025)", color="#e08a1e")
xs = np.linspace(train["log_budget_adj"].min(), train["log_budget_adj"].max(), 50)
ax.plot(xs, intercept + slope * xs, color="k", lw=2, label="the fitted line")
ax.set_xticks(ticks); ax.set_xticklabels(labels); ax.set_yticks(ticks); ax.set_yticklabels(labels)
ax.set(xlabel="Budget (2025 dollars)", ylabel="Domestic gross (2025 dollars)",
       title="Linear regression with one feature")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("reports/figures/simple_linear_fit.png", dpi=150)
print("\nsaved reports/figures/simple_linear_fit.png")

# Step 3: SEVERAL features -> the same idea with more inputs
cols = ["log_budget_adj", "is_sequel", "is_big_six", "runtime_min"]
several = LinearRegression().fit(train[cols], y_train)
print("\n--- Step 3: four features ---")
print("Each feature gets its own coefficient; the prediction is the intercept plus")
print("coefficient x feature value, added up.")
for name, c in zip(cols, several.coef_):
    print(f"  {name:<15} coefficient {c:+.3f}")
print(f"R2 on the test films: {r2_score(y_test, several.predict(test[cols])):.3f}")
