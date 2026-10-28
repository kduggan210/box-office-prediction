import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Step 1: the data, split by time 
films = pd.read_csv("data/processed/features.csv")
films = films[films["has_budget"] == 1]
train = films[films["release_year"] <= 2019]         # learn from 2000-2019
test = films[films["release_year"] >= 2022]          # predict 2022-2025
y_train, y_test = train["log_gross_adj"], test["log_gross_adj"]
cols = ["log_budget_adj", "is_sequel", "is_big_six", "runtime_min"]
print(f"{len(train)} training films, {len(test)} test films, {len(cols)} features")

# Step 2: rescale each feature to "standard deviations from the average"
scaler = StandardScaler().fit(train[cols])
X_train, X_test = scaler.transform(train[cols]), scaler.transform(test[cols])

# Step 3: fit ridge at several penalty strengths
print("\nalpha = 0 is plain linear regression. Coefficients are per 1 standard deviation of the feature.\n")
rows = []
for alpha in (0, 10, 100, 1000, 10000):
    model = Ridge(alpha=alpha).fit(X_train, y_train)
    rows.append({"alpha": alpha, **dict(zip(cols, model.coef_.round(3))),
                 "test R2": round(r2_score(y_test, model.predict(X_test)), 3)})
print(pd.DataFrame(rows).to_string(index=False))
print("\nAs alpha grows, every coefficient shrinks toward zero. A small penalty barely changes the"
      "\nanswer, and a very large one makes the model too timid (R2 falls). Ridge helps most when"
      "\nfeatures are many or move together; with four clean features there is little to fix.")

# Step 4: visualization
alphas = np.logspace(-1, 4.5, 40)
paths, scores = [], []
for alpha in alphas:
    model = Ridge(alpha=alpha).fit(X_train, y_train)
    paths.append(model.coef_)
    scores.append(r2_score(y_test, model.predict(X_test)))
paths = np.array(paths)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
for i, name in enumerate(cols):
    ax[0].plot(alphas, paths[:, i], label=name)
ax[0].axhline(0, color="k", lw=0.8)
ax[0].set(xscale="log", xlabel="Ridge penalty (alpha)", ylabel="Coefficient (per standard deviation)",
          title="Coefficients shrink toward zero as the penalty grows")
ax[0].legend(frameon=False, fontsize=8)
ax[1].plot(alphas, scores, color="#b23a48")
ax[1].set(xscale="log", xlabel="Ridge penalty (alpha)", ylabel="Test R$^2$ (2022-2025 films)",
          title="Too strong a penalty hurts the predictions")
for a in ax:
    a.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("reports/figures/simple_ridge_shrinkage.png", dpi=150)
print("\nsaved reports/figures/simple_ridge_shrinkage.png")
