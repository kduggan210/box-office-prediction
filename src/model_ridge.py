import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model_common import (load, load_groups, datasets, splits, evaluate, explain, save_tables,
                          plot_pred_vs_actual, plot_ablation, finish, FIG, TARGET, TITLES)

f, feats = load()
groups = load_groups()
rows, curves, abl_list, imp_list, preds = [], {}, [], [], {}


def ridge(alpha):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))


def rmse(model, X, y):
    return float(np.sqrt(np.mean((model.predict(X) - y) ** 2)))


for name, df in datasets(f).items():
    s, X, y = splits(df), df[feats], df[TARGET]
    tr, va, train, test = s["tune_train"], s["validate"], s["train"], s["test"]

    alphas = np.logspace(-2, 3, 30)
    curve = pd.DataFrame([(a, rmse(m := ridge(a).fit(X[tr], y[tr]), X[tr], y[tr]), rmse(m, X[va], y[va]))
                          for a in alphas], columns=["alpha", "train_rmse", "val_rmse"])

    best = float(curve.loc[curve["val_rmse"] <= curve["val_rmse"].min() * 1.002, "alpha"].max())
    curves[name] = (curve, best)
    print(f"\n{TITLES[name]}: chosen alpha = {best:.3g}")

    final = ridge(best).fit(X[train], y[train])
    test_pred, test_row = evaluate(rows, name, "Ridge", final, X, y, s)
    preds[name] = (y[test], test_pred)
    print(f"validation RMSE (2016-2019) {curve['val_rmse'].min():.3f} vs test RMSE (2022-2025) "
          f"{test_row['rmse_log']:.3f}; mean signed error on test {test_row['bias_log']:+.3f} (positive = over-prediction)")

    coefs = pd.Series(final[-1].coef_, index=feats)
    print("largest coefficients (per 1 standard deviation of the feature):")
    print(coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(10).round(3).to_string())

    def fit_predict(cols):
        return ridge(best).fit(X.loc[train, cols], y[train]).predict(X.loc[test, cols])

    a, i = explain(name, fit_predict, final, X, y, s, feats, groups)
    abl_list.append(a); imp_list.append(i)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
for a, (name, (curve, best)) in zip(ax, curves.items()):
    a.plot(curve["alpha"], curve["train_rmse"], label="training (2000-2015)", color="#3b6ea5")
    a.plot(curve["alpha"], curve["val_rmse"], label="validation (2016-2019)", color="#b23a48")
    a.axvline(best, color="k", ls="--", lw=1)
    a.set(xscale="log", xlabel="Ridge penalty (alpha)", title=TITLES[name])
ax[0].set_ylabel("RMSE of ln(gross)")
ax[0].legend(frameon=False)
fig.suptitle("Ridge regression: training and validation error by penalty strength")
fig.tight_layout()
fig.savefig(FIG / "ridge_validation_curve.png", dpi=150)
print(f"\nsaved {FIG / 'ridge_validation_curve.png'}")

save_tables("ridge", abl_list, imp_list)
plot_pred_vs_actual("ridge", "Ridge regression", preds)
plot_ablation("ridge", "Ridge regression", pd.concat(abl_list))
finish("ridge", rows)
