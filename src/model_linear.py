import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model_common import (load, load_groups, datasets, splits, evaluate, explain, save_tables,
                          plot_pred_vs_actual, plot_ablation, finish, TARGET, TITLES)

f, feats = load()
groups = load_groups()
rows, abl_list, imp_list, preds = [], [], [], {}


def linear():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())


for name, df in datasets(f).items():
    s, X, y = splits(df), df[feats], df[TARGET]
    train, test = s["train"], s["test"]                  # 2000-2019 / 2022-2025

    evaluate(rows, name, "Mean baseline", DummyRegressor().fit(X[train], y[train]), X, y, s)

    # 1) one feature: log budget (and whether a budget is recorded)
    budget_cols = ["log_budget_adj", "has_budget"]
    one = linear().fit(X.loc[train, budget_cols], y[train])
    evaluate(rows, name, "Linear: log budget only", one, X, y, s, cols=budget_cols)

    # 2) all features
    full = linear().fit(X[train], y[train])
    test_pred, _ = evaluate(rows, name, "Linear: all features", full, X, y, s)
    preds[name] = (y[test], test_pred)

    coefs = pd.Series(full[-1].coef_, index=feats)
    print(f"\n{TITLES[name]}: largest coefficients (per 1 standard deviation of the feature)")
    print(coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(10).round(3).to_string())

    def fit_predict(cols):                               # refit with only these columns
        return linear().fit(X.loc[train, cols], y[train]).predict(X.loc[test, cols])

    a, i = explain(name, fit_predict, full, X, y, s, feats, groups)
    abl_list.append(a); imp_list.append(i)

save_tables("linear", abl_list, imp_list)
plot_pred_vs_actual("linear", "Linear regression (all features)", preds)
plot_ablation("linear", "Linear regression", pd.concat(abl_list))
finish("linear", rows)
