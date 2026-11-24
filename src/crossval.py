import pandas as pd
import numpy as np
import lightgbm as lgb
from .metrics import metrics


def make_folds(df, cutoffs, horizon_days=48):
    """Return a list of (train, valid) pairs for rolling-origin validation."""
    folds = []
    for cutoff in cutoffs:
        tr = df[df["Date"] < cutoff]
        va = df[(df["Date"] >= cutoff) & (df["Date"] < cutoff + pd.Timedelta(days=horizon_days))]
        folds.append((tr, va))
    return folds

def cv_score_baseline(folds):
    """Return per-fold scores and their mean/std."""
    scores = []
    for tr, va in folds:
        # baseline
        baseline = tr.groupby(["Store", "DayOfWeek", "Promo"], observed=True)["Sales"].mean().rename('pred')
        va = va.merge(baseline, on=["Store", "DayOfWeek", "Promo"], how="left")
        pred = va["pred"]
        print("missing predictions:", va["pred"].isna().sum())   
        scores.append(metrics(va["Sales"], pred))
    per_fold = pd.concat(scores, ignore_index=True)
    summary = pd.DataFrame({"mean": per_fold.mean(), "std": per_fold.std()})
    return per_fold, summary

def cv_score(folds, features, params, early_stopping_rounds=100):
    """Fit one model per fold on log1p(Sales) with early stopping on the fold's validation."""
    scores, best_iters = [], []
    for tr, va in folds:
        model = lgb.LGBMRegressor(**params)
        model.fit(
            tr[features], np.log1p(tr["Sales"]),
            eval_set=[(va[features], np.log1p(va["Sales"]))],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
            )
        pred = np.expm1(model.predict(va[features]))
        scores.append(metrics(va["Sales"], pred))
        best_iters.append(model.best_iteration_)
    per_fold = pd.concat(scores, ignore_index=True).assign(best_iter=best_iters)
    summary = pd.DataFrame({"mean": per_fold.mean(), "std": per_fold.std()})
    return per_fold, summary
