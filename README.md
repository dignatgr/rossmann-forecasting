# Rossmann Store Sales: time-series forecasting

Daily sales forecasting for 1,115 Rossmann drugstores (Kaggle *Rossmann Store Sales*), built as a step-by-step workflow: exploratory analysis, baselines, a gradient boosting model, rolling-origin cross-validation, feature ablation and a final hold-out evaluation.

The goal is a clean, defensible forecasting pipeline rather than a leaderboard score: every modelling choice is checked against simple baselines and validated on data that comes strictly after the training period.

## Results

Hold-out period: trained on data before **2015-06-14**, evaluated on the following period (open stores only).

| Model | RMSPE | RMSE | MAE | WAPE |
|---|---|---|---|---|
| Baseline (store × day of week × promo mean) | 0.143 | 1090 | 738 | 10.3 % |
| LightGBM, default parameters | 0.137 | 1051 | 727 | 10.2 % |
| LightGBM, tuned + engineered features | **0.121** | **959** | **654** | **9.1 %** |

- The tuned model reduces RMSPE by about 15 % relative to the best baseline (0.121 vs 0.143).
- The gain is smaller on absolute-error metrics: MAE drops by 11 % (738 → 654) and WAPE goes from 10.3 % to 9.1 %.
- WAPE of 9.1 % means the total absolute error is about 9 % of total sales. MAE of about 654 is in sales units per store and day.

Rolling-origin cross-validation (3 folds, 48-day horizon, training data only):

| Setup | Mean RMSPE | Std |
|---|---|---|
| Baseline (store × day of week × promo) | 0.184 | 0.049 |
| LightGBM, tuned (`num_leaves=31`) | 0.151 | 0.012 |
| LightGBM, tuned + competition and holiday features | 0.127 | 0,020 |

The first fold (trained up to 2014-11-30, validated on the 48 days that follow) is the hardest, because the validation window covers the Christmas period, which the training data only partly describes.
The standard deviation across folds increases (0.012 → 0.020) while the mean RMSPE drops.
The new features improve the Easter and spring folds by about 20%, but the Christmas fold
only by about 9%: it remains the hardest to predict, so the gap between folds widens.

## Approach

1. **Exploratory analysis** (`01_eda.ipynb`)
   - 1,017,209 daily rows, 1,115 stores, plus store metadata (type, assortment, competition, Promo2).
   - 17 % of the days are closed days with zero sales; only open days are modelled. 54 open days with zero sales were also found.
   - Missing history: store 13 has a continuous gap of 184 days (2014-07-01 to 2014-12-31).
   - Weekly pattern, promotion effect and yearly seasonality (Easter, Christmas). The apparent Sunday peak comes from the few stores open on Sundays (mostly store type b, which sells more all week), not from a Sunday effect.
   - The sales distribution is strongly skewed, so the model is trained on `log1p(Sales)`.

2. **Baselines** (`02_baselines.ipynb`)
   - Store mean, store × day of week, store × day of week × promo, the same on the last 8 and 16 weeks only, and a seasonal naive.
   - Best baseline: store × day of week × promo (RMSPE 0.143, WAPE 10.3 %).

3. **Modelling** (`03_models.ipynb`)
   - LightGBM on `log1p(Sales)`, predictions back-transformed with `expm1`.
   - Calendar features (day, month, week of year), store attributes, promotions and holidays; `Store`, `StoreType`, `Assortment` and `StateHoliday` as categorical variables.
   - Temporal split at 2015-06-14. Cross-validation folds are built by date (cutoffs 2014-12-01, 2015-03-01, 2015-04-26, 48-day horizon), so validation always comes after training.
   - Tuning of `num_leaves` (7, 15, 31, 63) with early stopping (about 750 trees at `num_leaves=31`).
   - Feature ablation of competition, Promo2 and holiday features, alone and in combination. The best combination (competition + holidays) lowers CV RMSPE from 0.151 to 0.127, with early stopping pointing to 1,352 trees.
   - Final model: `num_leaves=31`, `n_estimators=1352`, `learning_rate=0.05`, 14 features.

## Metrics

- **RMSPE**: Kaggle's metric; penalises relative errors, so low-sales days weigh more.
- **RMSE** and **MAE**: errors in sales units.
- **WAPE**: sum of absolute errors divided by total sales; easy to read as "percentage of sales mispredicted".

## Validation notes and limitations

- Hyperparameters, the number of trees and the feature set were chosen on the cross-validation folds, which end before 2015-06-14. The hold-out period was not used to tune the model.
- The best baseline was selected by comparing variants on the hold-out period, which if anything favours the baseline.
- Early stopping inside each CV fold uses the fold's own validation data, so CV scores are slightly optimistic.
- Results come from a single hold-out window of the end of the data. Seasonal effects outside this window (Christmas, Easter) are only covered by the CV folds.
- Only open days are predicted. Store closures would need a separate model.
- Not done: tuning beyond `num_leaves`, lag or rolling-sales features, prediction intervals, comparison with classical time-series models (ETS, ARIMA), deployment.

## Repository structure

```
notebooks/
  01_eda.ipynb          exploration and data checks
  02_baselines.ipynb    naive and aggregate baselines
  03_models.ipynb       LightGBM, cross-validation, ablation, final evaluation
src/
  data.py               loading, merge with store metadata, date split
  features.py           calendar, competition, Promo2 and holiday features
  crossval.py           rolling-origin folds and CV scoring
  metrics.py            RMSPE, RMSE, MAE, WAPE
data/                   Kaggle files (not versioned)
```

## Reproduce

1. Download `train.csv` and `store.csv` from the [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) Kaggle competition and put them in `data/`.
2. Install the dependencies:
   ```
   pip install pandas numpy matplotlib seaborn lightgbm jupyter
   ```
3. Run the notebooks in order (`01_eda`, `02_baselines`, `03_models`).

## Stack

Python, pandas, NumPy, LightGBM, matplotlib, seaborn.