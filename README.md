# Demand Forecasting for Multi-Store Retail Sales

Forecasting weekly Store-Department sales using engineered time-series features, benchmarking a machine learning model (XGBoost) against a classical statistical model (SARIMA), and evaluating final performance on a fully held-out test window.

## Problem

Retailers need accurate weekly sales forecasts per store and department to plan inventory, staffing, and promotions. This project builds and evaluates forecasting models on Walmart's public multi-store sales dataset (via Kaggle's "Store Sales Forecasting" competition), with a particular focus on **holiday weeks**, where forecast errors are operationally the most costly.

## Dataset

- **Source:** Walmart Recruiting – Store Sales Forecasting (Kaggle)
- **Files used:** `train.csv`, `stores.csv`, `features.csv` (Kaggle's `test.csv` is unlabeled and excluded from evaluation — see below)
- **Scale:** 421,570 weekly records across 45 stores and 81 departments, Feb 2010 – Oct 2012

## Evaluation Metric — WMAE

All models are evaluated using **Weighted Mean Absolute Error (WMAE)**

```python
def wmae(y_true, y_pred, is_holiday):
    weights = np.where(is_holiday, 5, 1)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)
```

Holiday weeks are weighted 5x, reflecting their outsized business impact (higher volume, higher volatility, higher cost of getting it wrong). WMAE was chosen over RMSE/MAE because:
- It's **linear**, not squared, so a handful of extreme-volume weeks don't dominate the score the way RMSE would.
- It directly encodes the **business cost structure** — a miss during a holiday week matters more than a miss on a random week — which plain MAE ignores.

## Data Split — Train / Validation / Held-Out Test

A strict **3-way chronological split** is used to avoid data leakage and to keep the final evaluation genuinely unbiased:

| Split | Role |
|---|---|
| `tr` | Model fitting |
| `val` | Model selection, comparison, ablation, error analysis |
| `test_eval` (last 12 weeks) | **Never touched until the very last cell** — used only for final, unbiased evaluation |

Kaggle's `test.csv` has no `Weekly_Sales` labels, so it cannot be scored — it's used only to optionally generate a Kaggle-format submission file, which is explicitly *not* part of the evaluation.

## Approach

1. **Feature engineering** — lag features (previous weeks' sales), rolling mean/std, and date-derived features (week of year, month, quarter, holiday flag).
2. **Stationarity & seasonality diagnostics** — ADF (Augmented Dickey-Fuller) test and seasonal decomposition, used to justify and configure the SARIMA model.
3. **Baseline model** — Seasonal Naive: predicts each week as the historical average for that Store-Dept-WeekOfYear combination.
4. **Statistical model** — SARIMA, with a seasonal-naive fallback for low-volume Store-Dept combinations SARIMA doesn't reliably fit.
5. **ML model** — XGBoost, trained globally across all stores/departments using the engineered features.
6. **Ablation study** — measuring how much of XGBoost's accuracy specifically comes from the lag/rolling features.
7. **Rolling-origin cross-validation** — re-validating across 3 different time windows to confirm results aren't dependent on one lucky split.
8. **Error analysis** — breaking down residuals by store type and holiday weeks to identify where the model struggles.
9. **Final held-out evaluation** — scoring the selected model on `test_eval`, a window never used in training, validation, ablation, or CV.

## Results

| Model | WMAE (validation) |
|---|---|
| Seasonal Naive (baseline) | 2,015.00 |
| SARIMA (+ fallback) | 1,951.28 |
| **XGBoost (global)** | **1,353.35** |

- **XGBoost outperformed SARIMA by 31%** (WMAE), and the seasonal-naive baseline by 33%, on the validation split.
- **Lag/rolling-mean features accounted for 41%** of XGBoost's total accuracy (WMAE without them: 2,281.02 vs. with them: 1,353.35).
- **Rolling-origin CV** confirmed the improvement over baseline held consistently (~25–33%) across 3 independent time windows, not just one split.

### Final held-out test set (never seen during training/validation/CV)

| Model | WMAE (held-out test) |
|---|---|
| Seasonal Naive (baseline) | 1,916.23 |
| **XGBoost (global)** | **1,422.20** |

**XGBoost beat the seasonal-naive baseline by 26% on genuinely unseen data**, closely matching its validation-set performance (1,353.35) — evidence the model generalizes rather than overfitting to the validation split.

## Error Analysis Highlights

- Errors are highest during **holiday weeks** and for a small number of **high-volatility departments** (e.g., Dept 72), which show extreme swings in weekly sales.
- Mean percentage error is misleadingly large on near-zero-sales weeks due to division by small numbers; median percentage error is a more robust read on typical performance.

## Tech Stack

`pandas`, `numpy`, `scikit-learn`, `xgboost`, `statsmodels` (SARIMA, ADF test, seasonal decomposition), `matplotlib`, `seaborn`

