import numpy as np
import pandas as pd

def metrics(y_true, y_pred):
    """Return RMSPE, RMSE, MAE and WAPE as a one-row DataFrame."""
    mask = y_true > 0                      # ignore zero sales (RMSPE divides by y_true)
    y_true, y_pred = y_true[mask], y_pred[mask]
    err = y_true - y_pred
    return pd.DataFrame({
        "RMSPE": [np.sqrt(np.mean((err / y_true) ** 2))],
        "RMSE":  [np.sqrt(np.mean(err ** 2))],
        "MAE":   [np.mean(np.abs(err))],
        "WAPE":  [np.sum(np.abs(err)) / np.sum(y_true)],
    })
