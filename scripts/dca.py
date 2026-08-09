import numpy as np
import pandas as pd


def decision_curve(y_true, predicted_probability, thresholds):
    """Compute decision-curve net benefit for model, screen-all, and screen-none."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(predicted_probability)
    n = len(y)
    prevalence = y.mean()
    rows = []

    for pt in thresholds:
        predicted_positive = p >= pt
        tp = np.sum(predicted_positive & (y == 1))
        fp = np.sum(predicted_positive & (y == 0))
        net_benefit_model = tp / n - fp / n * pt / (1 - pt)
        net_benefit_all = prevalence - (1 - prevalence) * pt / (1 - pt)

        rows.append(
            {
                "threshold_probability": float(pt),
                "net_benefit_model": float(net_benefit_model),
                "net_benefit_screen_all": float(net_benefit_all),
                "net_benefit_screen_none": 0.0,
                "standardized_net_benefit": float(net_benefit_model / prevalence)
                if prevalence > 0
                else np.nan,
            }
        )

    return pd.DataFrame(rows)

