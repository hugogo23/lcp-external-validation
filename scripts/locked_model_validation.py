import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.interpolate import interp1d
from scipy.stats import chi2
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils import resample

try:
    from dca import decision_curve
except ImportError:
    from scripts.dca import decision_curve


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    project_dir = Path(path).resolve().parents[1]
    config["project_dir"] = str(project_dir)
    config["public_model_path"] = config["model_path"]

    for key in ["model_path", "output_dir"]:
        value = Path(config[key])
        if not value.is_absolute():
            config[key] = str(project_dir / value)

    for group in ["datasets", "cfps_outcome_sensitivity"]:
        if group in config:
            for name, value in config[group].items():
                path_value = Path(value)
                if not path_value.is_absolute():
                    config[group][name] = str(project_dir / path_value)

    return config


def model_sha256(model_path):
    return hashlib.sha256(Path(model_path).read_bytes()).hexdigest()


def calibration_intercept_slope(y_true, predicted_probability):
    eps = 1e-10
    p = np.clip(np.asarray(predicted_probability), eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    fitted = sm.Logit(np.asarray(y_true).astype(int), sm.add_constant(logit_p)).fit(disp=0)
    ci = fitted.conf_int()
    intercept_p = 2 * (1 - stats.norm.cdf(abs(fitted.params[0] / fitted.bse[0])))
    slope_p = 2 * (1 - stats.norm.cdf(abs((fitted.params[1] - 1) / fitted.bse[1])))
    return {
        "calibration_intercept": float(fitted.params[0]),
        "calibration_intercept_ci_low": float(ci[0, 0]),
        "calibration_intercept_ci_high": float(ci[0, 1]),
        "calibration_intercept_p_value": float(intercept_p),
        "calibration_slope": float(fitted.params[1]),
        "calibration_slope_ci_low": float(ci[1, 0]),
        "calibration_slope_ci_high": float(ci[1, 1]),
        "calibration_slope_p_value": float(slope_p),
    }


def ece_score(y_true, predicted_probability, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    predicted_probability = np.asarray(predicted_probability)
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, predicted_probability, n_bins=n_bins
    )

    if len(fraction_of_positives) < n_bins:
        bin_centers = np.linspace(0, 1, n_bins)
        existing_bin_centers = np.linspace(0, 1, len(fraction_of_positives))
        fraction_interp = interp1d(
            existing_bin_centers, fraction_of_positives, kind="linear", fill_value="extrapolate"
        )
        mean_pred_interp = interp1d(
            existing_bin_centers, mean_predicted_value, kind="linear", fill_value="extrapolate"
        )
        fraction_of_positives = fraction_interp(bin_centers)
        mean_predicted_value = mean_pred_interp(bin_centers)

    bin_sizes = np.histogram(predicted_probability, bins=n_bins)[0]
    return float(
        np.sum((bin_sizes / len(y_true)) * np.abs(fraction_of_positives - mean_predicted_value))
    )


def hosmer_lemeshow_test(y_true, predicted_probability, bins=10):
    y_true = np.asarray(y_true).astype(int)
    predicted_probability = np.asarray(predicted_probability)
    quantiles = np.linspace(0, 1, bins + 1)
    bin_cuts = np.quantile(predicted_probability, quantiles)
    bin_assignments = np.digitize(predicted_probability, bin_cuts, right=True)

    observed_events = []
    expected_events = []
    for i in range(1, bins + 1):
        bin_indices = np.where(bin_assignments == i)[0]
        observed_events.append(np.sum(y_true[bin_indices]))
        expected_events.append(np.sum(predicted_probability[bin_indices]))

    expected_events = np.asarray(expected_events)
    observed_events = np.asarray(observed_events)
    valid = expected_events > 0
    hl_statistic = np.sum((observed_events[valid] - expected_events[valid]) ** 2 / expected_events[valid])
    p_value = 1 - chi2.cdf(hl_statistic, bins - 2)
    return float(hl_statistic), float(p_value)


def brier_score_decomposition(y_true, predicted_probability, n_bins=10):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(predicted_probability)
    brier = float(np.mean((p - y) ** 2))
    prevalence = float(np.mean(y))

    if len(y) == 0:
        raise ValueError("Cannot calculate Brier decomposition for an empty dataset.")
    if n_bins <= 0 or n_bins > len(y):
        n_bins = min(len(y), 10)

    if np.all(p == p[0]):
        bins = np.zeros_like(p, dtype=int)
    else:
        try:
            bins = pd.qcut(p, q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            bins = np.zeros_like(p, dtype=int)

    reliability = 0.0
    resolution = 0.0
    for k in np.unique(bins):
        mask = bins == k
        n_k = np.sum(mask)
        if n_k == 0:
            continue
        observed_rate = np.mean(y[mask])
        mean_predicted = np.mean(p[mask])
        weight = n_k / len(y)
        reliability += weight * (observed_rate - mean_predicted) ** 2
        resolution += weight * (observed_rate - prevalence) ** 2

    uncertainty = prevalence * (1 - prevalence)
    return {
        "brier_reliability": float(reliability),
        "brier_resolution": float(resolution),
        "brier_uncertainty": float(uncertainty),
        "brier_decomposition_check": float(reliability - resolution + uncertainty),
        "brier_decomposition_abs_error": float(abs(brier - (reliability - resolution + uncertainty))),
    }


def threshold_metrics(y_true, predicted_probability, cutoff):
    y_true = np.asarray(y_true).astype(int)
    predicted_positive = (np.asarray(predicted_probability) >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted_positive).ravel()

    return {
        "model_probability_cutoff": float(cutoff),
        "sensitivity": float(recall_score(y_true, predicted_positive)),
        "specificity": float(tn / (tn + fp)),
        "ppv": float(precision_score(y_true, predicted_positive, zero_division=0)),
        "npv": float(tn / (tn + fn)),
        "f1": float(f1_score(y_true, predicted_positive, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predicted_positive)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positives_per_1000": float(fp / len(y_true) * 1000),
        "false_negatives_per_1000": float(fn / len(y_true) * 1000),
    }


def bootstrap_discrimination_metrics(
    y_true, predicted_probability, n_bootstraps=2000, random_state=42
):
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true).astype(int)
    predicted_probability = np.asarray(predicted_probability)
    n = len(y_true)
    rows = []

    for _ in range(n_bootstraps):
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        y_resampled, p_resampled = resample(
            y_true,
            predicted_probability,
            n_samples=n,
            random_state=seed,
            stratify=None,
        )
        if len(np.unique(y_resampled)) < 2:
            continue
        rows.append(
            {
                "auroc": roc_auc_score(y_resampled, p_resampled),
                "auprc": average_precision_score(y_resampled, p_resampled),
            }
        )

    frame = pd.DataFrame(rows)
    return {
        "auroc_bootstrap_mean": float(frame["auroc"].mean()),
        "auroc_ci_low": float(frame["auroc"].quantile(0.025)),
        "auroc_ci_high": float(frame["auroc"].quantile(0.975)),
        "auprc_bootstrap_mean": float(frame["auprc"].mean()),
        "auprc_ci_low": float(frame["auprc"].quantile(0.025)),
        "auprc_ci_high": float(frame["auprc"].quantile(0.975)),
        "bootstrap_repetitions": int(len(frame)),
    }


def decile_calibration_table(dataset_name, y_true, predicted_probability):
    frame = pd.DataFrame(
        {
            "dataset": dataset_name,
            "y_true": np.asarray(y_true).astype(int),
            "predicted_probability": np.asarray(predicted_probability),
        }
    )
    frame["risk_decile"] = pd.qcut(
        frame["predicted_probability"], 10, labels=False, duplicates="drop"
    )
    table = (
        frame.groupby("risk_decile", dropna=False)
        .agg(
            n=("y_true", "size"),
            events=("y_true", "sum"),
            observed_event_rate=("y_true", "mean"),
            mean_predicted_risk=("predicted_probability", "mean"),
        )
        .reset_index()
    )
    table["risk_decile"] = table["risk_decile"] + 1
    table["dataset"] = dataset_name
    table["observed_expected_ratio"] = table["events"] / (
        table["mean_predicted_risk"] * table["n"]
    )
    return table


def export_model_specification(model, config, output_dir):
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]
    spec = {
        "model_path": config.get("public_model_path", config["model_path"]),
        "model_sha256": model_sha256(config["model_path"]),
        "algorithm": type(classifier).__name__,
        "input_variables": config["input_variables"],
        "hyperparameters": {
            key: getattr(classifier, key, None)
            for key in [
                "learning_rate",
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "subsample",
                "random_state",
                "max_features",
            ]
        },
        "preprocessing": {},
    }

    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            imputer = transformer.named_steps["imputer"]
            spec["preprocessing"]["continuous"] = {
                "columns": list(columns),
                "imputation": "mean",
                "imputation_values": dict(zip(columns, map(float, imputer.statistics_))),
            }
        elif name == "cat":
            imputer = transformer.named_steps["imputer"]
            encoder = transformer.named_steps["encoder"]
            spec["preprocessing"]["categorical"] = {
                "columns": list(columns),
                "imputation": "most_frequent",
                "imputation_values": dict(zip(columns, map(str, imputer.statistics_))),
                "ordinal_encoder_categories": {
                    column: list(map(str, categories))
                    for column, categories in zip(columns, encoder.categories_)
                },
            }

    with open(output_dir / "final_model_specification.json", "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)

    return spec


def evaluate_dataset(dataset_name, data, model, config):
    variables = config["input_variables"]
    target = config["target_column"]
    cutoff = config["model_probability_cutoff"]
    y_true = data[target].astype(int).to_numpy()
    predicted_probability = model.predict_proba(data[variables])[:, 1]

    metrics = {
        "dataset": dataset_name,
        "n": int(len(data)),
        "events": int(y_true.sum()),
        "event_rate": float(y_true.mean()),
        "mean_predicted_risk": float(predicted_probability.mean()),
        "auroc": float(roc_auc_score(y_true, predicted_probability)),
        "auprc": float(average_precision_score(y_true, predicted_probability)),
        "brier": float(brier_score_loss(y_true, predicted_probability)),
    }
    metrics.update(
        bootstrap_discrimination_metrics(
            y_true,
            predicted_probability,
            n_bootstraps=config.get("bootstrap_repetitions", 2000),
            random_state=config.get("bootstrap_random_state", 42),
        )
    )
    metrics.update(calibration_intercept_slope(y_true, predicted_probability))
    metrics["ece"] = ece_score(y_true, predicted_probability)
    hl_statistic, hl_p_value = hosmer_lemeshow_test(y_true, predicted_probability, bins=10)
    metrics["hosmer_lemeshow"] = hl_statistic
    metrics["hosmer_lemeshow_p_value"] = hl_p_value
    metrics.update(brier_score_decomposition(y_true, predicted_probability, n_bins=10))
    metrics.update(threshold_metrics(y_true, predicted_probability, cutoff))

    predictions = pd.DataFrame(
        {
            "dataset": dataset_name,
            "ID": data["ID"].values if "ID" in data.columns else np.arange(len(data)),
            "y_true": y_true,
            "predicted_probability": predicted_probability,
        }
    )
    deciles = decile_calibration_table(dataset_name, y_true, predicted_probability)

    return metrics, predictions, deciles


def main(config_path):
    config = load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(config["model_path"])
    export_model_specification(model, config, output_dir)

    metric_rows = []
    prediction_tables = []
    decile_tables = []
    dca_tables = []

    threshold_grid = np.round(
        np.arange(
            config["decision_curve"]["threshold_min"],
            config["decision_curve"]["threshold_max"] + 1e-9,
            config["decision_curve"]["threshold_step"],
        ),
        3,
    )

    for dataset_name, dataset_path in config["datasets"].items():
        data = pd.read_csv(dataset_path)
        metrics, predictions, deciles = evaluate_dataset(dataset_name, data, model, config)
        metric_rows.append(metrics)
        prediction_tables.append(predictions)
        decile_tables.append(deciles)

        dca = decision_curve(
            predictions["y_true"].to_numpy(),
            predictions["predicted_probability"].to_numpy(),
            threshold_grid,
        )
        dca["dataset"] = dataset_name
        dca_tables.append(dca)

    pd.DataFrame(metric_rows).to_csv(output_dir / "locked_model_metrics.csv", index=False)
    pd.concat(prediction_tables, ignore_index=True).to_csv(
        output_dir / "locked_model_predictions.csv", index=False
    )
    pd.concat(decile_tables, ignore_index=True).to_csv(
        output_dir / "decile_calibration_table.csv", index=False
    )
    pd.concat(dca_tables, ignore_index=True).to_csv(
        output_dir / "decision_curve_results.csv", index=False
    )

    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/final_model_config.json")
    args = parser.parse_args()
    main(args.config)
