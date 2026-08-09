import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

try:
    from locked_model_validation import calibration_intercept_slope, threshold_metrics
except ImportError:
    from scripts.locked_model_validation import calibration_intercept_slope, threshold_metrics


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    project_dir = Path(path).resolve().parents[1]
    config["project_dir"] = str(project_dir)

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


def harmonize_known_cfps_categories(data):
    """Apply the same CFPS-to-model category coding used in the original validation notebook."""
    working = data.copy()
    mappings = {
        "shlt": {
            "1.unhealthy": "1.poor",
            "2.fair": "2.fair",
            "3.healthy": "3.good",
            "Poor": "1.poor",
            "Fair": "2.fair",
            "Good": "3.good",
            "Very good": "3.good",
            "Excellent": "3.good",
            "不健康": "1.poor",
            "一般": "2.fair",
            "比较健康": "3.good",
            "很健康": "3.good",
            "非常健康": "3.good",
        },
        "rural": {"乡村": "0.rural", "城镇": "1.urban", "城市": "1.urban"},
        "ragender": {"女": "0.female", "男": "1.male"},
        "smoke": {
            "1.current smoke": "1.current smoking",
            "2.ever smoke": "2.ever smoking",
            "3.never smoke": "3.never smoking",
        },
        "hukou": {
            "0.agricultural": "0.agriculture",
            "1.non-agricultural": "1.non-agriculture",
            "农业户口": "0.agriculture",
            "非农户口": "1.non-agriculture",
            "居民户口": "1.non-agriculture",
        },
        "job": {
            "0.agricultural": "0.agriculture",
            "1.non-agricultural": "1.non-agriculture",
            "Agricultural Work": "0.agriculture",
            "Non-agricultural Work": "1.non-agriculture",
            "农业工作": "0.agriculture",
            "非农工作": "1.non-agriculture",
        },
        "chronic": {
            "0.had chronic diseases": "0.yes",
            "1.no chronic diseases": "1.no",
            "No": "1.no",
            "Yes": "0.yes",
            "否": "1.no",
            "是": "0.yes",
        },
        "sinojapanese": {"0.noexperience": "0.no", "1.sinojapanese": "1.yes"},
        "civilwar": {"0.noexperience": "0.no", "1.civilwar": "1.yes"},
        "raeduc_c": {
            "文盲/半文盲": "1.illiterate",
            "小学": "2.primary school",
            "初中": "3.secondary school and above",
            "高中/中专/技校/职高/大专/本科/硕士/博士": "3.secondary school and above",
        },
    }
    for column, mapping in mappings.items():
        if column in working.columns:
            working[column] = working[column].replace(mapping)
    return working


def evaluate_outcome(data, model, config, outcome_column, label):
    variables = config["input_variables"]
    cutoff = config["model_probability_cutoff"]
    y_true = data[outcome_column].astype(int).to_numpy()
    predicted_probability = model.predict_proba(data[variables])[:, 1]

    metrics = {
        "outcome_definition": label,
        "n": int(len(data)),
        "events": int(y_true.sum()),
        "event_rate": float(y_true.mean()),
        "auroc": float(roc_auc_score(y_true, predicted_probability)),
        "auprc": float(average_precision_score(y_true, predicted_probability)),
        "brier": float(brier_score_loss(y_true, predicted_probability)),
    }
    metrics.update(calibration_intercept_slope(y_true, predicted_probability))
    metrics.update(threshold_metrics(y_true, predicted_probability, cutoff))
    return metrics


def add_age_education_residual_outcome(data, percentile=15):
    """Define low performance using residual cognition after age/education adjustment."""
    required = ["future_cognition", "age", "raeduc_c"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing columns for adjusted outcome: {missing}")

    working = data.copy()
    design = pd.get_dummies(working[["age", "raeduc_c"]], columns=["raeduc_c"], drop_first=True)
    design = design.fillna(design.median(numeric_only=True))
    y = working["future_cognition"].astype(float)

    fit = LinearRegression().fit(design, y)
    residual = y - fit.predict(design)
    threshold = np.nanpercentile(residual, percentile)
    column = f"age_education_residual_bottom_{percentile}"
    working[column] = residual <= threshold
    return working, column, float(threshold)


def main(config_path):
    config = load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(config["model_path"])
    rows = []

    for label, dataset_path in config["cfps_outcome_sensitivity"].items():
        data = harmonize_known_cfps_categories(pd.read_csv(dataset_path))
        rows.append(evaluate_outcome(data, model, config, config["target_column"], label))

    main_data = harmonize_known_cfps_categories(
        pd.read_csv(config["cfps_outcome_sensitivity"]["bottom_15_percent_main"])
    )
    adjusted_data, adjusted_column, residual_threshold = add_age_education_residual_outcome(
        main_data, percentile=15
    )
    adjusted_metrics = evaluate_outcome(
        adjusted_data,
        model,
        config,
        adjusted_column,
        "age_education_adjusted_residual_bottom_15_percent",
    )
    adjusted_metrics["residual_threshold"] = residual_threshold
    rows.append(adjusted_metrics)

    pd.DataFrame(rows).to_csv(output_dir / "cfps_outcome_sensitivity_metrics.csv", index=False)
    print(f"Wrote outcome sensitivity outputs to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/final_model_config.json")
    args = parser.parse_args()
    main(args.config)
