import argparse
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


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    project_dir = Path(path).resolve().parents[1]
    config["project_dir"] = project_dir
    return config


def resolve_path(project_dir, value):
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def normalize_sex(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"0.female", "female", "f", "0", "女"}:
        return "Female"
    if text in {"1.male", "male", "m", "1", "男"}:
        return "Male"
    return str(value)


def age_group(value):
    if pd.isna(value):
        return np.nan
    age = float(value)
    if 45 <= age <= 59:
        return "45-59"
    if age >= 60:
        return ">=60"
    return "<45"


def normalize_education(value):
    if pd.isna(value):
        return "Missing"
    text = str(value).strip().lower()
    if text in {"1.illiterate", "illiterate", "文盲/半文盲", "文盲", "半文盲"}:
        return "Illiterate"
    if text in {"2.primary school", "primary school", "小学"}:
        return "Primary school"
    if text in {"3.secondary school and above", "secondary school and above", "初中", "高中", "大专", "大学本科", "硕士", "博士"}:
        return "Secondary school and above"
    if "illiterate" in text or "文盲" in text:
        return "Illiterate"
    if "primary" in text or "小学" in text:
        return "Primary school"
    if "secondary" in text or "middle" in text or "above" in text or "初中" in text:
        return "Secondary school and above"
    return str(value)


def threshold_metrics(y_true, predicted_probability, cutoff):
    y_true = np.asarray(y_true).astype(int)
    predicted_positive = (np.asarray(predicted_probability) >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted_positive).ravel()
    return {
        "model_probability_cutoff": float(cutoff),
        "sensitivity": float(recall_score(y_true, predicted_positive, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "ppv": float(precision_score(y_true, predicted_positive, zero_division=0)),
        "npv": float(tn / (tn + fn)) if (tn + fn) else np.nan,
        "f1": float(f1_score(y_true, predicted_positive, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predicted_positive)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positives_per_1000": float(fp / len(y_true) * 1000),
        "false_negatives_per_1000": float(fn / len(y_true) * 1000),
    }


def bootstrap_discrimination(y_true, predicted_probability, n_bootstraps, random_state):
    y_true = np.asarray(y_true).astype(int)
    predicted_probability = np.asarray(predicted_probability)
    rng = np.random.default_rng(random_state)
    rows = []
    for _ in range(n_bootstraps):
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        y_resampled, p_resampled = resample(
            y_true,
            predicted_probability,
            n_samples=len(y_true),
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
    if frame.empty:
        return {
            "auroc_ci_low": np.nan,
            "auroc_ci_high": np.nan,
            "auprc_ci_low": np.nan,
            "auprc_ci_high": np.nan,
            "bootstrap_repetitions": 0,
        }
    return {
        "auroc_ci_low": float(frame["auroc"].quantile(0.025)),
        "auroc_ci_high": float(frame["auroc"].quantile(0.975)),
        "auprc_ci_low": float(frame["auprc"].quantile(0.025)),
        "auprc_ci_high": float(frame["auprc"].quantile(0.975)),
        "bootstrap_repetitions": int(len(frame)),
    }


def ece_score(y_true, predicted_probability, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    predicted_probability = np.asarray(predicted_probability)
    if len(np.unique(y_true)) < 2:
        return np.nan
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
    bins = min(bins, len(y_true))
    if bins < 3 or len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    bin_cuts = np.quantile(predicted_probability, np.linspace(0, 1, bins + 1))
    bin_assignments = np.digitize(predicted_probability, bin_cuts, right=True)
    observed_events = []
    expected_events = []
    for i in range(1, bins + 1):
        idx = np.where(bin_assignments == i)[0]
        if len(idx) == 0:
            continue
        observed_events.append(np.sum(y_true[idx]))
        expected_events.append(np.sum(predicted_probability[idx]))
    expected_events = np.asarray(expected_events)
    observed_events = np.asarray(observed_events)
    valid = expected_events > 0
    if valid.sum() < 3:
        return np.nan, np.nan
    hl_statistic = np.sum((observed_events[valid] - expected_events[valid]) ** 2 / expected_events[valid])
    p_value = 1 - chi2.cdf(hl_statistic, max(valid.sum() - 2, 1))
    return float(hl_statistic), float(p_value)


def calibration_intercept_slope(y_true, predicted_probability):
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return {
            "calibration_intercept": np.nan,
            "calibration_intercept_ci_low": np.nan,
            "calibration_intercept_ci_high": np.nan,
            "calibration_intercept_p_value": np.nan,
            "calibration_slope": np.nan,
            "calibration_slope_ci_low": np.nan,
            "calibration_slope_ci_high": np.nan,
            "calibration_slope_p_value": np.nan,
        }
    eps = 1e-10
    p = np.clip(np.asarray(predicted_probability), eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    try:
        fitted = sm.Logit(y_true, sm.add_constant(logit_p)).fit(disp=0)
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
    except Exception:
        return {
            "calibration_intercept": np.nan,
            "calibration_intercept_ci_low": np.nan,
            "calibration_intercept_ci_high": np.nan,
            "calibration_intercept_p_value": np.nan,
            "calibration_slope": np.nan,
            "calibration_slope_ci_low": np.nan,
            "calibration_slope_ci_high": np.nan,
            "calibration_slope_p_value": np.nan,
        }


def evaluate_subgroup(dataset_name, group_type, group, y_true, predicted_probability, cutoff, n_bootstraps, seed):
    y_true = np.asarray(y_true).astype(int)
    predicted_probability = np.asarray(predicted_probability)
    row = {
        "dataset": dataset_name,
        "group_type": group_type,
        "group": group,
        "n": int(len(y_true)),
        "events": int(y_true.sum()),
        "event_rate": float(y_true.mean()) if len(y_true) else np.nan,
        "mean_predicted_risk": float(predicted_probability.mean()) if len(y_true) else np.nan,
    }
    expected_events = row["mean_predicted_risk"] * row["n"]
    row["observed_expected_ratio"] = float(row["events"] / expected_events) if expected_events else np.nan
    if len(np.unique(y_true)) >= 2:
        row["auroc"] = float(roc_auc_score(y_true, predicted_probability))
        row["auprc"] = float(average_precision_score(y_true, predicted_probability))
        row["auprc_prevalence_baseline"] = row["event_rate"]
        row["auprc_enrichment_ratio"] = (
            float(row["auprc"] / row["event_rate"]) if row["event_rate"] else np.nan
        )
        row.update(bootstrap_discrimination(y_true, predicted_probability, n_bootstraps, seed))
        row["brier"] = float(brier_score_loss(y_true, predicted_probability))
        row["ece"] = ece_score(y_true, predicted_probability)
        hl, hl_p = hosmer_lemeshow_test(y_true, predicted_probability)
        row["hosmer_lemeshow"] = hl
        row["hosmer_lemeshow_p_value"] = hl_p
        row.update(calibration_intercept_slope(y_true, predicted_probability))
        row.update(threshold_metrics(y_true, predicted_probability, cutoff))
    else:
        for key in [
            "auroc",
            "auprc",
            "auprc_prevalence_baseline",
            "auprc_enrichment_ratio",
            "auroc_ci_low",
            "auroc_ci_high",
            "auprc_ci_low",
            "auprc_ci_high",
            "brier",
            "ece",
            "hosmer_lemeshow",
            "hosmer_lemeshow_p_value",
            "calibration_intercept",
            "calibration_intercept_ci_low",
            "calibration_intercept_ci_high",
            "calibration_intercept_p_value",
            "calibration_slope",
            "calibration_slope_ci_low",
            "calibration_slope_ci_high",
            "calibration_slope_p_value",
        ]:
            row[key] = np.nan
        row.update(threshold_metrics(y_true, predicted_probability, cutoff))
        row["bootstrap_repetitions"] = 0
    row["small_cell_flag"] = bool(row["n"] < 100 or row["events"] < 20 or (row["n"] - row["events"]) < 20)
    return row


def add_subgroup_columns(frame):
    frame = frame.copy()
    frame["subgroup_sex"] = frame["ragender"].map(normalize_sex)
    frame["subgroup_age"] = frame["age"].map(age_group)
    frame["subgroup_education"] = frame["raeduc_c"].map(normalize_education)
    return frame


SUBGROUP_BOOTSTRAP_OFFSETS = {
    ("overall", "Overall"): 0,
    ("sex", "Female"): 1,
    ("sex", "Male"): 2,
    ("age", "45-59"): 3,
    ("age", ">=60"): 4,
    ("education", "Illiterate"): 6,
    ("education", "Primary school"): 7,
    ("education", "Secondary school and above"): 8,
    ("education", "Missing"): 9,
}


def subgroup_rows_for_dataset(dataset_name, frame, model, config, n_bootstraps):
    features = config["input_variables"]
    target = config["target_column"]
    cutoff = config["model_probability_cutoff"]
    y_true = frame[target].astype(int).to_numpy()
    predicted_probability = model.predict_proba(frame[features])[:, 1]
    work = add_subgroup_columns(frame)
    work["predicted_probability"] = predicted_probability
    rows = [
        evaluate_subgroup(
            dataset_name,
            "overall",
            "Overall",
            y_true,
            predicted_probability,
            cutoff,
            n_bootstraps,
            config.get("bootstrap_random_state", 42)
            + SUBGROUP_BOOTSTRAP_OFFSETS[("overall", "Overall")],
        )
    ]
    subgroup_specs = [
        ("sex", "subgroup_sex", ["Female", "Male"]),
        ("age", "subgroup_age", ["45-59", ">=60"]),
        (
            "education",
            "subgroup_education",
            ["Illiterate", "Primary school", "Secondary school and above", "Missing"],
        ),
    ]
    for group_type, column, ordered_levels in subgroup_specs:
        for level in ordered_levels:
            subset = work[work[column] == level]
            if subset.empty:
                continue
            rows.append(
                evaluate_subgroup(
                    dataset_name,
                    group_type,
                    level,
                    subset[target].astype(int).to_numpy(),
                    subset["predicted_probability"].to_numpy(),
                    cutoff,
                    n_bootstraps,
                    config.get("bootstrap_random_state", 42)
                    + SUBGROUP_BOOTSTRAP_OFFSETS[(group_type, level)],
                )
            )
    return rows


def fmt_decimal(value, digits=3):
    if pd.isna(value):
        return ""
    return f"{value:.{digits}f}"


def fmt_ci(value, low, high):
    if pd.isna(value):
        return ""
    if pd.isna(low) or pd.isna(high):
        return fmt_decimal(value)
    return f"{value:.3f} ({low:.3f}-{high:.3f})"


def build_main_table(detailed):
    cfps = detailed[
        (detailed["dataset"] == "cfps_external_main")
        & (detailed["group_type"].isin(["sex", "age"]))
        & (detailed["group"] != "Missing")
    ].copy()
    table = pd.DataFrame(
        {
            "Subgroup": cfps["group_type"].map({"sex": "Sex", "age": "Age", "education": "Education"}),
            "Level": cfps["group"],
            "N": cfps["n"],
            "Events, n (%)": [
                f"{int(events)} ({rate * 100:.1f}%)"
                for events, rate in zip(cfps["events"], cfps["event_rate"])
            ],
            "AUROC (95% CI)": [
                fmt_ci(v, lo, hi)
                for v, lo, hi in zip(cfps["auroc"], cfps["auroc_ci_low"], cfps["auroc_ci_high"])
            ],
            "AUPRC (95% CI)": [
                fmt_ci(v, lo, hi)
                for v, lo, hi in zip(cfps["auprc"], cfps["auprc_ci_low"], cfps["auprc_ci_high"])
            ],
            "AUPRC prevalence baseline": [fmt_decimal(v) for v in cfps["auprc_prevalence_baseline"]],
            "AUPRC enrichment ratio": [fmt_decimal(v) for v in cfps["auprc_enrichment_ratio"]],
            "Brier score": [fmt_decimal(v) for v in cfps["brier"]],
            "O/E ratio": [fmt_decimal(v) for v in cfps["observed_expected_ratio"]],
            "Sensitivity": [fmt_decimal(v) for v in cfps["sensitivity"]],
            "Specificity": [fmt_decimal(v) for v in cfps["specificity"]],
            "PPV": [fmt_decimal(v) for v in cfps["ppv"]],
            "NPV": [fmt_decimal(v) for v in cfps["npv"]],
        }
    )
    return table


def build_supplement_table(detailed):
    keep = detailed[
        (detailed["group_type"].isin(["overall", "sex", "age", "education"]))
        & ~((detailed["group_type"] == "education") & (detailed["group"] == "Missing"))
    ].copy()
    return pd.DataFrame(
        {
            "Dataset": keep["dataset"].map(
                {
                    "charls_heldout": "CHARLS held-out validation",
                    "cfps_external_main": "CFPS external validation",
                }
            ),
            "Subgroup": keep["group_type"].map(
                {"overall": "Overall", "sex": "Sex", "age": "Age", "education": "Education"}
            ),
            "Level": keep["group"],
            "N": keep["n"],
            "Events, n (%)": [
                f"{int(events)} ({rate * 100:.1f}%)"
                for events, rate in zip(keep["events"], keep["event_rate"])
            ],
            "Mean predicted risk": [fmt_decimal(v) for v in keep["mean_predicted_risk"]],
            "Observed/expected ratio": [fmt_decimal(v) for v in keep["observed_expected_ratio"]],
            "AUROC (95% CI)": [
                fmt_ci(v, lo, hi)
                for v, lo, hi in zip(keep["auroc"], keep["auroc_ci_low"], keep["auroc_ci_high"])
            ],
            "AUPRC (95% CI)": [
                fmt_ci(v, lo, hi)
                for v, lo, hi in zip(keep["auprc"], keep["auprc_ci_low"], keep["auprc_ci_high"])
            ],
            "AUPRC prevalence baseline": [fmt_decimal(v) for v in keep["auprc_prevalence_baseline"]],
            "AUPRC enrichment ratio": [fmt_decimal(v) for v in keep["auprc_enrichment_ratio"]],
            "Brier score": [fmt_decimal(v) for v in keep["brier"]],
            "ECE": [fmt_decimal(v) for v in keep["ece"]],
            "Hosmer-Lemeshow p value": [fmt_decimal(v, 4) for v in keep["hosmer_lemeshow_p_value"]],
            "Calibration intercept (95% CI)": [
                fmt_ci(v, lo, hi)
                for v, lo, hi in zip(
                    keep["calibration_intercept"],
                    keep["calibration_intercept_ci_low"],
                    keep["calibration_intercept_ci_high"],
                )
            ],
            "Calibration slope (95% CI)": [
                fmt_ci(v, lo, hi)
                for v, lo, hi in zip(
                    keep["calibration_slope"],
                    keep["calibration_slope_ci_low"],
                    keep["calibration_slope_ci_high"],
                )
            ],
            "Sensitivity": [fmt_decimal(v) for v in keep["sensitivity"]],
            "Specificity": [fmt_decimal(v) for v in keep["specificity"]],
            "PPV": [fmt_decimal(v) for v in keep["ppv"]],
            "NPV": [fmt_decimal(v) for v in keep["npv"]],
        }
    )


def main(config_path, charls_heldout=None, cfps_data=None, n_bootstraps=None):
    config = load_config(config_path)
    project_dir = config["project_dir"]
    output_dir = resolve_path(project_dir, config.get("output_dir", "results"))
    master_dir = output_dir / "master_tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    master_dir.mkdir(parents=True, exist_ok=True)

    heldout_path = Path(charls_heldout) if charls_heldout else resolve_path(project_dir, config["datasets"]["charls_heldout"])
    cfps_path = Path(cfps_data) if cfps_data else resolve_path(project_dir, config["datasets"]["cfps_external_main"])
    model = joblib.load(resolve_path(project_dir, config["model_path"]))
    n_bootstraps = int(n_bootstraps or config.get("bootstrap_repetitions", 2000))

    datasets = {
        "charls_heldout": pd.read_csv(heldout_path),
        "cfps_external_main": pd.read_csv(cfps_path),
    }
    rows = []
    for dataset_name, frame in datasets.items():
        rows.extend(subgroup_rows_for_dataset(dataset_name, frame, model, config, n_bootstraps))

    detailed = pd.DataFrame(rows)
    main_table = build_main_table(detailed)
    supplement_table = build_supplement_table(detailed)

    detailed_path = output_dir / "subgroup_performance_detailed.csv"
    main_path = output_dir / "subgroup_performance_main_table.csv"
    supplement_path = output_dir / "subgroup_performance_supplement_table.csv"
    detailed.to_csv(detailed_path, index=False)
    main_table.to_csv(main_path, index=False)
    supplement_table.to_csv(supplement_path, index=False)
    detailed.to_csv(master_dir / "master_table_10_subgroup_performance_detailed.csv", index=False)
    main_table.to_csv(master_dir / "master_table_11_subgroup_performance_main_table.csv", index=False)
    supplement_table.to_csv(master_dir / "master_table_12_subgroup_performance_supplement_table.csv", index=False)
    print(f"Wrote {detailed_path}")
    print(f"Wrote {main_path}")
    print(f"Wrote {supplement_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Locked-model subgroup performance analyses.")
    parser.add_argument("--config", default="config/final_model_config.json")
    parser.add_argument("--charls-heldout", default=None)
    parser.add_argument("--cfps-data", default=None)
    parser.add_argument("--n-bootstraps", default=None)
    args = parser.parse_args()
    main(args.config, args.charls_heldout, args.cfps_data, args.n_bootstraps)
