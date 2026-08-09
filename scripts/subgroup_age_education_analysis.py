import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    project_dir = Path(path).resolve().parents[1]
    config["project_dir"] = project_dir
    return config


def resolve_path(project_dir, value):
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def threshold_metrics(y_true, pred_prob, cutoff):
    y_true = np.asarray(y_true).astype(int)
    pred_label = (np.asarray(pred_prob) >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred_label).ravel()
    return {
        "sensitivity": float(recall_score(y_true, pred_label, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "ppv": float(precision_score(y_true, pred_label, zero_division=0)),
        "npv": float(tn / (tn + fn)) if (tn + fn) else np.nan,
        "f1": float(f1_score(y_true, pred_label, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_model(name, model, frame, features, target, cutoff, dataset):
    x = frame[features]
    y = frame[target].astype(int)
    pred = model.predict_proba(x)[:, 1]
    row = {
        "model": name,
        "dataset": dataset,
        "n": int(len(frame)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "mean_predicted_risk": float(np.mean(pred)),
        "auroc": float(roc_auc_score(y, pred)),
        "auprc": float(average_precision_score(y, pred)),
        "brier": float(brier_score_loss(y, pred)),
        "model_probability_cutoff": float(cutoff),
    }
    row.update(threshold_metrics(y, pred, cutoff))
    return row


def build_age_education_baselines(config_path, train_data=None, heldout_data=None, cfps_data=None):
    config = load_config(config_path)
    project_dir = config["project_dir"]
    target = config["target_column"]
    cutoff = config["model_probability_cutoff"]
    output_dir = resolve_path(project_dir, config.get("output_dir", "results"))
    master_dir = output_dir / "master_tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    master_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(train_data) if train_data else resolve_path(project_dir, config["datasets"]["charls_train"])
    heldout_path = Path(heldout_data) if heldout_data else resolve_path(project_dir, config["datasets"]["charls_heldout"])
    cfps_path = Path(cfps_data) if cfps_data else resolve_path(project_dir, config["datasets"]["cfps_external_main"])
    full_model_path = resolve_path(project_dir, config["model_path"])

    train = pd.read_csv(train_path)
    datasets = {
        "charls_heldout": pd.read_csv(heldout_path),
        "cfps_external_main": pd.read_csv(cfps_path),
    }

    features = ["age", "raeduc_c"]

    logistic = Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        (
                            "age",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="mean")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            ["age"],
                        ),
                        (
                            "education",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            ["raeduc_c"],
                        ),
                    ]
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]
    )

    gbc = Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        ("age", SimpleImputer(strategy="mean"), ["age"]),
                        (
                            "education",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    (
                                        "encoder",
                                        OrdinalEncoder(
                                            handle_unknown="use_encoded_value",
                                            unknown_value=-1,
                                        ),
                                    ),
                                ]
                            ),
                            ["raeduc_c"],
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                GradientBoostingClassifier(
                    learning_rate=0.08416407511504528,
                    max_depth=2,
                    min_samples_leaf=3,
                    min_samples_split=6,
                    n_estimators=143,
                    random_state=42,
                ),
            ),
        ]
    )

    x_train = train[features]
    y_train = train[target].astype(int)
    logistic.fit(x_train, y_train)
    gbc.fit(x_train, y_train)
    full_model = joblib.load(full_model_path)
    full_features = config["input_variables"]

    rows = []
    for dataset_name, frame in datasets.items():
        rows.append(evaluate_model("full_15_predictor_locked_gbc", full_model, frame, full_features, target, cutoff, dataset_name))
        rows.append(evaluate_model("age_education_only_logistic_descriptive", logistic, frame, features, target, cutoff, dataset_name))
        rows.append(evaluate_model("age_education_only_gbc_descriptive", gbc, frame, features, target, cutoff, dataset_name))

    results = pd.DataFrame(rows)
    full = results[results["model"] == "full_15_predictor_locked_gbc"][["dataset", "auroc", "auprc", "brier"]].rename(
        columns={"auroc": "full_auroc", "auprc": "full_auprc", "brier": "full_brier"}
    )
    comparison = results[results["model"] != "full_15_predictor_locked_gbc"].merge(full, on="dataset", how="left")
    comparison["auroc_gap_full_minus_baseline"] = comparison["full_auroc"] - comparison["auroc"]
    comparison["auprc_gap_full_minus_baseline"] = comparison["full_auprc"] - comparison["auprc"]
    comparison["brier_gap_baseline_minus_full"] = comparison["brier"] - comparison["full_brier"]

    results.to_csv(output_dir / "age_education_only_baseline_comparison.csv", index=False)
    comparison.to_csv(output_dir / "age_education_only_baseline_gaps.csv", index=False)
    results.to_csv(master_dir / "master_table_8_age_education_only_baseline_comparison.csv", index=False)
    comparison.to_csv(master_dir / "master_table_9_age_education_only_baseline_gaps.csv", index=False)
    return results, comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Descriptive age+education-only baselines fit in CHARLS training and evaluated in held-out/CFPS."
    )
    parser.add_argument("--config", default="config/final_model_config.json")
    parser.add_argument("--train-data", default=None)
    parser.add_argument("--heldout-data", default=None)
    parser.add_argument("--cfps-data", default=None)
    args = parser.parse_args()
    build_age_education_baselines(args.config, args.train_data, args.heldout_data, args.cfps_data)
