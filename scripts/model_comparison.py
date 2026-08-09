import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import resample


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


def build_preprocessor(variables):
    continuous_vars = [
        "fathermental",
        "mothermental",
        "friendship",
        "comquality",
        "age",
        "mobilsev",
        "sleep",
        "children",
    ]
    numerical_vars = [var for var in variables if var in continuous_vars]
    categorical_vars = [var for var in variables if var not in continuous_vars]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))]),
                numerical_vars,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder()),
                    ]
                ),
                categorical_vars,
            ),
        ],
        remainder="drop",
    )


def make_pipeline(classifier, variables):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(variables)),
            ("classifier", classifier),
        ]
    )


def youden_threshold(y_true, predicted_probability):
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, predicted_probability)
    return float(thresholds[np.argmax(tpr - fpr)])


def threshold_metrics(y_true, predicted_probability, cutoff):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(predicted_probability) >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": float(cutoff),
        "sensitivity": float(recall_score(y_true, y_pred)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def point_metrics(dataset_name, model_name, y_true, predicted_probability, cutoff):
    metrics = {
        "model": model_name,
        "dataset": dataset_name,
        "n": int(len(y_true)),
        "events": int(np.sum(y_true)),
        "event_rate": float(np.mean(y_true)),
        "auroc": float(roc_auc_score(y_true, predicted_probability)),
        "auprc": float(average_precision_score(y_true, predicted_probability)),
    }
    metrics.update(threshold_metrics(y_true, predicted_probability, cutoff))
    return metrics


def bootstrap_metrics(y_true, predicted_probability, cutoff, n_bootstraps, n_samples, random_state):
    rng = np.random.default_rng(random_state)
    rows = []
    y_true = np.asarray(y_true).astype(int)
    predicted_probability = np.asarray(predicted_probability)

    for _ in range(n_bootstraps):
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        y_resampled, p_resampled = resample(
            y_true,
            predicted_probability,
            n_samples=n_samples,
            random_state=seed,
            stratify=None,
        )
        y_pred = (p_resampled >= cutoff).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_resampled, y_pred).ravel()
        rows.append(
            {
                "auroc": roc_auc_score(y_resampled, p_resampled),
                "auprc": average_precision_score(y_resampled, p_resampled),
                "precision": precision_score(y_resampled, y_pred, zero_division=0),
                "recall": recall_score(y_resampled, y_pred),
                "f1": f1_score(y_resampled, y_pred, zero_division=0),
                "sensitivity": recall_score(y_resampled, y_pred),
                "specificity": tn / (tn + fp),
            }
        )

    frame = pd.DataFrame(rows)
    summary = {}
    for column in frame.columns:
        summary[f"{column}_bootstrap_mean"] = float(frame[column].mean())
        summary[f"{column}_ci_low"] = float(frame[column].quantile(0.025))
        summary[f"{column}_ci_high"] = float(frame[column].quantile(0.975))
    return summary


def get_model_specs():
    specs = [
        (
            "LASSO Classifier",
            LogisticRegression(penalty="l1", solver="liblinear", C=0.5, random_state=42),
        ),
        (
            "MLP",
            MLPClassifier(
                solver="adam",
                hidden_layer_sizes=(10, 10),
                alpha=0.001,
                activation="tanh",
                random_state=42,
                max_iter=1000,
            ),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                max_depth=6,
                max_features=0.6912551485833529,
                min_samples_leaf=4,
                min_samples_split=3,
                n_estimators=40,
                random_state=42,
            ),
        ),
        (
            "GBC",
            GradientBoostingClassifier(
                learning_rate=0.08416407511504528,
                max_depth=2,
                min_samples_leaf=3,
                min_samples_split=6,
                n_estimators=143,
                random_state=42,
            ),
        ),
        (
            "HistGBC",
            HistGradientBoostingClassifier(
                min_samples_leaf=15,
                max_iter=100,
                max_depth=5,
                learning_rate=0.05,
                l2_regularization=0.01,
                random_state=42,
            ),
        ),
    ]

    try:
        import lightgbm as lgb

        specs.append(
            (
                "LightGBM",
                lgb.LGBMClassifier(
                    num_leaves=30,
                    n_estimators=150,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1,
                ),
            )
        )
    except ImportError:
        warnings.warn("LightGBM is not installed; skipping LightGBM.")

    try:
        from catboost import CatBoostClassifier

        specs.append(
            (
                "CatBoost",
                CatBoostClassifier(
                    learning_rate=0.2,
                    l2_leaf_reg=4,
                    iterations=150,
                    depth=5,
                    random_seed=42,
                    verbose=False,
                ),
            )
        )
    except ImportError:
        warnings.warn("CatBoost is not installed; skipping CatBoost.")

    return specs


def main(config_path, bootstrap=False, n_bootstraps=500, n_bootstrap_samples=1000):
    config = load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    variables = config["input_variables"]
    target = config["target_column"]
    train_data = pd.read_csv(config["datasets"]["charls_train"])
    heldout_data = pd.read_csv(config["datasets"]["charls_heldout"])
    cfps_data = pd.read_csv(config["datasets"]["cfps_external_main"])

    rows = []
    for model_name, classifier in get_model_specs():
        model = make_pipeline(classifier, variables)
        model.fit(train_data[variables], train_data[target])

        train_probability = model.predict_proba(train_data[variables])[:, 1]
        cutoff = youden_threshold(train_data[target], train_probability)

        for dataset_name, data in [
            ("charls_heldout", heldout_data),
            ("cfps_external_main", cfps_data),
        ]:
            y_true = data[target].astype(int).to_numpy()
            predicted_probability = model.predict_proba(data[variables])[:, 1]
            metrics = point_metrics(dataset_name, model_name, y_true, predicted_probability, cutoff)
            if bootstrap:
                metrics.update(
                    bootstrap_metrics(
                        y_true,
                        predicted_probability,
                        cutoff,
                        n_bootstraps=n_bootstraps,
                        n_samples=n_bootstrap_samples,
                        random_state=42,
                    )
                )
            rows.append(metrics)

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "model_comparison_metrics.csv", index=False)
    print(f"Wrote model comparison outputs to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/final_model_config.json")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--n-bootstraps", type=int, default=500)
    parser.add_argument("--n-bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    main(
        args.config,
        bootstrap=args.bootstrap,
        n_bootstraps=args.n_bootstraps,
        n_bootstrap_samples=args.n_bootstrap_samples,
    )
