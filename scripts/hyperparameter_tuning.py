import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


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


def candidate_search_spaces():
    spaces = {
        "LASSO Classifier": (
            LogisticRegression(penalty="l1", solver="liblinear", random_state=42),
            {"classifier__C": [0.1, 0.5, 1, 2, 3, 5, 10]},
        ),
        "MLP": (
            MLPClassifier(random_state=42, max_iter=1000),
            {
                "classifier__hidden_layer_sizes": [(10,), (10, 10), (20,), (10, 10, 10)],
                "classifier__activation": ["relu", "tanh"],
                "classifier__solver": ["adam", "lbfgs"],
                "classifier__alpha": [0.0001, 0.001, 0.01],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {
                "classifier__n_estimators": randint(10, 100),
                "classifier__max_depth": randint(3, 7),
                "classifier__min_samples_split": randint(2, 10),
                "classifier__min_samples_leaf": randint(1, 5),
                "classifier__max_features": uniform(0.1, 0.9),
            },
        ),
        "GBC": (
            GradientBoostingClassifier(random_state=42),
            {
                "classifier__n_estimators": randint(50, 200),
                "classifier__learning_rate": uniform(0.01, 0.2),
                "classifier__max_depth": randint(2, 10),
                "classifier__min_samples_split": randint(2, 10),
                "classifier__min_samples_leaf": randint(1, 5),
            },
        ),
        "HistGBC": (
            HistGradientBoostingClassifier(random_state=42),
            {
                "classifier__learning_rate": [0.05, 0.1, 0.2],
                "classifier__max_iter": [50, 100, 150, 200],
                "classifier__max_depth": [5, 10, 20, 30, None],
                "classifier__min_samples_leaf": [5, 10, 15, 20, 25],
                "classifier__l2_regularization": [0.0, 0.01, 0.05],
            },
        ),
    }

    try:
        import lightgbm as lgb

        spaces["LightGBM"] = (
            lgb.LGBMClassifier(random_state=42, verbose=-1),
            {
                "classifier__n_estimators": [50, 100, 150],
                "classifier__learning_rate": [0.01, 0.1, 0.2],
                "classifier__num_leaves": [20, 30, 40],
                "classifier__max_depth": [3, 4, 5],
            },
        )
    except ImportError:
        warnings.warn("LightGBM is not installed; skipping LightGBM tuning.")

    try:
        from catboost import CatBoostClassifier

        spaces["CatBoost"] = (
            CatBoostClassifier(random_seed=42, verbose=False),
            {
                "classifier__iterations": [50, 100, 150, 200, 300],
                "classifier__depth": [3, 5, 7, 9],
                "classifier__learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25],
                "classifier__l2_leaf_reg": [1, 2, 3, 4, 5],
            },
        )
    except ImportError:
        warnings.warn("CatBoost is not installed; skipping CatBoost tuning.")

    return spaces


def run_search(model_name, classifier, param_distributions, train_data, variables, target, n_iter):
    pipeline = make_pipeline(classifier, variables)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="roc_auc",
        error_score="raise",
        return_train_score=True,
        random_state=42,
    )
    search.fit(train_data[variables], train_data[target])

    best_index = search.best_index_
    return {
        "model": model_name,
        "best_params": search.best_params_,
        "best_cv_auroc": float(search.best_score_),
        "mean_train_auroc_at_best": float(search.cv_results_["mean_train_score"][best_index]),
        "train_auroc_refit": float(
            roc_auc_score(
                train_data[target],
                search.best_estimator_.predict_proba(train_data[variables])[:, 1],
            )
        ),
    }, pd.DataFrame(search.cv_results_).assign(model=model_name)


def main(config_path, n_iter):
    config = load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = pd.read_csv(config["datasets"]["charls_train"])
    variables = config["input_variables"]
    target = config["target_column"]

    summaries = []
    cv_results = []
    for model_name, (classifier, param_distributions) in candidate_search_spaces().items():
        summary, result = run_search(
            model_name,
            classifier,
            param_distributions,
            train_data,
            variables,
            target,
            n_iter=n_iter,
        )
        summaries.append(summary)
        cv_results.append(result)

    summary_frame = pd.DataFrame(summaries)
    summary_frame["best_params_json"] = summary_frame["best_params"].apply(
        lambda value: json.dumps(value, default=str, ensure_ascii=False)
    )
    summary_frame.drop(columns=["best_params"]).to_csv(
        output_dir / "hyperparameter_tuning_summary.csv", index=False
    )
    pd.concat(cv_results, ignore_index=True).to_csv(
        output_dir / "hyperparameter_tuning_cv_results.csv", index=False
    )

    selected = {
        row["model"]: row["best_params"]
        for row in summaries
    }
    with open(output_dir / "hyperparameter_tuning_selected_params.json", "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2, default=str)

    print(f"Wrote hyperparameter tuning outputs to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/final_model_config.json")
    parser.add_argument("--n-iter", type=int, default=20)
    args = parser.parse_args()
    main(args.config, n_iter=args.n_iter)
