import argparse
import json
from pathlib import Path

import pandas as pd


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    project_dir = Path(path).resolve().parents[1]
    config["project_dir"] = project_dir
    return config


def resolve_path(project_dir, value):
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def build_missingness_table(config_path):
    config = load_config(config_path)
    project_dir = config["project_dir"]
    output_dir = resolve_path(project_dir, config.get("output_dir", "results"))
    master_dir = output_dir / "master_tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    master_dir.mkdir(parents=True, exist_ok=True)

    variables = list(config["input_variables"]) + [config["target_column"]]
    rows = []
    for dataset, relative_path in config["datasets"].items():
        path = resolve_path(project_dir, relative_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing dataset for {dataset}: {path}. Public repositories should document "
                "where authorized users can obtain CHARLS/CFPS files and place harmonized files "
                "under the configured relative data paths."
            )
        frame = pd.read_csv(path)
        n = len(frame)
        target = config["target_column"]
        events = int(frame[target].sum()) if target in frame.columns else None
        event_rate = float(frame[target].mean()) if target in frame.columns else None
        for variable in variables:
            if variable not in frame.columns:
                rows.append({
                    "dataset": dataset,
                    "variable": variable,
                    "n": n,
                    "events": events,
                    "event_rate": event_rate,
                    "missing_n": None,
                    "missing_percent": None,
                    "available_n": None,
                    "variable_present": False,
                })
                continue
            missing_n = int(frame[variable].isna().sum())
            rows.append({
                "dataset": dataset,
                "variable": variable,
                "n": n,
                "events": events,
                "event_rate": event_rate,
                "missing_n": missing_n,
                "missing_percent": missing_n / n * 100 if n else None,
                "available_n": n - missing_n,
                "variable_present": True,
            })

    missingness = pd.DataFrame(rows)
    missingness.to_csv(output_dir / "missingness_table.csv", index=False)

    wide_parts = []
    for dataset in config["datasets"]:
        subset = missingness.loc[
            missingness["dataset"] == dataset,
            ["variable", "missing_n", "missing_percent", "available_n"],
        ].copy()
        subset = subset.rename(
            columns={
                "missing_n": f"{dataset} missing n",
                "missing_percent": f"{dataset} missing %",
                "available_n": f"{dataset} available n",
            }
        )
        wide_parts.append(subset)

    wide = wide_parts[0]
    for subset in wide_parts[1:]:
        wide = wide.merge(subset, on="variable", how="outer")
    wide.insert(0, "Variable", wide.pop("variable"))
    wide.to_csv(master_dir / "master_table_6_missingness.csv", index=False)
    return missingness, wide


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build missingness tables for final predictors.")
    parser.add_argument("--config", default="config/final_model_config.json")
    args = parser.parse_args()
    build_missingness_table(args.config)
