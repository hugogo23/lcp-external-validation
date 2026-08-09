import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fmt_ci(value, low, high, digits=3):
    return f"{value:.{digits}f} ({low:.{digits}f}-{high:.{digits}f})"


def fmt_value(value, digits=3):
    if pd.isna(value):
        return ""
    return f"{value:.{digits}f}"


def nearest_threshold_rows(dca, key_thresholds):
    rows = []
    for dataset, group in dca.groupby("dataset"):
        for key_threshold in key_thresholds:
            idx = (group["threshold_probability"] - key_threshold).abs().idxmin()
            row = group.loc[idx].copy()
            row["requested_threshold"] = key_threshold
            row["selected_threshold"] = row["threshold_probability"]
            row["model_better_than_screen_all"] = (
                row["net_benefit_model"] > row["net_benefit_screen_all"]
            )
            row["model_better_than_screen_none"] = row["net_benefit_model"] > 0
            rows.append(row)
    return pd.DataFrame(rows)


def build_model_comparison(results_dir, master_dir):
    source = pd.read_csv(results_dir / "model_comparison_metrics.csv")
    table = source.copy()
    if {"auroc_ci_low", "auroc_ci_high"}.issubset(table.columns):
        table["AUROC (95% CI)"] = table.apply(
            lambda row: fmt_ci(row["auroc"], row["auroc_ci_low"], row["auroc_ci_high"]),
            axis=1,
        )
    else:
        table["AUROC (95% CI)"] = table["auroc"].map(lambda value: fmt_value(value))

    if {"auprc_ci_low", "auprc_ci_high"}.issubset(table.columns):
        table["AUPRC (95% CI)"] = table.apply(
            lambda row: fmt_ci(row["auprc"], row["auprc_ci_low"], row["auprc_ci_high"]),
            axis=1,
        )
    else:
        table["AUPRC (95% CI)"] = table["auprc"].map(lambda value: fmt_value(value))

    table = table[
        [
            "model",
            "dataset",
            "n",
            "events",
            "event_rate",
            "threshold",
            "AUROC (95% CI)",
            "AUPRC (95% CI)",
            "sensitivity",
            "specificity",
            "precision",
            "f1",
            "cohen_kappa",
        ]
    ].rename(
        columns={
            "model": "Model",
            "dataset": "Dataset",
            "n": "N",
            "events": "Events",
            "event_rate": "Event rate",
            "threshold": "Operating threshold",
            "sensitivity": "Sensitivity",
            "specificity": "Specificity",
            "precision": "PPV",
            "f1": "F1 score",
            "cohen_kappa": "Cohen's kappa",
        }
    )
    table.to_csv(master_dir / "master_table_1_model_comparison.csv", index=False)
    return table


def build_final_validation(results_dir, master_dir):
    source = pd.read_csv(results_dir / "locked_model_metrics.csv")
    table = source.copy()
    if {"auroc_ci_low", "auroc_ci_high"}.issubset(table.columns):
        table["AUROC (95% CI)"] = table.apply(
            lambda row: fmt_ci(row["auroc"], row["auroc_ci_low"], row["auroc_ci_high"]),
            axis=1,
        )
    else:
        table["AUROC (95% CI)"] = table["auroc"].map(lambda value: fmt_value(value))

    if {"auprc_ci_low", "auprc_ci_high"}.issubset(table.columns):
        table["AUPRC (95% CI)"] = table.apply(
            lambda row: fmt_ci(row["auprc"], row["auprc_ci_low"], row["auprc_ci_high"]),
            axis=1,
        )
    else:
        table["AUPRC (95% CI)"] = table["auprc"].map(lambda value: fmt_value(value))

    validation = table[
        [
            "dataset",
            "n",
            "events",
            "event_rate",
            "mean_predicted_risk",
            "AUROC (95% CI)",
            "AUPRC (95% CI)",
            "brier",
            "model_probability_cutoff",
            "sensitivity",
            "specificity",
            "ppv",
            "npv",
            "f1",
            "accuracy",
            "false_positives_per_1000",
            "false_negatives_per_1000",
        ]
    ].rename(
        columns={
            "dataset": "Dataset",
            "n": "N",
            "events": "Events",
            "event_rate": "Event rate",
            "mean_predicted_risk": "Mean predicted risk",
            "brier": "Brier score",
            "model_probability_cutoff": "Model probability cutoff",
            "sensitivity": "Sensitivity",
            "specificity": "Specificity",
            "ppv": "PPV",
            "npv": "NPV",
            "f1": "F1 score",
            "accuracy": "Accuracy",
            "false_positives_per_1000": "False positives per 1,000",
            "false_negatives_per_1000": "False negatives per 1,000",
        }
    )
    validation.to_csv(master_dir / "master_table_2_final_gbc_validation.csv", index=False)

    calibration = source[
        [
            "dataset",
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
            "brier_reliability",
            "brier_resolution",
            "brier_uncertainty",
        ]
    ].rename(
        columns={
            "dataset": "Dataset",
            "brier": "Brier score",
            "ece": "ECE",
            "hosmer_lemeshow": "Hosmer-Lemeshow statistic",
            "hosmer_lemeshow_p_value": "Hosmer-Lemeshow p-value",
            "calibration_intercept": "Calibration intercept",
            "calibration_intercept_ci_low": "Intercept CI low",
            "calibration_intercept_ci_high": "Intercept CI high",
            "calibration_intercept_p_value": "Intercept p-value",
            "calibration_slope": "Calibration slope",
            "calibration_slope_ci_low": "Slope CI low",
            "calibration_slope_ci_high": "Slope CI high",
            "calibration_slope_p_value": "Slope p-value",
            "brier_reliability": "Brier reliability",
            "brier_resolution": "Brier resolution",
            "brier_uncertainty": "Brier uncertainty",
        }
    )
    calibration.to_csv(master_dir / "master_table_3_calibration.csv", index=False)
    return validation, calibration


def build_outcome_sensitivity(results_dir, master_dir):
    source = pd.read_csv(results_dir / "cfps_outcome_sensitivity_metrics.csv")
    table = source[
        [
            "outcome_definition",
            "n",
            "events",
            "event_rate",
            "auroc",
            "auprc",
            "brier",
            "calibration_intercept",
            "calibration_slope",
            "sensitivity",
            "specificity",
            "ppv",
            "npv",
            "f1",
        ]
    ].rename(
        columns={
            "outcome_definition": "Outcome definition",
            "n": "N",
            "events": "Events",
            "event_rate": "Event rate",
            "auroc": "AUROC",
            "auprc": "AUPRC",
            "brier": "Brier score",
            "calibration_intercept": "Calibration intercept",
            "calibration_slope": "Calibration slope",
            "sensitivity": "Sensitivity",
            "specificity": "Specificity",
            "ppv": "PPV",
            "npv": "NPV",
            "f1": "F1 score",
        }
    )
    table.to_csv(master_dir / "master_table_4_outcome_sensitivity.csv", index=False)
    return table


def build_dca(results_dir, master_dir):
    dca = pd.read_csv(results_dir / "decision_curve_results.csv")
    key = nearest_threshold_rows(dca, [0.05, 0.10, 0.1475, 0.15, 0.20, 0.30, 0.40])
    table = key[
        [
            "dataset",
            "requested_threshold",
            "selected_threshold",
            "net_benefit_model",
            "net_benefit_screen_all",
            "net_benefit_screen_none",
            "standardized_net_benefit",
            "model_better_than_screen_all",
            "model_better_than_screen_none",
        ]
    ].rename(
        columns={
            "dataset": "Dataset",
            "requested_threshold": "Requested threshold",
            "selected_threshold": "Selected threshold",
            "net_benefit_model": "Model net benefit",
            "net_benefit_screen_all": "Screen-all net benefit",
            "net_benefit_screen_none": "Screen-none net benefit",
            "standardized_net_benefit": "Standardized net benefit",
            "model_better_than_screen_all": "Model > screen all",
            "model_better_than_screen_none": "Model > screen none",
        }
    )
    table.to_csv(master_dir / "master_table_5_dca_key_thresholds.csv", index=False)
    return dca, table


def plot_dca(dca, figures_dir):
    labels = {
        "charls_train": "CHARLS train",
        "charls_heldout": "CHARLS held-out",
        "cfps_external_main": "CFPS external",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for ax, (dataset, group) in zip(axes, dca.groupby("dataset")):
        ax.plot(group["threshold_probability"], group["net_benefit_model"], label="Model", lw=2)
        ax.plot(
            group["threshold_probability"],
            group["net_benefit_screen_all"],
            label="Screen all",
            lw=1.5,
            linestyle="--",
        )
        ax.axhline(0, label="Screen none", color="black", lw=1, linestyle=":")
        ax.set_title(labels.get(dataset, dataset))
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure_dca_net_benefit.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "figure_dca_net_benefit.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_calibration(results_dir, figures_dir):
    deciles = pd.read_csv(results_dir / "decile_calibration_table.csv")
    labels = {
        "charls_train": "CHARLS train",
        "charls_heldout": "CHARLS held-out",
        "cfps_external_main": "CFPS external",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    for ax, (dataset, group) in zip(axes, deciles.groupby("dataset")):
        ax.plot(
            group["mean_predicted_risk"],
            group["observed_event_rate"],
            marker="o",
            lw=2,
        )
        max_axis = max(group["mean_predicted_risk"].max(), group["observed_event_rate"].max()) * 1.1
        max_axis = max(max_axis, 0.05)
        ax.plot([0, max_axis], [0, max_axis], color="black", linestyle=":", lw=1)
        ax.set_title(labels.get(dataset, dataset))
        ax.set_xlabel("Mean predicted risk")
        ax.set_ylabel("Observed event rate")
        ax.set_xlim(0, max_axis)
        ax.set_ylim(0, max_axis)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure_decile_calibration.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "figure_decile_calibration.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_outcome_sensitivity(table, figures_dir):
    plot_data = table.copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(plot_data))
    ax.plot(x, plot_data["AUROC"], marker="o", lw=2, label="AUROC")
    ax.plot(x, plot_data["AUPRC"], marker="o", lw=2, label="AUPRC")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_data["Outcome definition"], rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Performance")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure_outcome_sensitivity.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "figure_outcome_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def write_dca_interpretation(dca_key_table, output_path):
    lines = [
        "# DCA Interpretation Notes",
        "",
        "These notes summarize whether the model has positive net benefit and whether it exceeds the screen-all strategy at selected threshold probabilities.",
        "",
    ]
    for dataset, group in dca_key_table.groupby("Dataset"):
        positive_range = group[group["Model > screen none"]]
        screen_all_range = group[group["Model > screen all"]]
        lines.append(f"## {dataset}")
        if len(positive_range):
            thresholds = ", ".join(f"{x:.3f}" for x in positive_range["Selected threshold"])
            lines.append(f"- Positive net benefit at selected thresholds: {thresholds}.")
        else:
            lines.append("- No positive net benefit at the selected thresholds.")
        if len(screen_all_range):
            thresholds = ", ".join(f"{x:.3f}" for x in screen_all_range["Selected threshold"])
            lines.append(f"- Higher net benefit than screen-all at selected thresholds: {thresholds}.")
        else:
            lines.append("- Not higher than screen-all at the selected thresholds.")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main(project_dir):
    project_dir = Path(project_dir).resolve()
    results_dir = project_dir / "results"
    master_dir = results_dir / "master_tables"
    figures_dir = project_dir / "figures"
    master_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    build_model_comparison(results_dir, master_dir)
    build_final_validation(results_dir, master_dir)
    outcome_table = build_outcome_sensitivity(results_dir, master_dir)
    dca, dca_key_table = build_dca(results_dir, master_dir)

    plot_dca(dca, figures_dir)
    plot_calibration(results_dir, figures_dir)
    plot_outcome_sensitivity(outcome_table, figures_dir)
    write_dca_interpretation(dca_key_table, results_dir / "dca_interpretation_notes.md")

    print(f"Wrote master tables to {master_dir}")
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=".")
    args = parser.parse_args()
    main(args.project_dir)
