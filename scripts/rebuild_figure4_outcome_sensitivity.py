from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    project_dir = config_path.resolve().parents[1]

    value = Path(config["output_dir"])
    if not value.is_absolute():
        config["output_dir"] = str(project_dir / value)

    config["project_dir"] = str(project_dir)
    return config


def make_labels(values: list[str]) -> list[str]:
    mapping = {
        "bottom_10_percent": "Bottom 10%",
        "bottom_15_percent_main": "Bottom 15%\n(primary)",
        "bottom_20_percent": "Bottom 20%",
    }
    return [mapping[v] for v in values]


def main(config_path: str) -> None:
    config = load_config(Path(config_path))
    project_dir = Path(config["project_dir"])
    table_path = Path(config["output_dir"]) / "cfps_outcome_sensitivity_metrics.csv"
    figures_dir = project_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(table_path)
    plot = table[
        table["outcome_definition"].isin(
            ["bottom_10_percent", "bottom_15_percent_main", "bottom_20_percent"]
        )
    ].copy()
    order = ["bottom_10_percent", "bottom_15_percent_main", "bottom_20_percent"]
    plot["order"] = plot["outcome_definition"].map({v: i for i, v in enumerate(order)})
    plot = plot.sort_values("order")

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 12

    x = np.arange(len(plot))
    labels = make_labels(plot["outcome_definition"].tolist())

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.5), dpi=600)
    colors = {
        "AUROC": "#1f77b4",
        "AUPRC": "#ff7f0e",
        "Event rate": "#7f7f7f",
        "Sensitivity": "#2ca02c",
        "Specificity": "#d62728",
        "PPV": "#9467bd",
        "NPV": "#8c564b",
    }

    metric_columns = {
        "AUROC": "auroc",
        "AUPRC": "auprc",
        "Event rate": "event_rate",
        "Sensitivity": "sensitivity",
        "Specificity": "specificity",
        "PPV": "ppv",
        "NPV": "npv",
    }

    ax = axes[0]
    for metric, linestyle in [("AUROC", "-"), ("AUPRC", "-"), ("Event rate", "--")]:
        y = plot[metric_columns[metric]].to_numpy()
        ax.plot(
            x,
            y,
            marker="o",
            markersize=8,
            linewidth=3,
            linestyle=linestyle,
            color=colors[metric],
            label=metric,
        )
        for xi, yi in zip(x, y):
            ax.text(
                xi,
                yi + (0.02 if metric != "Event rate" else 0.015),
                f"{yi:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
                color=colors[metric],
            )

    ax.axvspan(0.75, 1.25, color="#e6e6e6", zorder=0)
    ax.set_title("Discrimination and precision-recall performance", fontsize=16, pad=12)
    ax.set_ylabel("Metric value", fontsize=14)
    ax.set_xlabel("CFPS lower-tail cutoff", fontsize=14, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 0.9)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.tick_params(axis="both", labelsize=12)

    ax = axes[1]
    for metric in ["Sensitivity", "Specificity", "PPV", "NPV"]:
        y = plot[metric_columns[metric]].to_numpy()
        ax.plot(
            x,
            y,
            marker="o",
            markersize=8,
            linewidth=3,
            color=colors[metric],
            label=metric,
        )
    ax.axvspan(0.75, 1.25, color="#e6e6e6", zorder=0)
    ax.set_title("Fixed-cutoff performance\n(model-probability cutoff = 0.148)", fontsize=16, pad=12)
    ax.set_ylabel("Metric value", fontsize=14)
    ax.set_xlabel("CFPS lower-tail cutoff", fontsize=14, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.tick_params(axis="both", labelsize=12)

    for panel, ax in zip(["A", "B"], axes):
        ax.text(-0.14, 1.06, panel, transform=ax.transAxes, fontsize=22, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    handles, legend_labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        legend_labels.extend(l)
    dedup = dict(zip(legend_labels, handles))
    fig.legend(
        dedup.values(),
        dedup.keys(),
        loc="lower center",
        ncol=7,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, -0.01),
        handlelength=2,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.28, wspace=0.27)

    png = figures_dir / "figure4_outcome_sensitivity_legend_below_600dpi.png"
    pdf = figures_dir / "figure4_outcome_sensitivity_legend_below_600dpi.pdf"
    fig.savefig(png, dpi=600)
    fig.savefig(pdf)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/final_model_config.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
