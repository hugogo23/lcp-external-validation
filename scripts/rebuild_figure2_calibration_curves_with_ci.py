from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from PIL import Image


TARGET_DEFAULT = "cog_impair"


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    project_dir = config_path.resolve().parents[1]

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

    config["project_dir"] = str(project_dir)
    return config


def fixed_decile_curve(y_true: np.ndarray, pred: np.ndarray, n_bins: int = 10):
    frame = pd.DataFrame({"y_true": y_true, "predicted_probability": pred})
    frame["risk_decile"] = pd.qcut(
        frame["predicted_probability"], n_bins, labels=False, duplicates="drop"
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
    table["risk_decile"] = table["risk_decile"].astype(int) + 1
    return table, frame["risk_decile"].astype(int).to_numpy()


def bootstrap_decile_ci(
    y_true: np.ndarray,
    original_bins: np.ndarray,
    n_bootstrap: int = 2000,
    random_state: int = 42,
):
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    bin_values = np.sort(np.unique(original_bins))
    n_bins = len(bin_values)
    rates = np.full((n_bootstrap, n_bins), np.nan)

    for b in range(n_bootstrap):
        sampled = rng.integers(0, n, size=n)
        y_b = y_true[sampled]
        bins_b = original_bins[sampled]
        for idx, bin_value in enumerate(bin_values):
            mask = bins_b == bin_value
            if mask.any():
                rates[b, idx] = y_b[mask].mean()

    low = np.nanpercentile(rates, 2.5, axis=0)
    high = np.nanpercentile(rates, 97.5, axis=0)
    return low, high


def predict_dataset(model, path: Path, variables: list[str], target: str):
    data = pd.read_csv(path)
    y_true = data[target].astype(int).to_numpy()
    pred = model.predict_proba(data[variables])[:, 1]
    return y_true, pred


def plot_panel(curve, ci_low, ci_high, title, panel, out_base, out_dir: Path, axis_max=0.5):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6.9, 5.05), dpi=600)

    x = curve["mean_predicted_risk"].to_numpy()
    y = curve["observed_event_rate"].to_numpy()

    ax.fill_between(
        x,
        ci_low,
        ci_high,
        color="#9E9E9E",
        alpha=0.16,
        linewidth=0,
        label="95% CI",
        zorder=1,
    )
    ax.plot(
        x,
        y,
        color="blue",
        marker="o",
        markersize=6,
        linewidth=2.0,
        label="Calibration curve",
        zorder=3,
    )
    ax.plot(
        [0, axis_max],
        [0, axis_max],
        color="black",
        linestyle="--",
        linewidth=1.6,
        label="Perfect calibration",
        zorder=2,
    )

    ax.set_xlim(0, axis_max)
    ax.set_ylim(0, axis_max)
    ax.set_xlabel("Mean predicted probability", fontsize=16)
    ax.set_ylabel("Observed LCP proportion", fontsize=16)
    ax.set_title(title, fontsize=18, pad=12)
    ax.grid(True, color="#D0D0D0", alpha=0.55, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=14, width=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
    ax.legend(
        loc="upper left",
        fontsize=9.5,
        frameon=True,
        handlelength=1.8,
        borderpad=0.6,
        labelspacing=0.5,
    )
    ax.text(
        -0.13,
        1.05,
        panel,
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        va="top",
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{out_base}.png", dpi=600, bbox_inches="tight")
    fig.savefig(out_dir / f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_decile_bar_panel(
    deciles: pd.DataFrame,
    metric_row: pd.Series,
    title: str,
    panel: str,
    out_base: str,
    out_dir: Path,
    axis_max: float,
) -> None:
    """Render a Figure 2 decile bar panel from the released aggregate tables."""
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6.9, 5.05), dpi=600)

    x = deciles["risk_decile"].to_numpy()
    observed = deciles["observed_event_rate"].to_numpy()
    predicted = deciles["mean_predicted_risk"].to_numpy()
    width = 0.36
    ax.bar(x - width / 2, observed, width, color="#1f77b4", label="Observed LCP")
    ax.bar(x + width / 2, predicted, width, color="#ff7f0e", label="Predicted probability")

    for xpos, value in zip(x - width / 2, observed):
        ax.text(xpos, value + 0.004, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    for xpos, value in zip(x + width / 2, predicted):
        ax.text(xpos, value + 0.004, f"{value:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlim(0.35, 10.65)
    ax.set_ylim(0, axis_max)
    ax.set_xticks(x)
    # Use the same 0.10 major-tick density in A and B as in C and D.
    ax.yaxis.set_major_locator(MultipleLocator(0.10))
    ax.set_xlabel("Predicted-probability decile", fontsize=16)
    ax.set_ylabel("Proportion", fontsize=16)
    ax.set_title(title, fontsize=18, pad=12)
    ax.grid(axis="y", color="#D0D0D0", alpha=0.55, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=14, width=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
    ax.legend(loc="upper left", fontsize=9.5, frameon=True)
    ax.text(
        -0.13,
        1.05,
        panel,
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        va="top",
        ha="left",
    )
    ax.text(
        0.50,
        0.78,
        f"HL p-value: {metric_row['hosmer_lemeshow_p_value']:.4f}\n"
        f"ECE: {metric_row['ece']:.4f}",
        transform=ax.transAxes,
        fontsize=10,
        ha="center",
        va="top",
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{out_base}.png", dpi=600, bbox_inches="tight")
    fig.savefig(out_dir / f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)


def combine_figure2_panels(figures_dir: Path) -> None:
    """Create the released 2x2 Figure 2 canvas from the four panel PNGs."""
    names = [
        "figure2A_calibration_bar_charls_lcp_v3_600dpi.png",
        "figure2B_calibration_bar_cfps_lcp_v3_600dpi.png",
        "figure2C_calibration_curve_charls_lcp_v3_600dpi.png",
        "figure2D_calibration_curve_cfps_lcp_v3_600dpi.png",
    ]
    panels = [Image.open(figures_dir / name).convert("RGB") for name in names]
    width, height = panels[0].size
    canvas = Image.new("RGB", (width * 2 + 90, height * 2 - 40), "white")
    canvas.paste(panels[0], (0, 0))
    canvas.paste(panels[1], (width + 90, 0))
    canvas.paste(panels[2], (0, height - 40))
    canvas.paste(panels[3], (width + 90, height - 40))
    output = figures_dir / "figure2_ABCD_calibration_lcp_v3_600dpi"
    canvas.save(f"{output}.png")
    canvas.save(f"{output}.pdf", resolution=600.0)


def main(config_path: str) -> None:
    config = load_config(Path(config_path))
    project_dir = Path(config["project_dir"])
    figures_dir = Path(config.get("figures_dir", project_dir / "figures"))
    output_dir = Path(config["output_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(config["model_path"])
    variables = config["input_variables"]
    target = config.get("target_column", TARGET_DEFAULT)
    n_bootstrap = config.get("bootstrap_repetitions", 2000)
    seed_base = config.get("bootstrap_random_state", 42)

    deciles = pd.read_csv(output_dir / "decile_calibration_table.csv")
    metrics = pd.read_csv(output_dir / "locked_model_metrics.csv").set_index("dataset")
    for dataset, title, panel, out_base, axis_max in [
        (
            "charls_heldout",
            "Calibration of survey-defined LCP prediction in CHARLS",
            "A",
            "figure2A_calibration_bar_charls_lcp_v3_600dpi",
            0.45,
        ),
        (
            "cfps_external_main",
            "Calibration of survey-defined LCP prediction in CFPS",
            "B",
            "figure2B_calibration_bar_cfps_lcp_v3_600dpi",
            0.37,
        ),
    ]:
        plot_decile_bar_panel(
            deciles.loc[deciles["dataset"] == dataset].sort_values("risk_decile"),
            metrics.loc[dataset],
            title,
            panel,
            out_base,
            figures_dir,
            axis_max,
        )

    rows = []
    for name, path, title, panel, out_base, seed in [
        (
            "charls_heldout",
            Path(config["datasets"]["charls_heldout"]),
            "Calibration curve of survey-defined LCP prediction in CHARLS",
            "C",
            "figure2C_calibration_curve_charls_lcp_v3_600dpi",
            seed_base,
        ),
        (
            "cfps_external",
            Path(config["datasets"]["cfps_external_main"]),
            "Calibration curve of survey-defined LCP prediction in CFPS",
            "D",
            "figure2D_calibration_curve_cfps_lcp_v3_600dpi",
            seed_base + 1,
        ),
    ]:
        y_true, pred = predict_dataset(model, path, variables, target)
        curve, original_bins = fixed_decile_curve(y_true, pred)
        ci_low, ci_high = bootstrap_decile_ci(
            y_true,
            original_bins,
            n_bootstrap=n_bootstrap,
            random_state=seed,
        )
        plot_panel(curve, ci_low, ci_high, title, panel, out_base, figures_dir)
        curve["dataset"] = name
        curve["observed_event_rate_ci_low"] = ci_low
        curve["observed_event_rate_ci_high"] = ci_high
        rows.append(curve)

    pd.concat(rows, ignore_index=True).to_csv(
        output_dir / "figure2_cd_bootstrap_ci_decile_values.csv", index=False
    )
    combine_figure2_panels(figures_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/final_model_config.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
