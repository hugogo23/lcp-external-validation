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
import shap
from PIL import Image, ImageDraw, ImageFont


INPUT_VARIABLES = [
    "province",
    "shlt",
    "rural",
    "sinojapanese",
    "chronic",
    "ragender",
    "smoke",
    "children",
    "age",
    "hukou",
    "raeduc_c",
    "civilwar",
    "job",
    "fedu",
    "sleep",
]

NAME_MAP = {
    "raeduc_c": "educational attainment",
    "age": "age",
    "sleep": "sleep duration",
    "ragender": "sex",
    "hukou": "hukou type",
    "rural": "residence",
    "children": "number of living children",
    "sinojapanese": "born in the Second Sino-Japanese era",
    "civilwar": "born in the Civil War era",
    "shlt": "self-rated health",
    "smoke": "smoking status",
    "fedu": "father's education",
    "chronic": "chronic diseases",
    "province": "region",
    "job": "occupation",
}


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


def build_panel(model, data_path: Path, title: str, out_png: Path, out_pdf: Path) -> None:
    data = pd.read_csv(data_path)
    x = data[INPUT_VARIABLES]

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    stripped_feature_names = [name.split("__", 1)[1] for name in feature_names]
    display_names = [NAME_MAP.get(name, name) for name in stripped_feature_names]

    transformed = model.named_steps["preprocessor"].transform(x)
    explainer = shap.TreeExplainer(
        model.named_steps["classifier"],
        feature_names=feature_names,
    )
    shap_values = explainer.shap_values(transformed)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(18, 5.5), dpi=300)
    ax = plt.gca()
    shap.summary_plot(
        shap_values,
        transformed,
        feature_names=display_names,
        plot_type="dot",
        show=False,
    )
    ax.set_ylabel("Top predictors", fontsize=16)
    ax.set_xlabel(f"SHAP values of {title}", fontsize=12)
    ylabels = [tick.get_text() for tick in ax.get_yticklabels()]
    ax.set_yticklabels(ylabels, fontsize=12, color="black")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def combine_panels(left: Path, right: Path, out_path: Path) -> None:
    left_img = Image.open(left).convert("RGB")
    right_img = Image.open(right).convert("RGB")
    gap = 120
    height = max(left_img.height, right_img.height)
    width = left_img.width + right_img.width + gap
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(left_img, (0, 0))
    canvas.paste(right_img, (left_img.width + gap, 0))

    draw = ImageDraw.Draw(canvas)
    font_candidates = [
        Path("/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ]
    label_font = None
    for font_path in font_candidates:
        try:
            label_font = ImageFont.truetype(str(font_path), 130)
            break
        except OSError:
            continue
    if label_font is None:
        label_font = ImageFont.load_default()

    draw.text((95, 75), "A", fill="black", font=label_font)
    draw.text((left_img.width + gap + 95, 75), "B", fill="black", font=label_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, dpi=(600, 600))


def main(config_path: str) -> None:
    config = load_config(Path(config_path))
    project_dir = Path(config["project_dir"])
    figures_dir = project_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(config["model_path"])

    fig1a_png = figures_dir / "figure1A_shap_charls_lcp_600dpi.png"
    fig1b_png = figures_dir / "figure1B_shap_cfps_lcp_600dpi.png"
    fig1a_pdf = figures_dir / "figure1A_shap_charls_lcp.pdf"
    fig1b_pdf = figures_dir / "figure1B_shap_cfps_lcp.pdf"
    combined = figures_dir / "figure1_shap_combined_lcp_600dpi.png"

    build_panel(
        model,
        Path(config["datasets"]["charls_heldout"]),
        "2-year LCP prediction in CHARLS",
        fig1a_png,
        fig1a_pdf,
    )
    build_panel(
        model,
        Path(config["datasets"]["cfps_external_main"]),
        "2-year LCP prediction in CFPS",
        fig1b_png,
        fig1b_pdf,
    )
    combine_panels(fig1a_png, fig1b_png, combined)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/final_model_config.json")
    return parser.parse_args()


if __name__ == "__main__":
    np.random.seed(42)
    args = parse_args()
    main(args.config)
