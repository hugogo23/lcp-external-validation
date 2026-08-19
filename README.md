# Survey-defined LCP External Validation

This repository contains the reproducible validation workflow for the manuscript:

**External Validation of a Machine Learning Prediction Model for Low Cognitive Performance**

## Scope

This repository supports model comparison in CHARLS and external validation of the final 15-predictor gradient boosting classifier.

The code does not redistribute raw CHARLS or CFPS data. Users must obtain the original datasets from the official study websites and place processed files at the expected paths or edit `config/final_model_config.json`.

## What Is Included

- `config/final_model_config.json`: locked model, predictors, paths, outcome definitions, and reporting settings.
- `config/model_comparison_hyperparameters.json`: candidate-model search spaces and selected hyperparameters.
- `scripts/model_comparison.py`: CHARLS-based model comparison and CFPS transport check for the candidate algorithms reported in the supplementary model-comparison table.
- `scripts/hyperparameter_tuning.py`: optional CHARLS-only hyperparameter tuning workflow for the candidate algorithms.
- `scripts/locked_model_validation.py`: primary validation workflow for the final GBC in CHARLS held-out and CFPS external validation.
- `scripts/build_analytic_cohorts.py`: constructs restricted one-row-per-participant CHARLS and CFPS analytic CSV files from the original Stata data, including the deterministic CFPS-to-CHARLS category mapping.
- `scripts/outcome_sensitivity.py`: CFPS outcome-definition sensitivity analyses.
- `scripts/dca.py`: decision-curve analysis helpers.
- `scripts/build_master_outputs.py`: creates manuscript-ready master tables and figures from the generated result CSV files.
- `scripts/subgroup_performance.py`: calculates subgroup-specific discrimination, calibration and threshold metrics.
- `scripts/subgroup_age_education_analysis.py`: calculates age- and education-only benchmark analyses.
- `scripts/missingness_table.py`: generates the item-missingness summary table.
- `scripts/rebuild_figure1_shap_sex.py`: selected Figure 1 SHAP regeneration workflow.
- `scripts/rebuild_figure2_calibration_curves_with_ci.py`: Figure 2 calibration-curve regeneration workflow with pointwise bootstrap confidence bands for panels C and D.
- `scripts/rebuild_figure4_outcome_sensitivity.py`: Figure 4 regeneration workflow.
- `data/README.md`: data availability and expected input files.
- `models/`: location for the locked model artifact.
- `results/`: generated aggregate result tables.
- `figures/`: manuscript-ready high-resolution figure files.
- `Dockerfile`: container definition for reproducing the Python analysis environment.
- `LICENSE`: MIT license for the code in this repository.

## Public Release Boundaries

This folder is intended to support public release of the validation code, model specification, aggregate result tables, and selected figure-generation workflow. Raw CHARLS/CFPS data are not redistributed. Row-level derived outputs that include participant identifiers, such as `results/locked_model_predictions.csv`, should not be committed to a public repository unless the applicable data-use agreements permit redistribution of such derived records.

## Important Definitions

### Survey-Defined Low Cognitive Performance

The outcome should be described as **survey-defined low cognitive performance**, not clinically diagnosed cognitive impairment or dementia.

### CFPS Main Outcome

The main CFPS target column is `cog_impair`, generated using a lower-tail cognitive-score threshold. The primary definition is the bottom 15% of the CFPS cognition score distribution, with bottom 10% and bottom 20% used as sensitivity analyses.

### Model Prediction Probability Cutoff

The fitted GBC model outputs predicted probabilities. A probability cutoff, currently `0.1475`, is used only for secondary threshold-based classification metrics such as sensitivity, specificity, PPV, and NPV. It is not a GBC hyperparameter and should not be confused with the cognitive-outcome definition threshold.

## Recommended Run Order

```bash
# 1. Optional: inspect or run cohort construction from raw Stata files after configuring paths
python scripts/build_analytic_cohorts.py charls --help
python scripts/build_analytic_cohorts.py cfps --help

# 2. Run the locked-model and sensitivity analyses from the analysis-ready datasets
python scripts/model_comparison.py --config config/final_model_config.json
python scripts/locked_model_validation.py --config config/final_model_config.json
python scripts/outcome_sensitivity.py --config config/final_model_config.json
python scripts/subgroup_performance.py --config config/final_model_config.json
python scripts/subgroup_age_education_analysis.py --config config/final_model_config.json
python scripts/missingness_table.py --config config/final_model_config.json

# 3. Assemble manuscript-ready aggregate tables and selected figures
python scripts/build_master_outputs.py --project-dir .
python scripts/rebuild_figure1_shap_sex.py --config config/final_model_config.json
python scripts/rebuild_figure2_calibration_curves_with_ci.py --config config/final_model_config.json
python scripts/rebuild_figure4_outcome_sensitivity.py --config config/final_model_config.json
```

The scripts write CSV/JSON outputs to `results/` and selected publication-ready files to `figures/`. The repository also includes aggregate result tables and manuscript-ready figure files generated for the revision.

To rerun the optional hyperparameter search on the CHARLS training partition:

```bash
python scripts/hyperparameter_tuning.py --config config/final_model_config.json --n-iter 20
```

To reproduce the Python analysis environment with Docker:

```bash
docker build -t survey-lcp-validation .
docker run --rm -it -v "$PWD":/workspace survey-lcp-validation
```

`LightGBM` and `CatBoost` are optional runtime dependencies for reproducing the full model-comparison table. If either package is not installed, `scripts/model_comparison.py` will skip that model and continue with the remaining algorithms.

## Current Reproducibility Scope

The current public-release scope covers the cohort-construction and Python validation workflow, selected hyperparameter grids, locked model artifact, aggregate result tables, selected figure-generation scripts, and a Dockerfile for reproducing the Python analysis environment. It does not redistribute raw CHARLS/CFPS data or row-level derived prediction outputs. Survey-weighted and multiple-imputation sensitivity analyses are reported as aggregate results in the manuscript/supplementary materials but are not included as full public reanalysis pipelines in this release unless separately added before submission.

## Data Sharing Note

Raw CHARLS and CFPS data are not included because the authors do not own redistribution rights. The manuscript and README should direct readers to the official CHARLS and CFPS data-access portals.

## License

Code in this repository is released under the MIT License. Raw CHARLS and CFPS data remain subject to their original data-use agreements and are not covered by this repository license.
