# Survey-defined LCP External Validation

This repository contains the reproducible validation workflow for the manuscript:

**External Validation of a Machine Learning Prediction Model for Low Cognitive Performance**

## Scope

This repository-style folder supports model comparison in CHARLS and external validation of the final 15-predictor gradient boosting classifier.

The code does not redistribute raw CHARLS or CFPS data. Users must obtain the original datasets from the official study websites and place processed files at the expected paths or edit `config/final_model_config.json`.

## What Is Included

- `config/final_model_config.json`: locked model, predictors, paths, outcome definitions, and reporting settings.
- `config/model_comparison_hyperparameters.json`: candidate-model search spaces and selected hyperparameters.
- `scripts/model_comparison.py`: CHARLS-based model comparison and CFPS transport check for the candidate algorithms reported in Table 2.
- `scripts/hyperparameter_tuning.py`: optional CHARLS-only hyperparameter tuning workflow for the candidate algorithms.
- `scripts/locked_model_validation.py`: primary validation workflow for the final GBC in CHARLS held-out and CFPS external validation.
- `scripts/outcome_sensitivity.py`: CFPS outcome-definition sensitivity analyses.
- `scripts/dca.py`: decision-curve analysis helpers.
- `scripts/build_master_outputs.py`: creates manuscript-ready master tables and figures from the generated result CSV files.
- `data/README.md`: data availability and expected input files.
- `models/`: location for the locked model artifact.
- `results/`: generated aggregate result tables.

## Public Release Boundaries

This folder is intended to support public release of the validation code, model specification, aggregate result tables, and selected figure-generation workflow. Raw CHARLS/CFPS data are not redistributed. Row-level derived outputs that include participant identifiers, such as `results/locked_model_predictions.csv`, should not be committed to a public repository unless the applicable data-use agreements permit redistribution of such derived records.

## Important Definitions

### Survey-Defined Low Cognitive Performance

The outcome should be described as **survey-defined low cognitive performance**, not clinically diagnosed cognitive impairment or dementia.

### CFPS Main Outcome

The main CFPS target column is `cog_impair`, generated using a lower-tail cognitive-score threshold. The planned main definition is bottom 15% of the CFPS cognition score distribution, with bottom 10% and bottom 20% used as sensitivity analyses.

### Model Prediction Probability Cutoff

The fitted GBC model outputs predicted probabilities. A probability cutoff, currently `0.1475`, is used only for secondary threshold-based classification metrics such as sensitivity, specificity, PPV, and NPV. It is not a GBC hyperparameter and should not be confused with the cognitive-outcome definition threshold.

## Recommended Run Order

```bash
python scripts/model_comparison.py --config config/final_model_config.json
python scripts/locked_model_validation.py --config config/final_model_config.json
python scripts/outcome_sensitivity.py --config config/final_model_config.json
python scripts/build_master_outputs.py --project-dir .
```

The scripts write CSV/JSON outputs to `results/`. The repository also includes aggregate result tables and manuscript-ready figure files generated for the revision.

To rerun the optional hyperparameter search on the CHARLS training partition:

```bash
python scripts/hyperparameter_tuning.py --config config/final_model_config.json --n-iter 20
```

`LightGBM` and `CatBoost` are optional runtime dependencies for reproducing the full model-comparison table. If either package is not installed, `scripts/model_comparison.py` will skip that model and continue with the remaining algorithms.

## Current Reproducibility Scope

The current public-release scope covers the Python validation workflow, selected hyperparameter grids, locked model artifact, aggregate result tables, and scripts used to generate selected validation figures. It does not currently include a Dockerfile, survey-weighted analyses, multiple-imputation sensitivity scripts, or complete SHAP figure-generation scripts. These should not be claimed in the manuscript or response letter unless they are added and verified before submission.

## Data Sharing Note

Raw CHARLS and CFPS data are not included because the authors do not own redistribution rights. The manuscript and README should direct readers to the official CHARLS and CFPS data-access portals.
