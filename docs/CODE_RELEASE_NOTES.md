# Code Release Notes

This repository-style folder is prepared for public release of the validation code, locked model specification, aggregate result tables, and selected figure-generation scripts.

Raw CHARLS and CFPS data are not redistributed. Users must obtain data from the official cohort portals and place processed analysis files according to `data/README.md` and `config/final_model_config.json`.

Do not commit row-level derived outputs with participant identifiers, such as `results/locked_model_predictions.csv`, unless the applicable data-use agreements permit redistribution.

The current release does not include a Dockerfile, survey-weighted analyses, multiple-imputation sensitivity scripts, or complete SHAP figure-generation scripts.
