# Data Inputs

Raw CHARLS and CFPS datasets are not redistributed in this folder.

## Cohort Construction From Restricted Stata Files

`scripts/build_analytic_cohorts.py` builds the restricted row-level analytic
CSV files from the study-specific Stata data. It implements the one-row-per-
participant, exact two-year predictor-to-outcome interval rule used in this
repository. It writes only local CSV files and JSON run summaries; do not
commit those outputs to a public repository.

Example commands, after obtaining the datasets through the official portals:

```bash
python scripts/build_analytic_cohorts.py charls \
  --input /path/to/charls0217.dta \
  --cognition-threshold 6.0 \
  --cohort-output data/intermediate/chals_2year_task.csv \
  --train-output data/intermediate/chals_2year_train.csv \
  --heldout-output data/intermediate/chals_2year_test.csv \
  --report-output data/intermediate/charls_cohort_report.json

python scripts/build_analytic_cohorts.py cfps \
  --input /path/to/cfps-v3-0809.dta \
  --output-dir data/intermediate \
  --report-output data/intermediate/cfps_cohort_report.json
```

The CFPS command restricts records to participants aged >=45 years at the
baseline predictor wave and writes 10th-, 15th- and 20th-percentile outcome-
definition datasets. Lower-tail cutoffs are calculated on the cleaned age-
eligible CFPS longitudinal person-wave cognition distribution before one interval
per participant is selected. The 15th-percentile cohort is additionally mapped to
the CHARLS category coding required by the locked model and written as
`cfps_retrospective_2years_process.csv`.

## Expected Processed Input Files

Place the processed analysis files in this folder before rerunning the scripts:

- `chals_2year_train.csv`
- `chals_2year_test.csv`
- `cfps_retrospective_2years_process.csv`
- `cfps_retrospective_2years_th10.csv`
- `cfps_retrospective_2years_th15.csv`
- `cfps_retrospective_2years_th20.csv`

These filenames correspond to the paths in `config/final_model_config.json`.
Users may either copy locally generated files from `data/intermediate/` to this
directory or modify the paths in the configuration file.

## Official Data Sources

- CHARLS: http://charls.pku.edu.cn
- CFPS: https://cfpsdata.pku.edu.cn/#/home
