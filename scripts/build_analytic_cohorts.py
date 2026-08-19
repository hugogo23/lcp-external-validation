#!/usr/bin/env python3
"""Build one-row-per-participant CHARLS and CFPS analytic cohorts from Stata data.

Raw CHARLS and CFPS data are restricted and are not distributed with this
repository. This script implements the cohort-construction rules used for the
survey-defined low cognitive performance (LCP) prediction workflow and writes
restricted, row-level CSV outputs to a user-selected local directory.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ID_COLUMN = "ID"
TIME_COLUMN = "time"
COGNITION_COLUMN = "cognition"
FUTURE_COGNITION_COLUMN = "future_cognition"
OUTCOME_COLUMN = "cog_impair"
HORIZON_YEARS = 2
CHARLS_COGNITION_THRESHOLD = 6.0

LOCKED_PREDICTORS = [
    # Historical input column name for the harmonized 3-category geographic
    # region (East/Central/West), referred to as "geographic region" in the
    # manuscript and supplement.
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

# CFPS value labels are mapped to the exact string coding used by the locked
# CHARLS model artifact. The `province` field is expected to be supplied already
# harmonized as 1.east/2.middle/3.west in the restricted source Stata file.
CFPS_CATEGORY_MAPPINGS = {
    "sinojapanese": {
        "1.sinojapanese": "1.yes",
        "0.noexperience": "0.no",
    },
    "chronic": {
        "1.no chronic diseases": "1.no",
        "0.had chronic diseases": "0.yes",
    },
    "shlt": {
        "1.unhealthy": "1.poor",
        "2.fair": "2.fair",
        "3.healthy": "3.good",
    },
    "rural": {
        "乡村": "0.rural",
        "城镇": "1.urban",
    },
    "ragender": {
        "男": "1.male",
        "女": "0.female",
    },
    "smoke": {
        "1.current smoke": "1.current smoking",
        "2.ever smoke": "2.ever smoking",
        "3.never smoke": "3.never smoking",
    },
    "raeduc_c": {
        "文盲/半文盲": "1.illiterate",
        "小学": "2.primary school",
        "初中": "3.secondary school and above",
    },
    "civilwar": {
        "0.noexperience": "0.no",
        "1.civilwar": "1.yes",
    },
    "job": {
        "0.agricultural": "0.agriculture",
        "1.non-agricultural": "1.non-agriculture",
    },
    "hukou": {
        "0.agricultural": "0.agriculture",
        "1.non-agricultural": "1.non-agriculture",
    },
    "fedu": {
        0: "0.illiterate",
        1: "1.literate",
        "0": "0.illiterate",
        "1": "1.literate",
        "0.illiterate": "0.illiterate",
        "1.literate": "1.literate",
        "文盲/半文盲": "0.illiterate",
        "文盲": "0.illiterate",
        "半文盲": "0.illiterate",
        "识字": "1.literate",
        "有读书": "1.literate",
    },
}


def require_columns(data: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(data.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def coerce_survey_time(values: pd.Series) -> pd.Series:
    """Convert CHARLS/CFPS wave labels to integer survey-time values."""
    cleaned = values.astype(str).str.strip()
    numeric = pd.to_numeric(cleaned, errors="coerce")
    if numeric.isna().any() or not np.allclose(numeric.dropna(), np.round(numeric.dropna())):
        raise ValueError("Survey time must contain integer-coded wave values.")
    return numeric.astype("int64")


def clean_longitudinal_data(data: pd.DataFrame, require_age: bool) -> tuple[pd.DataFrame, dict]:
    """Apply the minimal pre-cohort checks needed for the analytic records."""
    required = [ID_COLUMN, TIME_COLUMN, COGNITION_COLUMN] + LOCKED_PREDICTORS
    require_columns(data, required, "Input data")

    report = {"input_rows": int(len(data)), "input_participants": int(data[ID_COLUMN].nunique())}
    subset = [ID_COLUMN, TIME_COLUMN, COGNITION_COLUMN]
    if require_age:
        subset.append("age")
    cleaned = data.dropna(subset=subset).copy()
    report["rows_removed_missing_required_fields"] = int(len(data) - len(cleaned))

    cleaned[TIME_COLUMN] = coerce_survey_time(cleaned[TIME_COLUMN])
    before_deduplication = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=[ID_COLUMN, TIME_COLUMN], keep="first").copy()
    report["duplicate_id_time_rows_removed"] = int(before_deduplication - len(cleaned))
    cleaned = cleaned.sort_values([ID_COLUMN, TIME_COLUMN], kind="stable").reset_index(drop=True)
    report["cleaned_rows"] = int(len(cleaned))
    report["cleaned_participants"] = int(cleaned[ID_COLUMN].nunique())
    return cleaned, report


def build_one_interval_cohort(
    data: pd.DataFrame,
    is_lcp: Callable[[float], bool],
    control_selection_seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Select one exact two-year baseline-to-outcome interval per participant.

    For participants with an observed LCP onset, records after the first onset
    are not considered. The final eligible exact two-year interval is retained.
    For participants without observed LCP, one eligible non-event interval is
    selected with a fixed random seed. Every returned row contains predictors
    from the baseline wave and the cognition score from the outcome wave.
    """
    rng = random.Random(control_selection_seed)
    records: list[pd.Series] = []
    report = {
        "participants_considered": int(data[ID_COLUMN].nunique()),
        "participants_without_eligible_interval": 0,
        "participants_with_baseline_lcp_excluded": 0,
        "selected_event_intervals": 0,
        "selected_non_event_intervals": 0,
    }

    for _, person in data.groupby(ID_COLUMN, sort=False):
        person = person.sort_values(TIME_COLUMN, kind="stable")
        times = person[TIME_COLUMN].tolist()
        scores = person[COGNITION_COLUMN].tolist()

        onset_index = next((i for i, score in enumerate(scores) if is_lcp(score)), None)
        if onset_index == 0:
            report["participants_with_baseline_lcp_excluded"] += 1
            continue

        valid_end = onset_index + 1 if onset_index is not None else len(times)
        valid_times = times[:valid_end]
        eligible_pairs = [
            (start, end)
            for start, end in zip(valid_times[:-1], valid_times[1:])
            if end - start == HORIZON_YEARS
        ]
        if not eligible_pairs:
            report["participants_without_eligible_interval"] += 1
            continue

        start_time, end_time = eligible_pairs[-1] if onset_index is not None else rng.choice(eligible_pairs)
        baseline = person.loc[person[TIME_COLUMN] == start_time].iloc[0].copy()
        future_cognition = person.loc[person[TIME_COLUMN] == end_time, COGNITION_COLUMN].iloc[0]
        baseline[FUTURE_COGNITION_COLUMN] = future_cognition
        baseline["outcome_time"] = end_time
        baseline[OUTCOME_COLUMN] = bool(is_lcp(future_cognition))
        records.append(baseline)

        if baseline[OUTCOME_COLUMN]:
            report["selected_event_intervals"] += 1
        else:
            report["selected_non_event_intervals"] += 1

    if not records:
        raise ValueError("No eligible exact two-year intervals were constructed.")

    cohort = pd.DataFrame(records).reset_index(drop=True)
    report["analytic_rows"] = int(len(cohort))
    report["analytic_participants"] = int(cohort[ID_COLUMN].nunique())
    report["analytic_events"] = int(cohort[OUTCOME_COLUMN].sum())
    report["analytic_event_rate"] = float(cohort[OUTCOME_COLUMN].mean())
    return cohort, report


def map_cfps_categories(data: pd.DataFrame) -> pd.DataFrame:
    """Map CFPS category labels to the CHARLS coding used by the locked model."""
    mapped = data.copy()
    for column, mapping in CFPS_CATEGORY_MAPPINGS.items():
        original = mapped[column]
        transformed = original.map(mapping)
        unknown = original.notna() & transformed.isna()
        if unknown.any():
            values = sorted(map(str, original.loc[unknown].drop_duplicates().tolist()))
            raise ValueError(
                f"Unmapped non-missing CFPS values for {column}: {', '.join(values)}. "
                "Update CFPS_CATEGORY_MAPPINGS before continuing."
            )
        mapped[column] = transformed.where(original.notna(), pd.NA)
    return mapped


def write_csv(data: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination, index=False)


def write_report(report: dict, destination: Path | None) -> None:
    if destination is None:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def build_charls(args: argparse.Namespace) -> None:
    raw = pd.read_stata(args.input)
    cleaned, report = clean_longitudinal_data(raw, require_age=True)
    n_before_age_filter = len(cleaned)
    cleaned = cleaned[cleaned["age"] >= args.min_age].copy()
    report["rows_removed_age_lt_min_age"] = int(n_before_age_filter - len(cleaned))
    report["participants_after_age_filter"] = int(cleaned[ID_COLUMN].nunique())
    report["min_age"] = args.min_age
    cohort, cohort_report = build_one_interval_cohort(
        cleaned,
        is_lcp=lambda score: score <= args.cognition_threshold,
        control_selection_seed=args.control_selection_seed,
    )
    report.update(cohort_report)
    report.update(
        {
            "cohort": "CHARLS",
            "cognition_threshold": args.cognition_threshold,
            "control_selection_seed": args.control_selection_seed,
            "horizon_years": HORIZON_YEARS,
        }
    )
    write_csv(cohort, args.cohort_output)

    if args.train_output or args.heldout_output:
        if not args.train_output or not args.heldout_output:
            raise ValueError("Provide both --train-output and --heldout-output, or neither.")
        train_ids, heldout_ids = train_test_split(
            cohort[ID_COLUMN],
            test_size=args.heldout_fraction,
            stratify=cohort[OUTCOME_COLUMN],
            random_state=args.split_random_state,
        )
        write_csv(cohort[cohort[ID_COLUMN].isin(train_ids)].copy(), args.train_output)
        write_csv(cohort[cohort[ID_COLUMN].isin(heldout_ids)].copy(), args.heldout_output)
        report["training_rows"] = int(len(train_ids))
        report["heldout_rows"] = int(len(heldout_ids))
        report["split_random_state"] = args.split_random_state

    write_report(report, args.report_output)
    print(json.dumps(report, indent=2))


def build_cfps(args: argparse.Namespace) -> None:
    raw = pd.read_stata(args.input)
    cleaned, base_report = clean_longitudinal_data(raw, require_age=True)
    n_before_age_filter = len(cleaned)
    cleaned = cleaned[cleaned["age"] >= args.min_age].copy()
    base_report["rows_removed_age_lt_min_age"] = int(n_before_age_filter - len(cleaned))
    base_report["participants_after_age_filter"] = int(cleaned[ID_COLUMN].nunique())
    base_report["min_age"] = args.min_age
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    all_reports = []

    for percentile in args.percentiles:
        # Lower-tail cutoffs are calculated on the cleaned, age-eligible CFPS
        # longitudinal person-wave cognition distribution before one-interval-
        # per-participant selection, matching the manuscript definition.
        cutoff = float(np.percentile(cleaned[COGNITION_COLUMN].astype(float), percentile))
        cohort, report = build_one_interval_cohort(
            cleaned,
            is_lcp=lambda score, current_cutoff=cutoff: score <= current_cutoff,
            control_selection_seed=args.control_selection_seed,
        )
        report.update(base_report)
        report.update(
            {
                "cohort": "CFPS",
                "percentile": percentile,
                "cognition_cutoff": cutoff,
                "control_selection_seed": args.control_selection_seed,
                "horizon_years": HORIZON_YEARS,
            }
        )
        raw_output = output_dir / f"cfps_retrospective_2years_th{percentile}.csv"
        write_csv(cohort, raw_output)

        if percentile == args.main_percentile:
            processed = map_cfps_categories(cohort)
            write_csv(processed, output_dir / "cfps_retrospective_2years_process.csv")

        all_reports.append(report)

    write_report({"CFPS": all_reports}, args.report_output)
    print(json.dumps({"CFPS": all_reports}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="cohort", required=True)

    charls = subparsers.add_parser("charls", help="Build CHARLS cohort and optional ID-level split.")
    charls.add_argument("--input", type=Path, required=True, help="Restricted CHARLS .dta file.")
    charls.add_argument("--cohort-output", type=Path, required=True)
    charls.add_argument("--train-output", type=Path)
    charls.add_argument("--heldout-output", type=Path)
    charls.add_argument("--report-output", type=Path)
    charls.add_argument("--cognition-threshold", type=float, default=CHARLS_COGNITION_THRESHOLD)
    charls.add_argument("--control-selection-seed", type=int, default=42)
    charls.add_argument("--split-random-state", type=int, default=42)
    charls.add_argument("--heldout-fraction", type=float, default=0.20)
    charls.add_argument("--min-age", type=float, default=45.0, help="Minimum age at the baseline predictor wave.")
    charls.set_defaults(handler=build_charls)

    cfps = subparsers.add_parser("cfps", help="Build CFPS cohorts for lower-tail outcome definitions.")
    cfps.add_argument("--input", type=Path, required=True, help="Restricted CFPS .dta file.")
    cfps.add_argument("--output-dir", type=Path, required=True)
    cfps.add_argument("--report-output", type=Path)
    cfps.add_argument("--percentiles", type=int, nargs="+", default=[10, 15, 20])
    cfps.add_argument("--main-percentile", type=int, default=15)
    cfps.add_argument("--control-selection-seed", type=int, default=42)
    cfps.add_argument("--min-age", type=float, default=45.0, help="Minimum age at the baseline predictor wave.")
    cfps.set_defaults(handler=build_cfps)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.handler(arguments)
