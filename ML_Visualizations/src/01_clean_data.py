"""
01_clean_data.py
Clean the raw student burnout/exam-performance data and create dashboard-ready fields.

Works with either:
  - a plain CSV file
  - a ZIP file containing a CSV

Recommended command:
  python src/01_clean_data.py \
    --input /path/to/Burnout_Predictions_Final_1M.csv.zip \
    --out-dir data/processed \
    --dashboard-sample-size 100000

Main outputs:
  data/processed/student_burnout_cleaned.csv.gz
  data/processed/student_burnout_dashboard_sample.csv
  data/processed/cleaning_report.json
  data/processed/data_dictionary.csv
  data/processed/eda_summary_by_risk.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

RANDOM_STATE = 42

RISK_ORDER = ["Low", "Medium", "High"]
RISK_TO_CODE = {"Low": 0, "Medium": 1, "High": 2}
CODE_TO_RISK = {v: k for k, v in RISK_TO_CODE.items()}

RAW_FEATURES = [
    "age",
    "gender",
    "academic_year",
    "study_hours_per_day",
    "exam_pressure",
    "academic_performance",
    "stress_level",
    "anxiety_score",
    "depression_score",
    "sleep_hours",
    "physical_activity",
    "social_support",
    "screen_time",
    "internet_usage",
    "financial_stress",
    "family_expectation",
]

ENGINEERED_FEATURES = [
    "academic_resilience",
    "study_sleep_gap",
    "socio_economic_multiplier",
    "effort_reward_index",
    "total_cognitive_load",
    "anxiety_pressure_impact",
    "social_support_ratio",
    "combined_distress_index",
]

OUTCOME_COLUMNS = [
    "burnout_score",
    "mental_health_index",
    "risk_level",
    "dropout_risk",
    "target",
    "predicted_risk_index",
    "predicted_risk_label",
]

NUMERIC_DOMAIN_LIMITS = {
    "age": (16, 65),
    "academic_year": (1, 8),
    "study_hours_per_day": (0, 16),
    "exam_pressure": (0, 10),
    "academic_performance": (0, 100),
    "stress_level": (0, 10),
    "anxiety_score": (0, 10),
    "depression_score": (0, 10),
    "sleep_hours": (0, 14),
    "physical_activity": (0, 10),
    "social_support": (0, 10),
    "screen_time": (0, 16),
    "internet_usage": (0, 16),
    "financial_stress": (0, 10),
    "family_expectation": (0, 10),
    "burnout_score": (0, 10),
    "mental_health_index": (0, 10),
    "dropout_risk": (0, 10),
}

DATA_DICTIONARY = {
    "student_id": "Synthetic row identifier for joining dashboard tables.",
    "age": "Student age.",
    "gender": "Student gender category after standardization.",
    "academic_year": "Academic year level, treated as an ordered categorical variable.",
    "study_hours_per_day": "Average daily study hours.",
    "exam_pressure": "Self-reported exam pressure score on a 0-10 scale.",
    "academic_performance": "Academic performance score on a 0-100 scale.",
    "stress_level": "Self-reported stress score on a 0-10 scale.",
    "anxiety_score": "Self-reported anxiety score on a 0-10 scale.",
    "depression_score": "Self-reported depression score on a 0-10 scale.",
    "sleep_hours": "Average nightly sleep hours.",
    "physical_activity": "Average physical activity level/hours on the source scale.",
    "social_support": "Social support score on a 0-10 scale.",
    "screen_time": "Average daily screen time hours.",
    "internet_usage": "Average daily internet usage hours.",
    "financial_stress": "Financial stress score on a 0-10 scale.",
    "family_expectation": "Family expectation pressure score on a 0-10 scale.",
    "burnout_score": "Burnout score outcome. Keep for EDA; do not use as model input when predicting risk_level.",
    "mental_health_index": "Mental health index outcome. Keep for EDA; do not use as model input when predicting risk_level.",
    "dropout_risk": "Dropout-risk outcome. Keep for dashboard context; do not use as model input.",
    "risk_level": "Original Low/Medium/High burnout-risk label.",
    "target_risk_code": "Cleaned target encoding: Low=0, Medium=1, High=2.",
    "academic_resilience": "Engineered ratio: academic performance adjusted by pressure and stress.",
    "study_sleep_gap": "Study hours minus sleep hours. Positive values suggest study load is exceeding sleep.",
    "socio_economic_multiplier": "Financial stress multiplied by family expectation pressure.",
    "effort_reward_index": "Study effort divided by academic performance reward.",
    "total_cognitive_load": "Study hours plus screen time.",
    "anxiety_pressure_impact": "Anxiety score multiplied by exam pressure.",
    "social_support_ratio": "Social support divided by family expectation pressure plus one.",
    "combined_distress_index": "Anxiety score plus depression score.",
    "sleep_band": "Dashboard grouping for sleep hours.",
    "study_band": "Dashboard grouping for study hours.",
    "stress_band": "Dashboard grouping for stress level.",
    "behavior_quadrant": "Tableau segment combining sleep adequacy and study intensity.",
    "is_high_risk": "Binary flag where risk_level is High.",
}


def snake_case(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()


def find_csv_inside_zip(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX/")]
        if not csv_names:
            raise FileNotFoundError(f"No CSV file found inside ZIP: {path}")
        # Prefer the largest CSV file when multiple CSVs are present.
        csv_names = sorted(csv_names, key=lambda n: zf.getinfo(n).file_size, reverse=True)
        return csv_names[0]


def read_raw_data(input_path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() == ".zip":
        csv_name = find_csv_inside_zip(input_path)
        with zipfile.ZipFile(input_path) as zf:
            with zf.open(csv_name) as fh:
                df = pd.read_csv(fh, nrows=nrows, low_memory=False)
    else:
        df = pd.read_csv(input_path, nrows=nrows, low_memory=False)

    df.columns = [snake_case(c) for c in df.columns]
    return df


def standardize_gender(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.title()
    s = s.replace({
        "M": "Male",
        "Man": "Male",
        "Male ": "Male",
        "F": "Female",
        "Woman": "Female",
        "Female ": "Female",
        "Non Binary": "Other",
        "Non-Binary": "Other",
        "Prefer Not To Say": "Other",
        "Nan": pd.NA,
        "None": pd.NA,
        "": pd.NA,
    })
    allowed = {"Male", "Female", "Other"}
    s = s.where(s.isin(allowed), "Other")
    return s.fillna("Other")


def standardize_risk_level(df: pd.DataFrame) -> pd.Series:
    if "risk_level" in df.columns:
        s = df["risk_level"].astype("string").str.strip().str.title()
        s = s.replace({"Med": "Medium", "Moderate": "Medium", "Hi": "High"})
        s = s.where(s.isin(RISK_ORDER))
    elif "target" in df.columns:
        # Common version in this project had High=0, Low=1, Medium=2.
        target_map = {0: "High", 1: "Low", 2: "Medium"}
        s = pd.to_numeric(df["target"], errors="coerce").map(target_map).astype("string")
    elif "burnout_score" in df.columns:
        b = pd.to_numeric(df["burnout_score"], errors="coerce")
        s = pd.cut(
            b,
            bins=[-np.inf, 3.5, 6.5, np.inf],
            labels=["Low", "Medium", "High"],
            ordered=True,
        ).astype("string")
    else:
        raise ValueError(
            "No target column found. Need risk_level, target, or burnout_score to build the ML target."
        )

    # Fill missing target labels from burnout score if possible; otherwise use mode.
    if s.isna().any() and "burnout_score" in df.columns:
        b = pd.to_numeric(df["burnout_score"], errors="coerce")
        fallback = pd.cut(
            b,
            bins=[-np.inf, 3.5, 6.5, np.inf],
            labels=["Low", "Medium", "High"],
            ordered=True,
        ).astype("string")
        s = s.fillna(fallback)

    if s.isna().any():
        mode = s.dropna().mode()
        fill_value = mode.iloc[0] if not mode.empty else "Low"
        s = s.fillna(fill_value)
    return s


def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clip_domain_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    clip_report: Dict[str, Dict[str, float]] = {}
    for col, (lo, hi) in NUMERIC_DOMAIN_LIMITS.items():
        if col not in df.columns:
            continue
        before_low = int((df[col] < lo).sum(skipna=True))
        before_high = int((df[col] > hi).sum(skipna=True))
        df[col] = df[col].clip(lower=lo, upper=hi)
        clip_report[col] = {"below_min_clipped": before_low, "above_max_clipped": before_high, "min": lo, "max": hi}
    return df, clip_report


def impute_missing(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    missing_before = df.isna().sum().astype(int).to_dict()
    for col in df.select_dtypes(include=["number"]).columns:
        if df[col].isna().any():
            median = df[col].median()
            if math.isnan(median):
                median = 0.0
            df[col] = df[col].fillna(median)
    for col in df.select_dtypes(exclude=["number"]).columns:
        if df[col].isna().any():
            mode = df[col].dropna().mode()
            fill = mode.iloc[0] if not mode.empty else "Unknown"
            df[col] = df[col].fillna(fill)
    return df, {k: v for k, v in missing_before.items() if v > 0}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    # These formulas are intentionally transparent so you can explain every feature in presentation/Q&A.
    eps = 1e-6
    required = [
        "academic_performance",
        "exam_pressure",
        "stress_level",
        "study_hours_per_day",
        "sleep_hours",
        "financial_stress",
        "family_expectation",
        "screen_time",
        "anxiety_score",
        "depression_score",
        "social_support",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot engineer burnout features. Missing columns: {missing}")

    df["academic_resilience"] = df["academic_performance"] / (100.0 * (1.0 + df["exam_pressure"] + df["stress_level"] + eps))
    df["study_sleep_gap"] = df["study_hours_per_day"] - df["sleep_hours"]
    df["socio_economic_multiplier"] = df["financial_stress"] * df["family_expectation"]
    df["effort_reward_index"] = df["study_hours_per_day"] / (1.0 + df["academic_performance"] + eps)
    df["total_cognitive_load"] = df["study_hours_per_day"] + df["screen_time"]
    df["anxiety_pressure_impact"] = df["anxiety_score"] * df["exam_pressure"]
    df["social_support_ratio"] = df["social_support"] / (1.0 + df["family_expectation"] + eps)
    df["combined_distress_index"] = df["anxiety_score"] + df["depression_score"]

    return df


def add_dashboard_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "student_id", np.arange(1, len(df) + 1, dtype=np.int64))

    df["risk_level"] = standardize_risk_level(df)
    df["target_risk_code"] = df["risk_level"].map(RISK_TO_CODE).astype("int8")
    df["is_high_risk"] = (df["risk_level"] == "High").astype("int8")

    df["sleep_band"] = pd.cut(
        df["sleep_hours"],
        bins=[-np.inf, 5, 6, 7, 8, np.inf],
        labels=["<5h", "5-6h", "6-7h", "7-8h", "8h+"],
        ordered=True,
    ).astype("string")
    df["study_band"] = pd.cut(
        df["study_hours_per_day"],
        bins=[-np.inf, 2, 4, 6, 8, np.inf],
        labels=["<2h", "2-4h", "4-6h", "6-8h", "8h+"],
        ordered=True,
    ).astype("string")
    df["stress_band"] = pd.cut(
        df["stress_level"],
        bins=[-np.inf, 3, 6, 8, np.inf],
        labels=["Low stress", "Moderate stress", "High stress", "Severe stress"],
        ordered=True,
    ).astype("string")
    df["screen_band"] = pd.cut(
        df["screen_time"],
        bins=[-np.inf, 3, 5, 7, np.inf],
        labels=["Low screen", "Medium screen", "High screen", "Very high screen"],
        ordered=True,
    ).astype("string")

    high_study = df["study_hours_per_day"] >= df["study_hours_per_day"].median()
    enough_sleep = df["sleep_hours"] >= 7.0
    df["behavior_quadrant"] = np.select(
        [high_study & enough_sleep, high_study & ~enough_sleep, ~high_study & enough_sleep, ~high_study & ~enough_sleep],
        ["High study / adequate sleep", "High study / low sleep", "Lower study / adequate sleep", "Lower study / low sleep"],
        default="Unclassified",
    )

    df["sleep_deficit_hours"] = np.maximum(0.0, 7.0 - df["sleep_hours"])
    df["overstudy_flag"] = (df["study_hours_per_day"] > 8.0).astype("int8")
    df["low_sleep_flag"] = (df["sleep_hours"] < 6.0).astype("int8")
    df["high_pressure_flag"] = (df["exam_pressure"] >= 8.0).astype("int8")
    return df


def build_summary_tables(df: pd.DataFrame, out_dir: Path) -> None:
    risk_summary = (
        df.groupby("risk_level", observed=False)
        .agg(
            students=("student_id", "count"),
            avg_burnout_score=("burnout_score", "mean") if "burnout_score" in df.columns else ("student_id", "count"),
            avg_sleep_hours=("sleep_hours", "mean"),
            avg_study_hours=("study_hours_per_day", "mean"),
            avg_stress_level=("stress_level", "mean"),
            avg_exam_pressure=("exam_pressure", "mean"),
            avg_social_support=("social_support", "mean"),
            avg_financial_stress=("financial_stress", "mean"),
        )
        .reset_index()
    )
    risk_summary.to_csv(out_dir / "eda_summary_by_risk.csv", index=False)

    behavior_summary = (
        df.groupby(["sleep_band", "study_band", "behavior_quadrant"], observed=False)
        .agg(
            students=("student_id", "count"),
            high_risk_rate=("is_high_risk", "mean"),
            avg_stress=("stress_level", "mean"),
            avg_exam_pressure=("exam_pressure", "mean"),
            avg_burnout=("burnout_score", "mean") if "burnout_score" in df.columns else ("student_id", "count"),
            avg_mental_health=("mental_health_index", "mean") if "mental_health_index" in df.columns else ("student_id", "count"),
        )
        .reset_index()
    )
    behavior_summary.to_csv(out_dir / "behavior_sleep_study_summary.csv", index=False)

    dictionary_rows = [{"column": k, "description": v} for k, v in DATA_DICTIONARY.items()]
    pd.DataFrame(dictionary_rows).to_csv(out_dir / "data_dictionary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean student burnout data for Module 3 and Module 4 dashboards.")
    parser.add_argument("--input", required=True, help="Path to raw CSV or ZIP containing CSV.")
    parser.add_argument("--out-dir", default="data/processed", help="Output directory for cleaned files.")
    parser.add_argument("--nrows", type=int, default=0, help="Optional row limit for testing. 0 means full data.")
    parser.add_argument("--dashboard-sample-size", type=int, default=100000, help="Rows to export as lightweight dashboard sample.")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nrows = args.nrows if args.nrows and args.nrows > 0 else None
    df = read_raw_data(input_path, nrows=nrows)
    original_shape = df.shape
    original_columns = df.columns.tolist()

    # Standardize known categorical fields first.
    if "gender" in df.columns:
        df["gender"] = standardize_gender(df["gender"])
    else:
        df["gender"] = "Other"

    numeric_cols = [c for c in set(RAW_FEATURES + ENGINEERED_FEATURES + OUTCOME_COLUMNS) if c in df.columns and c not in {"gender", "risk_level", "predicted_risk_label"}]
    df = coerce_numeric(df, numeric_cols)

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        df = df.drop_duplicates().reset_index(drop=True)

    df, missing_report = impute_missing(df)
    df, clip_report = clip_domain_values(df)

    # Recompute the project-specific interaction metrics so the feature definitions are transparent.
    df = add_engineered_features(df)
    df = add_dashboard_fields(df)

    # Keep columns in a logical dashboard/modeling order.
    ordered_cols = (
        ["student_id"]
        + [c for c in RAW_FEATURES if c in df.columns]
        + [c for c in ENGINEERED_FEATURES if c in df.columns]
        + [c for c in ["burnout_score", "mental_health_index", "dropout_risk", "risk_level", "target_risk_code", "is_high_risk"] if c in df.columns]
        + [c for c in ["sleep_band", "study_band", "stress_band", "screen_band", "behavior_quadrant", "sleep_deficit_hours", "overstudy_flag", "low_sleep_flag", "high_pressure_flag"] if c in df.columns]
        + [c for c in df.columns if c not in set(RAW_FEATURES + ENGINEERED_FEATURES + OUTCOME_COLUMNS + ["student_id", "target_risk_code", "is_high_risk", "sleep_band", "study_band", "stress_band", "screen_band", "behavior_quadrant", "sleep_deficit_hours", "overstudy_flag", "low_sleep_flag", "high_pressure_flag"])]
    )
    df = df.loc[:, ordered_cols]

    cleaned_path = out_dir / "student_burnout_cleaned.csv.gz"
    df.to_csv(cleaned_path, index=False, compression={"method": "gzip", "compresslevel": 1})

    sample_size = min(args.dashboard_sample_size, len(df))
    sample = df.sample(n=sample_size, random_state=RANDOM_STATE) if sample_size < len(df) else df.copy()
    sample.to_csv(out_dir / "student_burnout_dashboard_sample.csv", index=False)

    build_summary_tables(df, out_dir)

    report = {
        "input_file": str(input_path),
        "original_shape": list(original_shape),
        "cleaned_shape": list(df.shape),
        "original_columns": original_columns,
        "duplicate_rows_removed": duplicate_count,
        "missing_values_imputed": missing_report,
        "domain_clipping": clip_report,
        "risk_distribution": df["risk_level"].value_counts().reindex(RISK_ORDER).fillna(0).astype(int).to_dict(),
        "outputs": {
            "cleaned_dataset": str(cleaned_path),
            "dashboard_sample": str(out_dir / "student_burnout_dashboard_sample.csv"),
            "eda_summary_by_risk": str(out_dir / "eda_summary_by_risk.csv"),
            "behavior_sleep_study_summary": str(out_dir / "behavior_sleep_study_summary.csv"),
            "data_dictionary": str(out_dir / "data_dictionary.csv"),
        },
        "notes": [
            "Do not train the risk classifier using burnout_score, mental_health_index, dropout_risk, risk_level, target, or previous predicted columns.",
            "The eight engineered features are recomputed from raw behavioral/academic inputs for explainability.",
        ],
    }
    with open(out_dir / "cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
