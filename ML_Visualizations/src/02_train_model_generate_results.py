"""
02_train_model_generate_results.py
Train a burnout-risk classifier and generate Module 3/4 dashboard outputs.

Recommended command after cleaning:
  python src/02_train_model_generate_results.py \
    --cleaned data/processed/student_burnout_cleaned.csv.gz \
    --out-dir outputs \
    --max-rows 0 \
    --device cpu

For faster laptop testing:
  python src/02_train_model_generate_results.py \
    --cleaned data/processed/student_burnout_cleaned.csv.gz \
    --out-dir outputs \
    --max-rows 250000 \
    --device cpu

Outputs:
  outputs/models/burnout_risk_model.joblib
  outputs/dashboard/dashboard_predictions_full.csv.gz
  outputs/dashboard/powerbi_dashboard_sample.csv
  outputs/dashboard/tableau_sleep_study_grid.csv
  outputs/dashboard/tableau_sleep_study_scatter_sample.csv
  outputs/reports/model_metrics.json
  outputs/reports/classification_report.csv
  outputs/reports/confusion_matrix.csv
  outputs/reports/feature_importance.csv
  outputs/reports/shap_feature_importance.csv, when SHAP succeeds
  outputs/figures/*.html
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover
    raise ImportError("xgboost is required. Install with: pip install xgboost") from exc

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    px = None
    go = None

RANDOM_STATE = 42
RISK_ORDER = ["Low", "Medium", "High"]
RISK_TO_CODE = {"Low": 0, "Medium": 1, "High": 2}
CODE_TO_RISK = {v: k for k, v in RISK_TO_CODE.items()}

BASE_FEATURES = [
    "age",
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

CATEGORICAL_FEATURES = ["gender"]
NUMERIC_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

LEAKAGE_COLUMNS = {
    "burnout_score",
    "mental_health_index",
    "risk_level",
    "target",
    "target_risk_code",
    "is_high_risk",
    "dropout_risk",
    "predicted_risk_index",
    "predicted_risk_label",
    "predicted_risk_label_model",
    "predicted_risk_code_model",
    "p_low",
    "p_medium",
    "p_high",
    "prediction_confidence",
}


def read_cleaned(path: Path, max_rows: int = 0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cleaned file not found: {path}")
    nrows = None if max_rows == 0 else max_rows
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in MODEL_FEATURES + ["target_risk_code", "risk_level"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns. Re-run 01_clean_data.py. Missing: {missing}")


def build_pipeline(device: str = "cpu") -> Pipeline:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", encoder, CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # XGBoost 2+ uses tree_method='hist' and device='cuda' for GPU when available.
    model = XGBClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.075,
        subsample=0.85,
        colsample_bytree=0.85,
        max_bin=128,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        device=device,
        random_state=RANDOM_STATE,
        n_jobs=4,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def get_feature_names(pipeline: Pipeline) -> List[str]:
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    cleaned = []
    for name in names:
        cleaned.append(name.replace("cat__", "").replace("num__", ""))
    return cleaned


def evaluate_model(y_true: np.ndarray, y_proba: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1, 2])),
    }
    try:
        metrics["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, y_proba, labels=[0, 1, 2], multi_class="ovr", average="macro"))
        metrics["roc_auc_ovr_weighted"] = float(roc_auc_score(y_true, y_proba, labels=[0, 1, 2], multi_class="ovr", average="weighted"))
    except Exception:
        metrics["roc_auc_ovr_macro"] = float("nan")
        metrics["roc_auc_ovr_weighted"] = float("nan")
    return metrics


def write_json(path: Path, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def make_output_dirs(out_dir: Path) -> Dict[str, Path]:
    dirs = {
        "models": out_dir / "models",
        "dashboard": out_dir / "dashboard",
        "reports": out_dir / "reports",
        "figures": out_dir / "figures",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def write_metric_tables(y_test: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, dirs: Dict[str, Path]) -> None:
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=RISK_ORDER,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(dirs["reports"] / "classification_report.csv")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    cm_df = pd.DataFrame(cm, index=[f"actual_{c}" for c in RISK_ORDER], columns=[f"predicted_{c}" for c in RISK_ORDER])
    cm_df.to_csv(dirs["reports"] / "confusion_matrix.csv")

    actual_counts = pd.Series(y_test).map(CODE_TO_RISK).value_counts().reindex(RISK_ORDER, fill_value=0)
    pred_counts = pd.Series(y_pred).map(CODE_TO_RISK).value_counts().reindex(RISK_ORDER, fill_value=0)
    pd.DataFrame({"risk_level": RISK_ORDER, "actual_count": actual_counts.values, "predicted_count": pred_counts.values}).to_csv(
        dirs["reports"] / "prediction_distribution.csv", index=False
    )

    # ROC and precision-recall curve points for dashboard/Plotly use.
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    roc_rows = []
    pr_rows = []
    for i, cls in enumerate(RISK_ORDER):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, roc_thresholds = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_rows.extend({"class": cls, "fpr": a, "tpr": b, "threshold": t} for a, b, t in zip(fpr, tpr, roc_thresholds))
        precision, recall, pr_thresholds = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        # precision_recall_curve returns one extra point without a threshold.
        thresholds_padded = list(pr_thresholds) + [np.nan]
        pr_rows.extend({"class": cls, "precision": p, "recall": r, "threshold": t} for p, r, t in zip(precision, recall, thresholds_padded))
    pd.DataFrame(roc_rows).to_csv(dirs["reports"] / "roc_curve_points.csv", index=False)
    pd.DataFrame(pr_rows).to_csv(dirs["reports"] / "precision_recall_curve_points.csv", index=False)


def write_feature_importance(pipeline: Pipeline, dirs: Dict[str, Path]) -> pd.DataFrame:
    feature_names = get_feature_names(pipeline)
    booster = pipeline.named_steps["model"]
    importances = getattr(booster, "feature_importances_", np.zeros(len(feature_names)))
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df.to_csv(dirs["reports"] / "feature_importance.csv", index=False)
    return df


def write_shap_importance(pipeline: Pipeline, X_sample: pd.DataFrame, dirs: Dict[str, Path], max_rows: int = 300) -> None:
    try:
        import shap

        sample = X_sample.sample(n=min(max_rows, len(X_sample)), random_state=RANDOM_STATE)
        X_trans = pipeline.named_steps["preprocessor"].transform(sample)
        model = pipeline.named_steps["model"]
        feature_names = get_feature_names(pipeline)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_trans)

        arr = np.asarray(shap_values)
        # SHAP versions may return (classes, rows, features) or (rows, features, classes).
        if arr.ndim == 3 and arr.shape[0] == 3:
            abs_mean = np.mean(np.abs(arr), axis=(0, 1))
        elif arr.ndim == 3 and arr.shape[-1] == 3:
            abs_mean = np.mean(np.abs(arr), axis=(0, 2))
        elif arr.ndim == 2:
            abs_mean = np.mean(np.abs(arr), axis=0)
        else:
            raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")

        shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": abs_mean})
        shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        shap_df.to_csv(dirs["reports"] / "shap_feature_importance.csv", index=False)

        # A compact long-form sample for dashboards. Keep only top 15 features to avoid huge files.
        top_features = shap_df["feature"].head(15).tolist()
        top_idx = [feature_names.index(f) for f in top_features]
        if arr.ndim == 3 and arr.shape[-1] == 3:
            local_strength = np.mean(np.abs(arr[:, top_idx, :]), axis=2)
        elif arr.ndim == 3 and arr.shape[0] == 3:
            local_strength = np.mean(np.abs(arr[:, :, top_idx]), axis=0)
        else:
            local_strength = np.abs(arr[:, top_idx])
        local_df = pd.DataFrame(local_strength, columns=top_features)
        local_df.insert(0, "sample_row", np.arange(1, len(local_df) + 1))
        local_df.to_csv(dirs["reports"] / "shap_local_strength_sample.csv", index=False)
    except Exception as exc:
        warnings.warn(f"SHAP output skipped: {exc}")


def predict_in_batches(pipeline: Pipeline, df: pd.DataFrame, batch_size: int = 100000) -> Tuple[np.ndarray, np.ndarray]:
    probs = []
    preds = []
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start : start + batch_size][MODEL_FEATURES]
        p = pipeline.predict_proba(batch)
        probs.append(p)
        preds.append(np.argmax(p, axis=1))
    return np.vstack(probs), np.concatenate(preds)


def write_dashboard_outputs(df: pd.DataFrame, y_proba_full: np.ndarray, y_pred_full: np.ndarray, dirs: Dict[str, Path], sample_size: int, write_full: bool = False) -> None:
    dashboard = df.copy()
    dashboard["predicted_risk_code_model"] = y_pred_full.astype("int8")
    dashboard["predicted_risk_label_model"] = pd.Series(y_pred_full).map(CODE_TO_RISK).values
    dashboard["p_low"] = y_proba_full[:, 0]
    dashboard["p_medium"] = y_proba_full[:, 1]
    dashboard["p_high"] = y_proba_full[:, 2]
    dashboard["prediction_confidence"] = y_proba_full.max(axis=1)
    dashboard["prediction_correct"] = (dashboard["target_risk_code"].values == y_pred_full).astype("int8")
    dashboard["error_type"] = np.where(dashboard["prediction_correct"] == 1, "Correct", dashboard["risk_level"] + " -> " + dashboard["predicted_risk_label_model"])

    if write_full:
        full_path = dirs["dashboard"] / "dashboard_predictions_full.csv.gz"
        dashboard.to_csv(full_path, index=False, compression={"method": "gzip", "compresslevel": 1})

    sample_n = min(sample_size, len(dashboard))
    sample = dashboard.sample(n=sample_n, random_state=RANDOM_STATE) if sample_n < len(dashboard) else dashboard
    sample.to_csv(dirs["dashboard"] / "powerbi_dashboard_sample.csv", index=False)
    sample.to_csv(dirs["dashboard"] / "streamlit_dashboard_sample.csv", index=False)

    powerbi_kpi = pd.DataFrame(
        {
            "metric": [
                "students",
                "actual_high_risk_rate",
                "predicted_high_risk_rate",
                "avg_predicted_high_probability",
                "avg_sleep_hours",
                "avg_study_hours",
                "avg_stress_level",
                "avg_exam_pressure",
                "model_accuracy_on_full_scored_data",
            ],
            "value": [
                len(dashboard),
                dashboard["is_high_risk"].mean(),
                (dashboard["predicted_risk_label_model"] == "High").mean(),
                dashboard["p_high"].mean(),
                dashboard["sleep_hours"].mean(),
                dashboard["study_hours_per_day"].mean(),
                dashboard["stress_level"].mean(),
                dashboard["exam_pressure"].mean(),
                dashboard["prediction_correct"].mean(),
            ],
        }
    )
    powerbi_kpi.to_csv(dirs["dashboard"] / "powerbi_kpi_summary.csv", index=False)

    segment_cols = ["gender", "academic_year", "sleep_band", "study_band", "stress_band", "behavior_quadrant"]
    segment_rows = []
    for col in segment_cols:
        g = (
            dashboard.groupby(col, observed=False)
            .agg(
                students=("student_id", "count"),
                actual_high_risk_rate=("is_high_risk", "mean"),
                predicted_high_risk_rate=("p_high", "mean"),
                avg_burnout=("burnout_score", "mean") if "burnout_score" in dashboard.columns else ("student_id", "count"),
                avg_mental_health=("mental_health_index", "mean") if "mental_health_index" in dashboard.columns else ("student_id", "count"),
                avg_sleep=("sleep_hours", "mean"),
                avg_study=("study_hours_per_day", "mean"),
                avg_stress=("stress_level", "mean"),
            )
            .reset_index()
            .rename(columns={col: "segment_value"})
        )
        g.insert(0, "segment_name", col)
        segment_rows.append(g)
    pd.concat(segment_rows, ignore_index=True).to_csv(dirs["dashboard"] / "powerbi_risk_by_segment.csv", index=False)

    tableau_grid = (
        dashboard.groupby(["sleep_band", "study_band", "behavior_quadrant"], observed=False)
        .agg(
            students=("student_id", "count"),
            actual_high_risk_rate=("is_high_risk", "mean"),
            avg_predicted_high_probability=("p_high", "mean"),
            avg_burnout_score=("burnout_score", "mean") if "burnout_score" in dashboard.columns else ("student_id", "count"),
            avg_mental_health_index=("mental_health_index", "mean") if "mental_health_index" in dashboard.columns else ("student_id", "count"),
            avg_sleep_hours=("sleep_hours", "mean"),
            avg_study_hours=("study_hours_per_day", "mean"),
            avg_stress_level=("stress_level", "mean"),
            avg_exam_pressure=("exam_pressure", "mean"),
            avg_social_support=("social_support", "mean"),
        )
        .reset_index()
    )
    tableau_grid.to_csv(dirs["dashboard"] / "tableau_sleep_study_grid.csv", index=False)

    scatter_cols = [
        "student_id",
        "sleep_hours",
        "study_hours_per_day",
        "stress_level",
        "exam_pressure",
        "academic_performance",
        "screen_time",
        "social_support",
        "risk_level",
        "predicted_risk_label_model",
        "p_high",
        "behavior_quadrant",
    ]
    sample[scatter_cols].to_csv(dirs["dashboard"] / "tableau_sleep_study_scatter_sample.csv", index=False)


def write_plotly_figures(dirs: Dict[str, Path], feature_importance: pd.DataFrame, y_test: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, dashboard_sample_path: Path) -> None:
    if px is None or go is None:
        return

    # 1. Confusion matrix.
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    fig_cm = px.imshow(cm, x=RISK_ORDER, y=RISK_ORDER, text_auto=True, labels={"x": "Predicted", "y": "Actual", "color": "Students"}, title="ML Visual 1: Confusion Matrix")
    fig_cm.write_html(dirs["figures"] / "01_confusion_matrix.html")

    # 2. Feature importance.
    fi = feature_importance.head(15).sort_values("importance", ascending=True)
    fig_fi = px.bar(fi, x="importance", y="feature", orientation="h", title="ML Visual 2: XGBoost Feature Importance")
    fig_fi.write_html(dirs["figures"] / "02_feature_importance.html")

    # 3. ROC curve.
    roc_df = pd.read_csv(dirs["reports"] / "roc_curve_points.csv")
    fig_roc = px.line(roc_df, x="fpr", y="tpr", color="class", title="ML Visual 3: One-vs-Rest ROC Curves")
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    fig_roc.write_html(dirs["figures"] / "03_roc_curves.html")

    # 4. Precision-recall curve.
    pr_df = pd.read_csv(dirs["reports"] / "precision_recall_curve_points.csv")
    fig_pr = px.line(pr_df, x="recall", y="precision", color="class", title="ML Visual 4: Precision-Recall Curves")
    fig_pr.write_html(dirs["figures"] / "04_precision_recall_curves.html")

    # 5. Probability distribution.
    sample = pd.read_csv(dashboard_sample_path)
    prob_long = sample[["risk_level", "p_low", "p_medium", "p_high"]].melt(id_vars="risk_level", var_name="probability", value_name="score")
    fig_prob = px.histogram(prob_long, x="score", color="probability", facet_row="risk_level", nbins=40, title="ML Visual 5: Predicted Probability Distributions by Actual Risk")
    fig_prob.write_html(dirs["figures"] / "05_probability_distribution.html")

    # 6. Sleep vs study heatmap for behavioral deep dive.
    grid = pd.read_csv(dirs["dashboard"] / "tableau_sleep_study_grid.csv")
    pivot = grid.pivot_table(index="sleep_band", columns="study_band", values="avg_predicted_high_probability", aggfunc="mean", observed=False)
    fig_heat = px.imshow(pivot, text_auto=".2%", aspect="auto", title="ML Visual 6: Predicted High-Risk Probability by Sleep and Study Bands")
    fig_heat.write_html(dirs["figures"] / "06_sleep_study_high_risk_heatmap.html")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train burnout-risk model and generate BI/Tableau/Streamlit outputs.")
    parser.add_argument("--cleaned", required=True, help="Path to student_burnout_cleaned.csv.gz from 01_clean_data.py.")
    parser.add_argument("--out-dir", default="outputs", help="Root output directory.")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional row limit for model training and scoring. 0 means all rows.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction for model evaluation.")
    parser.add_argument("--dashboard-sample-size", type=int, default=100000, help="Rows to export as uncompressed dashboard sample.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Use cuda only in a GPU environment such as Colab with RAPIDS/XGBoost GPU.")
    parser.add_argument("--shap-sample-size", type=int, default=300, help="Rows for SHAP explainability. Use 0 to skip SHAP for speed.")
    parser.add_argument("--write-full-dashboard", action="store_true", help="Also write dashboard_predictions_full.csv.gz. This can be slow for 1M rows.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    dirs = make_output_dirs(out_dir)
    df = read_cleaned(Path(args.cleaned), max_rows=args.max_rows)
    validate_columns(df)

    # Ensure target is numeric and has the expected order.
    y = pd.to_numeric(df["target_risk_code"], errors="coerce").astype(int)
    X = df[MODEL_FEATURES].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    pipeline = build_pipeline(device=args.device)

    try:
        pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)
    except Exception as exc:
        if args.device == "cuda":
            warnings.warn(f"CUDA training failed; retrying on CPU. Error: {exc}")
            pipeline = build_pipeline(device="cpu")
            pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)
        else:
            raise

    y_proba = pipeline.predict_proba(X_test)
    y_pred = np.argmax(y_proba, axis=1)
    metrics = evaluate_model(y_test.to_numpy(), y_proba, y_pred)
    metrics.update({
        "rows_used": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_order": RISK_ORDER,
        "feature_count_before_encoding": len(MODEL_FEATURES),
        "feature_count_after_encoding": len(get_feature_names(pipeline)),
        "leakage_columns_excluded": sorted(list(LEAKAGE_COLUMNS)),
        "device_requested": args.device,
    })

    write_json(dirs["reports"] / "model_metrics.json", metrics)
    write_metric_tables(y_test.to_numpy(), y_pred, y_proba, dirs)
    feature_importance = write_feature_importance(pipeline, dirs)
    if args.shap_sample_size > 0:
        write_shap_importance(pipeline, X_test, dirs, max_rows=args.shap_sample_size)

    # Score the full dataset used by this run and build BI/Tableau/Streamlit tables.
    full_proba, full_pred = predict_in_batches(pipeline, df)
    write_dashboard_outputs(df, full_proba, full_pred, dirs, sample_size=args.dashboard_sample_size, write_full=args.write_full_dashboard)

    model_package = {
        "pipeline": pipeline,
        "model_features": MODEL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "class_order": RISK_ORDER,
        "risk_to_code": RISK_TO_CODE,
        "code_to_risk": CODE_TO_RISK,
        "metrics": metrics,
    }
    joblib.dump(model_package, dirs["models"] / "burnout_risk_model.joblib")

    write_plotly_figures(
        dirs,
        feature_importance=feature_importance,
        y_test=y_test.to_numpy(),
        y_pred=y_pred,
        y_proba=y_proba,
        dashboard_sample_path=dirs["dashboard"] / "powerbi_dashboard_sample.csv",
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved model and dashboard outputs under: {out_dir}")


if __name__ == "__main__":
    main()
