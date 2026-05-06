"""
Streamlit web app for Module 3 self-taught tool requirement.

Run after cleaning and training:
  streamlit run streamlit_app.py

Expected files, created by the scripts in /src:
  outputs/dashboard/streamlit_dashboard_sample.csv
  outputs/dashboard/tableau_sleep_study_grid.csv
  outputs/reports/model_metrics.json
  outputs/reports/feature_importance.csv
  outputs/models/burnout_risk_model.joblib

The app still opens without the model file, but the Risk Simulator requires the model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DASHBOARD_FILE = APP_DIR / "outputs" / "dashboard" / "streamlit_dashboard_sample.csv"
DEFAULT_GRID_FILE = APP_DIR / "outputs" / "dashboard" / "tableau_sleep_study_grid.csv"
DEFAULT_MODEL_FILE = APP_DIR / "outputs" / "models" / "burnout_risk_model.joblib"
DEFAULT_METRICS_FILE = APP_DIR / "outputs" / "reports" / "model_metrics.json"
DEFAULT_IMPORTANCE_FILE = APP_DIR / "outputs" / "reports" / "feature_importance.csv"

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

MODEL_FEATURES = ["gender"] + BASE_FEATURES + ENGINEERED_FEATURES


st.set_page_config(
    page_title="Student Burnout Predictive Analytics",
    page_icon="🎓",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_dashboard_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["risk_level", "predicted_risk_label_model", "sleep_band", "study_band", "stress_band", "behavior_quadrant", "gender"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


@st.cache_data(show_spinner=False)
def load_metrics(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_feature_importance(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_sleep_study_grid(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_resource(show_spinner=False)
def load_model_package(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return joblib.load(p)


def add_engineered_features(row: Dict) -> Dict:
    eps = 1e-6
    row["academic_resilience"] = row["academic_performance"] / (100.0 * (1.0 + row["exam_pressure"] + row["stress_level"] + eps))
    row["study_sleep_gap"] = row["study_hours_per_day"] - row["sleep_hours"]
    row["socio_economic_multiplier"] = row["financial_stress"] * row["family_expectation"]
    row["effort_reward_index"] = row["study_hours_per_day"] / (1.0 + row["academic_performance"] + eps)
    row["total_cognitive_load"] = row["study_hours_per_day"] + row["screen_time"]
    row["anxiety_pressure_impact"] = row["anxiety_score"] * row["exam_pressure"]
    row["social_support_ratio"] = row["social_support"] / (1.0 + row["family_expectation"] + eps)
    row["combined_distress_index"] = row["anxiety_score"] + row["depression_score"]
    return row


def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Dashboard filters")
    filtered = df.copy()

    if "gender" in filtered.columns:
        genders = sorted(filtered["gender"].dropna().unique().tolist())
        selected = st.sidebar.multiselect("Gender", genders, default=genders)
        if selected:
            filtered = filtered[filtered["gender"].isin(selected)]

    if "academic_year" in filtered.columns:
        years = sorted(filtered["academic_year"].dropna().unique().tolist())
        selected_years = st.sidebar.multiselect("Academic year", years, default=years)
        if selected_years:
            filtered = filtered[filtered["academic_year"].isin(selected_years)]

    if "risk_level" in filtered.columns:
        risks = [r for r in RISK_ORDER if r in set(filtered["risk_level"].dropna())]
        selected_risk = st.sidebar.multiselect("Actual risk level", risks, default=risks)
        if selected_risk:
            filtered = filtered[filtered["risk_level"].isin(selected_risk)]

    if "sleep_hours" in filtered.columns:
        low, high = float(filtered["sleep_hours"].min()), float(filtered["sleep_hours"].max())
        sleep_range = st.sidebar.slider("Sleep hours", low, high, (low, high), 0.1)
        filtered = filtered[(filtered["sleep_hours"] >= sleep_range[0]) & (filtered["sleep_hours"] <= sleep_range[1])]

    if "study_hours_per_day" in filtered.columns:
        low, high = float(filtered["study_hours_per_day"].min()), float(filtered["study_hours_per_day"].max())
        study_range = st.sidebar.slider("Study hours per day", low, high, (low, high), 0.1)
        filtered = filtered[(filtered["study_hours_per_day"] >= study_range[0]) & (filtered["study_hours_per_day"] <= study_range[1])]

    return filtered


def metric_card(label: str, value, help_text: str = "") -> None:
    st.metric(label=label, value=value, help=help_text if help_text else None)


def render_overview(df: pd.DataFrame, metrics: Dict) -> None:
    st.subheader("Executive overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Students", f"{len(df):,}")
    with c2:
        high_rate = df["is_high_risk"].mean() if "is_high_risk" in df.columns else np.nan
        metric_card("Actual high risk", f"{high_rate:.1%}")
    with c3:
        p_high = df["p_high"].mean() if "p_high" in df.columns else np.nan
        metric_card("Avg predicted high-risk probability", f"{p_high:.1%}")
    with c4:
        metric_card("Avg sleep", f"{df['sleep_hours'].mean():.2f}h")
    with c5:
        metric_card("Model macro F1", f"{metrics.get('macro_f1', np.nan):.3f}")

    left, right = st.columns([1.1, 1])
    with left:
        risk_counts = df["predicted_risk_label_model"].value_counts().reindex(RISK_ORDER, fill_value=0).reset_index()
        risk_counts.columns = ["predicted_risk", "students"]
        fig = px.bar(risk_counts, x="predicted_risk", y="students", text="students", title="Predicted burnout-risk distribution")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        if "risk_level" in df.columns and "predicted_risk_label_model" in df.columns:
            cm = pd.crosstab(df["risk_level"], df["predicted_risk_label_model"]).reindex(index=RISK_ORDER, columns=RISK_ORDER, fill_value=0)
            fig = px.imshow(cm, text_auto=True, aspect="auto", title="Actual vs predicted risk matrix")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**What to say in presentation:** this page is the Power BI-style executive story: how many students are at risk, where predictions concentrate, and whether the model is making consistent decisions.")


def render_behavior_deep_dive(df: pd.DataFrame, grid: pd.DataFrame) -> None:
    st.subheader("Behavioral deep-dive: Sleep vs Study")
    left, right = st.columns([1, 1])

    with left:
        heat_source = grid if not grid.empty else df
        if not grid.empty and {"sleep_band", "study_band", "avg_predicted_high_probability"}.issubset(heat_source.columns):
            heat = heat_source.pivot_table(index="sleep_band", columns="study_band", values="avg_predicted_high_probability", aggfunc="mean", observed=False)
            title = "Average predicted high-risk probability by sleep and study band"
        else:
            heat = df.pivot_table(index="sleep_band", columns="study_band", values="p_high", aggfunc="mean", observed=False)
            title = "Average predicted high-risk probability by sleep and study band"
        fig = px.imshow(heat, text_auto=".1%", aspect="auto", title=title)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        sample = df.sample(n=min(20000, len(df)), random_state=42) if len(df) > 20000 else df
        fig = px.scatter(
            sample,
            x="sleep_hours",
            y="study_hours_per_day",
            color="predicted_risk_label_model",
            size="p_high",
            hover_data=["stress_level", "exam_pressure", "academic_performance", "social_support"],
            title="Student-level Sleep vs Study interaction",
        )
        st.plotly_chart(fig, use_container_width=True)

    seg = (
        df.groupby("behavior_quadrant", observed=False)
        .agg(
            students=("student_id", "count"),
            actual_high_risk_rate=("is_high_risk", "mean"),
            avg_predicted_high_probability=("p_high", "mean"),
            avg_sleep=("sleep_hours", "mean"),
            avg_study=("study_hours_per_day", "mean"),
            avg_stress=("stress_level", "mean"),
        )
        .reset_index()
        .sort_values("avg_predicted_high_probability", ascending=False)
    )
    fig = px.bar(
        seg,
        x="behavior_quadrant",
        y="avg_predicted_high_probability",
        hover_data=["students", "actual_high_risk_rate", "avg_sleep", "avg_study", "avg_stress"],
        title="Behavior quadrants ranked by predicted high-risk probability",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(seg, use_container_width=True)

    st.markdown("**Tableau dashboard angle:** use this same view as a heatmap + scatter + quadrant ranking. It directly answers whether sleep loss changes the risk pattern among heavy-study students.")


def render_predictive_insights(df: pd.DataFrame, importance: pd.DataFrame, metrics: Dict) -> None:
    st.subheader("Predictive insights and model explainability")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics.get('accuracy', np.nan):.3f}")
    c2.metric("Balanced accuracy", f"{metrics.get('balanced_accuracy', np.nan):.3f}")
    c3.metric("Macro F1", f"{metrics.get('macro_f1', np.nan):.3f}")
    c4.metric("ROC-AUC OVR", f"{metrics.get('roc_auc_ovr_macro', np.nan):.3f}")

    left, right = st.columns([1, 1])
    with left:
        if not importance.empty:
            top = importance.head(15).sort_values("importance", ascending=True)
            fig = px.bar(top, x="importance", y="feature", orientation="h", title="Global feature importance")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance file not found. Run src/02_train_model_generate_results.py first.")

    with right:
        prob_long = df[["risk_level", "p_low", "p_medium", "p_high"]].melt(id_vars="risk_level", var_name="probability", value_name="score")
        fig = px.histogram(prob_long, x="score", color="probability", facet_row="risk_level", nbins=35, title="Predicted probability distributions by actual risk")
        st.plotly_chart(fig, use_container_width=True)

    errors = df[df["prediction_correct"] == 0] if "prediction_correct" in df.columns else pd.DataFrame()
    if not errors.empty:
        st.markdown("#### Error analysis")
        err = errors["error_type"].value_counts().reset_index()
        err.columns = ["error_type", "students"]
        fig = px.bar(err, x="error_type", y="students", title="Most common misclassification paths")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(errors.head(1000), use_container_width=True)


def make_prediction(model_package, input_row: Dict) -> Tuple[str, np.ndarray, pd.DataFrame]:
    row = add_engineered_features(input_row.copy())
    X_one = pd.DataFrame([row])[MODEL_FEATURES]
    pipeline = model_package["pipeline"]
    proba = pipeline.predict_proba(X_one)[0]
    pred_code = int(np.argmax(proba))
    pred_label = CODE_TO_RISK[pred_code]
    return pred_label, proba, X_one


def get_local_explanation(model_package, X_one: pd.DataFrame, pred_label: str, importance: pd.DataFrame) -> pd.DataFrame:
    # Prefer SHAP local explanations when shap is installed; otherwise fall back to global importance.
    try:
        import shap

        pipeline = model_package["pipeline"]
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        X_trans = preprocessor.transform(X_one)
        feature_names = [name.replace("cat__", "").replace("num__", "") for name in preprocessor.get_feature_names_out()]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_trans)
        arr = np.asarray(shap_values)
        pred_code = RISK_TO_CODE[pred_label]
        if arr.ndim == 3 and arr.shape[-1] == 3:
            values = arr[0, :, pred_code]
        elif arr.ndim == 3 and arr.shape[0] == 3:
            values = arr[pred_code, 0, :]
        elif arr.ndim == 2:
            values = arr[0, :]
        else:
            raise ValueError(f"Unexpected SHAP shape: {arr.shape}")
        expl = pd.DataFrame({"feature": feature_names, "local_shap_contribution": values})
        expl["absolute_contribution"] = expl["local_shap_contribution"].abs()
        return expl.sort_values("absolute_contribution", ascending=False).head(12)
    except Exception:
        fallback = importance.copy()
        if fallback.empty:
            return pd.DataFrame(columns=["feature", "local_shap_contribution", "absolute_contribution"])
        fallback = fallback.head(12).rename(columns={"importance": "absolute_contribution"})
        fallback["local_shap_contribution"] = fallback["absolute_contribution"]
        return fallback[["feature", "local_shap_contribution", "absolute_contribution"]]


def render_risk_simulator(model_package, importance: pd.DataFrame) -> None:
    st.subheader("Real-time risk simulator")
    if model_package is None:
        st.warning("Model file not found. Run src/02_train_model_generate_results.py first, then restart Streamlit.")
        return

    st.markdown("Adjust a student's academic, behavioral, and mental-health indicators. The app recomputes the engineered features and returns a risk prediction.")

    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        age = st.slider("Age", 17, 35, 23)
        academic_year = st.selectbox("Academic year", [1, 2, 3, 4], index=1)
        academic_performance = st.slider("Academic performance", 0.0, 100.0, 71.0, 0.5)
        study_hours = st.slider("Study hours per day", 0.0, 14.0, 5.0, 0.1)
    with c2:
        sleep_hours = st.slider("Sleep hours", 3.0, 10.0, 6.5, 0.1)
        physical_activity = st.slider("Physical activity", 0.0, 7.0, 3.0, 0.1)
        screen_time = st.slider("Screen time", 1.0, 12.0, 5.0, 0.1)
        internet_usage = st.slider("Internet usage", 1.0, 14.0, 5.0, 0.1)
        social_support = st.slider("Social support", 0.0, 10.0, 5.0, 0.1)
    with c3:
        exam_pressure = st.slider("Exam pressure", 0.0, 10.0, 6.0, 0.1)
        stress_level = st.slider("Stress level", 0.0, 10.0, 4.2, 0.1)
        anxiety_score = st.slider("Anxiety score", 0.0, 10.0, 3.0, 0.1)
        depression_score = st.slider("Depression score", 0.0, 10.0, 1.3, 0.1)
        financial_stress = st.slider("Financial stress", 0.0, 10.0, 5.0, 0.1)
        family_expectation = st.slider("Family expectation", 0.0, 10.0, 6.0, 0.1)

    input_row = {
        "gender": gender,
        "age": age,
        "academic_year": academic_year,
        "study_hours_per_day": study_hours,
        "exam_pressure": exam_pressure,
        "academic_performance": academic_performance,
        "stress_level": stress_level,
        "anxiety_score": anxiety_score,
        "depression_score": depression_score,
        "sleep_hours": sleep_hours,
        "physical_activity": physical_activity,
        "social_support": social_support,
        "screen_time": screen_time,
        "internet_usage": internet_usage,
        "financial_stress": financial_stress,
        "family_expectation": family_expectation,
    }

    pred_label, proba, X_one = make_prediction(model_package, input_row)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted risk", pred_label)
    c2.metric("P(Low)", f"{proba[0]:.1%}")
    c3.metric("P(Medium)", f"{proba[1]:.1%}")
    c4.metric("P(High)", f"{proba[2]:.1%}")

    prob_df = pd.DataFrame({"risk": RISK_ORDER, "probability": proba})
    fig = px.bar(prob_df, x="risk", y="probability", range_y=[0, 1], title="Predicted class probabilities")
    st.plotly_chart(fig, use_container_width=True)

    expl = get_local_explanation(model_package, X_one, pred_label, importance)
    if not expl.empty:
        fig = px.bar(
            expl.sort_values("absolute_contribution", ascending=True),
            x="local_shap_contribution",
            y="feature",
            orientation="h",
            title=f"Local explanation for predicted class: {pred_label}",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(expl, use_container_width=True)

    with st.expander("Engineered features for this simulated student"):
        st.dataframe(X_one[ENGINEERED_FEATURES].T.rename(columns={0: "value"}), use_container_width=True)


def render_data_download(df: pd.DataFrame) -> None:
    st.subheader("Dashboard data preview and download")
    st.dataframe(df.head(1000), use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered dashboard data", csv, "filtered_student_burnout_dashboard.csv", "text/csv")


def main() -> None:
    st.title("Student Burnout Predictive Analytics")
    st.caption("Module 3: Power BI predictive insights + Streamlit self-taught app. Module 4: Sleep vs Study behavioral deep-dive for Tableau.")

    with st.sidebar.expander("File paths", expanded=False):
        dashboard_file = st.text_input("Dashboard sample CSV", str(DEFAULT_DASHBOARD_FILE))
        grid_file = st.text_input("Sleep-study grid CSV", str(DEFAULT_GRID_FILE))
        metrics_file = st.text_input("Metrics JSON", str(DEFAULT_METRICS_FILE))
        importance_file = st.text_input("Feature importance CSV", str(DEFAULT_IMPORTANCE_FILE))
        model_file = st.text_input("Model package", str(DEFAULT_MODEL_FILE))

    if not Path(dashboard_file).exists():
        st.error("Dashboard CSV not found. Run src/01_clean_data.py and src/02_train_model_generate_results.py first.")
        st.stop()

    df = load_dashboard_data(dashboard_file)
    grid = load_sleep_study_grid(grid_file)
    metrics = load_metrics(metrics_file)
    importance = load_feature_importance(importance_file)
    model_package = load_model_package(model_file)
    filtered = apply_sidebar_filters(df)

    st.sidebar.markdown(f"Rows after filters: **{len(filtered):,}**")

    tabs = st.tabs(["Overview", "Behavioral Deep-Dive", "Predictive Insights", "Risk Simulator", "Data"])
    with tabs[0]:
        render_overview(filtered, metrics)
    with tabs[1]:
        render_behavior_deep_dive(filtered, grid)
    with tabs[2]:
        render_predictive_insights(filtered, importance, metrics)
    with tabs[3]:
        render_risk_simulator(model_package, importance)
    with tabs[4]:
        render_data_download(filtered)


if __name__ == "__main__":
    main()
