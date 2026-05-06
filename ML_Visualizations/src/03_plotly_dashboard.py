"""
03_plotly_dashboard.py
Create an interactive Plotly HTML dashboard from the ML dashboard output.

Command:
  python src/03_plotly_dashboard.py \
    --data outputs/dashboard/powerbi_dashboard_sample.csv \
    --out outputs/figures/plotly_behavior_ml_dashboard.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RISK_ORDER = ["Low", "Medium", "High"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Plotly dashboard HTML for student burnout analytics.")
    parser.add_argument("--data", required=True, help="Path to powerbi_dashboard_sample.csv or streamlit_dashboard_sample.csv")
    parser.add_argument("--out", default="outputs/figures/plotly_behavior_ml_dashboard.html", help="Output HTML path")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    # Figure 1: Executive risk distribution.
    risk_counts = df["predicted_risk_label_model"].value_counts().reindex(RISK_ORDER, fill_value=0).reset_index()
    risk_counts.columns = ["risk_level", "students"]
    fig1 = px.bar(risk_counts, x="risk_level", y="students", title="Predicted Burnout Risk Distribution")

    # Figure 2: Sleep vs study behavioral interaction.
    heat = df.pivot_table(
        index="sleep_band",
        columns="study_band",
        values="p_high",
        aggfunc="mean",
        observed=False,
    )
    fig2 = px.imshow(heat, text_auto=".1%", aspect="auto", title="Predicted High-Risk Probability: Sleep vs Study")

    # Figure 3: Scatter for individual-level drill-down.
    fig3 = px.scatter(
        df.sample(n=min(15000, len(df)), random_state=42),
        x="sleep_hours",
        y="study_hours_per_day",
        color="predicted_risk_label_model",
        size="p_high",
        hover_data=["stress_level", "exam_pressure", "academic_performance", "social_support"],
        title="Student-Level Sleep vs Study Drill-Down",
    )

    # Figure 4: Top segment risk.
    seg = (
        df.groupby("behavior_quadrant", observed=False)
        .agg(students=("student_id", "count"), avg_high_probability=("p_high", "mean"), actual_high_rate=("is_high_risk", "mean"))
        .reset_index()
        .sort_values("avg_high_probability", ascending=False)
    )
    fig4 = px.bar(seg, x="behavior_quadrant", y="avg_high_probability", hover_data=["students", "actual_high_rate"], title="Behavior Segments Ranked by Predicted High-Risk Probability")

    # Combine into one HTML page.
    html_parts = [
        "<html><head><title>Student Burnout Plotly Dashboard</title></head><body>",
        "<h1>Student Burnout Predictive Analytics Dashboard</h1>",
        "<p>Use this HTML for the Plotly part of the project. The Power BI and Tableau dashboards should use the same output tables for consistency.</p>",
        fig1.to_html(full_html=False, include_plotlyjs="cdn"),
        fig2.to_html(full_html=False, include_plotlyjs=False),
        fig3.to_html(full_html=False, include_plotlyjs=False),
        fig4.to_html(full_html=False, include_plotlyjs=False),
        "</body></html>",
    ]
    out_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"Saved Plotly dashboard to {out_path}")


if __name__ == "__main__":
    main()
