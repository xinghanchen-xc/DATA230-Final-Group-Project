# DATA230-Final-Group-Project

***

# Scalable Predictive Analytics of Student Burnout: A GPU-Accelerated Study of 1M Records

## Project Overview
This project presents a production-ready analytics pipeline designed to identify mental health risks within a massive "Digital Twin" student population of **1,000,000 records**. By leveraging **NVIDIA RAPIDS** and **XGBoost 2.0**, we successfully bridged the gap between big data engineering and explainable AI, providing actionable decision support for academic institutions.

### Technical Core (Lead Contribution)
- **Scale:** High-fidelity processing of 1,000,000 rows x 20 features.
- **Hardware Acceleration:** Implemented a GPU-based pipeline using **cuDF** and **XGBoost (CUDA)**, achieving a **[Insert your speedup, e.g., 20.5x] speedup** over traditional CPU-based Pandas.
- **Explainable AI (XAI):** Integrated **SHAP (SHapley Additive exPlanations)** to transform a "black-box" model into a transparent decision-support tool.

---

## 📂 Data Management & Scalability
Due to the significant scale of the serialized output (**~300MB CSV**), we implemented a multi-tier data management strategy to comply with GitHub's 100MB file limit while ensuring reproducibility:

1.  **Full Analytic Dataset (1,000,000 Records):** Hosted on **Google Drive** for seamless integration with Power BI and Streamlit. [Access Full Dataset Here](插入你的Google_Drive链接)
2.  **GitHub Preview Subset (`Burnout_Predictions_Preview_50k.csv`):** A statistically representative 5% sample provided in this repository for code verification and structure review.
3.  **Serialized Features:** Used **Apache Parquet** format for internal data reconstitution to preserve metadata and optimize I/O performance.

---

## Project Architecture

### Phase 1: Data Engineering (`Task_1_Data_Engineering.ipynb`)
- **GPU Preprocessing:** Automated cleaning and standardization using `cuDF`.
- **Feature Engineering:** Developed 8 domain-specific interaction metrics (e.g., *Academic Resilience Ratio*, *Effort-Reward Disparity*) to capture non-linear burnout triggers.
- **Benchmarking:** Explicit performance comparison between CPU and GPU latencies.

### Phase 2: Predictive Modeling & XAI (`Task_2_ML_Interpretability.ipynb`)
- **XGBoost 2.0:** Implemented stochastic gradient boosting with `tree_method='hist'` and `device='cuda'`.
- **Interpretability:** Generated **SHAP Summary Plots** to quantify feature impact, identifying **Financial Stress** and **Sleep Hygiene** as the primary drivers of student risk.

---

## Multi-Tool Visualization Suite
To satisfy institutional reporting requirements, the project integrates four distinct visualization platforms:
1.  **Plotly:** High-density EDA and statistical distribution analysis.
2.  **Tableau:** Multivariate behavioral interaction mapping.
3.  **Power BI:** Executive-level predictive reporting (Actual vs. Predicted).
4.  **Streamlit:** A self-taught web application for real-time risk assessment and SHAP-based local explanations.

---

## Preliminary ML Direction
- **Task:** Multi-Class Classification (Predicting Low, Medium, High risk).
- **Justification:** EDA revealed a **"Performance Cliff"** at a stress threshold of 8/10, indicating a non-linear relationship that necessitates ensemble-based learning over simple regression.
- **Ethics & Privacy:** Treated the synthetic dataset as a **Scalable Stress-Test Sandbox**, ensuring the pipeline is "Privacy-by-Design" and ready for de-identified real-world deployment.

---

## Team Contribution Matrix
| Role | Responsibility | Primary Tools |
| :--- | :--- | :--- |
| Xinghan Chen | GPU Pipeline, Advanced ML, XAI, IEEE Methodology | NVIDIA RAPIDS, XGBoost, SHAP |
| **Member** | BI Suite, Self-Taught Tool, Dashboard Integration | Power BI, Streamlit, Tableau |
| **Member** | *Task Assignment: IEEE Synthesis & Slides (Pending)* | Google Docs, GitHub |

---

## Execution Instructions
1.  Open Google Colab and enable a **T4 GPU** runtime.
2.  Run `Task_1_Data_Engineering.ipynb` to download and preprocess the Kaggle dataset.
3.  Run `Task_2_ML_Interpretability.ipynb` to train the model and generate interpretability artifacts.
4.  Access the `Final_Predictions_for_Dashboard.csv` for BI visualization.

---
