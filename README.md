# DATA230-Final-Group-Project

***

# Scalable Predictive Analytics of Student Burnout: A GPU-Accelerated Study of 1M Records

## 🎓 Project Overview
This project presents a high-performance analytics pipeline designed to identify mental health risks within a massive student population of **1,000,000 records**. By leveraging **NVIDIA RAPIDS (cuDF)** and **XGBoost 2.0**, we successfully bridged the gap between big data engineering and **Explainable AI (XAI)**, providing actionable decision support for academic institutions.

### 🚀 Key Technical Achievements (Project Lead)
- **Extreme Scale:** High-fidelity processing of 1,000,000 rows across 20+ features.
- **Hardware Acceleration:** Implemented a GPU-based pipeline achieving a measured **21.67x throughput gain** over CPU-based Pandas.
- **Explainable AI:** Integrated **SHAP (SHapley Additive exPlanations)** to transform a "black-box" model into a transparent decision-support tool.
- **Tool Diversity:** Integrated 4 distinct platforms (Plotly, Tableau, Power BI, Streamlit) to satisfy institutional reporting needs.

---

## 🛠️ Repository Architecture
The repository is structured into modular tasks to ensure technical consistency and scalability:

### [Task 1: GPU Data Engineering](https://github.com/xinghanchen-xc/DATA230-Final-Group-Project/tree/main/Task%201%20-%20GPU%20Data%20Engineering)
- **NVIDIA RAPIDS Pipeline:** Automated cleaning, standardization, and benchmarking.
- **Feature Engineering:** Developed 8 interaction metrics (e.g., *Academic Resilience Ratio*) to capture non-linear burnout triggers.

### [Task 2: Advanced ML & Interpretability](https://github.com/xinghanchen-xc/DATA230-Final-Group-Project/tree/main/Task%202%20-%20ML%20%26%20Interpretability)
- **XGBoost 2.0 Optimization:** Hardware-accelerated training using the `hist` tree method and `cuda` device mapping.
- **XAI Suite:** Generated **SHAP Beeswarm Plots** to quantify directional feature impact.
- **Diagnostics:** Comprehensive monitoring via multi-logloss and classification error curves.

### [Task 3: Visual Analytics (Power BI)](https://github.com/xinghanchen-xc/DATA230-Final-Group-Project/tree/main/Task%203%20-%20Visual%20Analytics%20PowerBI)
- **Executive Suite:** Macro-level distribution and stressor intensity analysis.
- **Predictive Dashboard:** Integrated ML outputs featuring a high-fidelity **Confusion Matrix** and **Performance Cliff** analysis.

### [Task 4: Self-Taught Tool (Streamlit)](https://github.com/xinghanchen-xc/DATA230-Final-Group-Project/tree/main/Task%204%20-%20Streamlit%20Self-Taught%20Tool)
- **Real-time Simulator:** A Python-based web app for clinical risk triaging.
- **Decision Support:** Provides evidence-based reasoning and intervention strategies based on real-time user inputs.

---

## 📂 Data Management & Reproducibility
- **Dataset:** Sourced via `kagglehub` API (1M Records).
- **GitHub Preview:** Due to size limits (>300MB), a statistically representative 5% sample (`Burnout_Predictions_Preview_50k.csv`) is hosted here. 
- **Cloud Storage:** The full analytic dataset is maintained on **Google Drive** for Power BI integration.

---

## 👥 Accountability & Collaboration Proof
As per the Professor's requirement for "Collaboration Proof," this repository maintains a full audit log of all technical contributions:
- **Xinghan Chen (Technical Architect)**: Responsible for the end-to-end architecture of Tasks 1, 2, 3, and 4. This includes the development of the GPU-accelerated pipeline, the implementation of the core Machine Learning logic (XGBoost 2.0 & SHAP), and the structural design of the visualization suite.
- **Abhijith Reddy (Technical Assistant)**: Assisted with Machine Learning diagnostic tasks, while also providing support in dashboard refinement and final technical reporting.
- **Sanjana Reddy (Communications & Documentation Lead)**: Responsible for the synthesis of the final IEEE Technical Report and the creation of the presentation materials.

---

## ⚙️ How to Run
1. Open Google Colab with a **T4 GPU** runtime.
2. Execute `Task_1_Data_Engineering.ipynb` for data acquisition and preprocessing.
3. Execute `Task_2_ML_Interpretability.ipynb` for model training and SHAP analysis.
4. Run `streamlit run Task_4_.../app.py` to launch the local simulator.

*Developed for **DATA-230: Data Visualization** at **San José State University (SJSU)**.*
