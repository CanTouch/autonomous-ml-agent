"""
Autonomous ML Agent - Streamlit Web Application
A self-contained web interface for automated tabular machine learning.
"""

import os
import sys
import json
import time
import shutil
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.agent import MLAgent


st.set_page_config(
    page_title="Autonomous ML Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .champion-banner {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 18px 24px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_benchmark_datasets():
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        return {}
    return {p.name: p for p in sorted(raw_dir.glob("*.csv"))}


def parse_final_report(dataset_stem: str):
    reports = list(Path("reports").glob(f"FINAL_REPORT_{dataset_stem}_*.md"))
    if not reports:
        return None, ""
    latest_path = sorted(reports)[-1]
    content = latest_path.read_text()
    return latest_path, content


# --- SIDEBAR: Data Input & Mode Selection ---
st.sidebar.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
st.sidebar.title("ML Agent Setup")

data_source = st.sidebar.radio(
    "Data Source",
    ["📂 Choose Benchmark Dataset", "📤 Upload Custom CSV"],
)

benchmarks = get_benchmark_datasets()
selected_dataset_path = None
uploaded_df = None

if data_source == "📂 Choose Benchmark Dataset":
    if benchmarks:
        choice = st.sidebar.selectbox("Select Benchmark", list(benchmarks.keys()))
        selected_dataset_path = benchmarks[choice]
        uploaded_df = pd.read_csv(selected_dataset_path)
    else:
        st.sidebar.warning("No benchmark datasets found in data/raw/")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            temp_path = Path("data/raw") / uploaded_file.name
            uploaded_df.to_csv(temp_path, index=False)
            selected_dataset_path = temp_path
            st.sidebar.success(f"Loaded '{uploaded_file.name}'")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("Operation Mode")
mode = st.sidebar.radio(
    "Execution Mode",
    ["⚡ 1-Click Autonomous (Full Auto)", "⚙️ Custom Settings"],
)

agent = MLAgent()

# --- MAIN CONTENT ---
st.markdown('<div class="main-header">🤖 Autonomous ML Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload any raw tabular dataset. The agent automatically inspects, cleans, balances, tests multiple ML algorithms, tunes the winner, and presents final findings.</div>',
    unsafe_allow_html=True,
)

if uploaded_df is None or selected_dataset_path is None:
    st.info("👈 Please select a benchmark dataset or upload your own CSV in the sidebar to get started.")
    st.stop()

# Dataset Overview Cards
info = agent.inspect_dataset(selected_dataset_path)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Records", f"{info['rows']:,}")
with c2:
    st.metric("Features", f"{info['columns']:,}")
with c3:
    st.metric("Missing Values", f"{info['missing_total']:,}")
with c4:
    st.metric("Duplicate Rows", f"{info['duplicate_rows']:,}")

# Preview & Target Detection
with st.expander("🔍 Dataset Preview (First 10 rows)", expanded=False):
    st.dataframe(uploaded_df.head(10), use_container_width=True)

# Determine Target & Task
inferred_target = info["candidate_targets"][0] if info["candidate_targets"] else None
inferred_task = agent.infer_task_type(uploaded_df, inferred_target)

final_target = inferred_target
final_task = inferred_task
final_model = "auto"
final_feature_selection = True
final_class_balancing = None

if mode == "⚙️ Custom Settings":
    col_a, col_b = st.columns(2)
    with col_a:
        options = ["(None - Unsupervised)"] + uploaded_df.columns.tolist()
        default_idx = options.index(inferred_target) if inferred_target in options else 0
        tgt_choice = st.selectbox("Target Column to Predict", options, index=default_idx)
        final_target = None if tgt_choice == "(None - Unsupervised)" else tgt_choice

    with col_b:
        task_options = ["classification", "regression", "unsupervised"]
        default_task_idx = task_options.index(inferred_task) if inferred_task in task_options else 0
        final_task = st.selectbox("ML Task", task_options, index=default_task_idx)

    col_c, col_d = st.columns(2)
    with col_c:
        if final_task != "unsupervised":
            final_model = st.selectbox(
                "Model Strategy",
                ["auto (Tournament Winner)", "xgboost", "random_forest", "gradient_boosting", "adaboost", "svm", "logistic_regression", "linear_regression"],
            )
            if final_model.startswith("auto"):
                final_model = "auto"

    with col_d:
        if final_task == "classification":
            final_class_balancing = st.selectbox("Class Balancing", ["none", "undersample", "smote", "class_weight"])
            if final_class_balancing == "none":
                final_class_balancing = None

    final_feature_selection = st.checkbox("Enable 3-Tier Feature Selection", value=True)
else:
    # Autonomous Defaults
    st.markdown(
        f"""
        > 💡 **Autonomous Decision:**
        > Target: **`{final_target}`** | Inferred Task: **`{final_task.upper()}`** | Model: **`Auto-Tournament (Evaluates All Models)`**
        """
    )
    if final_task == "classification" and final_target:
        y_counts = uploaded_df[final_target].value_counts(normalize=True)
        if len(y_counts) > 0 and y_counts.iloc[0] > 0.60:
            final_class_balancing = "undersample"
            st.info(f"⚖️ Class imbalance detected ({y_counts.iloc[0]*100:.1f}% majority). Automatic training undersampling enabled.")

# --- EXECUTION BUTTON ---
start_run = st.button("🚀 Run Full Data Science Pipeline", type="primary", use_container_width=True)

if start_run:
    stem = selected_dataset_path.stem
    clean_cfg = Path(f"{stem}_cleaning_config.json")
    feat_cfg = Path(f"{stem}_feature_engineering_config.json")

    # Generate configs if not present
    if not clean_cfg.exists() or not feat_cfg.exists():
        with st.spinner("Synthesizing data cleaning & feature engineering strategies..."):
            clean_cfg, feat_cfg = agent.auto_generate_configs(uploaded_df, final_target, stem)

    # Progress container
    progress_bar = st.progress(0)
    status_text = st.empty()

    stages = [
        ("Data Profiling & Health Scan", 15),
        ("Automated Data Cleaning & Outlier Imputation", 30),
        ("Feature Engineering & Encoding", 45),
        ("Feature Selection & Collinearity Filtering", 60),
        ("Model Tournament & Champion Selection", 80),
        ("Hyperparameter Tuning & Final Evaluation", 95),
        ("Final Master Report Generation", 100),
    ]

    status_text.text("🤖 Launching Autonomous Data Scientist Pipeline...")
    start_time = time.time()

    with st.spinner("Agent is analyzing dataset and training models..."):
        returncode = agent.run_pipeline(
            dataset_path=selected_dataset_path,
            target=final_target,
            task=final_task,
            model=final_model,
            cleaning_config=clean_cfg,
            feature_config=feat_cfg,
            feature_selection=final_feature_selection,
            class_balancing=final_class_balancing,
        )

    elapsed = time.time() - start_time
    progress_bar.progress(100)

    if returncode == 0:
        status_text.success(f"✨ Pipeline completed successfully in {elapsed:.1f} seconds!")
        st.balloons()
    else:
        status_text.error(f"Pipeline encountered an issue (exit code {returncode}).")
        st.stop()

# --- DISPLAY RESULTS (IF REPORT EXISTS) ---
stem = selected_dataset_path.stem
report_path, report_content = parse_final_report(stem)

if report_path and report_path.exists():
    st.markdown("---")
    st.subheader("📊 Executive Data Scientist Report")

    # Tournament Leaderboard Table (if Model Selection report exists)
    ms_reports = list(Path("reports").glob(f"model_selection*_{stem}_*.md")) + list(Path("reports").glob(f"model_selection*_{stem}_selected_*.md"))
    if ms_reports:
        latest_ms = sorted(ms_reports)[-1]
        ms_text = latest_ms.read_text()
        table_lines = [l for l in ms_text.split("\n") if l.startswith("|") and not l.startswith("|---")]
        if len(table_lines) > 1:
            header = [h.strip() for h in table_lines[0].split("|")[1:-1]]
            rows = []
            for line in table_lines[1:]:
                row_vals = [c.strip() for c in line.split("|")[1:-1]]
                if len(row_vals) == len(header):
                    rows.append(row_vals)
            if rows:
                leaderboard_df = pd.DataFrame(rows, columns=header)
                st.write("### 🏆 Algorithm Tournament Leaderboard")
                st.dataframe(leaderboard_df, use_container_width=True)

    # Generated Plot Artifacts
    plot_files = list(Path("reports").glob(f"*_{stem}_*.png"))
    if plot_files:
        st.write("### 📈 Visualizations & Diagnostics")
        plot_cols = st.columns(min(len(plot_files), 2))
        for idx, ppath in enumerate(plot_files):
            with plot_cols[idx % 2]:
                st.image(str(ppath), caption=ppath.name)

    # Export Trained Model and Predictions Buttons
    st.write("### 💾 Export Model & Predictions")
    dl_c1, dl_c2, dl_c3 = st.columns(3)

    # 1. Download Master Report
    with dl_c1:
        st.download_button(
            label="📄 Download Executive Report",
            data=report_content,
            file_name=report_path.name,
            mime="text/markdown",
            use_container_width=True,
        )

    # 2. Download Trained Model (.joblib)
    saved_models = list(Path("reports").glob(f"model_*_{stem}.joblib"))
    with dl_c2:
        if saved_models:
            latest_model = sorted(saved_models)[-1]
            with open(latest_model, "rb") as mf:
                st.download_button(
                    label=f"💾 Download Model ({latest_model.stem.split('_')[1]})",
                    data=mf.read(),
                    file_name=latest_model.name,
                    mime="application/octet-stream",
                    use_container_width=True,
                )
        else:
            st.button("💾 Model (.joblib)", disabled=True, use_container_width=True)

    # 3. Download Predictions CSV
    saved_preds = list(Path("reports").glob(f"predictions_*_{stem}.csv"))
    with dl_c3:
        if saved_preds:
            latest_pred = sorted(saved_preds)[-1]
            pred_text = latest_pred.read_text()
            st.download_button(
                label="📊 Download Predictions (CSV)",
                data=pred_text,
                file_name=latest_pred.name,
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("📊 Predictions (CSV)", disabled=True, use_container_width=True)

    # Live Batch Scoring
    if saved_models:
        latest_model_path = sorted(saved_models)[-1]
        with st.expander("🔮 Batch Prediction on New Unlabelled Data", expanded=False):
            st.write("Upload a new dataset with the same features to generate predictions using the trained champion model.")
            batch_file = st.file_uploader("Upload CSV for Scoring", type=["csv"], key="batch_scoring_uploader")
            if batch_file is not None:
                import joblib
                batch_df = pd.read_csv(batch_file)
                st.write(f"Loaded {len(batch_df)} rows for prediction.")
                try:
                    loaded_model = joblib.load(latest_model_path)
                    pred_res = batch_df.copy()
                    # Align feature columns if possible
                    feature_cols = [c for c in loaded_model.feature_names_in_ if c in batch_df.columns] if hasattr(loaded_model, "feature_names_in_") else batch_df.columns
                    preds = loaded_model.predict(batch_df[feature_cols])
                    pred_res["model_prediction"] = preds
                    if hasattr(loaded_model, "predict_proba"):
                        pred_res["prediction_probability"] = loaded_model.predict_proba(batch_df[feature_cols])[:, 1]
                    st.dataframe(pred_res.head(10))
                    st.download_button(
                        label="📥 Download Scored CSV",
                        data=pred_res.to_csv(index=False),
                        file_name=f"scored_{batch_file.name}",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"Prediction error: {e}. Ensure features match the trained model.")

    # Full Master Report Expander
    with st.expander("📄 Full Markdown Master Report", expanded=False):
        st.markdown(report_content)
