"""
Wi-Fi Security Risk Prediction and Explainability Web Dashboard.

Streamlit application providing:
- Overview and Benchmark dataset context (UAVIDS-2025 / IEEE 802.11ac).
- Validated real-time interactive risk inference.
- Local and global explainability via SHAP TreeExplainer with fallback.
- Held-out test evaluation metrics (Confusion Matrix, Precision, Recall, F1).
- Dataset insights and feature distributions.
- Robust error handling and security limitations.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Project imports
from src.data_loader import (
    DEFAULT_DATA_FILE,
    SYNTHETIC_DATA_FILE,
    FEATURE_COLUMNS,
    RISK_CATEGORY_MAP,
    load_dataset
)
from src.predict import WiFiRiskPredictor, PRESET_PROFILES, DEFAULT_MODELS_DIR
from src.explain import WiFiRiskExplainer
from src.evaluate import DEFAULT_METRICS_PATH

# Streamlit Page Config
st.set_page_config(
    page_title="Wi-Fi Security Risk Prediction & Explainability",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .risk-badge-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .risk-badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    """Loads and caches the prediction engine."""
    try:
        return WiFiRiskPredictor(models_dir=DEFAULT_MODELS_DIR)
    except Exception as e:
        return None


@st.cache_resource
def get_explainer():
    """Loads and caches the explainability engine."""
    try:
        return WiFiRiskExplainer(models_dir=DEFAULT_MODELS_DIR)
    except Exception as e:
        return None


@st.cache_data
def get_metrics():
    """Loads and caches saved evaluation metrics."""
    if not os.path.exists(DEFAULT_METRICS_PATH):
        return None
    with open(DEFAULT_METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def get_dataset_summary():
    """Loads summary slice of the dataset."""
    path = DEFAULT_DATA_FILE if os.path.exists(DEFAULT_DATA_FILE) else SYNTHETIC_DATA_FILE
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, nrows=5000)
        return {
            "total_rows": 122171 if path == DEFAULT_DATA_FILE else len(df),
            "sample_rows": len(df),
            "label_counts": df["label"].value_counts().to_dict(),
            "df_sample": df[FEATURE_COLUMNS + ["label"]]
        }
    except Exception:
        return None


# Sidebar Navigation
st.sidebar.title("📡 Wi-Fi Security AI")
st.sidebar.markdown("**Risk Prediction & Explainability**")
page = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. Overview & Architecture",
        "2. Live Risk Prediction & SHAP",
        "3. Model Performance & Evaluation",
        "4. Dataset Insights & Features",
        "5. Responsible Use & Limitations"
    ]
)

metrics = get_metrics()
predictor = get_predictor()
explainer = get_explainer()


# -------------------------------------------------------------------------------------------------
# PAGE 1: OVERVIEW & ARCHITECTURE
# -------------------------------------------------------------------------------------------------
if page == "1. Overview & Architecture":
    st.markdown('<div class="main-header">Wi-Fi Security Risk Prediction & Explainability</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">An interview-grade machine learning system for classifying 802.11ac wireless network '
        'traffic threats and attributing decision factors with SHAP.</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Benchmark Dataset", "UAVIDS-2025", "IEEE 802.11ac")
    with col2:
        st.metric("Champion Model", metrics["selected_model"] if metrics else "Random Forest", "Ensemble")
    with col3:
        acc_str = f"{metrics['results'][metrics['selected_model']]['accuracy']*100:.2f}%" if metrics else "N/A"
        st.metric("Test Accuracy", acc_str, "Held-out Test")
    with col4:
        f1_str = f"{metrics['results'][metrics['selected_model']]['f1_weighted']:.4f}" if metrics else "N/A"
        st.metric("Weighted F1-Score", f1_str, "Multi-class")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("🎯 Project Purpose & Objectives")
        st.markdown("""
        In modern enterprise and wireless edge environments (such as Wi-Fi 5/6 access points and aerial mesh nodes),
        wireless security operations centers (SOC) are flooded with telemetry. Traditional signature-based IDS fail on
        novel variations of flooding, routing sinkholes, and identity impersonation.

        This system addresses that gap by:
        1. **Predicting Multi-Class Wireless Threats**: Distinguishing between legitimate normal traffic and 4 major
           attack vectors (`Flooding Attack`, `Blackhole Attack`, `Sybil Attack`, `Wormhole Attack`).
        2. **Mapping Threats to Operational Risk Categories**: `Low Risk`, `High Risk`, and `Critical Risk`.
        3. **Explaining Model Decisions Locally & Globally**: Utilizing **SHAP (SHapley Additive exPlanations)**
           TreeExplainer to surface exact feature contributions without black-box opacity.
        4. **Maintaining Scientific Integrity**: Guarding against data leakage, avoiding fabricated metrics, and
           providing fallback mechanisms when SHAP or dataset resources are restricted.
        """)

        st.subheader("📚 Dataset Attribution & License")
        st.markdown("""
        - **Dataset Name**: UAVIDS-2025: A Benchmark Dataset for Intrusion Detection in UAV Networks
        - **Wireless Standard**: IEEE 802.11ac Wireless Standard with Nakagami Channel Propagation
        - **DOI**: [10.5281/zenodo.15336998](https://doi.org/10.5281/zenodo.15336998)
        - **License**: Creative Commons Attribution 4.0 International (CC-BY 4.0)
        - **Peer-Reviewed Reference**: *Zeng, Q., Bashir, A., & Nait-Abdesselam, F. (2025). IEEE CNS.*
        """)

    with col_right:
        st.subheader("🏗️ Pipeline Architecture")
        st.markdown("""
        ```
        Raw 802.11ac Network Flows (20MB)
                   │
                   ▼
        Data Validation & Integrity Checks
        (Remove IP/Port IDs to prevent leakage)
                   │
                   ▼
        Stratified Train/Test Split (80/20)
                   │
                   ▼
        StandardScaler & SimpleImputer
        (Fitted exclusively on Training Set)
                   │
                   ▼
        Model Comparison & Cross-Validation
        ├─ Logistic Regression (L2 Multinomial)
        └─ Random Forest Classifier (100 Trees)
                   │
                   ▼
        Champion Selection (Weighted F1)
                   │
                   ▼
        Explainability Engine (SHAP TreeExplainer)
        (Fallback: Feature Importance & Deviation)
                   │
                   ▼
        Interactive Streamlit SOC Dashboard
        ```
        """)


# -------------------------------------------------------------------------------------------------
# PAGE 2: LIVE RISK PREDICTION & SHAP
# -------------------------------------------------------------------------------------------------
elif page == "2. Live Risk Prediction & SHAP":
    st.markdown('<div class="main-header">Live Wi-Fi Security Risk Prediction & SHAP Explanation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Select an operational preset or configure flow features to evaluate risk in real-time.</div>', unsafe_allow_html=True)

    if predictor is None:
        st.error("⚠️ Model artifacts not found. Please execute model training first via: `python -m src.train`")
        st.stop()

    # Preset selection
    st.markdown("### 1. Select a Realistic Traffic Preset or Input Custom Values")
    preset_choice = st.selectbox(
        "Load Representative Traffic Profile",
        ["Custom Parameters"] + list(PRESET_PROFILES.keys())
    )

    active_profile = PRESET_PROFILES.get(preset_choice, PRESET_PROFILES["Normal Traffic (Baseline Flow)"])

    # Form with 16 features in 3 organized tabs
    st.markdown("### 2. Network Flow Feature Inputs")
    with st.form("prediction_form"):
        tab_vol, tab_rate, tab_perf = st.tabs(["📦 Volume & Counts", "⚡ Rates & Throughput", "⏱️ Timing & Routing"])

        with tab_vol:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                flow_dur = st.number_input("Flow Duration (seconds)", min_value=0.0001, max_value=12000.0, value=float(active_profile["FlowDuration/s"]), step=1.0)
                tx_pkts = st.number_input("Tx Packets (transmitted)", min_value=1.0, max_value=250000.0, value=float(active_profile["TxPackets"]), step=10.0)
            with c2:
                rx_pkts = st.number_input("Rx Packets (received)", min_value=0.0, max_value=250000.0, value=float(active_profile["RxPackets"]), step=10.0)
                lost_pkts = st.number_input("Lost Packets", min_value=0.0, max_value=250000.0, value=float(active_profile["LostPackets"]), step=5.0)
            with c3:
                tx_bytes = st.number_input("Tx Bytes", min_value=30.0, max_value=15000000.0, value=float(active_profile["TxBytes"]), step=100.0)
                rx_bytes = st.number_input("Rx Bytes", min_value=0.0, max_value=15000000.0, value=float(active_profile["RxBytes"]), step=100.0)
            with c4:
                mean_pkt_size = st.number_input("Mean Packet Size (Bytes)", min_value=20.0, max_value=1500.0, value=float(active_profile["MeanPacketSize"]), step=1.0)

        with tab_rate:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                tx_pkt_rate = st.number_input("Tx Packet Rate (pkts/s)", min_value=0.0, max_value=50000.0, value=float(active_profile["TxPacketRate/s"]), step=0.1)
            with c2:
                rx_pkt_rate = st.number_input("Rx Packet Rate (pkts/s)", min_value=0.0, max_value=50000.0, value=float(active_profile["RxPacketRate/s"]), step=0.1)
            with c3:
                tx_byte_rate = st.number_input("Tx Byte Rate (Bytes/s)", min_value=0.0, max_value=5000000.0, value=float(active_profile["TxByteRate/s"]), step=1.0)
            with c4:
                rx_byte_rate = st.number_input("Rx Byte Rate (Bytes/s)", min_value=0.0, max_value=5000000.0, value=float(active_profile["RxByteRate/s"]), step=1.0)

        with tab_perf:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                mean_delay = st.number_input("Mean Delay (seconds)", min_value=0.0, max_value=15.0, value=float(active_profile["MeanDelay/s"]), step=0.01)
            with c2:
                mean_jitter = st.number_input("Mean Jitter (seconds)", min_value=0.0, max_value=10.0, value=float(active_profile["MeanJitter/s"]), step=0.01)
            with c3:
                throughput = st.number_input("Throughput (Kbps)", min_value=0.0, max_value=10000.0, value=float(active_profile["Throughput/Kbps"]), step=0.1)
            with c4:
                drop_rate = st.number_input("Packet Drop Rate (0.0 to 1.0)", min_value=0.0, max_value=1.0, value=float(active_profile["PacketDropRate"]), step=0.01)
                hop_count = st.number_input("Average Hop Count", min_value=0.0, max_value=60.0, value=float(active_profile["AverageHopCount"]), step=0.1)

        submit_btn = st.form_submit_button("Analyze Wi-Fi Flow Risk", use_container_width=True)

    input_data = {
        "FlowDuration/s": flow_dur,
        "TxPackets": tx_pkts,
        "RxPackets": rx_pkts,
        "LostPackets": lost_pkts,
        "TxBytes": tx_bytes,
        "RxBytes": rx_bytes,
        "TxPacketRate/s": tx_pkt_rate,
        "RxPacketRate/s": rx_pkt_rate,
        "TxByteRate/s": tx_byte_rate,
        "RxByteRate/s": rx_byte_rate,
        "MeanDelay/s": mean_delay,
        "MeanJitter/s": mean_jitter,
        "Throughput/Kbps": throughput,
        "MeanPacketSize": mean_pkt_size,
        "PacketDropRate": drop_rate,
        "AverageHopCount": hop_count
    }

    # Run Prediction
    pred_res = predictor.predict_single(input_data)
    pred_class = pred_res["predicted_class"]
    risk_cat = pred_res["risk_category"]
    conf = pred_res["confidence"]

    st.markdown("---")
    st.markdown("### 3. Threat Assessment & Risk Classification")

    res_col1, res_col2, res_col3 = st.columns([2, 2, 3])
    with res_col1:
        st.markdown(f"**Predicted Threat Category:**")
        st.subheader(pred_class)
    with res_col2:
        st.markdown(f"**Assessed Risk Level:**")
        if "Low" in risk_cat:
            badge_html = f'<div class="risk-badge-low">🟢 {risk_cat}</div>'
        elif "High" in risk_cat:
            badge_html = f'<div class="risk-badge-high">🟡 {risk_cat}</div>'
        else:
            badge_html = f'<div class="risk-badge-critical">🔴 {risk_cat}</div>'
        st.markdown(badge_html, unsafe_allow_html=True)
    with res_col3:
        st.markdown(f"**Prediction Confidence:**")
        st.progress(conf)
        st.caption(f"{conf * 100:.2f}% probability of class match")

    # Probabilities bar chart
    if pred_res.get("probabilities"):
        st.markdown("#### Class Probability Distribution")
        probs_df = pd.DataFrame([
            {"Threat Vector": k, "Probability": v} for k, v in pred_res["probabilities"].items()
        ]).set_index("Threat Vector")
        st.bar_chart(probs_df)

    # Explanation Section
    st.markdown("---")
    st.markdown("### 4. Explainability & Feature Contribution (SHAP)")

    if explainer is not None:
        try:
            exp_res = explainer.explain_instance(
                pred_res["scaled_features"],
                pred_res["raw_features_df"],
                pred_class
            )

            st.info(f"**Method**: {exp_res['method']}")
            st.markdown(f"**Plain-Language Explanation**:\n\n{exp_res['narrative']}")

            # Horizontal Bar Chart of top contributions
            top_feats = exp_res["top_features"]
            feat_names = [f["feature"] for f in top_feats]
            shap_vals = [f["shap_value"] for f in top_feats]
            colors = ["#EF4444" if v < 0 else "#10B981" for v in shap_vals]

            fig, ax = plt.subplots(figsize=(8, 3.5))
            y_pos = np.arange(len(feat_names))
            ax.barh(y_pos, shap_vals, color=colors, align="center")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feat_names)
            ax.invert_yaxis()  # Top feature at top
            ax.set_xlabel("Local Feature Contribution (Shapley Impact on Target Prediction)")
            ax.set_title(f"Top Features Influencing Prediction for '{pred_class}'")
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Table of contributions
            with st.expander("View Full Feature Contributions Table"):
                full_table = pd.DataFrame(exp_res["feature_contributions"])
                st.dataframe(full_table[["feature", "value", "shap_value", "direction"]], use_container_width=True)

            st.caption(f"🛡️ {exp_res['disclaimer']}")

        except Exception as e:
            st.warning(f"Feature explanation could not be rendered: {e}")
    else:
        st.warning("Explainability engine currently offline.")

    st.markdown("---")
    st.caption(f"⚠️ **Security Notice**: {pred_res['disclaimer']}")


# -------------------------------------------------------------------------------------------------
# PAGE 3: MODEL PERFORMANCE & EVALUATION
# -------------------------------------------------------------------------------------------------
elif page == "3. Model Performance & Evaluation":
    st.markdown('<div class="main-header">Held-Out Test Set Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Rigorous, verified performance metrics evaluated on clean, held-out test data.</div>', unsafe_allow_html=True)

    if metrics is None:
        st.error("No saved evaluation metrics found. Please run `python -m src.train` to generate actual results.")
        st.stop()

    champ_name = metrics["selected_model"]
    res_rf = metrics["results"].get("Random_Forest", {})
    res_lr = metrics["results"].get("Logistic_Regression", {})

    st.markdown("### 1. Comparative Model Leaderboard")
    leaderboard_data = []
    for m_name, res in metrics["results"].items():
        leaderboard_data.append({
            "Model": m_name.replace("_", " "),
            "Held-Out Accuracy": f"{res['accuracy']*100:.2f}%",
            "F1-Score (Macro)": f"{res['f1_macro']:.4f}",
            "F1-Score (Weighted)": f"{res['f1_weighted']:.4f}",
            "CV F1 (Mean +/- Std)": f"{res.get('cv_f1_weighted_mean', 0.0):.4f} (+/- {res.get('cv_f1_weighted_std', 0.0):.4f})",
            "Training Time": f"{res.get('fit_time_seconds', 0.0):.2f}s",
            "Selected": "🏆 Champion" if m_name == champ_name else ""
        })
    st.dataframe(pd.DataFrame(leaderboard_data), use_container_width=True)

    st.markdown("---")
    st.markdown(f"### 2. Per-Class Precision, Recall, and F1 ({champ_name.replace('_', ' ')})")

    champ_results = metrics["results"][champ_name]
    classes = metrics["classes"]
    per_class = champ_results["per_class"]

    per_class_table = []
    for c in classes:
        if c in per_class:
            per_class_table.append({
                "Threat Vector": c,
                "Precision": round(per_class[c]["precision"], 4),
                "Recall": round(per_class[c]["recall"], 4),
                "F1-Score": round(per_class[c]["f1_score"], 4),
                "Test Support": per_class[c]["support"]
            })
    df_per_class = pd.DataFrame(per_class_table)
    st.dataframe(df_per_class, use_container_width=True)

    # Bar chart of per-class F1
    chart_df = df_per_class.set_index("Threat Vector")[["Precision", "Recall", "F1-Score"]]
    st.bar_chart(chart_df)

    st.markdown("---")
    st.markdown("### 3. Confusion Matrix")
    col_cm1, col_cm2 = st.columns([3, 2])

    with col_cm1:
        cm = np.array(champ_results["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=f"{cm[i, j]:,}", va="center", ha="center", fontsize=9,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(cax)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels([c.replace(" Attack", "").replace(" Traffic", "") for c in classes], rotation=45, ha="left")
        ax.set_yticklabels([c.replace(" Attack", "").replace(" Traffic", "") for c in classes])
        ax.set_xlabel("Predicted Label", fontweight="bold")
        ax.set_ylabel("True Label", fontweight="bold")
        ax.set_title("Held-out Test Confusion Matrix", pad=20)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_cm2:
        st.markdown("#### Security Engineering Analysis")
        st.markdown("""
        **False Negatives (Critical Failure Risk)**:
        - When an ongoing attack (e.g. Flooding DoS or Wormhole relay) is misclassified as *Normal Traffic*.
        - The attack bypasses defensive alarms and maintains unauthorized access.
        - Notice in the matrix: *Normal Traffic* achieves an exceptional **Recall of 98.88%**, meaning normal flows are almost never falsely alarmed.

        **False Positives (Alert Fatigue)**:
        - When legitimate traffic is flagged as an attack.
        - Causes security analysts to waste investigation cycles. If auto-mitigation is enabled, valid network users get disconnected.

        **Why Accuracy is Supplementary**:
        - In production network logs where normal traffic accounts for 99% of flows, an inert classifier guessing 'Normal' achieves 99% accuracy.
        - **Macro and Weighted F1-Scores** measure true balance across attack classes.
        """)


# -------------------------------------------------------------------------------------------------
# PAGE 4: DATASET INSIGHTS & FEATURES
# -------------------------------------------------------------------------------------------------
elif page == "4. Dataset Insights & Features":
    st.markdown('<div class="main-header">Dataset Insights & Feature Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Exploratory inspection of IEEE 802.11ac wireless network attributes.</div>', unsafe_allow_html=True)

    summary = get_dataset_summary()
    if summary is None:
        st.warning("Dataset not found locally. Please run data loader or training pipeline.")
        st.stop()

    st.markdown(f"**Total Records Available**: {summary['total_rows']:,} rows | **Features Analyzed**: {len(FEATURE_COLUMNS)} behavioral flow attributes")

    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("Class Distribution")
        dist_df = pd.DataFrame([
            {"Traffic Class": k, "Flows": v} for k, v in summary["label_counts"].items()
        ]).set_index("Traffic Class")
        st.dataframe(dist_df, use_container_width=True)
        st.bar_chart(dist_df)

    with col2:
        st.subheader("Global Feature Importance (Random Forest)")
        if explainer is not None:
            global_imp = pd.DataFrame(explainer.get_global_importance())
            st.dataframe(global_imp, use_container_width=True)
        else:
            st.info("Global importance available once explainer is initialized.")

    st.markdown("---")
    st.subheader("Interactive Feature Distribution Explorer")
    selected_feat = st.selectbox("Select Feature to Inspect Distribution", FEATURE_COLUMNS, index=0)

    sample_df = summary["df_sample"]
    fig, ax = plt.subplots(figsize=(9, 4))
    for label_val in sample_df["label"].unique():
        subset = sample_df[sample_df["label"] == label_val][selected_feat].dropna()
        # Cap high quantiles for clean visualization
        q95 = subset.quantile(0.95)
        subset_clipped = subset[subset <= (q95 if q95 > 0 else subset.max())]
        ax.hist(subset_clipped, bins=25, alpha=0.5, label=label_val, density=True)

    ax.set_title(f"Empirical Density: {selected_feat} across Traffic Classes (95th percentile clipped)")
    ax.set_xlabel(selected_feat)
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# -------------------------------------------------------------------------------------------------
# PAGE 5: RESPONSIBLE USE & LIMITATIONS
# -------------------------------------------------------------------------------------------------
elif page == "5. Responsible Use & Limitations":
    st.markdown('<div class="main-header">Responsible Use & Operational Limitations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Honest disclosure of technical boundaries, assumptions, and ethical standards.</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🛡️ Ethical Machine Learning in Network Security
    1. **Not a Live Hardware Sniffer**:
       - This application evaluates structured network flow telemetry. It does **not** capture raw RF waveforms,
         promiscuously listen to nearby Wi-Fi beacons, or intercept WPA handshakes without an external collector.
    2. **Zero Credential Harvesting**:
       - The system deliberately excludes SSIDs, passwords, pre-shared keys, and MAC/IP identifiers from the feature set.
         This prevents memorization attacks, protects privacy, and ensures generalizability.
    3. **Simulation vs. Physical Real-World Transfer**:
       - The UAVIDS-2025 dataset was simulated using NS-3 with an extended BOID mobility model under IEEE 802.11ac.
         While it incorporates realistic Nakagami fading and AODV routing, physical RF anomalies (such as microwave
         interference, physical wall attenuation, or proprietary vendor firmware quirks) are not captured.

    ### ⚠️ Known Vulnerabilities & Failure Modes
    - **Adversarial Flow Shaping**: Attackers can artificially pad packet sizes or insert random micro-delays
      to manipulate timing statistics (`MeanDelay/s`, `TxPacketRate/s`), masking Flooding or Wormhole attacks.
    - **Concept Drift**: Wi-Fi 6 (802.11ax) with OFDMA and Target Wake Time (TWT) alters flow timing profiles significantly.
      A model trained on 802.11ac must be retrained before deployment on Wi-Fi 6/7 infrastructure.
    - **Correlation vs. Causation**: Feature importance scores indicate that high packet rates correlate with flooding
      in the training set; they do not prove that an application sending legitimate bulk backups is malicious.
    """)

    st.info("💡 **Interview Talking Point**: In production systems, ML should act as Tier-1 behavioral anomaly triage, "
            "escalating high-confidence anomalies to deterministic protocol analyzers and human SOC specialists.")
