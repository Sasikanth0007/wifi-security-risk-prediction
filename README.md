# Wi-Fi Security Risk Prediction and Explainability

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-UAVIDS--2025-orange.svg)](https://doi.org/10.5281/zenodo.15336998)
[![Tests](https://img.shields.io/badge/Tests-16%20Passed-brightgreen.svg)]()

A reproducible, interview-grade Machine Learning pipeline and interactive SOC dashboard that predicts wireless network threat categories from IEEE 802.11ac flow attributes and explains individual predictions using **SHAP (SHapley Additive exPlanations)**.

---

## 1. Project Overview

Wireless networks in modern enterprise and edge environments are exposed to diverse layer-2 and layer-3 attacks, including deauthentication/flooding Denial of Service, packet drop sinkholes, and identity impersonation. Traditional intrusion detection systems (IDS) rely heavily on rigid signatures that fail against mutated attack profiles, while deep learning models often operate as opaque "black boxes".

This project delivers:
- **Multi-Class Threat Classification**: Classifies wireless network flows into 5 traffic classes (`Normal Traffic`, `Flooding Attack`, `Blackhole Attack`, `Sybil Attack`, `Wormhole Attack`).
- **Operational Risk Categorization**: Maps granular threat predictions into actionable SOC severity categories (`Low Risk`, `High Risk`, `Critical Risk`).
- **Explainable AI (XAI)**: Uses **SHAP TreeExplainer** (with automated permutation importance fallback) to show exactly which behavioral features drove a prediction.
- **Zero-Contamination Preprocessing**: Strict stratified train/test partitioning prior to fitting transformers, completely preventing data leakage.
- **Interactive SOC Dashboard**: Streamlit web interface with real-time validated inference, preset profiles, local SHAP waterfalls, and performance analysis.

---

## 2. Dataset Attribution, Source, & Feature Definitions

### A. Source & License
- **Dataset Name**: UAVIDS-2025: A Benchmark Dataset for Intrusion Detection in UAV Networks Using Machine Learning Techniques
- **DOI**: [10.5281/zenodo.15336998](https://doi.org/10.5281/zenodo.15336998)
- **License**: Check the [Zenodo record](https://doi.org/10.5281/zenodo.15336998) for the current dataset license and attribution terms before redistributing the dataset. The dataset file is not included in this repository.
- **Peer-Reviewed Reference**:
  ```bibtex
  @inproceedings{zeng2025uavids,
    title={UAVIDS-2025: A Benchmark Dataset for Intrusion Detection in UAV Networks Using Machine Learning Techniques},
    author={Zeng, Qingli and Bashir, Abdalrahman and Nait-Abdesselam, Farid},
    booktitle={2025 IEEE Conference on Communications and Network Security (CNS)},
    pages={1--9},
    year={2025},
    organization={IEEE}
  }
  ```
- **Physical Channel Model**: IEEE 802.11ac wireless standard with Nakagami wireless propagation channel and AODV routing simulated in NS-3.

### B. Target Classes & Risk Severity Mapping
| Class Label | Threat Vector Description | Operational Risk Category |
| :--- | :--- | :--- |
| **Normal Traffic** | Legitimate, healthy IEEE 802.11ac data flow | `Low Risk (Normal Operation)` |
| **Flooding Attack** | High-volume wireless packet saturation / DoS buffer exhaustion | `High Risk (DoS / Buffer Exhaustion)` |
| **Sybil Attack** | Attacker transmits using multiple forged identities / MAC spoofing | `High Risk (Identity Impersonation)` |
| **Blackhole Attack** | Attacker node attracts routing flows and silently drops all packets | `Critical Risk (Packet Drop Sinkhole)` |
| **Wormhole Attack** | Attacker tunnels packets out-of-band to disrupt wireless mesh topology | `Critical Risk (Unauthorized Tunnel)` |

### C. Feature Space (16 Behavioral Attributes)
| Feature Name | Type | Physical Meaning |
| :--- | :--- | :--- |
| `FlowDuration/s` | Float | Duration of the network flow in seconds |
| `TxPackets` | Int | Number of transmitted packets |
| `RxPackets` | Int | Number of received packets |
| `LostPackets` | Int | Number of lost/dropped packets |
| `TxBytes` | Int | Total transmitted volume in bytes |
| `RxBytes` | Int | Total received volume in bytes |
| `TxPacketRate/s` | Float | Transmission rate in packets per second |
| `RxPacketRate/s` | Float | Reception rate in packets per second |
| `TxByteRate/s` | Float | Transmission rate in bytes per second |
| `RxByteRate/s` | Float | Reception rate in bytes per second |
| `MeanDelay/s` | Float | Average end-to-end packet delay in seconds |
| `MeanJitter/s` | Float | Statistical variance in packet arrival delay |
| `Throughput/Kbps`| Float | Observed bandwidth utilization in Kilobits per second |
| `MeanPacketSize` | Int | Mean payload size per packet in bytes |
| `PacketDropRate` | Float | Ratio of lost packets to total transmitted packets ($[0.0, 1.0]$) |
| `AverageHopCount`| Float | Mean number of wireless routing hops traversed |

> **Data Integrity Notice**: Identifiers (`FlowID`, `SrcAddr`, `DstAddr`, `SrcPort`, `DstPort`) and constant columns (`Protocol` = UDP) are **strictly excluded** from training to prevent data leakage and memorization.

---

## 3. Architecture & Project Structure

```text
WiFi Risk/
??? app.py                         # Streamlit dashboard
??? requirements.txt
??? .gitignore
??? README.md
??? data/
?   ??? sample_synthetic_wifi_dev.csv  # Small synthetic sample for development/tests
??? models/
?   ??? training_metrics.json      # Evaluation scores from the included training run
??? src/                           # Data loading, preprocessing, training, prediction, explainability
??? tests/                          # Pytest suite
```

Training generates the model artifacts under `models/` locally. They and the full research dataset are not committed.

---

## 4. Prerequisites & Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12 (Tested on Python 3.12.4 Windows 64-bit).
- PowerShell or standard terminal.

### Installation Steps
Open PowerShell in the project directory:

```powershell
# Navigate to the project root
cd "C:\Users\91939\OneDrive\Desktop\OffCampus\WiFi Risk"

# Install dependencies from requirements.txt
python -m pip install -r requirements.txt
```

---

## 5. Dataset Preparation

The dataset loader automatically downloads UAVIDS-2025 from Zenodo when training if `data/UAVIDS-2025.csv` is absent. The dataset is excluded from this repository; review the [Zenodo record](https://doi.org/10.5281/zenodo.15336998) for its current license and attribution terms. To download it manually, save the CSV as `data/UAVIDS-2025.csv`.

```powershell
# Trigger automated download and verification
python -c "from src.data_loader import download_dataset; download_dataset()"
```

If you are offline or testing in a restricted CI environment, generate the synthetic development sample:
```powershell
python -c "from src.data_loader import create_synthetic_dev_sample; create_synthetic_dev_sample()"
```

---

## 6. Model Training & Evaluation

### A. Train and Compare Models
Trains both **Logistic Regression** and **Random Forest** with 5-fold Stratified Cross-Validation on training data and evaluates on held-out test data:

```powershell
# Standard training on 25,000 stratified samples (reproducible, ~8 seconds)
python -m src.train --sample-size 25000

# Optional: Train on the entire 122,171 dataset
python -m src.train --sample-size -1

# Optional: Rapid offline test using the synthetic development sample
python -m src.train --use-synthetic
```

Training creates the model files used by the dashboard under `models/`. These generated files are not committed; rerun training after cloning before starting the dashboard.

### B. Display Evaluation & Metrics Report
```powershell
python -m src.evaluate
```

---

## 7. How to Launch the Streamlit Dashboard

Run the Streamlit application from PowerShell:

```powershell
python -m streamlit run app.py
```

The application will launch in your browser at:
`http://localhost:8501`

### Dashboard Sections:
1. **1. Overview & Architecture**: Executive summary, dataset DOI, high-level metrics, and pipeline diagram.
2. **2. Live Risk Prediction & SHAP**: Interactive form with realistic presets, instant inference, probability distributions, local SHAP horizontal contribution charts, and plain-language security narratives.
3. **3. Model Performance & Evaluation**: Model comparison leaderboard, per-class Precision/Recall/F1 table, interactive confusion matrix heatmap, and deep False Positive vs. False Negative analysis.
4. **4. Dataset Insights & Features**: Class balance charts, global feature importance ranking, and empirical density distribution histograms for all 16 flow features.
5. **5. Responsible Use & Limitations**: Ethical considerations, evasion risks, concept drift, and SOC integration guidance.

---

## 8. Reproducing Verified Metrics

The actual evaluation scores produced on the held-out test set ($N = 5,000$ test samples, fixed `random_state=42`):

| Model Name | Test Accuracy | F1-Score (Macro) | F1-Score (Weighted) | 5-Fold CV F1 (Mean $\pm$ Std) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** (L2 Multinomial) | 79.96% | 0.8022 | 0.8015 | $0.8034 \pm 0.0053$ |
| **Random Forest Classifier** (100 Trees) | **94.28%** | **0.9448** | **0.9428** | **$0.9418 \pm 0.0058$** |

### Per-Class Performance Breakdown (Champion: Random Forest)
| Traffic Class | Precision | Recall | F1-Score | Test Support |
| :--- | :---: | :---: | :---: | :---: |
| **Normal Traffic** | **0.9824** | **0.9888** | **0.9856** | 1,071 |
| **Flooding Attack** | **0.9727** | **0.9703** | **0.9715** | 807 |
| **Sybil Attack** | **0.9808** | **0.9868** | **0.9838** | 985 |
| **Blackhole Attack** | **0.9534** | **0.8428** | **0.8947** | 1,069 |
| **Wormhole Attack** | **0.8459** | **0.9354** | **0.8884** | 1,068 |

---

## 9. Running Automated Tests

Run the full pytest suite:

```powershell
python -m pytest -v
```

The suite was run with Python 3.12.4 using `python -m pytest -v --basetemp=.pytest-tmp` (16 passed). This option keeps temporary files inside the project folder on systems where pytest cannot access the default temporary directory.

**Test Coverage Summary**:
- `test_data_loader.py`: Verifies schema validation, target column detection, missing feature detection, and synthetic generation.
- `test_preprocessing.py`: Verifies stratified split dimensions, absence of data leakage, and transformer behavior.
- `test_train_and_predict.py`: Verifies training artifacts generation, single inference, input range validation, and batch inference.
- `test_explain.py`: Verifies SHAP TreeExplainer calculation, global importance rankings, narrative generation, and fallback execution.

---

## 10. Explainability Method & Known Limitations

### Explanation Method
- **Algorithm**: SHAP (SHapley Additive exPlanations) using `shap.TreeExplainer` for the ensemble model.
- **Mathematical Foundation**: Computes cooperative game theoretic Shapley values to allocate the marginal contribution of each network feature to the log-odds of a specific threat class.
- **Fallback**: If SHAP experiences an execution failure or unsupported feature type, the pipeline automatically switches to a documented permutation importance and baseline-deviation fallback, ensuring the web dashboard never crashes.

### Technical & Operational Limitations
1. **Correlation vs. Causation**: SHAP values indicate feature importance within the learned mathematical distribution of the training dataset. They are **not** causal proofs that a specific parameter caused the intrusion.
2. **No Hardware Sniffing**: The system is designed for flow telemetry analysis. It does not contain an active packet sniffer (e.g. Scapy/Aircrack) and does not capture physical RF signals directly.
3. **Wi-Fi Generation Differences**: The dataset reflects IEEE 802.11ac parameters. Deploying against Wi-Fi 6 (802.11ax) or Wi-Fi 7 (802.11be) requires domain adaptation due to differences in OFDMA scheduling and multi-link operations.
4. **Adversarial Evasion**: Sophisticated attackers may shape traffic rates to mimic normal traffic baseline delays. Production deployments should combine behavioral ML with signature firewalls and human SOC verification.
