# PV MPP Modeling using Linear, Ridge, and ANN Regression

This project implements and compares classical linear models and a neural network–based approach for predicting the Maximum Power Point (MPP) of a photovoltaic (PV) system under varying irradiance and temperature conditions.

The goal is to evaluate whether nonlinear regression provides a meaningful advantage over linear baselines, particularly under dynamic operating regimes.

---

## Project Motivation

Maximum Power Point Tracking (MPPT) is critical for efficient photovoltaic energy extraction. While classical methods and linear models capture coarse trends, PV system behavior under dynamic environmental conditions is inherently nonlinear.

This project investigates:

- how far linear and regularized regression models can go, and
- whether an Artificial Neural Network (ANN) offers practical benefits beyond these baselines.

---

## System Overview

The overall workflow of the project is shown below:

PV System Simulation (MATLAB / Simulink)
|
v
Dataset Generation (Irradiance, Temperature → Vmpp, Impp)
|
v
Train / Validation / Test Split
|
v
Models Implemented

- Linear Regression
- Ridge Regression
- Artificial Neural Network (ANN)
  |
  v
  Evaluation (Vmpp, Impp)
  |
  v
  Derived Output (Pmpp = Vmpp × Impp)
  |
  v
  Explainability & Error Analysis

---

## Data Generation and Description

- Data is generated using a photovoltaic system simulation under varying irradiance and temperature conditions.
- Step changes are introduced to capture dynamic and transient behavior.
- Simulation-based data allows controlled experimentation and repeatability.

**Inputs:**

- Irradiance
- Temperature
- (Optional: voltage/current depending on configuration)

**Targets:**

- Vmpp (voltage at maximum power point)
- Impp (current at maximum power point)

**Derived Quantity:**

- Pmpp = Vmpp × Impp

---

## Models Implemented

### Linear Regression

- Serves as a baseline model.
- Assumes a linear relationship between inputs and outputs.
- Limited in capturing nonlinear and interaction effects.

### Ridge Regression

- Regularized linear regression.
- Mitigates multicollinearity and improves coefficient stability.
- Retains linear expressiveness.

### Artificial Neural Network (ANN)

- Nonlinear regression model.
- Designed to capture complex dependencies between environmental conditions and MPP.
- Trained using supervised learning on simulation-generated data.

---

## Evaluation Strategy and Metrics

All models are evaluated using identical train–test splits to ensure a fair comparison.

**Metrics used:**

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

Performance is evaluated for:

- Vmpp
- Impp
- Pmpp (derived from predicted Vmpp and Impp)

---

## Results Summary

- Linear and ridge regression models capture general trends but struggle under dynamic operating conditions.
- Ridge regression provides modest improvements over plain linear regression.
- The ANN demonstrates improved predictive performance, particularly during transient regimes, though residual error remains non-negligible.

These results indicate that nonlinear modeling offers advantages, but does not eliminate prediction error entirely.

---

## Explainability and Error Analysis

Explainability techniques are used to understand model behavior and failure modes rather than solely to improve predictive accuracy.

- Linear and ridge models are interpreted using regression coefficients.
- ANN predictions are analyzed using SHAP-based feature attribution.
- Error behavior is examined across different operating regimes to identify conditions where model performance degrades.

---

## Project Scope and Limitations

- Data is simulation-based and does not include field or hardware measurements.
- Real-time MPPT implementation is not considered.
- ANN performance may degrade under distribution shifts not represented in the simulation data.

**Future extensions include:**

- Hardware-in-the-loop validation
- Real-time control integration
- Evaluation on field data

---

## Repository Structure

````text
pv-mpp-regression/
├── 01_models/
│ ├── backup/
│ │ ├── GenerateInputs.asv
│ │ └── po_mppt.asv
│ └── pv_mpp.slx
│
├── 02_src/
│ ├── slprj/
│ ├── ExtractMPP.m
│ ├── GenerateInputs.m
│ └── po_mppt.m
│
├── 03_data/
│ └── PV_results.csv
│
├── 04_analysis/
│ └── step_ip_analysis.ipynb
│
├── 05_ann/
│ ├── **pycache**/
│ ├── metrics/
│ ├── models/
│ ├── compute_pmpp.py
│ ├── train_baselines.py
│ ├── train_impp.py
│ ├── train_pmpp_direct.py
│ ├── train_vmpp.py
│ └── utils.py
│
├── 06_graphs/
│ ├── pmpp/
│ ├── impp_train_vs_val_loss.png
│ ├── predictedImpp_vs_actualImpp.png
│ ├── predictedVmpp_vs_actualVmpp.png
│ └── vmpp_train_vs_val_loss.png
│
├── slprj/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

---

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt

2. Run train_vmpp.py, train_impp.py and then compute_pmpp.py

````

A detailed methodological explanation and analysis is available in:

docs/technical_note.md

## Notes

- The repository is licensed under the **GNU General Public License v3.0 (GPL-3.0)**, ensuring that all modifications and derivative works remain open and reproducible.
