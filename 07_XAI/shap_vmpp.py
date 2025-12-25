import os
import sys
import shap
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ---- Add project training utils to path ----
sys.path.append(os.path.abspath("05_training"))

from utils import load_data, prepare_X_y_vmpp

# ---- Create output directory ----
os.makedirs("06_graphs/explainable_graphs", exist_ok=True)

# ---- Load trained Vmpp model ----
model = load_model("05_training/models/vmpp_model.h5", compile=False)

# ---- Load data ----
df = load_data("03_data/PV_results.csv")
X, y = prepare_X_y_vmpp(df)   # X = [G, Temp]

# ---- Train-test split (same as training) ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# ---- Scale X (fit on TRAIN only) ----
scaler_X = MinMaxScaler()
X_train_s = scaler_X.fit_transform(X_train)
X_test_s = scaler_X.transform(X_test)

# ---- SHAP explainer ----
explainer = shap.Explainer(
    model,
    X_train_s[:100],          # background
    feature_names=["G", "Temp"]
)

# ---- Explain ENTIRE test set ----
shap_values = explainer(X_test_s)

# ---- Save summary plot ----
plt.figure()
shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()
plt.savefig(
    "06_graphs/explainable_graphs/shap_vmpp_summary_full_test.png",
    dpi=300
)
plt.close()

print("Saved SHAP Vmpp summary plot (full test set).")