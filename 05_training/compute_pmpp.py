import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import joblib
import os

from utils import (
    load_data,
    prepare_X_y_vmpp,
    prepare_X_y_impp,
    evaluate_regression,
    append_metrics,
)

# ====== CONFIG ======
DATA_PATH = "03_data/PV_results.csv"
VMPP_MODEL_PATH = "05_ann/models/vmpp_model.h5"
IMPP_MODEL_PATH = "05_ann/models/impp_model.h5"
TEST_SIZE = 0.15
RANDOM_STATE = 42
# ====================


def main():
    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Prepare Vmpp and Impp datasets
    X_vmpp, y_vmpp = prepare_X_y_vmpp(df)
    X_impp, y_impp = prepare_X_y_impp(df)

    # 3. IDENTICAL test split (must match training)
    _, X_test, _, y_vmpp_test = train_test_split(
        X_vmpp, y_vmpp, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    _, _, _, y_impp_test = train_test_split(
        X_impp, y_impp, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 4. Load trained models (NO recompiling)
    vmpp_model = load_model(VMPP_MODEL_PATH, compile=False)
    impp_model = load_model(IMPP_MODEL_PATH, compile=False)

    # 5. Load scalers saved during training
    vmpp_scaler_X = joblib.load("05_ann/models/vmpp_scaler_X.pkl")
    vmpp_scaler_y = joblib.load("05_ann/models/vmpp_scaler_y.pkl")

    impp_scaler_X = joblib.load("05_ann/models/impp_scaler_X.pkl")
    impp_scaler_y = joblib.load("05_ann/models/impp_scaler_y.pkl")

    # 6. Scale TEST inputs using training scalers
    X_test_scaled = vmpp_scaler_X.transform(X_test)

    # 7. Predict Vmpp and Impp
    Vmpp_pred = vmpp_scaler_y.inverse_transform(
        vmpp_model.predict(X_test_scaled)
    )

    Impp_pred = impp_scaler_y.inverse_transform(
        impp_model.predict(X_test_scaled)
    )

    # 8. Compute Pmpp using physics
    Pmpp_pred = Vmpp_pred * Impp_pred
    Pmpp_true = y_vmpp_test * y_impp_test

    # 9. Evaluate Pmpp on TEST set
    pmpp_metrics = evaluate_regression(Pmpp_true, Pmpp_pred)

    print("\n=== Computed Pmpp Test Metrics ===")
    for k, v in pmpp_metrics.items():
        print(f"{k.upper():4s}: {v:.4f}")

    # 10. Plot Actual vs Predicted Pmpp
    os.makedirs("06_graphs/pmpp", exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(Pmpp_true, Pmpp_pred, s=15)
    min_val = min(Pmpp_true.min(), Pmpp_pred.min())
    max_val = max(Pmpp_true.max(), Pmpp_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("Actual Pmpp (W)")
    plt.ylabel("Predicted Pmpp (W)")
    plt.title("Computed Pmpp: Actual vs Predicted (Test Set)")
    plt.tight_layout()
    plt.savefig("06_graphs/pmpp/predicted_vs_actual_pmpp.png")
    plt.close()

    # 11. Log metrics
    append_metrics(
        filepath="05_ann/metrics/pmpp_computed_metrics.txt",
        metrics_dict={
            "MSE":  f"{pmpp_metrics['mse']:.4f}",
            "RMSE": f"{pmpp_metrics['rmse']:.4f}",
            "MAE":  f"{pmpp_metrics['mae']:.4f}",
            "R2":   f"{pmpp_metrics['r2']:.4f}",
        },
        notes="Pmpp computed from predicted Vmpp × Impp (test set)"
    )

    print("Saved computed Pmpp metrics and plot.")


if __name__ == "__main__":
    main()

