import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler
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
TEST_SIZE = 0.15
RANDOM_STATE = 42
RIDGE_ALPHA = 1.0
# ====================


def main():
    os.makedirs("06_graphs/pmpp_baselines", exist_ok=True)
    os.makedirs("05_ann/metrics", exist_ok=True)

    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Prepare datasets
    X_vmpp, y_vmpp = prepare_X_y_vmpp(df)
    X_impp, y_impp = prepare_X_y_impp(df)

    # 3. IDENTICAL train/test split (critical for fairness)
    X_train, X_test, y_vmpp_train, y_vmpp_test = train_test_split(
        X_vmpp, y_vmpp, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    _, _, y_impp_train, y_impp_test = train_test_split(
        X_impp, y_impp, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 4. Scale inputs (fit on TRAIN only)
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # =========================
    # Linear Regression
    # =========================
    lin_vmpp = LinearRegression()
    lin_impp = LinearRegression()

    lin_vmpp.fit(X_train_scaled, y_vmpp_train)
    lin_impp.fit(X_train_scaled, y_impp_train)

    Vmpp_pred_lin = lin_vmpp.predict(X_test_scaled)
    Impp_pred_lin = lin_impp.predict(X_test_scaled)

    Pmpp_pred_lin = Vmpp_pred_lin * Impp_pred_lin
    Pmpp_true = y_vmpp_test * y_impp_test

    lin_metrics = evaluate_regression(Pmpp_true, Pmpp_pred_lin)

    print("\n=== Linear Regression Pmpp (Computed) ===")
    for k, v in lin_metrics.items():
        print(f"{k.upper():4s}: {v:.4f}")

    append_metrics(
        filepath="05_ann/metrics/pmpp_linear_baseline_metrics.txt",
        metrics_dict={
            "MSE":  f"{lin_metrics['mse']:.4f}",
            "RMSE": f"{lin_metrics['rmse']:.4f}",
            "MAE":  f"{lin_metrics['mae']:.4f}",
            "R2":   f"{lin_metrics['r2']:.4f}",
        },
        notes="Pmpp computed from Linear Regression Vmpp × Impp"
    )

    # Plot: Linear baseline
    plt.figure(figsize=(6, 6))
    plt.scatter(Pmpp_true, Pmpp_pred_lin, s=15)
    min_val = min(Pmpp_true.min(), Pmpp_pred_lin.min())
    max_val = max(Pmpp_true.max(), Pmpp_pred_lin.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("Actual Pmpp (W)")
    plt.ylabel("Predicted Pmpp (W)")
    plt.title("Linear Baseline: Computed Pmpp (Test Set)")
    plt.tight_layout()
    plt.savefig("06_graphs/pmpp_baselines/pmpp_linear_pred_vs_actual.png")
    plt.close()

    # =========================
    # Ridge Regression
    # =========================
    ridge_vmpp = Ridge(alpha=RIDGE_ALPHA)
    ridge_impp = Ridge(alpha=RIDGE_ALPHA)

    ridge_vmpp.fit(X_train_scaled, y_vmpp_train)
    ridge_impp.fit(X_train_scaled, y_impp_train)

    Vmpp_pred_ridge = ridge_vmpp.predict(X_test_scaled)
    Impp_pred_ridge = ridge_impp.predict(X_test_scaled)

    Pmpp_pred_ridge = Vmpp_pred_ridge * Impp_pred_ridge

    ridge_metrics = evaluate_regression(Pmpp_true, Pmpp_pred_ridge)

    print("\n=== Ridge Regression Pmpp (Computed) ===")
    for k, v in ridge_metrics.items():
        print(f"{k.upper():4s}: {v:.4f}")

    append_metrics(
        filepath="05_ann/metrics/pmpp_ridge_baseline_metrics.txt",
        metrics_dict={
            "MSE":  f"{ridge_metrics['mse']:.4f}",
            "RMSE": f"{ridge_metrics['rmse']:.4f}",
            "MAE":  f"{ridge_metrics['mae']:.4f}",
            "R2":   f"{ridge_metrics['r2']:.4f}",
        },
        notes=f"Pmpp computed from Ridge Regression Vmpp × Impp (alpha={RIDGE_ALPHA})"
    )

    # Plot: Ridge baseline
    plt.figure(figsize=(6, 6))
    plt.scatter(Pmpp_true, Pmpp_pred_ridge, s=15)
    min_val = min(Pmpp_true.min(), Pmpp_pred_ridge.min())
    max_val = max(Pmpp_true.max(), Pmpp_pred_ridge.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("Actual Pmpp (W)")
    plt.ylabel("Predicted Pmpp (W)")
    plt.title("Ridge Baseline: Computed Pmpp (Test Set)")
    plt.tight_layout()
    plt.savefig("06_graphs/pmpp_baselines/pmpp_ridge_pred_vs_actual.png")
    plt.close()


if __name__ == "__main__":
    main()
