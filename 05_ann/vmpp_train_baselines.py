import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
import os

print("Running from:", os.getcwd())

from utils import (
    load_data,
    prepare_X_y_vmpp,
    scale_data,
    evaluate_regression,
    append_metrics,
)

# ====== CONFIG ======
DATA_PATH = "03_data/PV_results.csv"
RIDGE_ALPHA = 1.0
# ====================


def plot_actual_vs_pred(y_true, y_pred, title, save_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=15)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("Actual Vmpp (V)")
    plt.ylabel("Predicted Vmpp (V)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    np.random.seed(42)

    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Prepare X, y
    X, y = prepare_X_y_vmpp(df)

    # 3. Explicit Train / Validation / Test split (70 / 15 / 15)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42
    )

    # 4. Scale data (fit scalers on TRAIN only)
    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train_scaled,
        y_val_scaled,
        y_test_scaled,
        scaler_X,
        scaler_y,
    ) = scale_data(X_train, X_val, X_test, y_train, y_val, y_test)

    # sklearn expects 1D y
    y_train_scaled = y_train_scaled.ravel()
    y_test_scaled = y_test_scaled.ravel()

    y_test_orig = scaler_y.inverse_transform(
        y_test_scaled.reshape(-1, 1)
    )

    # ==================================================
    # LINEAR REGRESSION
    # ==================================================
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_train_scaled)

    y_test_pred_scaled = lin_reg.predict(X_test_scaled)
    y_test_pred = scaler_y.inverse_transform(
        y_test_pred_scaled.reshape(-1, 1)
    )

    lin_metrics = evaluate_regression(y_test_orig, y_test_pred)

    print("\n=== Linear Regression Test Metrics ===")
    for k, v in lin_metrics.items():
        print(f"{k.upper():4s}: {v:.4f}")

    plot_actual_vs_pred(
        y_test_orig,
        y_test_pred,
        "Linear Regression: Actual vs Predicted Vmpp (Test Set)",
        "06_graphs/vmpp_linear_regression_actual_vs_pred.png"
    )

    append_metrics(
        filepath="05_ann/metrics/vmpp_linear_regression_metrics.txt",
        metrics_dict={
            "MSE":  f"{lin_metrics['mse']:.4f}",
            "RMSE": f"{lin_metrics['rmse']:.4f}",
            "MAE":  f"{lin_metrics['mae']:.4f}",
            "R2":   f"{lin_metrics['r2']:.4f}",
        },
        notes="Linear regression baseline for vmpp(same split & scaling as ANN)"
    )

    # ==================================================
    # RIDGE REGRESSION
    # ==================================================
    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(X_train_scaled, y_train_scaled)

    y_test_pred_scaled = ridge.predict(X_test_scaled)
    y_test_pred = scaler_y.inverse_transform(
        y_test_pred_scaled.reshape(-1, 1)
    )

    ridge_metrics = evaluate_regression(y_test_orig, y_test_pred)

    print("\n=== Ridge Regression Test Metrics ===")
    for k, v in ridge_metrics.items():
        print(f"{k.upper():4s}: {v:.4f}")

    plot_actual_vs_pred(
        y_test_orig,
        y_test_pred,
        "Ridge Regression: Actual vs Predicted Vmpp (Test Set)",
        "06_graphs/vmpp_ridge_regression_actual_vs_pred.png"
    )

    append_metrics(
        filepath="05_ann/metrics/vmpp_ridge_regression_metrics.txt",
        metrics_dict={
            "MSE":  f"{ridge_metrics['mse']:.4f}",
            "RMSE": f"{ridge_metrics['rmse']:.4f}",
            "MAE":  f"{ridge_metrics['mae']:.4f}",
            "R2":   f"{ridge_metrics['r2']:.4f}",
        },
        notes=f"Ridge regression baseline for Vmpp (alpha={RIDGE_ALPHA})"
    )


if __name__ == "__main__":
    main()
