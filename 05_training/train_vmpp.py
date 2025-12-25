import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

import os
import joblib

print("Running from:", os.getcwd())

from utils import (
    load_data,
    prepare_X_y_vmpp,
    scale_data,
    build_model,
    evaluate_regression,
    append_metrics
)

# ====== CONFIG ======
DATA_PATH = "03_data/PV_results.csv"
EPOCHS = 100
BATCH_SIZE = 32
# ====================


def main():
    # Reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

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

    # 5. Build model
    vmpp_model = build_model(input_dim=X_train_scaled.shape[1])
    vmpp_model.summary()

    # 6. Train model with explicit validation
    history = vmpp_model.fit(
        X_train_scaled,
        y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    # 7. Plot training vs validation loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="Train Loss (MSE)")
    plt.plot(history.history["val_loss"], label="Validation Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss of Vmpp")
    plt.legend()
    plt.tight_layout()
    plt.savefig("06_graphs/vmpp_train_vs_val_loss.png")
    plt.close()

    # 8. Predictions (back to original scale)

    # --- Train ---
    y_train_pred_scaled = vmpp_model.predict(X_train_scaled)
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
    y_train_orig = scaler_y.inverse_transform(y_train_scaled)

    # --- Validation ---
    y_val_pred_scaled = vmpp_model.predict(X_val_scaled)
    y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled)
    y_val_orig = scaler_y.inverse_transform(y_val_scaled)

    # --- Test ---
    y_test_pred_scaled = vmpp_model.predict(X_test_scaled)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
    y_test_orig = scaler_y.inverse_transform(y_test_scaled)

    # 9. Metrics on ORIGINAL scale
    train_metrics = evaluate_regression(y_train_orig, y_train_pred)
    val_metrics = evaluate_regression(y_val_orig, y_val_pred)
    test_metrics = evaluate_regression(y_test_orig, y_test_pred)


    print("\n=== Train Metrics ===")
    for k, v in train_metrics.items():
        print(f"{k.upper():4s}: {v:.4f}")

    print("\n=== Validation Metrics ===")
    for k, v in val_metrics.items():
        print(f"{k.upper():4s}: {v:.4f}")

    print("\n=== Test Metrics ===")
    for k, v in test_metrics.items():
        print(f"{k.upper():4s}: {v:.4f}")

    # 10. Actual vs Predicted (Test set)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_orig, y_test_pred, s=15)
    min_val = min(y_test_orig.min(), y_test_pred.min())
    max_val = max(y_test_orig.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("Actual Vmpp (V)")
    plt.ylabel("Predicted Vmpp (V)")
    plt.title("Actual vs Predicted Vmpp (Test Set)")
    plt.tight_layout()
    plt.savefig("06_graphs/predictedVmpp_vs_actualVmpp_test.png")
    plt.close()

    # 11. Append test metrics to file
    results_path = "05_ann/metrics/vmpp_metrics.txt"

    append_metrics(
         filepath=results_path,
            metrics_dict={
        "MSE":  f"{test_metrics['mse']:.4f}",
        "RMSE": f"{test_metrics['rmse']:.4f}",
        "MAE":  f"{test_metrics['mae']:.4f}",
        "R2":   f"{test_metrics['r2']:.4f}",
    },
    notes="Final test set evaluation (unseen data)"
)

    print("Saved metrics to:", results_path)

    # 12. Save model
    os.makedirs("05_ann/models", exist_ok=True)
    vmpp_model.save("05_ann/models/vmpp_model.h5")
    
    # 13. Save scalers
    joblib.dump(scaler_X, "05_ann/models/vmpp_scaler_X.pkl")
    joblib.dump(scaler_y, "05_ann/models/vmpp_scaler_y.pkl")


if __name__ == "__main__":
    main()
