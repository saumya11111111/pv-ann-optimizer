import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import os
print("Running from:", os.getcwd())

from utils import (
    load_data,
    prepare_X_y,
    train_test_scale,
    build_model,
    evaluate_regression,
)

# ====== CONFIG ======
DATA_PATH = "03_data/PV_results.csv"   
EPOCHS = 100
BATCH_SIZE = 32
TEST_SIZE = 0.2
# ====================


def main():
    # Reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Prepare X, y
    X, y = prepare_X_y(df)

    # 3. Train–test split + scaling
    (
        X_train_scaled,
        X_test_scaled,
        y_train_scaled,
        y_test_scaled,
        scaler_X,
        scaler_y,
    ) = train_test_scale(X, y, test_size=TEST_SIZE)

    # 4. Build model
    model = build_model(input_dim=X_train_scaled.shape[1])
    model.summary()

    # 5. Train model
    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1,
    )

    # 6. Plot training vs validation loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="Train Loss (MSE)")
    plt.plot(history.history["val_loss"], label="Val Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 7. Predict on TRAIN and TEST sets (back to original scale)
    # --- Train set ---
    y_train_pred_scaled = model.predict(X_train_scaled)
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
    y_train_orig = scaler_y.inverse_transform(y_train_scaled)

    # --- Test set ---
    y_test_pred_scaled = model.predict(X_test_scaled)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
    y_test_orig = scaler_y.inverse_transform(y_test_scaled)

    # 8. Metrics on ORIGINAL DATA SCALE
    train_metrics = evaluate_regression(y_train_orig, y_train_pred)
    test_metrics = evaluate_regression(y_test_orig, y_test_pred)

    print("\n=== Train Metrics (original Pmpp units) ===")
    for name, value in train_metrics.items():
        print(f"{name.upper():4s}: {value:.4f}")

    print("\n=== Test Metrics (original Pmpp units) ===")
    for name, value in test_metrics.items():
        print(f"{name.upper():4s}: {value:.4f}")

    # 9. Actual vs Predicted Pmpp (Test data)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_orig, y_test_pred, s=15)
    min_val = min(y_test_orig.min(), y_test_pred.min())
    max_val = max(y_test_orig.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("Actual Pmpp (W)")
    plt.ylabel("Predicted Pmpp (W)")
    plt.title("Actual vs Predicted Pmpp (Test Set)")
    plt.tight_layout()
    plt.show()

    results_path = "05_ann/ann_test_metrics.txt"
    mse_test = test_metrics['mse']
    rmse_test = test_metrics['rmse']
    mae_test = test_metrics['mae']
    r2_test = test_metrics['r2']

    with open(results_path, "w") as f:
        f.write("=== ANN Test Metrics ===\n")
        f.write(f"MSE  : {mse_test:.4f}\n")
        f.write(f"RMSE : {rmse_test:.4f}\n")
        f.write(f"MAE  : {mae_test:.4f}\n")
        f.write(f"R²   : {r2_test:.4f}\n")

    print("Saved metrics to:", results_path)


if __name__ == "__main__":
    main()

    
