import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


REQUIRED_COLS = ['G', 'Temp', 'Vmpp', 'Impp', 'Pmpp']


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load PV MPP data from CSV and check required columns.
    """
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    return df


def prepare_X_y(df: pd.DataFrame):
    """
    Split dataframe into features X and target y.
    X: [G, Temp, Vmpp, Impp]
    y: [Pmpp]
    """
    X = df[['G', 'Temp', 'Vmpp', 'Impp']].values
    y = df['Pmpp'].values.reshape(-1, 1)
    return X, y


def train_test_scale(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Train–test split + MinMax scaling for X and y.

    Returns:
        X_train_scaled, X_test_scaled,
        y_train_scaled, y_test_scaled,
        scaler_X, scaler_y
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train_scaled,
        y_test_scaled,
        scaler_X,
        scaler_y,
    )


def build_model(input_dim: int = 4, learning_rate: float = 0.001):
    """
    Build and compile a simple ANN for Pmpp regression.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(4, activation='relu'),
        layers.Dense(1, activation='linear')  # Pmpp (regression)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    return model


def evaluate_regression(y_true, y_pred) -> dict:
    """
    Compute regression metrics (MSE, RMSE, MAE, R²) on ORIGINAL-scale data.

    y_true and y_pred must be in the same units (here: Pmpp in Watts).
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }
