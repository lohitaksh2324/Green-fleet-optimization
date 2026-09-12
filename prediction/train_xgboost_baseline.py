"""
Deliverable 1 baseline: XGBoost fuel consumption predictor.
Trains on the synthetic (EU-MRV-calibrated) fleet voyage dataset and reports
RMSE / MAE / R^2 as specified in the plan (Day 5).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib

DATA_PATH = "../data/synthetic_fleet_voyages.csv"
MODEL_PATH = "../data/xgb_fuel_model.joblib"

NUMERIC_FEATURES = [
    "displacement_tons", "cargo_load_pct", "speed_knots",
    "distance_nm", "voyage_hours",
]
CATEGORICAL_FEATURES = ["ship_type", "fuel_type", "weather"]
TARGET = "fuel_kg"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    # power_kw and co2_kg/cost_usd are downstream of fuel_kg / leak the target — excluded from features
    return df


def build_pipeline():
    preprocessor = ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def main():
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = np.log1p(df[TARGET])  # log-target: fuel_kg spans orders of magnitude across ship types

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    preds_log = pipe.predict(X_test)
    y_test_actual = np.expm1(y_test)
    preds = np.expm1(preds_log)

    rmse = np.sqrt(mean_squared_error(y_test_actual, preds))
    mae = mean_absolute_error(y_test_actual, preds)
    r2 = r2_score(y_test_actual, preds)
    mape = np.mean(np.abs((y_test_actual - preds) / y_test_actual)) * 100

    print(f"Test set size: {len(y_test)}")
    print(f"RMSE: {rmse:.2f} kg")
    print(f"MAE:  {mae:.2f} kg")
    print(f"R^2:  {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # per-ship-type error breakdown (useful for report + spotting where the model struggles)
    breakdown = X_test.copy()
    breakdown["actual"] = y_test_actual.values
    breakdown["pred"] = preds
    breakdown["abs_pct_err"] = np.abs((breakdown.actual - breakdown.pred) / breakdown.actual) * 100
    print("\nMAPE by ship_type:")
    print(breakdown.groupby("ship_type")["abs_pct_err"].mean().round(2))

    joblib.dump(pipe, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


if __name__ == "__main__":
    main()
