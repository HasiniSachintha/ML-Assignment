import pandas as pd
import numpy as np
import os
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

import xgboost as xgb
from scipy.stats import randint, uniform


# ============================================================
# METRICS
# ============================================================

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    denom = np.where(np.abs(y_true) < 1.0, 1.0, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100


# ============================================================
# LOAD DATA
# ============================================================

def load_cleaned_data(file_path):
    print(f"Loading cleaned data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows")
    return df


# ============================================================
# FEATURE ENGINEERING (NO LEAKAGE)
# ============================================================

def prepare_features(df):
    print("Performing feature engineering...")

    df = df.sort_values(["District_encoded", "Year"]).copy()

    # Lag features
    df["Yield_Lag1"] = df.groupby("District_encoded")["Yield"].shift(1)
    df["Yield_Lag2"] = df.groupby("District_encoded")["Yield"].shift(2)

    # Rolling mean (shifted properly to avoid leakage)
    df["Yield_RollingMean3"] = (
        df.groupby("District_encoded")["Yield"]
        .shift(1)
        .rolling(3)
        .mean()
        .reset_index(0, drop=True)
    )

    # Trend
    df["Year_Index"] = df["Year"] - df["Year"].min()

    # Interaction
    df["Extent_x_PrevYield"] = df["Extent"] * df["Prev_Yield"]

    # Crisis indicator
    df["CrisisPeriod"] = (df["Year"] >= 2020).astype(int)

    # Remove rows with NaN from lagging
    df = df.dropna().reset_index(drop=True)

    feature_columns = [
        "Year_Index",
        "Extent",
        "Prev_Yield",
        "Yield_Lag1",
        "Yield_Lag2",
        "Yield_RollingMean3",
        "Extent_x_PrevYield",
        "District_encoded",
        "Season_encoded",
        "CrisisPeriod"
    ]

    X = df[feature_columns]
    y = df["Yield"]

    print(f"Final dataset size: {X.shape}")

    return X, y, df


# ============================================================
# SIMPLE FORECASTING SPLIT
# Train ≤ 2022 | Test = 2023
# ============================================================

def split_data(X, y, df):
    print("\nSimple forecasting setup:")
    print("Train ≤ 2022 | Test = 2023")

    train_mask = df["Year"] <= 2022
    test_mask = df["Year"] == 2023

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    print(f"Train years: {df[train_mask]['Year'].min()}–{df[train_mask]['Year'].max()}")
    print("Test year: 2023")

    return X_train, X_test, y_train, y_test


# ============================================================
# MODELS
# ============================================================

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    print("Training Random Forest...")

    model = RandomForestRegressor(
        n_estimators=600,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    print("Training XGBoost...")

    base_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    param_dist = {
        "n_estimators": randint(300, 700),
        "max_depth": randint(4, 8),
        "learning_rate": uniform(0.05, 0.1),
        "subsample": uniform(0.7, 0.3),
        "colsample_bytree": uniform(0.7, 0.3)
    }

    search = RandomizedSearchCV(
        base_model,
        param_dist,
        n_iter=25,
        scoring="neg_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    print("Best XGBoost params:", search.best_params_)
    return search.best_estimator_


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"\n{name}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

    return {
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "MAPE (%)": mape
    }


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_cleaned_data("data/processed/cleaned_ginger_data.csv")

    X, y, df_processed = prepare_features(df)

    X_train, X_test, y_train, y_test = split_data(X, y, df_processed)

    lr_model = train_linear_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    results = []
    results.append(evaluate_model(lr_model, X_test, y_test, "Linear Regression"))
    results.append(evaluate_model(rf_model, X_test, y_test, "Random Forest"))
    results.append(evaluate_model(xgb_model, X_test, y_test, "XGBoost"))

    results_df = pd.DataFrame(results)

    print("\nFinal Model Comparison:")
    print(results_df)

    # Feature Importance (RF)
    print("\nRandom Forest Feature Importance:")
    importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print(importance)

    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_model, "models/best_model.pkl")

    print("\nTraining complete.")
    print("Best model (Random Forest) saved.")


if __name__ == "__main__":
    main()