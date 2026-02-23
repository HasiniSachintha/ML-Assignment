import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import os


# ============================================================
# METRICS HELPER (Not required but useful if needed later)
# ============================================================

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    denom = np.where(np.abs(y_true) < 1.0, 1.0, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100


# ============================================================
# LOAD MODEL + DATA
# ============================================================

def load_model_and_data(model_path, data_path):
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    print("Model loaded successfully")

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Data loaded: {len(df)} rows")

    return model, df


# ============================================================
# FEATURE ENGINEERING (MUST MATCH train.py EXACTLY)
# ============================================================

def prepare_features(df):
    print("Recreating training features for SHAP...")

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

    return X, df, feature_columns


# ============================================================
# SAME SPLIT AS train.py
# Train ≤ 2022 | Test = 2023
# ============================================================

def prepare_test_data(X, df):
    print("Preparing SHAP test set (Year = 2023)...")

    test_mask = df["Year"] == 2023
    X_test = X[test_mask]

    print(f"Test set size (2023): {X_test.shape}")

    return X_test


# ============================================================
# SHAP EXPLAINER
# ============================================================

def calculate_shap(model, X_test):

    print("\nInitializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)

    print("Calculating SHAP values...")
    shap_values = explainer.shap_values(X_test)

    print(f"SHAP values shape: {np.array(shap_values).shape}")

    return explainer, shap_values


# ============================================================
# PLOTS
# ============================================================

def plot_shap_summary(shap_values, X_test, output_path):
    print("Generating SHAP summary plot...")

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_feature_importance(shap_values, feature_names, output_path):

    mean_shap = np.abs(shap_values).mean(0)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_|SHAP|": mean_shap
    }).sort_values("Mean_|SHAP|", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(importance_df["Feature"], importance_df["Mean_|SHAP|"])
    plt.xlabel("Mean |SHAP Value|")
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")

    print("\nSHAP Feature Ranking:")
    print(importance_df.sort_values("Mean_|SHAP|", ascending=False))


# ============================================================
# TEXT INTERPRETATION
# ============================================================

def generate_interpretation(shap_values, feature_names, output_path):

    mean_shap = np.abs(shap_values).mean(0)
    avg_shap = shap_values.mean(0)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_|SHAP|": mean_shap,
        "Avg_SHAP": avg_shap
    }).sort_values("Mean_|SHAP|", ascending=False)

    with open(output_path, "w") as f:
        f.write("SHAP EXPLAINABILITY REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(importance_df.to_string(index=False))
        f.write("\n\n")
        f.write("Interpretation:\n")
        f.write("- Higher |SHAP| = more important feature\n")
        f.write("- Positive Avg_SHAP increases yield prediction\n")
        f.write("- Negative Avg_SHAP decreases yield prediction\n")

    print(f"Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    model_path = "models/best_model.pkl"
    data_path = "data/processed/cleaned_ginger_data.csv"

    os.makedirs("reports", exist_ok=True)

    summary_plot_path = "reports/shap_summary.png"
    importance_plot_path = "reports/shap_importance.png"
    interpretation_path = "reports/shap_interpretation.txt"

    model, df = load_model_and_data(model_path, data_path)

    X, df_processed, feature_names = prepare_features(df)

    X_test = prepare_test_data(X, df_processed)

    explainer, shap_values = calculate_shap(model, X_test)

    plot_shap_summary(shap_values, X_test, summary_plot_path)
    plot_feature_importance(shap_values, feature_names, importance_plot_path)
    generate_interpretation(shap_values, feature_names, interpretation_path)

    print("\nSHAP Analysis Complete.")
    print("Results saved in 'reports/' folder.")


if __name__ == "__main__":
    main()