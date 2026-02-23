import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import os
from scipy.stats import randint, uniform

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def load_cleaned_data(file_path):
    """Load cleaned dataset."""
    print(f"Loading cleaned data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows")
    return df

def prepare_features(df):
    """
    Prepare features and target variable.
    Features: Year, Extent, Prev_Yield, District_encoded, Season_encoded
    Target: Yield
    """
    print("Preparing features and target...")
    
    feature_columns = ['Year', 'Extent', 'Prev_Yield', 'District_encoded', 'Season_encoded']
    X = df[feature_columns].copy()
    y = df['Yield'].copy()
    
    print(f"Features: {feature_columns}")
    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets using a random 80/20 split."""
    print(f"Splitting data: {1-test_size:.0%} train, {test_size:.0%} test (random split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test

def train_linear_regression(X_train, y_train):
    """Train Linear Regression model (baseline)."""
    print("\n" + "="*50)
    print("Training Linear Regression (Baseline)...")
    print("="*50)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression trained successfully")
    return model

def train_random_forest(X_train, y_train):
    """Train Random Forest Regressor."""
    print("\n" + "="*50)
    print("Training Random Forest Regressor...")
    print("="*50)
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Random Forest trained successfully")
    return model

def train_xgboost_with_tuning(X_train, y_train):
    """
    Train XGBoost Regressor with hyperparameter tuning using RandomizedSearchCV.
    """
    print("\n" + "="*50)
    print("Training XGBoost Regressor with Hyperparameter Tuning...")
    print("="*50)
    
    # Base XGBoost model
    base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    
    # Hyperparameter search space
    param_distributions = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 5)
    }
    
    print("Performing RandomizedSearchCV...")
    print(f"Parameter distributions: {param_distributions}")
    
    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=50,  # Number of parameter settings sampled
        cv=5,  # 5-fold cross-validation
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    print("\nBest parameters found:")
    print(random_search.best_params_)
    print(f"Best CV score (neg MSE): {random_search.best_score_:.4f}")
    
    best_model = random_search.best_estimator_
    print("XGBoost trained successfully")
    
    return best_model, random_search.best_params_

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model using RMSE, MAE, R², and MAPE.
    """
    print(f"\nEvaluating {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    metrics = {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE (%)': mape
    }
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    return metrics, y_pred

def save_model(model, file_path):
    """Save trained model to file."""
    print(f"\nSaving model to {file_path}...")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    print("Model saved successfully")

def save_comparison_table(results, output_path):
    """Save model comparison table to file."""
    print(f"\nSaving comparison table to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df_results = pd.DataFrame(results)
    df_results = df_results.round(4)
    
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL COMPARISON RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(df_results.to_string(index=False))
        f.write("\n\n")
        f.write("="*60 + "\n")
        f.write("METRICS EXPLANATION\n")
        f.write("="*60 + "\n")
        f.write("RMSE (Root Mean Squared Error): Lower is better\n")
        f.write("MAE (Mean Absolute Error): Lower is better\n")
        f.write("R² (Coefficient of Determination): Higher is better (max 1.0)\n")
        f.write("MAPE (Mean Absolute Percentage Error): Lower is better\n")
    
    print("Comparison table saved")
    print("\n" + df_results.to_string(index=False))

def main():
    """Main training pipeline."""
    # File paths
    cleaned_data_path = 'data/processed/cleaned_ginger_data.csv'
    model_output_path = 'models/xgboost_model.pkl'
    comparison_output_path = 'reports/model_comparison.txt'
    
    # Load cleaned data
    df = load_cleaned_data(cleaned_data_path)
    
    # Prepare features and target
    X, y = prepare_features(df)
    
    # Split data (random 80/20 split)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train models
    lr_model = train_linear_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    xgb_model, best_params = train_xgboost_with_tuning(X_train, y_train)
    
    # Evaluate models
    results = []
    
    lr_metrics, _ = evaluate_model(lr_model, X_test, y_test, "Linear Regression")
    results.append(lr_metrics)
    
    rf_metrics, _ = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    results.append(rf_metrics)
    
    xgb_metrics, _ = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    results.append(xgb_metrics)
    
    # Save comparison table
    save_comparison_table(results, comparison_output_path)
    
    # Save best model (XGBoost)
    save_model(xgb_model, model_output_path)
    
    # Save feature names for inference
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Best model (XGBoost) saved to: {model_output_path}")
    print(f"Comparison results saved to: {comparison_output_path}")
    
    return xgb_model, results

if __name__ == "__main__":
    model, results = main()
