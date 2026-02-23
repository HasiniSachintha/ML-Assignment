import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import os

def load_model_and_data(model_path, data_path):
    """Load trained model and dataset."""
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    print("Model loaded successfully")
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Data loaded: {len(df)} rows")
    
    return model, df

def prepare_test_data(df, feature_names):
    """Prepare test data with same features used in training."""
    print("Preparing test data...")
    X_test = df[feature_names].copy()
    print(f"Test data shape: {X_test.shape}")
    return X_test

def initialize_shap_explainer(model, X_test, sample_size=100):
    """
    Initialize SHAP TreeExplainer and calculate SHAP values.
    Uses a sample of test data for efficiency if dataset is large.
    """
    print("\n" + "="*50)
    print("Initializing SHAP TreeExplainer...")
    print("="*50)
    
    # Sample data if too large (for faster computation)
    if len(X_test) > sample_size:
        print(f"Sampling {sample_size} instances from test set for SHAP analysis...")
        X_sample = X_test.sample(n=sample_size, random_state=42)
    else:
        X_sample = X_test
        print(f"Using all {len(X_sample)} instances for SHAP analysis")
    
    # Initialize TreeExplainer (optimized for tree-based models)
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    print("Calculating SHAP values...")
    shap_values = explainer.shap_values(X_sample)
    
    print(f"SHAP values calculated: shape {shap_values.shape}")
    
    return explainer, shap_values, X_sample

def plot_shap_summary(shap_values, X_sample, feature_names, output_path):
    """
    Generate SHAP summary plot showing feature importance and impact direction.
    """
    print(f"\nGenerating SHAP summary plot...")
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plot saved to {output_path}")

def plot_feature_importance(shap_values, feature_names, output_path):
    """
    Generate feature importance plot (bar plot of mean absolute SHAP values).
    """
    print(f"\nGenerating feature importance plot...")
    
    # Calculate mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(0)
    
    # Create DataFrame for easier plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_shap
    }).sort_values('Importance', ascending=True)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Mean |SHAP Value|', fontsize=12)
    plt.title('Feature Importance (SHAP)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature importance plot saved to {output_path}")
    
    # Print feature importance ranking
    print("\nFeature Importance Ranking:")
    print(importance_df.to_string(index=False))

def plot_dependence(shap_values, X_sample, feature_names, output_path):
    """
    Generate SHAP dependence plot showing interaction effects.
    Plots dependence for the most important feature.
    """
    print(f"\nGenerating SHAP dependence plot...")
    
    # Find most important feature
    mean_shap = np.abs(shap_values).mean(0)
    most_important_idx = np.argmax(mean_shap)
    most_important_feature = feature_names[most_important_idx]
    
    print(f"Most important feature: {most_important_feature}")
    
    # Create dependence plot
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        most_important_idx,
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False
    )
    plt.title(f'SHAP Dependence Plot: {most_important_feature}', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Dependence plot saved to {output_path}")

def generate_interpretation_insights(shap_values, X_sample, feature_names, output_path):
    """
    Generate text file with interpretation insights from SHAP analysis.
    """
    print(f"\nGenerating interpretation insights...")
    
    # Calculate feature importance
    mean_shap = np.abs(shap_values).mean(0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_|SHAP|': mean_shap
    }).sort_values('Mean_|SHAP|', ascending=False)
    
    # Calculate average SHAP values (direction of impact)
    avg_shap = shap_values.mean(0)
    
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("SHAP EXPLAINABILITY ANALYSIS\n")
        f.write("="*60 + "\n\n")
        
        f.write("FEATURE IMPORTANCE RANKING\n")
        f.write("-"*60 + "\n")
        f.write(importance_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("FEATURE IMPACT DIRECTION\n")
        f.write("-"*60 + "\n")
        f.write("Positive SHAP values increase predicted yield\n")
        f.write("Negative SHAP values decrease predicted yield\n\n")
        
        for i, feature in enumerate(feature_names):
            f.write(f"{feature}:\n")
            f.write(f"  Mean SHAP value: {avg_shap[i]:.4f}\n")
            f.write(f"  Mean |SHAP| value: {mean_shap[i]:.4f}\n")
            if avg_shap[i] > 0:
                f.write(f"  Impact: Generally increases yield\n")
            else:
                f.write(f"  Impact: Generally decreases yield\n")
            f.write("\n")
        
        f.write("="*60 + "\n")
        f.write("INTERPRETATION NOTES\n")
        f.write("="*60 + "\n")
        f.write("1. Higher absolute SHAP values indicate greater feature importance\n")
        f.write("2. Positive SHAP values suggest the feature increases yield prediction\n")
        f.write("3. Negative SHAP values suggest the feature decreases yield prediction\n")
        f.write("4. Dependence plots show how feature interactions affect predictions\n")
    
    print(f"Interpretation insights saved to {output_path}")

def main():
    """Main explainability pipeline."""
    # File paths
    model_path = 'models/xgboost_model.pkl'
    data_path = 'data/processed/cleaned_ginger_data.csv'
    
    # Output paths
    summary_plot_path = 'reports/shap_summary_plot.png'
    importance_plot_path = 'reports/shap_feature_importance.png'
    dependence_plot_path = 'reports/shap_dependence_plot.png'
    insights_path = 'reports/shap_interpretation.txt'
    
    # Create reports directory
    os.makedirs('reports', exist_ok=True)
    
    # Load model and data
    model, df = load_model_and_data(model_path, data_path)
    
    # Load feature names
    feature_names = joblib.load('models/feature_names.pkl')
    print(f"Feature names: {feature_names}")
    
    # Prepare test data
    X_test = prepare_test_data(df, feature_names)
    
    # Initialize SHAP explainer and calculate values
    explainer, shap_values, X_sample = initialize_shap_explainer(model, X_test)
    
    # Generate visualizations
    plot_shap_summary(shap_values, X_sample, feature_names, summary_plot_path)
    plot_feature_importance(shap_values, feature_names, importance_plot_path)
    plot_dependence(shap_values, X_sample, feature_names, dependence_plot_path)
    
    # Generate interpretation insights
    generate_interpretation_insights(shap_values, X_sample, feature_names, insights_path)
    
    print("\n" + "="*50)
    print("SHAP ANALYSIS COMPLETE")
    print("="*50)
    print(f"Summary plot: {summary_plot_path}")
    print(f"Feature importance plot: {importance_plot_path}")
    print(f"Dependence plot: {dependence_plot_path}")
    print(f"Interpretation insights: {insights_path}")
    
    return explainer, shap_values, X_sample

if __name__ == "__main__":
    explainer, shap_values, X_sample = main()
