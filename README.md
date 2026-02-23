# Ginger Yield Prediction - Machine Learning Assignment

A machine learning project for predicting district-level ginger yield variations in Sri Lanka using XGBoost regression with SHAP explainability.

## Project Overview

This project implements a complete machine learning pipeline for agricultural yield prediction:

- **Dataset**: Sri Lankan ginger production data (2001-2025)
- **Algorithm**: XGBoost Regressor (with Linear Regression and Random Forest for comparison)
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **Frontend**: Streamlit web application

## Project Structure

```
Assignment/
├── data/
│   ├── raw/
│   │   └── ginger_data.csv          # Raw dataset
│   └── processed/
│       └── cleaned_ginger_data.csv  # Processed dataset (generated)
├── src/
│   ├── preprocess.py                 # Data preprocessing script
│   ├── train.py                      # Model training script
│   └── explain.py                    # SHAP explainability script
├── models/
│   ├── xgboost_model.pkl            # Trained XGBoost model (generated)
│   ├── district_encoder.pkl         # District label encoder (generated)
│   ├── season_encoder.pkl            # Season label encoder (generated)
│   └── feature_names.pkl             # Feature names (generated)
├── reports/
│   ├── shap_summary_plot.png         # SHAP summary plot (generated)
│   ├── shap_feature_importance.png   # Feature importance plot (generated)
│   ├── shap_dependence_plot.png      # Dependence plot (generated)
│   ├── shap_interpretation.txt       # Interpretation insights (generated)
│   └── model_comparison.txt          # Model comparison results (generated)
├── app/
│   └── streamlit_app.py              # Streamlit web application
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── report.md                         # Academic write-up
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone or Download the Project

Navigate to the project directory:

```bash
cd /path/to/Assignment
```

### Step 2: Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Data Preprocessing

Run the preprocessing script to clean and transform the raw data:

```bash
python src/preprocess.py
```

This script will:
- Load the raw CSV file
- Reshape data from wide to long format
- Remove rows with zero extent
- Compute Yield = Production / Extent
- Create Prev_Yield lag feature
- Encode categorical variables
- Save cleaned data to `data/processed/cleaned_ginger_data.csv`

**Output**: Cleaned dataset and encoders saved to `models/` directory

### Step 2: Model Training

Train the machine learning models:

```bash
python src/train.py
```

This script will:
- Load the cleaned dataset
- Split data into train/test sets (80/20)
- Train three models:
  - Linear Regression (baseline)
  - Random Forest Regressor
  - XGBoost Regressor (with hyperparameter tuning)
- Evaluate models using RMSE, MAE, R², and MAPE
- Save the best XGBoost model to `models/xgboost_model.pkl`
- Generate comparison table in `reports/model_comparison.txt`

**Output**: Trained models and evaluation metrics

### Step 3: SHAP Explainability Analysis

Generate SHAP explanations and visualizations:

```bash
python src/explain.py
```

This script will:
- Load the trained XGBoost model
- Calculate SHAP values for test set
- Generate visualizations:
  - SHAP summary plot
  - Feature importance plot
  - Dependence plot
- Save plots to `reports/` directory
- Generate interpretation insights

**Output**: SHAP plots and interpretation file in `reports/` directory

### Step 4: Run Streamlit Application

Launch the interactive web application:

```bash
streamlit run app/streamlit_app.py
```

The application will open in your default web browser. You can:
- Input district, year, season, extent, and previous yield
- Get yield predictions
- View SHAP explanations for predictions

**Note**: Make sure you've completed Steps 1-3 before running the Streamlit app, as it requires the trained model and processed data.

## Workflow Summary

```
Raw Data → Preprocessing → Training → Explainability → Frontend
   ↓            ↓             ↓            ↓              ↓
CSV file   Cleaned CSV   Model.pkl   SHAP plots    Web App
```

## Key Features

### Data Preprocessing
- Wide-to-long format conversion
- Lag feature creation (Prev_Yield)
- Categorical encoding
- Missing value handling

### Model Training
- Multiple model comparison
- Hyperparameter tuning (RandomizedSearchCV)
- Comprehensive evaluation metrics
- Model persistence

### Explainability
- SHAP TreeExplainer for XGBoost
- Multiple visualization types
- Feature importance analysis
- Interaction effect analysis

### Frontend
- Interactive prediction interface
- Real-time SHAP explanations
- User-friendly visualization
- Model information display

## Model Details

### Features Used
- **Year**: Temporal variable
- **Extent**: Cultivated area in hectares
- **Prev_Yield**: Previous year yield (lag feature)
- **District_encoded**: District label encoding
- **Season_encoded**: Season label encoding (Yala=0, Maha=1)

### Target Variable
- **Yield**: Production / Extent (tons/hectare)

### Evaluation Metrics
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **MAPE**: Mean Absolute Percentage Error

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Make sure you run scripts in the correct order (preprocess → train → explain)

2. **Import Errors**: Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

3. **SHAP Plotting Issues**: If SHAP plots fail, ensure matplotlib backend is set correctly:
   ```python
   import matplotlib
   matplotlib.use('Agg')  # For non-interactive backends
   ```

4. **Streamlit App Errors**: Ensure model files exist in `models/` directory before running the app

## Academic Report

See `report.md` for the complete academic write-up including:
- Problem definition
- Algorithm selection rationale
- Model evaluation strategy
- Explainability interpretation
- Critical discussion
- Limitations and ethical considerations

## License

This project is developed for academic purposes as part of a Machine Learning assignment.

## Contact

For questions or issues, please refer to the project documentation or contact the course instructor.

---

**Note**: This project uses real agricultural data from Sri Lanka. All predictions and analyses should be interpreted in the context of agricultural variability and data limitations.
