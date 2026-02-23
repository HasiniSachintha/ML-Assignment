import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Page configuration
st.set_page_config(
    page_title="Ginger Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 1rem 0;
    }
    .prediction-box {
        background-color: #E8F5E9;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_encoders():
    """Load trained model and encoders (cached for performance)."""
    try:
        model = joblib.load('models/xgboost_model.pkl')
        district_encoder = joblib.load('models/district_encoder.pkl')
        season_encoder = joblib.load('models/season_encoder.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return model, district_encoder, season_encoder, feature_names
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please run preprocessing and training scripts first. Error: {e}")
        st.stop()

@st.cache_resource
def load_shap_explainer():
    """Load SHAP explainer (cached for performance)."""
    try:
        model = joblib.load('models/xgboost_model.pkl')
        # Load sample data for SHAP background
        df = pd.read_csv('data/processed/cleaned_ginger_data.csv')
        feature_names = joblib.load('models/feature_names.pkl')

        # Recreate any derived features used during training but not stored
        # directly in the cleaned CSV (e.g., CrisisPeriod).
        if 'Year' in df.columns and 'CrisisPeriod' in feature_names and 'CrisisPeriod' not in df.columns:
            df['CrisisPeriod'] = (df['Year'] >= 2020).astype(int)

        X_sample = df[feature_names].sample(min(100, len(df)), random_state=42)
        explainer = shap.TreeExplainer(model)
        return explainer, X_sample
    except Exception as e:
        st.warning(f"Could not load SHAP explainer: {e}")
        return None, None

def get_available_districts():
    """Get list of available districts from the dataset."""
    try:
        df = pd.read_csv('data/processed/cleaned_ginger_data.csv')
        districts = sorted(df['District'].unique().tolist())
        return districts
    except:
        return ['Colombo', 'Gampaha', 'Kalutara', 'Kandy', 'Matale', 'Nuwara Eliya', 
                'Galle', 'Matara', 'Hambantota', 'Jaffna', 'Vanni', 'Batticaloa', 
                'Digamadulla', 'Trincomalee', 'Kurunegala', 'Puttalam', 'Anuradhapura', 
                'Polonnaruwa', 'Badulla', 'Moneragala', 'Ratnapura', 'Kegalle']

def preprocess_input(district, year, season, extent, prev_yield, district_encoder, season_encoder):
    """Preprocess user input to match model format."""
    # Encode district
    district_encoded = district_encoder.transform([district])[0]

    # Encode season
    season_encoded = season_encoder.transform([season])[0]

    # Crisis indicator for post-2020 years, to mirror the training pipeline
    crisis_period = 1 if year >= 2020 else 0

    # Create feature array matching the training feature order
    # [Year, Extent, Prev_Yield, District_encoded, Season_encoded, CrisisPeriod]
    features = np.array([[year, extent, prev_yield, district_encoded, season_encoded, crisis_period]])

    return features

def predict_yield(model, features):
    """Predict yield using the trained model."""
    prediction = model.predict(features)[0]
    return max(0, prediction)  # Ensure non-negative yield

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<p class="main-header">🌾 Ginger Yield Prediction System</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model and encoders
    model, district_encoder, season_encoder, feature_names = load_model_and_encoders()
    
    # Sidebar for inputs
    st.sidebar.header("📝 Input Parameters")
    
    # Get available districts
    available_districts = get_available_districts()
    
    # User inputs
    district = st.sidebar.selectbox(
        "District",
        options=available_districts,
        index=0
    )
    
    year = st.sidebar.number_input(
        "Year",
        min_value=2000,
        max_value=2030,
        value=2023,
        step=1
    )
    
    season = st.sidebar.selectbox(
        "Season",
        options=['Yala', 'Maha'],
        index=1
    )
    
    extent = st.sidebar.number_input(
        "Extent (hectares)",
        min_value=0.0,
        value=100.0,
        step=1.0,
        format="%.2f"
    )
    
    prev_yield = st.sidebar.number_input(
        "Previous Year Yield (optional)",
        min_value=0.0,
        value=10.0,
        step=0.1,
        format="%.2f",
        help="Yield from the previous year for the same district-season combination"
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🔮 Prediction")
        
        if st.button("Predict Yield", type="primary"):
            # Preprocess input
            features = preprocess_input(
                district, year, season, extent, prev_yield,
                district_encoder, season_encoder
            )
            
            # Predict
            predicted_yield = predict_yield(model, features)
            
            # Display prediction
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.metric(
                label="Predicted Yield",
                value=f"{predicted_yield:.2f}",
                delta=f"{predicted_yield:.2f} tons/hectare"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Store prediction for SHAP explanation
            st.session_state['prediction'] = predicted_yield
            st.session_state['features'] = features
            st.session_state['input_data'] = {
                'district': district,
                'year': year,
                'season': season,
                'extent': extent,
                'prev_yield': prev_yield
            }
    
    with col2:
        st.header("📊 Model Information")
        
        st.info("""
        **Model:** XGBoost Regressor
        
        **Features Used:**
        - Year
        - Extent (hectares)
        - Previous Year Yield
        - District (encoded)
        - Season (encoded)
        
        **Target:** Yield (tons/hectare)
        """)
        
        # Display feature importance if available
        try:
            st.subheader("Feature Importance")
            # Load feature importance from SHAP if available
            if os.path.exists('reports/shap_interpretation.txt'):
                with open('reports/shap_interpretation.txt', 'r') as f:
                    content = f.read()
                    # Extract feature importance section
                    if 'FEATURE IMPORTANCE RANKING' in content:
                        st.text("Check reports/shap_interpretation.txt for detailed feature importance")
        except:
            pass
    
    # SHAP Explanation Section
    if 'prediction' in st.session_state:
        st.markdown("---")
        st.header("🔍 SHAP Explanation")
        
        explainer, X_sample = load_shap_explainer()
        
        if explainer is not None and X_sample is not None:
            try:
                # Calculate SHAP values for the input
                shap_values = explainer.shap_values(st.session_state['features'])
                
                # Create DataFrame for display
                input_df = pd.DataFrame(
                    st.session_state['features'],
                    columns=feature_names
                )
                
                # Display input features
                st.subheader("Input Features")
                display_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Value': st.session_state['features'][0]
                })
                st.dataframe(display_df, use_container_width=True)
                
                # SHAP values visualization
                st.subheader("SHAP Values (Feature Contributions)")
                shap_df = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP Value': shap_values[0]
                }).sort_values('SHAP Value', key=abs, ascending=False)
                
                st.dataframe(shap_df, use_container_width=True)
                
                # Waterfall plot
                st.subheader("SHAP Waterfall Plot")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[0],
                        base_values=explainer.expected_value,
                        data=st.session_state['features'][0],
                        feature_names=feature_names
                    ),
                    show=False
                )
                st.pyplot(fig)
                plt.close()
                
                # Interpretation
                st.subheader("Interpretation")
                st.info(f"""
                **Base Value:** {explainer.expected_value:.2f} tons/hectare
                
                **Predicted Value:** {st.session_state['prediction']:.2f} tons/hectare
                
                Features with positive SHAP values increase the prediction.
                Features with negative SHAP values decrease the prediction.
                """)
                
            except Exception as e:
                st.error(f"Error generating SHAP explanation: {e}")
        else:
            st.warning("SHAP explainer not available. Please run explain.py script first.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Machine Learning Assignment - Ginger Yield Prediction</p>
        <p>Dataset: Sri Lankan Agricultural Data</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
