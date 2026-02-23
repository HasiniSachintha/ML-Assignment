import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def load_raw_data(file_path):
    """Load raw CSV file."""
    print(f"Loading raw data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

def remove_total_columns(df):
    """Remove Extent_Total and Production_Total columns."""
    print("Removing total columns...")
    df = df.drop(columns=['Extent_Total', 'Production_Total'], errors='ignore')
    return df

def reshape_to_long_format(df):
    """
    Convert wide format to long format.
    Creates Season column and combines Extent_Yala/Extent_Maha into Extent,
    and Production_Yala/Production_Maha into Production.
    """
    print("Reshaping from wide to long format...")
    
    # Create list to store long format rows
    long_data = []
    
    for _, row in df.iterrows():
        district = row['District']
        year = row['Year']
        
        # Process Yala season
        if pd.notna(row['Extent_Yala']) and row['Extent_Yala'] != 0:
            long_data.append({
                'District': district,
                'Year': year,
                'Season': 'Yala',
                'Extent': row['Extent_Yala'],
                'Production': row['Production_Yala']
            })
        
        # Process Maha season
        if pd.notna(row['Extent_Maha']) and row['Extent_Maha'] != 0:
            long_data.append({
                'District': district,
                'Year': year,
                'Season': 'Maha',
                'Extent': row['Extent_Maha'],
                'Production': row['Production_Maha']
            })
    
    df_long = pd.DataFrame(long_data)
    print(f"After reshaping: {len(df_long)} rows")
    return df_long

def compute_yield(df):
    """Compute Yield = Production / Extent."""
    print("Computing Yield...")
    df['Yield'] = df['Production'] / df['Extent']
    
    # Handle any division by zero or invalid values
    df['Yield'] = df['Yield'].replace([np.inf, -np.inf], np.nan)
    
    # Remove rows with invalid yield
    df = df.dropna(subset=['Yield'])
    
    print(f"After computing yield: {len(df)} rows")
    return df

def create_lag_feature(df):
    """
    Create Prev_Yield feature: previous year yield per district-season combination.
    Groups by District and Season, then shifts Yield by 1 year.
    """
    print("Creating lag feature Prev_Yield...")
    
    # Sort by District, Season, and Year
    df = df.sort_values(['District', 'Season', 'Year']).reset_index(drop=True)
    
    # Group by District and Season, then shift Yield by 1
    df['Prev_Yield'] = df.groupby(['District', 'Season'])['Yield'].shift(1)
    
    # Fill missing Prev_Yield values with median yield per district-season combination
    median_yield = df.groupby(['District', 'Season'])['Yield'].transform('median')
    df['Prev_Yield'] = df['Prev_Yield'].fillna(median_yield)
    
    # If still missing, fill with overall median
    if df['Prev_Yield'].isna().any():
        overall_median = df['Yield'].median()
        df['Prev_Yield'] = df['Prev_Yield'].fillna(overall_median)
    
    print(f"Prev_Yield created. Missing values: {df['Prev_Yield'].isna().sum()}")
    return df

def encode_categorical_variables(df):
    """
    Encode categorical variables: District and Season.
    Uses Label Encoding for simplicity.
    """
    print("Encoding categorical variables...")
    
    # Encode District
    district_encoder = LabelEncoder()
    df['District_encoded'] = district_encoder.fit_transform(df['District'])
    
    # Encode Season (Yala=0, Maha=1)
    season_encoder = LabelEncoder()
    df['Season_encoded'] = season_encoder.fit_transform(df['Season'])
    
    # Save encoders for later use (inference)
    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(district_encoder, 'models/district_encoder.pkl')
    joblib.dump(season_encoder, 'models/season_encoder.pkl')
    
    print(f"District encoding: {len(district_encoder.classes_)} unique districts")
    print(f"Season encoding: {season_encoder.classes_}")
    
    return df, district_encoder, season_encoder

def save_cleaned_data(df, output_path):
    """Save cleaned dataset to CSV."""
    print(f"Saving cleaned data to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved: {len(df)} rows, {len(df.columns)} columns")
    return df

def main():
    """Main preprocessing pipeline."""
    # File paths
    raw_data_path = 'data/raw/ginger_data.csv'
    output_path = 'data/processed/cleaned_ginger_data.csv'
    
    # Load raw data
    df = load_raw_data(raw_data_path)
    
    # Remove total columns
    df = remove_total_columns(df)
    
    # Reshape to long format
    df = reshape_to_long_format(df)
    
    # Compute yield
    df = compute_yield(df)
    
    # Create lag feature
    df = create_lag_feature(df)
    
    # Encode categorical variables
    df, district_encoder, season_encoder = encode_categorical_variables(df)
    
    # Save cleaned data
    save_cleaned_data(df, output_path)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("PREPROCESSING SUMMARY")
    print("="*50)
    print(f"Final dataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nYield statistics:")
    print(df['Yield'].describe())
    print(f"\nDistricts: {df['District'].nunique()}")
    print(f"Years: {df['Year'].min()} - {df['Year'].max()}")
    print(f"Seasons: {df['Season'].unique()}")
    print(f"\nMissing values:")
    print(df.isna().sum())
    
    return df

if __name__ == "__main__":
    df_cleaned = main()
