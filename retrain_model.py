#!/usr/bin/env python3
"""
Retrain the Random Forest model properly to fix the stuck prediction issue
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

print("=== RETRAINING THE MODEL ===")

# Load and clean the data
print("Loading data...")
df = pd.read_csv('redfin_properties_all_cities.csv')
print(f"Original data shape: {df.shape}")

# Clean the data (simplified version)
def clean_price(price):
    if pd.isna(price) or price == 0:
        return 0
    if isinstance(price, str):
        cleaned = price.replace('$', '').replace(',', '').strip()
        try:
            return float(cleaned)
        except:
            return 0
    return float(price)

def clean_numeric(value):
    if pd.isna(value):
        return 0
    if isinstance(value, str):
        import re
        numbers = re.findall(r'\d+\.?\d*', str(value))
        if numbers:
            return float(numbers[0])
        else:
            return 0
    try:
        return float(value)
    except:
        return 0

# Apply cleaning
df['Price'] = df['Price'].apply(clean_price)
df['Beds'] = df['Beds'].apply(clean_numeric)
df['Baths'] = df['Baths'].apply(clean_numeric) 
df['Sqft'] = df['Sqft'].apply(clean_numeric)

# Filter valid data
valid_mask = (
    (df['Price'] > 50000) & (df['Price'] < 2000000) &  # Reasonable price range
    (df['Beds'] >= 1) & (df['Beds'] <= 8) &            # Reasonable bedroom count
    (df['Baths'] >= 1) & (df['Baths'] <= 6) &          # Reasonable bathroom count
    (df['Sqft'] > 500) & (df['Sqft'] < 6000)           # Reasonable square footage
)

df_clean = df[valid_mask].copy()
print(f"After filtering: {df_clean.shape}")

if df_clean.shape[0] < 100:
    print("Not enough data after filtering!")
    exit(1)

# Feature engineering
print("Engineering features...")
df_clean['Total_rooms'] = df_clean['Beds'] + df_clean['Baths']
df_clean['Bath_bed_ratio'] = df_clean['Baths'] / df_clean['Beds']
df_clean['Sqft_per_room'] = df_clean['Sqft'] / df_clean['Total_rooms']
df_clean['Log_sqft'] = np.log1p(df_clean['Sqft'])
df_clean['Log_price'] = np.log1p(df_clean['Price'])

# Interaction features
df_clean['Beds_Baths_interaction'] = df_clean['Beds'] * df_clean['Baths']
df_clean['Beds_Sqft_interaction'] = df_clean['Beds'] * df_clean['Sqft']
df_clean['Baths_Sqft_interaction'] = df_clean['Baths'] * df_clean['Sqft']

# Define features
feature_columns = [
    'Beds', 'Baths', 'Sqft', 'Total_rooms', 'Bath_bed_ratio',
    'Sqft_per_room', 'Log_sqft', 'Beds_Baths_interaction',
    'Beds_Sqft_interaction', 'Baths_Sqft_interaction'
]

X = df_clean[feature_columns].copy()
y = df_clean['Log_price'].copy()

# Remove any remaining invalid entries
valid_indices = np.isfinite(X).all(axis=1) & np.isfinite(y)
X = X[valid_indices]
y = y[valid_indices]

print(f"Final training data: {X.shape[0]} samples, {X.shape[1]} features")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Train a new Random Forest model
print("Training new Random Forest model...")
rf_model = RandomForestRegressor(
    n_estimators=100,  # Reduced for faster training
    max_depth=20,      # Increased depth
    min_samples_split=2,  # Allow more splits
    min_samples_leaf=1,   # Allow smaller leaves
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Test the model
print("Testing model...")
y_pred = rf_model.predict(X_test)

# Convert back to actual prices
y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(y_pred)

# Calculate metrics
r2 = r2_score(y_test_actual, y_pred_actual)
mae = mean_absolute_error(y_test_actual, y_pred_actual)

print(f"Model performance:")
print(f"  R² Score: {r2:.4f}")
print(f"  MAE: ${mae:,.0f}")

# Test with different inputs to ensure variation
print("\nTesting prediction variation:")
def test_prediction(beds, baths, sqft):
    # Feature engineering for single prediction
    total_rooms = beds + baths
    bath_bed_ratio = baths / beds
    sqft_per_room = sqft / total_rooms
    log_sqft = np.log1p(sqft)
    beds_baths_interaction = beds * baths
    beds_sqft_interaction = beds * sqft
    baths_sqft_interaction = baths * sqft
    
    features = pd.DataFrame([[
        beds, baths, sqft, total_rooms, bath_bed_ratio,
        sqft_per_room, log_sqft, beds_baths_interaction,
        beds_sqft_interaction, baths_sqft_interaction
    ]], columns=feature_columns)
    
    log_pred = rf_model.predict(features)[0]
    return np.expm1(log_pred)

# Test with various inputs
test_cases = [
    (2, 1, 1000),
    (2, 1, 1500),
    (2, 1, 2000),
    (3, 2, 1500),
    (3, 2, 2000),
    (3, 2, 2500),
    (4, 3, 2000),
    (4, 3, 3000),
]

for beds, baths, sqft in test_cases:
    price = test_prediction(beds, baths, sqft)
    print(f"  {beds} beds, {baths} baths, {sqft} sqft -> ${price:,.0f}")

# Save the new model
print("\nSaving new model...")
new_model_data = {'Random Forest': rf_model}
joblib.dump(new_model_data, 'random_forest_model_fixed.joblib')
print("New model saved as 'random_forest_model_fixed.joblib'")

print("\nDone! The new model should have varying predictions.")
