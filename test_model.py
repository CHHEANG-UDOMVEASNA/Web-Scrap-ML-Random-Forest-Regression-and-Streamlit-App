#!/usr/bin/env python3
import joblib
import numpy as np
import pandas as pd

# Load model 
model_data = joblib.load('random_forest_model.joblib')
model = model_data['Random Forest']

print("=== MODEL INFO ===")
print(f"Model feature names: {model.feature_names_in_}")
print(f"Model n_features: {model.n_features_in_}")

# Define feature names as expected by the model
feature_names = ['Beds', 'Baths', 'Sqft', 'Total_rooms', 'Bath_bed_ratio', 
                'Sqft_per_room', 'Log_sqft', 'Beds_Baths_interaction', 
                'Beds_Sqft_interaction', 'Baths_Sqft_interaction']

def engineer_features_with_names(beds, baths, sqft):
    """Engineer features with proper naming"""
    total_rooms = beds + baths
    bath_bed_ratio = baths / beds if beds > 0 else 0
    sqft_per_room = sqft / total_rooms if total_rooms > 0 else 0
    log_sqft = np.log1p(sqft)
    beds_baths_interaction = beds * baths
    beds_sqft_interaction = beds * sqft
    baths_sqft_interaction = baths * sqft
    
    # Create DataFrame with proper feature names
    features_dict = {
        'Beds': beds,
        'Baths': baths, 
        'Sqft': sqft,
        'Total_rooms': total_rooms,
        'Bath_bed_ratio': bath_bed_ratio,
        'Sqft_per_room': sqft_per_room,
        'Log_sqft': log_sqft,
        'Beds_Baths_interaction': beds_baths_interaction,
        'Beds_Sqft_interaction': beds_sqft_interaction,
        'Baths_Sqft_interaction': baths_sqft_interaction
    }
    
    return pd.DataFrame([features_dict])[feature_names]

print("\n=== TESTING DIFFERENT SQUARE FOOTAGES ===")
beds, baths = 3, 2

for sqft in [1000, 1500, 2000, 2500, 3000]:
    features_df = engineer_features_with_names(beds, baths, sqft)
    log_pred = model.predict(features_df)[0]
    price = np.expm1(log_pred)
    print(f"Sqft: {sqft:4d} | Log pred: {log_pred:.4f} | Price: ${price:8,.0f}")

print("\n=== TESTING DIFFERENT BEDROOMS (2000 sqft) ===")
sqft, baths = 2000, 2

for beds in [1, 2, 3, 4, 5]:
    features_df = engineer_features_with_names(beds, baths, sqft)
    log_pred = model.predict(features_df)[0]
    price = np.expm1(log_pred)
    print(f"Beds: {beds} | Log pred: {log_pred:.4f} | Price: ${price:8,.0f}")
