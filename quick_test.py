#!/usr/bin/env python3
"""Quick test of app functions"""

import joblib
import pandas as pd
import numpy as np

# Load model directly
model_data = joblib.load('random_forest_model.joblib')
model = model_data['Random Forest']

def engineer_features(beds, baths, sqft):
    """Engineer features as DataFrame (fixed version)"""
    total_rooms = beds + baths
    bath_bed_ratio = baths / beds if beds > 0 else 0
    sqft_per_room = sqft / total_rooms if total_rooms > 0 else 0
    log_sqft = np.log1p(sqft)
    beds_baths_interaction = beds * baths
    beds_sqft_interaction = beds * sqft
    baths_sqft_interaction = baths * sqft
    
    feature_names = [
        'Beds', 'Baths', 'Sqft', 'Total_rooms', 'Bath_bed_ratio',
        'Sqft_per_room', 'Log_sqft', 'Beds_Baths_interaction',
        'Beds_Sqft_interaction', 'Baths_Sqft_interaction'
    ]
    
    feature_values = [
        beds, baths, sqft, total_rooms, bath_bed_ratio,
        sqft_per_room, log_sqft, beds_baths_interaction,
        beds_sqft_interaction, baths_sqft_interaction
    ]
    
    return pd.DataFrame([feature_values], columns=feature_names)

def predict_price(model, beds, baths, sqft):
    """Make prediction"""
    features = engineer_features(beds, baths, sqft)
    log_price_pred = model.predict(features)[0]
    return np.expm1(log_price_pred)

print("=== TESTING FIXED APP ===")
beds, baths = 3, 2

for sqft in [1000, 1500, 2000, 2500, 3000]:
    price = predict_price(model, beds, baths, sqft)
    print(f"Sqft: {sqft} | Price: ${price:,.0f}")
