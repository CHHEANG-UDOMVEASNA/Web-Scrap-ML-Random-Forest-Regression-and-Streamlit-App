#!/usr/bin/env python3
import joblib
import pandas as pd
import numpy as np

print("=== DEBUGGING THE STUCK PREDICTION ===")

# Load the model
model_data = joblib.load('random_forest_model.joblib')
model = model_data['Random Forest']

print(f"Model type: {type(model)}")
print(f"Model feature names: {list(model.feature_names_in_)}")

def debug_predict(beds, baths, sqft):
    """Debug version of feature engineering"""
    print(f"\nInput: beds={beds}, baths={baths}, sqft={sqft}")
    
    # Calculate features step by step
    total_rooms = beds + baths
    bath_bed_ratio = baths / beds if beds > 0 else 0
    sqft_per_room = sqft / total_rooms if total_rooms > 0 else 0
    log_sqft = np.log1p(sqft)
    beds_baths_interaction = beds * baths
    beds_sqft_interaction = beds * sqft
    baths_sqft_interaction = baths * sqft
    
    # Show calculated features
    print(f"  total_rooms: {total_rooms}")
    print(f"  bath_bed_ratio: {bath_bed_ratio}")
    print(f"  sqft_per_room: {sqft_per_room}")
    print(f"  log_sqft: {log_sqft}")
    print(f"  beds_baths_interaction: {beds_baths_interaction}")
    print(f"  beds_sqft_interaction: {beds_sqft_interaction}")
    print(f"  baths_sqft_interaction: {baths_sqft_interaction}")
    
    # Create feature DataFrame
    feature_values = [
        beds, baths, sqft, total_rooms, bath_bed_ratio,
        sqft_per_room, log_sqft, beds_baths_interaction,
        beds_sqft_interaction, baths_sqft_interaction
    ]
    
    feature_names = [
        'Beds', 'Baths', 'Sqft', 'Total_rooms', 'Bath_bed_ratio',
        'Sqft_per_room', 'Log_sqft', 'Beds_Baths_interaction',
        'Beds_Sqft_interaction', 'Baths_Sqft_interaction'
    ]
    
    df = pd.DataFrame([feature_values], columns=feature_names)
    print(f"  Feature DataFrame shape: {df.shape}")
    print(f"  Feature values: {feature_values}")
    
    # Make prediction
    log_pred = model.predict(df)[0]
    price = np.expm1(log_pred)
    
    print(f"  Log prediction: {log_pred}")
    print(f"  Final price: ${price:,.0f}")
    
    return price

# Test with different inputs
test_cases = [
    (1, 1, 1000),
    (2, 1.5, 1500), 
    (3, 2, 2000),
    (4, 3, 2500),
    (5, 4, 3000),
    (3, 2, 1000),  # Same beds/baths, different sqft
    (3, 2, 4000),  # Same beds/baths, different sqft
]

for beds, baths, sqft in test_cases:
    price = debug_predict(beds, baths, sqft)
