#!/usr/bin/env python3
import joblib
import numpy as np
import pandas as pd

def engineer_features(beds, baths, sqft):
    """Engineer features exactly as in the app"""
    total_rooms = beds + baths
    bath_bed_ratio = baths / beds if beds > 0 else 0
    sqft_per_room = sqft / total_rooms if total_rooms > 0 else 0
    log_sqft = np.log1p(sqft)
    beds_baths_interaction = beds * baths
    beds_sqft_interaction = beds * sqft
    baths_sqft_interaction = baths * sqft
    
    features = [
        beds, baths, sqft, total_rooms, bath_bed_ratio,
        sqft_per_room, log_sqft, beds_baths_interaction,
        beds_sqft_interaction, baths_sqft_interaction
    ]
    return np.array(features).reshape(1, -1)

# Load model
print("Loading model...")
model_data = joblib.load('random_forest_model.joblib')
model = model_data['Random Forest']
print(f"Model type: {type(model)}")
print(f"Model n_estimators: {model.n_estimators}")

# Test with different square footages
print("\n=== Testing predictions with different square footages ===")
beds, baths = 3, 2

test_sqfts = [1000, 1500, 2000, 2500, 3000, 3500, 4000]

for sqft in test_sqfts:
    features = engineer_features(beds, baths, sqft)
    print(f"\nSqft: {sqft}")
    print(f"Features: {features[0]}")
    
    log_pred = model.predict(features)[0]
    price = np.expm1(log_pred)
    
    print(f"Log prediction: {log_pred:.6f}")
    print(f"Final price: ${price:,.0f}")

# Test if the model is actually working by varying other features
print("\n=== Testing with different bedrooms (sqft=2000) ===")
sqft = 2000
baths = 2
for beds in [1, 2, 3, 4, 5]:
    features = engineer_features(beds, baths, sqft)
    log_pred = model.predict(features)[0]
    price = np.expm1(log_pred)
    print(f"Beds: {beds}, Price: ${price:,.0f}")

print("\n=== Testing with different bathrooms (sqft=2000, beds=3) ===")
sqft = 2000
beds = 3
for baths in [1, 1.5, 2, 2.5, 3]:
    features = engineer_features(beds, baths, sqft)
    log_pred = model.predict(features)[0]
    price = np.expm1(log_pred)
    print(f"Baths: {baths}, Price: ${price:,.0f}")
