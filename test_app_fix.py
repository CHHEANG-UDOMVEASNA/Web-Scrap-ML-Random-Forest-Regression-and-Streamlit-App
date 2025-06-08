#!/usr/bin/env python3
"""Test the fixed app prediction logic"""

import sys
sys.path.append('.')

# Import the functions from app.py
from app import load_model, engineer_features, predict_price
import numpy as np

print("=== TESTING FIXED APP PREDICTION LOGIC ===")

# Load model
model = load_model()
if model is None:
    print("ERROR: Could not load model!")
    exit(1)

print(f"Model loaded successfully: {type(model)}")

# Test predictions with different square footages
beds, baths = 3, 2
print(f"\nTesting with {beds} beds, {baths} baths:")

test_sqfts = [1000, 1500, 2000, 2500, 3000, 3500, 4000]

for sqft in test_sqfts:
    try:
        predicted_price = predict_price(model, beds, baths, sqft)
        if predicted_price is not None:
            print(f"Sqft: {sqft:4d} | Price: ${predicted_price:,.0f}")
        else:
            print(f"Sqft: {sqft:4d} | ERROR in prediction")
    except Exception as e:
        print(f"Sqft: {sqft:4d} | EXCEPTION: {e}")

print("\n=== Testing different beds/baths with 2000 sqft ===")
sqft = 2000
test_configs = [(1, 1), (2, 1), (3, 2), (4, 3), (5, 4)]

for beds, baths in test_configs:
    try:
        predicted_price = predict_price(model, beds, baths, sqft)
        if predicted_price is not None:
            print(f"Beds: {beds}, Baths: {baths} | Price: ${predicted_price:,.0f}")
        else:
            print(f"Beds: {beds}, Baths: {baths} | ERROR in prediction")
    except Exception as e:
        print(f"Beds: {beds}, Baths: {baths} | EXCEPTION: {e}")
