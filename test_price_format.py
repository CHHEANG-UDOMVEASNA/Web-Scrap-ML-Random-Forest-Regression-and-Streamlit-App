#!/usr/bin/env python3
"""
Test the compact price formatting
"""

def format_price_compact(price):
    if price >= 1000000:
        return f"${price/1000000:.1f}M"
    elif price >= 1000:
        return f"${price/1000:.0f}K"
    else:
        return f"${price:.0f}"

# Test with different price ranges
test_prices = [
    2288728,  # From the screenshot
    1945419,  # Low range
    2632037,  # High range
    450000,   # Medium price
    850000,   # Close to 1M
    1200000,  # Over 1M
    150000,   # Lower price
]

print("=== Testing Compact Price Formatting ===")
for price in test_prices:
    compact = format_price_compact(price)
    full = f"${price:,.0f}"
    print(f"{full:>12} -> {compact:>8}")

# Test the range formatting specifically
predicted_price = 2288728
low_price = predicted_price * 0.85
high_price = predicted_price * 1.15

print(f"\n=== Price Range Example ===")
print(f"Predicted: ${predicted_price:,.0f}")
print(f"Low:       ${low_price:,.0f} -> {format_price_compact(low_price)}")
print(f"High:      ${high_price:,.0f} -> {format_price_compact(high_price)}")
print(f"Range:     {format_price_compact(low_price)} - {format_price_compact(high_price)}")
