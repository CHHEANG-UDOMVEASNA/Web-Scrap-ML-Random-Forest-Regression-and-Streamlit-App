print("=== Testing Price Range Formatting ===")

# Simulate the exact scenario from the screenshot
predicted_price = 2288728

# Original long format (problematic)
low_price = predicted_price * 0.85
high_price = predicted_price * 1.15
original_range = f"${low_price:,.0f} - ${high_price:,.0f}"

# New compact format
def format_price_compact(price):
    if price >= 1000000:
        return f"${price/1000000:.1f}M"
    elif price >= 1000:
        return f"${price/1000:.0f}K"
    else:
        return f"${price:.0f}"

compact_range = f"{format_price_compact(low_price)} - {format_price_compact(high_price)}"

print(f"Predicted Price: ${predicted_price:,.0f}")
print(f"Low Price:       ${low_price:,.0f}")
print(f"High Price:      ${high_price:,.0f}")
print()
print(f"Original Format: {original_range}")
print(f"Length:          {len(original_range)} characters")
print()
print(f"Compact Format:  {compact_range}")
print(f"Length:          {len(compact_range)} characters")
print(f"Space Saved:     {len(original_range) - len(compact_range)} characters")
