import pandas as pd
import random

# List of major Indian logistics hubs
cities = [
    "Delhi (NCR)", "Mumbai", "Bangalore", "Chennai", "Kolkata", 
    "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Surat", 
    "Lucknow", "Indore", "Nagpur", "Patna", "Bhopal"
]

# Generate 100 dummy loads
data = []
for i in range(100):
    source = random.choice(cities)
    dest = random.choice([c for c in cities if c != source])
    
    # Logic: Create "Return Load" opportunities specifically for your demo
    # If a truck goes Delhi -> Mumbai, we want to find a load Mumbai -> Delhi
    if source == "Mumbai" and random.random() > 0.6:
        dest = "Delhi (NCR)"
    if source == "Bangalore" and random.random() > 0.6:
        dest = "Delhi (NCR)"

    # Pricing Logic: Approx ₹2000-4000 per ton depending on distance (randomized)
    weight = random.choice([9, 15, 20, 25]) # Standard truck sizes
    price = random.randint(25000, 85000)

    data.append({
        "Load_ID": f"L{1000+i}",
        "Source": source,
        "Destination": dest,
        "Weight_Tons": weight,
        "Price": price,
        "Status": "Available",
        "Cargo_Type": random.choice(["Textiles", "Auto Parts", "FMCG", "Electronics", "Steel Coils", "Pharma"])
    })

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("loads_data.csv", index=False)
print("✅ loads_data.csv generated with 100 records!")
print(df.head())
