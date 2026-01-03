import pandas as pd

# Cities Data
cities_data = {
    'City': ['Delhi (NCR)', 'Mumbai', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Kanpur', 'Agra', 'Gwalior', 'Indore', 'Nagpur', 'Surat', 'Vadodara', 'Bhopal', 'Visakhapatnam', 'Kochi'],
    'Latitude': [28.7041, 19.0760, 12.9716, 17.3850, 13.0827, 22.5726, 18.5204, 23.0225, 26.9124, 26.8467, 26.4499, 27.1767, 26.2183, 22.7196, 21.1458, 21.1702, 22.3072, 23.1815, 17.6869, 9.9312],
    'Longitude': [77.1025, 72.8777, 77.5946, 78.4867, 80.2707, 88.3639, 73.8567, 72.5714, 75.7873, 80.9462, 80.3319, 78.0081, 78.1627, 75.8577, 79.0882, 72.8311, 73.1812, 79.9864, 83.2185, 76.2673],
    'Tier': ['Tier 1', 'Tier 1', 'Tier 1', 'Tier 1', 'Tier 1', 'Tier 1', 'Tier 2', 'Tier 2', 'Tier 2', 'Tier 2', 'Tier 2', 'Tier 2', 'Tier 2', 'Tier 2', 'Tier 2', 'Tier 2', 'Tier 2', 'Tier 2', 'Tier 2', 'Tier 2'],
    'State': ['Delhi', 'Maharashtra', 'Karnataka', 'Telangana', 'Tamil Nadu', 'West Bengal', 'Maharashtra', 'Gujarat', 'Rajasthan', 'Uttar Pradesh', 'Uttar Pradesh', 'Uttar Pradesh', 'Madhya Pradesh', 'Madhya Pradesh', 'Maharashtra', 'Gujarat', 'Gujarat', 'Madhya Pradesh', 'Andhra Pradesh', 'Kerala']
}

vehicles_data = {
    'Vehicle_Type': ['small_truck', 'medium_truck', 'large_truck'],
    'Capacity_Tons': [3, 8, 20],
    'Cost_Per_KM': [22.5, 27.5, 35],
    'Daily_Cost': [1500, 1800, 2200],
    'Axles': [2, 3, 4],
    'Description': ['Small Truck (1-3 tons)', 'Medium Truck (5-10 tons)', 'Large Truck (16+ tons)']
}

# Create DataFrames
cities_df = pd.DataFrame(cities_data)
vehicles_df = pd.DataFrame(vehicles_data)

# Save to CSV
cities_df.to_csv('cities_data.csv', index=False)
vehicles_df.to_csv('vehicles_data.csv', index=False)

print("✅ CSV files created successfully!")
print(f"📍 Cities: {len(cities_df)} cities")
print(f"🚛 Vehicles: {len(vehicles_df)} vehicle types")
