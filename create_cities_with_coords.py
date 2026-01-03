# D:\route\create_cities_with_coords.py

import pandas as pd

cities = pd.DataFrame({
    'City': ['Delhi (NCR)', 'Mumbai', 'Bengaluru', 'Hyderabad', 'Pune', 'Agra', 'Kanpur', 'Ghaziabad', 'Faridabad', 'Noida'],
    'Latitude': [28.7041, 19.0760, 12.9716, 17.3850, 18.5204, 27.1767, 26.4499, 28.6692, 28.4089, 28.5921],
    'Longitude': [77.1025, 72.8777, 77.5946, 78.4867, 73.8567, 78.0081, 80.3319, 77.5979, 77.2948, 77.3910],
    'Tier': ['Tier-1', 'Tier-1', 'Tier-1', 'Tier-1', 'Tier-1', 'Tier-3', 'Tier-3', 'Tier-2', 'Tier-2', 'Tier-2'],
    'State': ['Delhi', 'Maharashtra', 'Karnataka', 'Telangana', 'Maharashtra', 'UP', 'UP', 'UP', 'Haryana', 'UP'],
    'Average_Travel_Time_10km_Minutes': [18, 20, 16, 14, 15, 14, 15, 17, 16, 17],
    'Congestion_Level_Peak_Percent': [45, 50, 40, 35, 38, 30, 32, 42, 40, 42],
    'Average_Speed_Peak_Hours_kmph': [28, 25, 32, 35, 33, 35, 34, 30, 31, 30],
    'Average_Speed_Off_Peak_kmph': [40, 35, 45, 48, 45, 48, 46, 42, 43, 42],
    'Delivery_Locations': [45000, 52000, 38000, 32000, 28000, 15000, 16000, 21000, 19500, 20000],
    'Avg_Daily_Orders': [12000, 14000, 11000, 9000, 8500, 4500, 4800, 6200, 5800, 6000]
})

cities.to_csv('data/cities_data.csv', index=False)
print("✓ Cities with coordinates created")
