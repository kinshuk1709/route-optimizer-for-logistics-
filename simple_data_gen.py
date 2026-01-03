import pandas as pd
import numpy as np

# Load cities
cities_df = pd.read_csv('cities_data.csv')
cities = list(zip(
    cities_df['City'],
    cities_df['Latitude'],
    cities_df['Longitude']
))
print(f"✅ Loaded {len(cities)} cities\n")

# Vehicle cost per km
vehicles = {'3.5T': 25, '7.5T': 35, '12T': 45, '18T': 55, '25T': 65}

routes = []
print("📊 Generating 1000 routes...\n")

for i in range(1000):
    if (i + 1) % 200 == 0:
        print(f"Progress: {i + 1}/1000")
    
    # Random cities
    from_city = cities[np.random.randint(0, len(cities))]
    to_city = cities[np.random.randint(0, len(cities))]
    if from_city == to_city:
        continue
    
    # Distance
    lat_diff = abs(to_city[1] - from_city[1])
    lon_diff = abs(to_city[2] - from_city[2])
    distance_km = ((lat_diff**2 + lon_diff**2)**0.5) * 111 * 1.25
    
    # Vehicle
    vehicle = np.random.choice(list(vehicles.keys()))
    
    # Cargo
    cargo = np.random.randint(1, 20)
    
    # Peak hours
    is_peak = np.random.randint(0, 2)
    is_weekend = np.random.randint(0, 2)
    
    # Cost
    cost = distance_km * vehicles[vehicle] + (distance_km/10)*85 + distance_km*1.5 + 500*is_peak + 300*is_weekend + cargo*50
    cost = cost * np.random.uniform(0.95, 1.05)
    
    routes.append({
        'from_city': from_city[0],
        'to_city': to_city[0],
        'distance_km': round(distance_km, 2),
        'vehicle': vehicle,
        'cargo': cargo,
        'is_peak': is_peak,
        'is_weekend': is_weekend,
        'cost': round(cost, 2)
    })

df = pd.DataFrame(routes)
df.to_csv('synthetic_training_data.csv', index=False)

print(f"\n✅ Created {len(df)} routes!")
print(f"💾 File: synthetic_training_data.csv\n")
