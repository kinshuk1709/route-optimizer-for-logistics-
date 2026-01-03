# D:\route\test_route_optimizer.py
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.route_optimizer import RouteOptimizer

def main():
    csv_path = os.path.join(os.path.dirname(__file__), 'src', 'cities_data.csv')
    
    print(f"📍 Loading cities from: {csv_path}")
    optimizer = RouteOptimizer(csv_path)
    
    origin = "Delhi"
    intermediate = ["Pune", "Bangalore", "Hyderabad"]
    destination = "Chennai"
    
    print(f"\n🛣️  FINDING TOP 3 ROUTES: {origin} → ??? → {destination}")
    print("=" * 80)
    
    # 🚚 TEST 3 VEHICLE TYPES = DIFFERENT COSTS!
    vehicle_configs = [
        {"capacity": 4.0, "cost_per_km": 22, "name": "Truck (4T)"},
        {"capacity": 6.0, "cost_per_km": 28, "name": "Heavy Truck (6T)"}, 
        {"capacity": 2.0, "cost_per_km": 18, "name": "Light Van (2T)"}
    ]
    
    all_routes = []
    
    # Generate routes for EACH vehicle type
    for config in vehicle_configs:
        routes = optimizer.find_top_3_routes(
            origin, intermediate, destination,
            vehicle_capacity_tons=config["capacity"],
            vehicle_cost_per_km=config["cost_per_km"]
        )
        for route in routes:
            route['vehicle'] = config["name"]
            all_routes.append(route)
    
    # Show TOP 3 across all vehicle types
    top_3 = sorted(all_routes, key=lambda x: x['predicted_cost'])[:3]
    
    for i, route in enumerate(top_3, 1):
        rank = f"{i}️⃣"
        print(f"{rank} {route['vehicle']} Route: {' → '.join(route['route'])}")
        print(f"   📏 Distance: {route['distance_km']} km")
        print(f"   ⏱️  Time: {route['estimated_time_hours']:.1f} hrs ({route['num_stops']} stops)")
        print(f"   💰 Cost: ₹{route['predicted_cost']:,.0f} (₹{route['cost_per_km']:.0f}/km)")
        print()

if __name__ == "__main__":
    main()
