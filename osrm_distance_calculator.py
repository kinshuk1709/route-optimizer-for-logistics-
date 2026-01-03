import requests
import pandas as pd
from math import radians, cos, sin, asin, sqrt

class DistanceCalculator:
    """Calculate distances using OSRM (Open Source Routing Machine)"""
    
    def __init__(self):
        self.osrm_url = "http://router.project-osrm.org/route/v1/driving"
    
    def get_road_distance(self, lat1, lon1, lat2, lon2):
        """Get actual road distance between two coordinates using OSRM"""
        try:
            url = f"{self.osrm_url}/{lon1},{lat1};{lon2},{lat2}?overview=full"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('routes') and len(data['routes']) > 0:
                    route = data['routes'][0]
                    distance_m = route.get('distance', 0)
                    duration_s = route.get('duration', 0)
                    
                    distance_km = distance_m / 1000
                    duration_hours = duration_s / 3600
                    
                    return distance_km, duration_hours
        
        except Exception as e:
            print(f"OSRM error: {e}. Using fallback...")
        
        return self.haversine_distance(lat1, lon1, lat2, lon2)
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate straight-line distance as fallback"""
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        r = 6371
        straight_line_km = c * r
        distance_km = straight_line_km * 1.25
        
        avg_speed = 55
        duration_hours = distance_km / avg_speed
        
        return distance_km, duration_hours
    
    def get_distance_between_cities(self, lat1, lon1, lat2, lon2, use_osrm=True):
        """Get distance between two cities"""
        if use_osrm:
            return self.get_road_distance(lat1, lon1, lat2, lon2)
        else:
            return self.haversine_distance(lat1, lon1, lat2, lon2)


def calculate_route_distance(cities_df, route_cities, use_osrm=True):
    """Calculate total distance and time for a route"""
    calc = DistanceCalculator()
    
    total_distance = 0
    total_time = 0
    
    for i in range(len(route_cities) - 1):
        city1 = route_cities[i]
        city2 = route_cities[i + 1]
        
        row1 = cities_df[cities_df["City"] == city1]
        row2 = cities_df[cities_df["City"] == city2]
        
        if row1.empty or row2.empty:
            continue
        
        lat1 = float(row1.iloc[0]["Latitude"])
        lon1 = float(row1.iloc[0]["Longitude"])
        lat2 = float(row2.iloc[0]["Latitude"])
        lon2 = float(row2.iloc[0]["Longitude"])
        
        distance, time = calc.get_distance_between_cities(
            lat1, lon1, lat2, lon2, use_osrm=use_osrm
        )
        
        total_distance += distance
        total_time += time
    
    return total_distance, total_time


def format_time(hours):
    """Convert hours to readable format"""
    h = int(hours)
    m = int((hours - h) * 60)
    return f"{h}h {m}m"


if __name__ == "__main__":
    calc = DistanceCalculator()
    
    print("="*60)
    print("Testing OSRM Distance Calculator")
    print("="*60)
    
    print("\nDelhi to Jaipur:")
    dist, time = calc.get_road_distance(28.7041, 77.1025, 26.9124, 75.7873)
    print(f"  Distance: {dist:.2f} km")
    print(f"  Time: {format_time(time)}")
    
    print("\nMumbai to Bangalore:")
    dist, time = calc.get_road_distance(19.076, 72.8777, 12.9716, 77.5946)
    print(f"  Distance: {dist:.2f} km")
    print(f"  Time: {format_time(time)}")
    
    print("\nDelhi to Mumbai:")
    dist, time = calc.get_road_distance(28.7041, 77.1025, 19.076, 72.8777)
    print(f"  Distance: {dist:.2f} km")
    print(f"  Time: {format_time(time)}")
    
    print("\n" + "="*60)
