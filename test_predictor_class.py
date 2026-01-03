# D:\route\test_predictor_class.py

from src.predictor import RouteOptimizer

# Initialize RouteOptimizer with paths relative to D:\route
optimizer = RouteOptimizer(
    model_path='models/gb_route_optimizer.pkl',
    scaler_path='models/feature_scaler_phase2.pkl',
    features_path='models/feature_names_phase2.pkl'
)

# Minimal example feature dict.
# For now, fill engineered features with reasonable placeholders.
features = {
    'Distance_from_Hub_KM': 35,
    'Delivery_Radius_KM': 45,
    'Travel_Time_10km_Minutes': 23.4,
    'Peak_Congestion_Percent': 30.86,
    'Off_Peak_Speed_kmph': 37.0,
    'Peak_Speed_kmph': 24.6,
    'Vehicle_Capacity_Tons': 4.0,
    'Cost_Per_KM_INR': 22,
    'Average_Daily_Orders': 12000,
    'Delivery_Locations': 45000,
    'Warehouse_Stock_Tons': 5000,

    # Engineered features – temporary example values
    'Efficiency_Score': 100,
    'Congestion_Factor': 1.1,
    'Speed_Ratio': 0.66,
    'Order_Capacity_Ratio': 0.27,
    'Travel_Cost_Per_Location': 27.2,
    'Warehouse_Utilization': 0.24,
    'Traffic_Impact_Hours': 1.86,
    'Effective_Speed': 30.8,
    'Distance_Time_Ratio': 1.5,
    'Vehicle_Cost_Efficiency': 0.18,
    'Delivery_Radius_Efficiency': 1.29
}

cost = optimizer.predict_cost(features)
print(f"Predicted cost: ₹{cost:.2f}")
