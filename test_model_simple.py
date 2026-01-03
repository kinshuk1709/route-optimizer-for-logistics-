import pickle
import numpy as np

model = pickle.load(open('models/gb_route_optimizer.pkl', 'rb'))
scaler = pickle.load(open('models/feature_scaler_phase2.pkl', 'rb'))
features = pickle.load(open('models/feature_names_phase2.pkl', 'rb'))

print("✓ Model loaded successfully!")
print(f"✓ Number of trees: {model.n_estimators}")
print(f"✓ Number of features: {len(features)}")

sample = np.array([[35, 45, 23.4, 30.86, 37.0, 24.6, 4.0, 22,
                    12000, 45000, 5000, 100, 50, 25, 80, 24,
                    10, 18, 80, 100, 70, 90]])

scaled = scaler.transform(sample)
prediction = model.predict(scaled)[0]

print(f"✓ Sample prediction: ₹{prediction:.2f}")
