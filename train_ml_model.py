import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import json

print("="*70)
print("🤖 ML Model Training - Route Cost Prediction")
print("="*70)

# Load data
print("\n📂 Loading training data...")
df = pd.read_csv('synthetic_training_data.csv')
print(f"✅ Loaded {len(df)} routes")

# Prepare features
print("\n🔧 Preparing features...")
feature_columns = ['distance_km', 'cargo', 'is_peak', 'is_weekend']
X = df[feature_columns].copy()
y = df['cost'].copy()

print(f"✅ Selected features: {feature_columns}")
print(f"   Distance: {X['distance_km'].min():.0f} - {X['distance_km'].max():.0f} km")
print(f"   Cost: ₹{y.min():.0f} - ₹{y.max():.0f}")

# Split data
print("\n✂️ Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✅ Train: {len(X_train)} | Test: {len(X_test)}")

# Scale features
print("\n⚖️ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
print("\n🔨 Training Gradient Boosting model...")
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train_scaled, y_train)
print("✅ Training complete!")

# Evaluate
print("\n📈 Model Evaluation:\n")

y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

train_mae = mean_absolute_error(y_train, y_pred_train)
train_mape = mean_absolute_percentage_error(y_train, y_pred_train)
train_r2 = r2_score(y_train, y_pred_train)

test_mae = mean_absolute_error(y_test, y_pred_test)
test_mape = mean_absolute_percentage_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print("TRAINING SET:")
print(f"  MAE: ₹{train_mae:.2f}")
print(f"  MAPE: {train_mape:.2%}")
print(f"  R²: {train_r2:.4f}")

print("\nTEST SET:")
print(f"  MAE: ₹{test_mae:.2f}")
print(f"  MAPE: {test_mape:.2%}")
print(f"  R²: {test_r2:.4f}")

# Cross-validation
print("\n🔄 Cross-validation (5-fold)...")
cv_scores = cross_val_score(
    model, X_train_scaled, y_train, 
    cv=5, 
    scoring='neg_mean_absolute_percentage_error'
)
cv_mape = -cv_scores.mean()
print(f"✅ CV MAPE: {cv_mape:.2%}")

# Feature importance
print("\n🎯 Feature Importance:")
for feat, imp in zip(feature_columns, model.feature_importances_):
    bar = '█' * int(imp * 100)
    print(f"   {feat:20s} {bar} {imp:.4f}")

# Save model
print("\n💾 Saving model files...")
with open('route_cost_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

model_info = {
    'model_type': 'GradientBoostingRegressor',
    'n_samples': len(df),
    'test_mape': float(test_mape),
    'test_r2': float(test_r2),
    'features': feature_columns
}
with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("✅ Saved: route_cost_model.pkl")
print("✅ Saved: feature_scaler.pkl")
print("✅ Saved: feature_names.pkl")
print("✅ Saved: model_info.json")

print("\n" + "="*70)
print(f"✅ MODEL TRAINING COMPLETE!")
print(f"   Test MAPE: {test_mape:.2%}")
print(f"   Test R²: {test_r2:.4f}")
print("="*70 + "\n")
