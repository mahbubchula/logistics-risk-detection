import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🔧 DATA PREPROCESSING & PREPARATION")
print("="*80)

# Create directories
os.makedirs('../data/processed', exist_ok=True)
os.makedirs('../models', exist_ok=True)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n📊 Loading dataset...")
df = pd.read_csv('../data/raw/dynamic_supply_chain_logistics_dataset.csv')

# Convert timestamp to datetime and extract features
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['hour'] = df['timestamp'].dt.hour
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

print(f"✅ Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("🛠️ FEATURE ENGINEERING")
print("="*80)

# Create interaction features
print("\n🔄 Creating interaction features...")

# Critical interactions identified from EDA
df['traffic_weather_interaction'] = df['traffic_congestion_level'] * df['weather_condition_severity']
df['driver_fatigue_interaction'] = df['driver_behavior_score'] * df['fatigue_monitoring_score']
df['route_port_risk'] = df['route_risk_level'] * df['port_congestion_level']
df['inventory_fulfillment_ratio'] = df['warehouse_inventory_level'] / (df['historical_demand'] + 1)
df['cost_per_hour'] = df['shipping_costs'] / (df['loading_unloading_time'] + 1)

# Risk indicators
df['high_risk_indicator'] = (
    (df['route_risk_level'] > 7) & 
    (df['traffic_congestion_level'] > 7)
).astype(int)

df['delivery_pressure'] = (
    (df['eta_variation_hours'] > 3) & 
    (df['fatigue_monitoring_score'] > 0.7)
).astype(int)

print(f"✅ Created {7} interaction features")

# ============================================================================
# 3. DEFINE FEATURES AND TARGETS
# ============================================================================
print("\n" + "="*80)
print("🎯 DEFINING FEATURES AND TARGETS")
print("="*80)

# Features to exclude
exclude_cols = [
    'timestamp',
    'risk_classification',  # Target 1
    'disruption_likelihood_score',  # Target 2
    'delay_probability',  # Target 3
    'delivery_time_deviation'  # Target 4
]

# Get all feature columns
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"\n📊 Total features: {len(feature_cols)}")
print(f"   - Original features: 21")
print(f"   - Temporal features: 5 (year, month, day_of_week, hour, is_weekend)")
print(f"   - Engineered features: 7")

# Define targets
targets = {
    'risk_classification': 'classification',
    'disruption_likelihood_score': 'regression',
    'delay_probability': 'regression',
    'delivery_time_deviation': 'regression'
}

print(f"\n🎯 Target variables: {len(targets)}")
for target, task_type in targets.items():
    print(f"   - {target}: {task_type}")

# ============================================================================
# 4. TRAIN-TEST SPLIT (Temporal Split)
# ============================================================================
print("\n" + "="*80)
print("✂️ TRAIN-TEST SPLIT (Temporal)")
print("="*80)

# Sort by timestamp for temporal split
df = df.sort_values('timestamp')

# Use 80% for training, 20% for testing (temporal split)
split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

print(f"\n📊 Training set: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
print(f"   Date range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")

print(f"\n📊 Test set: {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
print(f"   Date range: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

# ============================================================================
# 5. PREPARE FEATURES (X) AND TARGETS (y)
# ============================================================================
print("\n" + "="*80)
print("📦 PREPARING FEATURES AND TARGETS")
print("="*80)

X_train = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()

# Prepare each target
y_train_dict = {}
y_test_dict = {}

for target in targets.keys():
    y_train_dict[target] = train_df[target].copy()
    y_test_dict[target] = test_df[target].copy()

print(f"\n✅ X_train shape: {X_train.shape}")
print(f"✅ X_test shape: {X_test.shape}")

# ============================================================================
# 6. FEATURE SCALING
# ============================================================================
print("\n" + "="*80)
print("📏 FEATURE SCALING (StandardScaler)")
print("="*80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)

print("✅ Features scaled using StandardScaler")

# Save scaler
joblib.dump(scaler, '../models/feature_scaler.pkl')
print("✅ Scaler saved to: ../models/feature_scaler.pkl")

# ============================================================================
# 7. ENCODE RISK CLASSIFICATION TARGET
# ============================================================================
print("\n" + "="*80)
print("🏷️ ENCODING RISK CLASSIFICATION")
print("="*80)

le = LabelEncoder()
y_train_risk_encoded = le.fit_transform(y_train_dict['risk_classification'])
y_test_risk_encoded = le.transform(y_test_dict['risk_classification'])

print(f"\n📊 Class mapping:")
for idx, class_name in enumerate(le.classes_):
    train_count = np.sum(y_train_risk_encoded == idx)
    test_count = np.sum(y_test_risk_encoded == idx)
    print(f"   {class_name}: {idx}")
    print(f"      Train: {train_count:,} ({train_count/len(y_train_risk_encoded)*100:.2f}%)")
    print(f"      Test:  {test_count:,} ({test_count/len(y_test_risk_encoded)*100:.2f}%)")

# Save label encoder
joblib.dump(le, '../models/label_encoder.pkl')
print("\n✅ Label encoder saved to: ../models/label_encoder.pkl")

# ============================================================================
# 8. APPLY SMOTE FOR CLASS BALANCING
# ============================================================================
print("\n" + "="*80)
print("⚖️ APPLYING SMOTE FOR CLASS BALANCING")
print("="*80)

print("\n📊 Before SMOTE:")
unique, counts = np.unique(y_train_risk_encoded, return_counts=True)
for u, c in zip(unique, counts):
    print(f"   Class {le.classes_[u]}: {c:,} samples")

# Apply SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train_risk_encoded)

print(f"\n📊 After SMOTE:")
unique, counts = np.unique(y_train_balanced, return_counts=True)
for u, c in zip(unique, counts):
    print(f"   Class {le.classes_[u]}: {c:,} samples")

print(f"\n✅ Training set size increased from {len(X_train_scaled):,} to {len(X_train_balanced):,}")

# ============================================================================
# 9. SAVE PROCESSED DATA
# ============================================================================
print("\n" + "="*80)
print("💾 SAVING PROCESSED DATA")
print("="*80)

# Save processed data
np.save('../data/processed/X_train_scaled.npy', X_train_scaled.values)
np.save('../data/processed/X_test_scaled.npy', X_test_scaled.values)
np.save('../data/processed/X_train_balanced.npy', X_train_balanced)
np.save('../data/processed/y_train_risk_encoded.npy', y_train_risk_encoded)
np.save('../data/processed/y_test_risk_encoded.npy', y_test_risk_encoded)
np.save('../data/processed/y_train_balanced.npy', y_train_balanced)

# Save regression targets
for target in ['disruption_likelihood_score', 'delay_probability', 'delivery_time_deviation']:
    np.save(f'../data/processed/y_train_{target}.npy', y_train_dict[target].values)
    np.save(f'../data/processed/y_test_{target}.npy', y_test_dict[target].values)

# Save feature names
pd.Series(feature_cols).to_csv('../data/processed/feature_names.csv', index=False, header=['feature'])

print("\n✅ Saved processed datasets:")
print("   - X_train_scaled.npy")
print("   - X_test_scaled.npy")
print("   - X_train_balanced.npy (SMOTE)")
print("   - y_train/test for all targets")
print("   - feature_names.csv")

# ============================================================================
# 10. SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("📊 PREPROCESSING SUMMARY")
print("="*80)

summary = {
    'Total Samples': len(df),
    'Training Samples': len(X_train),
    'Test Samples': len(X_test),
    'Training Samples (SMOTE)': len(X_train_balanced),
    'Total Features': len(feature_cols),
    'Original Features': 21,
    'Temporal Features': 5,
    'Engineered Features': 7,
    'Target Variables': len(targets),
    'Train-Test Split': '80-20 (Temporal)',
    'Scaling Method': 'StandardScaler',
    'Class Balancing': 'SMOTE'
}

for key, value in summary.items():
    print(f"   {key}: {value}")

print("\n" + "="*80)
print("✅ DATA PREPROCESSING COMPLETE!")
print("="*80)
print("\n🚀 Ready for model training!")