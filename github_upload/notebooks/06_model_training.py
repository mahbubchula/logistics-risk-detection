import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report)
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🤖 MACHINE LEARNING MODEL TRAINING")
print("="*80)

# ============================================================================
# 1. LOAD PREPROCESSED DATA
# ============================================================================
print("\n📊 Loading preprocessed data...")

# Load balanced training data (with SMOTE)
X_train = np.load('../data/processed/X_train_balanced.npy')
y_train = np.load('../data/processed/y_train_balanced.npy')

# Load original test data
X_test = np.load('../data/processed/X_test_scaled.npy')
y_test = np.load('../data/processed/y_test_risk_encoded.npy')

# Load feature names
feature_names = pd.read_csv('../data/processed/feature_names.csv')['feature'].tolist()

# Load label encoder
le = joblib.load('../models/label_encoder.pkl')

print(f"\n✅ Training data: {X_train.shape}")
print(f"✅ Test data: {X_test.shape}")
print(f"✅ Number of features: {len(feature_names)}")
print(f"✅ Classes: {le.classes_}")

# ============================================================================
# 2. DEFINE MODELS
# ============================================================================
print("\n" + "="*80)
print("🎯 DEFINING MACHINE LEARNING MODELS")
print("="*80)

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ),
    
    'XGBoost': XGBClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    ),
    
    'LightGBM': LGBMClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ),
    
    'CatBoost': CatBoostClassifier(
        iterations=200,
        depth=10,
        learning_rate=0.1,
        random_state=42,
        verbose=False
    ),
    
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        random_state=42
    ),
    
    'SVM': SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=42,
        probability=True
    ),
    
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=256,
        learning_rate='adaptive',
        max_iter=200,
        random_state=42,
        early_stopping=True
    )
}

print(f"\n📋 Models to train: {len(models)}")
for model_name in models.keys():
    print(f"   ✓ {model_name}")

# ============================================================================
# 3. TRAIN AND EVALUATE MODELS
# ============================================================================
print("\n" + "="*80)
print("🏋️ TRAINING MODELS")
print("="*80)

results = []
trained_models = {}

for model_name, model in models.items():
    print(f"\n{'='*80}")
    print(f"🤖 Training: {model_name}")
    print(f"{'='*80}")
    
    # Train model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"✅ Training completed in {training_time:.2f} seconds")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # ROC-AUC (multiclass)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    except:
        roc_auc = 0.0
    
    # Per-class metrics
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    # Print results
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {roc_auc:.4f}")
    
    print(f"\n📊 PER-CLASS METRICS:")
    for idx, class_name in enumerate(le.classes_):
        print(f"   {class_name}:")
        print(f"      Precision: {precision_per_class[idx]:.4f}")
        print(f"      Recall:    {recall_per_class[idx]:.4f}")
        print(f"      F1-Score:  {f1_per_class[idx]:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 CONFUSION MATRIX:")
    print(f"   Classes: {le.classes_}")
    print(cm)
    
    # Store results
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'Training_Time': training_time,
        'High_Risk_Precision': precision_per_class[0],
        'High_Risk_Recall': recall_per_class[0],
        'High_Risk_F1': f1_per_class[0],
        'Low_Risk_Precision': precision_per_class[1],
        'Low_Risk_Recall': recall_per_class[1],
        'Low_Risk_F1': f1_per_class[1],
        'Moderate_Risk_Precision': precision_per_class[2],
        'Moderate_Risk_Recall': recall_per_class[2],
        'Moderate_Risk_F1': f1_per_class[2]
    })
    
    # Save model
    trained_models[model_name] = model
    joblib.dump(model, f'../models/{model_name.replace(" ", "_").lower()}_model.pkl')
    print(f"\n💾 Model saved: {model_name.replace(' ', '_').lower()}_model.pkl")

# ============================================================================
# 4. RESULTS COMPARISON
# ============================================================================
print("\n" + "="*80)
print("📊 MODEL COMPARISON RESULTS")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('F1-Score', ascending=False)

print("\n🏆 OVERALL PERFORMANCE RANKING (by F1-Score):")
print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].to_string(index=False))

print("\n🎯 HIGH RISK CLASS PERFORMANCE:")
print(results_df[['Model', 'High_Risk_Precision', 'High_Risk_Recall', 'High_Risk_F1']].to_string(index=False))

print("\n🎯 LOW RISK CLASS PERFORMANCE:")
print(results_df[['Model', 'Low_Risk_Precision', 'Low_Risk_Recall', 'Low_Risk_F1']].to_string(index=False))

print("\n🎯 MODERATE RISK CLASS PERFORMANCE:")
print(results_df[['Model', 'Moderate_Risk_Precision', 'Moderate_Risk_Recall', 'Moderate_Risk_F1']].to_string(index=False))

# Save results
results_df.to_csv('../data/results/model_comparison_results.csv', index=False)
print("\n💾 Results saved to: ../data/results/model_comparison_results.csv")

# ============================================================================
# 5. BEST MODEL IDENTIFICATION
# ============================================================================
print("\n" + "="*80)
print("🏆 BEST MODEL SELECTION")
print("="*80)

best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]
best_f1 = results_df.iloc[0]['F1-Score']
best_accuracy = results_df.iloc[0]['Accuracy']

print(f"\n🥇 Best Model: {best_model_name}")
print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"   F1-Score: {best_f1:.4f}")

# Save best model separately
joblib.dump(best_model, '../models/best_model.pkl')
print(f"\n💾 Best model saved as: best_model.pkl")

# ============================================================================
# 6. DETAILED CLASSIFICATION REPORT
# ============================================================================
print("\n" + "="*80)
print(f"📋 DETAILED CLASSIFICATION REPORT - {best_model_name}")
print("="*80)

y_pred_best = best_model.predict(X_test)
print("\n", classification_report(y_test, y_pred_best, target_names=le.classes_))

print("\n" + "="*80)
print("✅ MODEL TRAINING COMPLETE!")
print("="*80)
print(f"\n📁 All models saved in: ../models/")
print(f"📊 Results saved in: ../data/results/")