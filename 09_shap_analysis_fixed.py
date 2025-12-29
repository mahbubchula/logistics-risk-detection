import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lightgbm import LGBMClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🔍 SHAP ANALYSIS - EXPLAINABLE AI (FIXED)")
print("="*80)

# Load data
X_train = np.load('../data/processed/X_train_balanced.npy')
y_train = np.load('../data/processed/y_train_balanced.npy')
X_test = np.load('../data/processed/X_test_scaled.npy')
y_test = np.load('../data/processed/y_test_risk_encoded.npy')
feature_names = pd.read_csv('../data/processed/feature_names.csv')['feature'].tolist()
le = joblib.load('../models/label_encoder.pkl')

# Create DataFrames
X_train_df = pd.DataFrame(X_train, columns=feature_names)
X_test_df = pd.DataFrame(X_test, columns=feature_names)

print(f"\n📊 Data loaded:")
print(f"   Features: {len(feature_names)}")
print(f"   Training samples: {len(X_train):,}")
print(f"   Test samples: {len(X_test):,}")

# ============================================================================
# TRAIN MODEL
# ============================================================================
print("\n" + "="*80)
print("🤖 TRAINING LIGHTGBM FOR SHAP ANALYSIS")
print("="*80)

model = LGBMClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    force_col_wise=True
)

model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# ============================================================================
# SHAP EXPLAINER
# ============================================================================
print("\n" + "="*80)
print("🔬 CREATING SHAP EXPLAINER")
print("="*80)

print("\n📊 Initializing SHAP TreeExplainer...")
explainer = shap.TreeExplainer(model)

# Calculate SHAP values for test set (use subset for speed)
sample_size = min(500, len(X_test))  # Reduced for stability
X_test_sample = X_test_df.sample(n=sample_size, random_state=42)

print(f"📊 Calculating SHAP values for {sample_size} samples...")
shap_values = explainer.shap_values(X_test_sample)

print("✅ SHAP values calculated!")
print(f"   SHAP values shape: {np.array(shap_values).shape}")

# ============================================================================
# GLOBAL FEATURE IMPORTANCE - MANUAL CALCULATION
# ============================================================================
print("\n" + "="*80)
print("📊 GLOBAL FEATURE IMPORTANCE")
print("="*80)

# Calculate mean absolute SHAP values across all classes
if isinstance(shap_values, list):
    # Multi-class case
    mean_shap = np.abs(np.array(shap_values)).mean(axis=(0, 1))
else:
    mean_shap = np.abs(shap_values).mean(axis=0)

# Create feature importance DataFrame
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': mean_shap
}).sort_values('importance', ascending=False)

print("\n📊 Top 20 Most Important Features (Global):")
print(feature_importance_df.head(20).to_string(index=False))

# Save feature importance
feature_importance_df.to_csv('../data/results/shap_global_feature_importance.csv', index=False)
print("\n✅ Saved: shap_global_feature_importance.csv")

# Plot feature importance
fig, ax = plt.subplots(figsize=(12, 10))
top_20 = feature_importance_df.head(20)
ax.barh(range(len(top_20)), top_20['importance'])
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['feature'])
ax.set_xlabel('Mean |SHAP value|', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
ax.set_title('Top 20 Feature Importance (SHAP)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('../figures/shap_global_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: shap_global_importance.png")

# ============================================================================
# PER-CLASS SHAP ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("📊 PER-CLASS SHAP ANALYSIS")
print("="*80)

class_importances = {}

for class_idx, class_name in enumerate(le.classes_):
    print(f"\n🎯 Analyzing: {class_name}")
    
    # Get SHAP values for this class
    if isinstance(shap_values, list):
        class_shap = shap_values[class_idx]
    else:
        class_shap = shap_values
    
    # Calculate mean absolute SHAP for this class
    mean_class_shap = np.abs(class_shap).mean(axis=0)
    class_importances[class_name] = mean_class_shap
    
    # Create DataFrame
    class_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_class_shap
    }).sort_values('importance', ascending=False)
    
    print(f"\n   Top 10 features for {class_name}:")
    print(class_importance_df.head(10).to_string(index=False))
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    top_15 = class_importance_df.head(15)
    ax.barh(range(len(top_15)), top_15['importance'])
    ax.set_yticks(range(len(top_15)))
    ax.set_yticklabels(top_15['feature'])
    ax.set_xlabel('Mean |SHAP value|', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(f'Top 15 Features for {class_name}', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'../figures/shap_importance_{class_name.replace(" ", "_").lower()}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: shap_importance_{class_name.replace(' ', '_').lower()}.png")

# ============================================================================
# CLASS COMPARISON
# ============================================================================
print("\n" + "="*80)
print("📊 FEATURE IMPORTANCE COMPARISON ACROSS CLASSES")
print("="*80)

# Create comparison DataFrame
importance_comparison_df = pd.DataFrame(class_importances, index=feature_names)
importance_comparison_df['total'] = importance_comparison_df.sum(axis=1)
importance_comparison_df = importance_comparison_df.sort_values('total', ascending=False).head(15)
importance_comparison_df = importance_comparison_df.drop('total', axis=1)

print("\n📊 Top 15 Features Across All Classes:")
print(importance_comparison_df.round(4))

# Plot comparison
fig, ax = plt.subplots(figsize=(14, 8))
importance_comparison_df.plot(kind='barh', ax=ax, width=0.8)
ax.set_xlabel('Mean |SHAP value|', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
ax.set_title('Top 15 Features - Importance Comparison Across Risk Classes', 
             fontsize=14, fontweight='bold')
ax.legend(title='Risk Class', fontsize=10, loc='lower right')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('../figures/shap_class_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: shap_class_comparison.png")

# Save comparison
importance_comparison_df.to_csv('../data/results/shap_importance_by_class.csv')
print("✅ Saved: shap_importance_by_class.csv")

# ============================================================================
# BEESWARM PLOTS (Alternative to Summary)
# ============================================================================
print("\n" + "="*80)
print("📊 CREATING SHAP BEESWARM PLOTS")
print("="*80)

try:
    for class_idx, class_name in enumerate(le.classes_):
        print(f"\n🐝 Creating beeswarm for {class_name}...")
        
        if isinstance(shap_values, list):
            class_shap = shap_values[class_idx]
        else:
            class_shap = shap_values
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create explanation object
        shap_exp = shap.Explanation(
            values=class_shap,
            base_values=np.array([explainer.expected_value[class_idx]] * len(class_shap)),
            data=X_test_sample.values,
            feature_names=feature_names
        )
        
        shap.plots.beeswarm(shap_exp, max_display=20, show=False)
        plt.title(f'SHAP Beeswarm Plot - {class_name}', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'../figures/shap_beeswarm_{class_name.replace(" ", "_").lower()}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: shap_beeswarm_{class_name.replace(' ', '_').lower()}.png")
except Exception as e:
    print(f"\n⚠️ Beeswarm plots failed: {str(e)}")
    print("   Continuing with other visualizations...")

# ============================================================================
# WATERFALL PLOTS - INDIVIDUAL EXAMPLES
# ============================================================================
print("\n" + "="*80)
print("📊 WATERFALL PLOTS - INDIVIDUAL PREDICTIONS")
print("="*80)

# Get predictions
y_pred = model.predict(X_test_sample)

# Find examples of each class
examples = {}
for class_idx, class_name in enumerate(le.classes_):
    class_samples = np.where(y_pred == class_idx)[0]
    if len(class_samples) > 0:
        examples[class_name] = class_samples[0]
        print(f"   Found example for {class_name}: sample {class_samples[0]}")

# Create waterfall plots
for class_name, sample_idx in examples.items():
    print(f"\n💧 Creating waterfall for {class_name}...")
    
    try:
        pred_class = y_pred[sample_idx]
        
        if isinstance(shap_values, list):
            sample_shap = shap_values[pred_class][sample_idx]
        else:
            sample_shap = shap_values[sample_idx]
        
        # Create explanation
        shap_exp = shap.Explanation(
            values=sample_shap,
            base_values=explainer.expected_value[pred_class],
            data=X_test_sample.iloc[sample_idx].values,
            feature_names=feature_names
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.plots.waterfall(shap_exp, max_display=15, show=False)
        plt.title(f'SHAP Waterfall - {class_name} Prediction', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'../figures/shap_waterfall_{class_name.replace(" ", "_").lower()}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: shap_waterfall_{class_name.replace(' ', '_').lower()}.png")
    except Exception as e:
        print(f"   ❌ Error creating waterfall for {class_name}: {str(e)}")

# ============================================================================
# DEPENDENCE PLOTS
# ============================================================================
print("\n" + "="*80)
print("📊 SHAP DEPENDENCE PLOTS")
print("="*80)

# Get top 6 features
top_6_features = feature_importance_df.head(6)['feature'].tolist()
print(f"\n📈 Creating dependence plots for top 6 features:")
for feat in top_6_features:
    print(f"   - {feat}")

try:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, feat_name in enumerate(top_6_features):
        feat_idx = feature_names.index(feat_name)
        
        # Use High Risk class (most important)
        if isinstance(shap_values, list):
            class_shap = shap_values[0]  # High Risk
        else:
            class_shap = shap_values
        
        # Create scatter plot manually
        x = X_test_sample[feat_name].values
        y = class_shap[:, feat_idx]
        
        axes[idx].scatter(x, y, alpha=0.5, s=10)
        axes[idx].set_xlabel(feat_name, fontsize=10)
        axes[idx].set_ylabel('SHAP value', fontsize=10)
        axes[idx].set_title(f'{feat_name}\n(Impact on High Risk)', fontsize=10)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figures/shap_dependence_top6.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✅ Saved: shap_dependence_top6.png")
except Exception as e:
    print(f"\n⚠️ Dependence plots failed: {str(e)}")

print("\n" + "="*80)
print("✅ SHAP ANALYSIS COMPLETE!")
print("="*80)
print("\n📁 All visualizations saved in: ../figures/")
print("📊 SHAP data saved in: ../data/results/")
print("\n🎯 Key outputs:")
print("   - Global feature importance")
print("   - Per-class feature importance")
print("   - Class comparison chart")
print("   - Beeswarm plots (if successful)")
print("   - Waterfall plots")
print("   - Dependence plots")