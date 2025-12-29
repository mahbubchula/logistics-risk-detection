import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("📊 BINARY CLASSIFICATION VISUALIZATIONS (HIGH RISK DETECTION)")
print("="*80)

import os
os.makedirs('../figures/binary', exist_ok=True)

# Load data
X_test = np.load('../data/processed/X_test_scaled.npy')
y_test = np.load('../data/processed/y_test_risk_encoded.npy')
le = joblib.load('../models/label_encoder.pkl')

# Convert to binary: High Risk (0) vs Non-High Risk (1 or 2)
y_test_binary = (y_test == 0).astype(int)  # 1 = High Risk, 0 = Non-High Risk

# Load best model
model = joblib.load('../models/lightgbm_model.pkl')

# Get predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Convert predictions to binary
y_pred_binary = (y_pred == 0).astype(int)  # 1 = High Risk, 0 = Non-High Risk
y_pred_proba_binary = y_pred_proba[:, 0]  # Probability of High Risk

print(f"\n✅ Loaded LightGBM model and test data")
print(f"   Test samples: {len(y_test):,}")
print(f"   High Risk: {np.sum(y_test_binary == 1):,}")
print(f"   Non-High Risk: {np.sum(y_test_binary == 0):,}")

# ============================================================================
# FIGURE 1: MODEL COMPARISON - BINARY CLASSIFICATION
# ============================================================================
print("\n📊 Creating Figure 1: Binary Model Comparison...")

# Load all model results and convert to binary
results_df = pd.read_csv('../data/results/model_comparison_results.csv')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: High Risk F1-Score
ax = axes[0, 0]
models_sorted = results_df.sort_values('High_Risk_F1', ascending=True)
colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(models_sorted)))
bars = ax.barh(models_sorted['Model'], models_sorted['High_Risk_F1'], color=colors)
ax.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
ax.set_title('High Risk Detection Performance (F1-Score)', fontsize=14, fontweight='bold')
ax.set_xlim([0.75, 0.90])
for i, (v, model) in enumerate(zip(models_sorted['High_Risk_F1'], models_sorted['Model'])):
    ax.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=10)
    if model == 'LightGBM':
        bars[i].set_edgecolor('gold')
        bars[i].set_linewidth(3)

# Plot 2: High Risk Recall
ax = axes[0, 1]
models_sorted = results_df.sort_values('High_Risk_Recall', ascending=True)
colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(models_sorted)))
bars = ax.barh(models_sorted['Model'], models_sorted['High_Risk_Recall'], color=colors)
ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_title('High Risk Detection Recall', fontsize=14, fontweight='bold')
ax.set_xlim([0.75, 1.0])
for i, (v, model) in enumerate(zip(models_sorted['High_Risk_Recall'], models_sorted['Model'])):
    ax.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=10)
    if model == 'LightGBM':
        bars[i].set_edgecolor('gold')
        bars[i].set_linewidth(3)

# Plot 3: High Risk Precision
ax = axes[1, 0]
models_sorted = results_df.sort_values('High_Risk_Precision', ascending=True)
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(models_sorted)))
bars = ax.barh(models_sorted['Model'], models_sorted['High_Risk_Precision'], color=colors)
ax.set_xlabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('High Risk Detection Precision', fontsize=14, fontweight='bold')
ax.set_xlim([0.75, 0.80])
for i, (v, model) in enumerate(zip(models_sorted['High_Risk_Precision'], models_sorted['Model'])):
    ax.text(v + 0.0005, i, f'{v:.4f}', va='center', fontsize=10)
    if model == 'LightGBM':
        bars[i].set_edgecolor('gold')
        bars[i].set_linewidth(3)

# Plot 4: Precision-Recall Trade-off
ax = axes[1, 1]
ax.scatter(results_df['High_Risk_Recall'], results_df['High_Risk_Precision'],
          s=200, alpha=0.6, c=range(len(results_df)), cmap='rainbow', edgecolors='black')
for i, model in enumerate(results_df['Model']):
    ax.annotate(model, 
                (results_df.iloc[i]['High_Risk_Recall'], 
                 results_df.iloc[i]['High_Risk_Precision']),
                fontsize=9, ha='center', va='bottom')
ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Precision-Recall Trade-off (High Risk)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim([0.78, 1.0])
ax.set_ylim([0.74, 0.78])

plt.tight_layout()
plt.savefig('../figures/binary/01_model_comparison_binary.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 01_model_comparison_binary.png")

# ============================================================================
# FIGURE 2: CONFUSION MATRIX - BINARY (LARGE & CLEAR)
# ============================================================================
print("\n📊 Creating Figure 2: Binary Confusion Matrix...")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

cm_binary = confusion_matrix(y_test_binary, y_pred_binary)

fig, ax = plt.subplots(figsize=(10, 8))

# Create heatmap
sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-High Risk', 'High Risk'],
            yticklabels=['Non-High Risk', 'High Risk'],
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 16, 'weight': 'bold'},
            ax=ax, linewidths=2, linecolor='black')

ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.set_title('Confusion Matrix: High Risk Detection (LightGBM)', 
             fontsize=16, fontweight='bold', pad=20)

# Calculate metrics
accuracy = accuracy_score(y_test_binary, y_pred_binary)
precision = precision_score(y_test_binary, y_pred_binary)
recall = recall_score(y_test_binary, y_pred_binary)
f1 = f1_score(y_test_binary, y_pred_binary)

# Add metrics text
metrics_text = f"""
Performance Metrics:
━━━━━━━━━━━━━━━━━
Accuracy:  {accuracy:.2%}
Precision: {precision:.2%}
Recall:    {recall:.2%}
F1-Score:  {f1:.4f}

Key Insight:
Captures {recall:.1%} of all
High Risk scenarios!
"""

ax.text(1.15, 0.5, metrics_text, transform=ax.transAxes, 
        fontsize=12, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('../figures/binary/02_confusion_matrix_binary.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 02_confusion_matrix_binary.png")

# ============================================================================
# FIGURE 3: ROC CURVE - BINARY
# ============================================================================
print("\n📊 Creating Figure 3: ROC Curve...")

fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_proba_binary)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(fpr, tpr, color='#ff6b6b', lw=3, 
        label=f'LightGBM (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

# Mark operating point
operating_idx = np.argmin(np.abs(thresholds - 0.5))
ax.plot(fpr[operating_idx], tpr[operating_idx], 'go', markersize=15, 
        label=f'Operating Point (Threshold=0.5)\nTPR={tpr[operating_idx]:.3f}, FPR={fpr[operating_idx]:.3f}')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
ax.set_ylabel('True Positive Rate (Recall)', fontsize=13, fontweight='bold')
ax.set_title('ROC Curve: High Risk Detection', fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../figures/binary/03_roc_curve_binary.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 03_roc_curve_binary.png")

# ============================================================================
# FIGURE 4: PERFORMANCE SUMMARY TABLE
# ============================================================================
print("\n📊 Creating Figure 4: Performance Summary Table...")

# Top 5 models for binary
top_models = results_df.nlargest(5, 'High_Risk_F1')[
    ['Model', 'Accuracy', 'High_Risk_Precision', 'High_Risk_Recall', 'High_Risk_F1']
].copy()

top_models.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
top_models = top_models.round(4)

fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=top_models.values,
                colLabels=top_models.columns,
                cellLoc='center',
                loc='center',
                colColours=['#f0f0f0']*5)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Highlight best model (LightGBM)
for j in range(5):
    table[(1, j)].set_facecolor('#90EE90')
    table[(1, j)].set_text_props(weight='bold')

plt.title('Top 5 Models: High Risk Detection Performance', 
          fontsize=16, fontweight='bold', pad=20)
plt.savefig('../figures/binary/04_performance_table.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 04_performance_table.png")

# ============================================================================
# FIGURE 5: SHAP IMPORTANCE (BINARY FOCUS)
# ============================================================================
print("\n📊 Creating Figure 5: SHAP Feature Importance...")

# Load SHAP results
shap_importance = pd.read_csv('../data/results/shap_global_feature_importance.csv')
top_15 = shap_importance.head(15)

fig, ax = plt.subplots(figsize=(12, 8))

colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_15)))
bars = ax.barh(range(len(top_15)), top_15['importance'], color=colors, edgecolor='black')

ax.set_yticks(range(len(top_15)))
ax.set_yticklabels(top_15['feature'].str.replace('_', ' ').str.title())
ax.set_xlabel('Mean |SHAP Value|', fontsize=13, fontweight='bold')
ax.set_title('Top 15 Features for High Risk Detection (SHAP Analysis)', 
             fontsize=15, fontweight='bold', pad=20)
ax.invert_yaxis()

# Add values
for i, v in enumerate(top_15['importance']):
    ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('../figures/binary/05_shap_importance_binary.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 05_shap_importance_binary.png")

print("\n" + "="*80)
print("✅ ALL BINARY VISUALIZATIONS COMPLETE!")
print("="*80)
print("\n📁 All figures saved in: ../figures/binary/")
print("\n🎯 Generated 5 publication-ready figures:")
print("   1. Model Comparison (4 subplots)")
print("   2. Confusion Matrix with Metrics")
print("   3. ROC Curve")
print("   4. Performance Summary Table")
print("   5. SHAP Feature Importance")