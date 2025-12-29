import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

print("="*80)
print("📊 LOGISTICS & SUPPLY CHAIN DATASET - INITIAL EXPLORATION")
print("="*80)

# Load dataset
print("\n🔄 Loading dataset...")
df = pd.read_csv('../data/raw/dynamic_supply_chain_logistics_dataset.csv')

print("✅ Dataset loaded successfully!")
print("\n" + "="*80)

# Basic Information
print("\n📈 DATASET SHAPE")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

print("\n" + "="*80)
print("\n📋 COLUMN NAMES & DATA TYPES")
print(df.dtypes)

print("\n" + "="*80)
print("\n🔍 FIRST 5 ROWS")
print(df.head())

print("\n" + "="*80)
print("\n📊 STATISTICAL SUMMARY")
print(df.describe())

print("\n" + "="*80)
print("\n❓ MISSING VALUES")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Percentage': missing_pct
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
if len(missing_df) > 0:
    print(missing_df)
else:
    print("✅ No missing values found!")

print("\n" + "="*80)
print("\n🎯 TARGET VARIABLES IDENTIFIED")
target_vars = [
    'Disruption Likelihood Score',
    'Delay Probability', 
    'Risk Classification',
    'Delivery Time Deviation'
]

for var in target_vars:
    if var in df.columns:
        print(f"\n✅ {var}")
        if df[var].dtype == 'object':
            print(f"   Type: Categorical")
            print(f"   Unique values: {df[var].nunique()}")
            print(f"   Distribution:\n{df[var].value_counts()}")
        else:
            print(f"   Type: Numerical")
            print(f"   Min: {df[var].min():.4f}")
            print(f"   Max: {df[var].max():.4f}")
            print(f"   Mean: {df[var].mean():.4f}")
            print(f"   Std: {df[var].std():.4f}")

print("\n" + "="*80)
print("\n💾 Saving dataset info to processed folder...")

# Save basic info
with open('../data/processed/dataset_info.txt', 'w') as f:
    df.info(buf=f)
    
df.describe().to_csv('../data/processed/statistical_summary.csv')

print("✅ Exploration complete!")
print("="*80)