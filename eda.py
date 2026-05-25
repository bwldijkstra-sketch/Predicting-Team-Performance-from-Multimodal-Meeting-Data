import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")
sns.set_palette("muted")

# Load data
struct = pd.read_csv('ami_structural_features.csv')
y_df   = pd.read_csv('y_performance_final.csv')

# Keep only meetings present in both
df = pd.merge(y_df, struct, on='meeting_id')
print(f"Meetings after merge: {len(df)}")
print(f"Structural features:  {struct.shape[1] - 1} columns")

targets = ['overall_performance', 'satisfaction', 
           'cohesiveness', 'leadership', 'information_processing']

print(df[targets].describe().round(3))

fig, axes = plt.subplots(1, 5, figsize=(18, 4))

for ax, col in zip(axes, targets):
    ax.hist(df[col], bins=15, edgecolor='white', linewidth=0.5)
    ax.axvline(df[col].median(), color='red', 
               linestyle='--', linewidth=1.5, label='Median')
    ax.set_title(col.replace('_', '\n'), fontsize=10)
    ax.set_xlabel('Score (1–7)')
    ax.legend(fontsize=8)

plt.suptitle('Distribution of Performance Dimensions', 
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('eda_y_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

for col in targets:
    median = df[col].median()
    df[f'{col}_binary'] = (df[col] > median).astype(int)
    counts = df[f'{col}_binary'].value_counts()
    print(f"{col}: median={median:.3f} | "
          f"low={counts[0]} ({counts[0]/len(df)*100:.1f}%) | "
          f"high={counts[1]} ({counts[1]/len(df)*100:.1f}%)")
    

fig, ax = plt.subplots(figsize=(7, 6))
corr = df[targets].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', 
            cmap='coolwarm', center=0, ax=ax,
            linewidths=0.5, vmin=-1, vmax=1)
ax.set_title('Correlation Between Performance Dimensions', fontsize=12)
plt.tight_layout()
plt.savefig('eda_target_correlations.png', dpi=150, bbox_inches='tight')
plt.show()

feature_cols = [c for c in struct.columns if c != 'meeting_id']
print(f"Number of structural features: {len(feature_cols)}")
print(f"\nFeature names:\n{feature_cols}")

# Check for missing values
missing = df[feature_cols].isnull().sum()
print(f"\nFeatures with missing values: {(missing > 0).sum()}")