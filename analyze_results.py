# analyze_results.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

print("Analyzing Collision Predictions...")
print("=" * 60)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
predictions_file = os.path.join(script_dir, 'predictions_fixed.csv')

# Load predictions
if not os.path.exists(predictions_file):
    raise FileNotFoundError(f"File not found: {predictions_file}")
predictions = pd.read_csv(predictions_file)

# Check required columns
required_columns = ['risk_probability', 'high_risk', 'id']
for col in required_columns:
    if col not in predictions.columns:
        raise ValueError(f"Missing required column: {col}")

print(f"\n📊 PREDICTION OVERVIEW")
print(f"{'='*60}")
print(f"Total predictions: {len(predictions):,}")
print(f"\nColumns: {list(predictions.columns)}")

print(f"\n📈 RISK STATISTICS")
print(f"{'='*60}")
print(f"Mean risk probability: {predictions['risk_probability'].mean():.4%}")
print(f"Median risk probability: {predictions['risk_probability'].median():.4%}")
print(f"Max risk probability: {predictions['risk_probability'].max():.4%}")
print(f"Min risk probability: {predictions['risk_probability'].min():.4%}")

print(f"\n⚠️ HIGH-RISK EVENTS")
print(f"{'='*60}")
high_risk_count = predictions['high_risk'].sum()
print(f"Total high-risk events: {high_risk_count} ({high_risk_count/len(predictions)*100:.2f}%)")

# Show top 10 highest risk events
print(f"\n🔴 TOP 10 HIGHEST RISK COLLISIONS:")
print(f"{'='*60}")
top_10 = predictions.nlargest(10, 'risk_probability')
for idx, row in top_10.iterrows():
    print(f"  ID: {int(row['id']):5d}  |  Risk: {row['risk_probability']:.2%}  |  High-Risk: {'YES' if row['high_risk']==1 else 'NO'}")

# Risk categories
print(f"\n📊 RISK CATEGORIES")
print(f"{'='*60}")
categories = {
    'Very Low (0-1%)': (predictions['risk_probability'] < 0.01).sum(),
    'Low (1-5%)': ((predictions['risk_probability'] >= 0.01) & (predictions['risk_probability'] < 0.05)).sum(),
    'Medium (5-20%)': ((predictions['risk_probability'] >= 0.05) & (predictions['risk_probability'] < 0.20)).sum(),
    'High (20-50%)': ((predictions['risk_probability'] >= 0.20) & (predictions['risk_probability'] < 0.50)).sum(),
    'Very High (50%+)': (predictions['risk_probability'] >= 0.50).sum(),
}

for category, count in categories.items():
    percentage = count/len(predictions)*100
    bar = '█' * int(percentage)
    print(f"  {category:20s}: {count:5d} ({percentage:5.2f}%) {bar}")

# Create visualizations
print(f"\n📊 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Risk Distribution (Histogram)
axes[0, 0].hist(predictions['risk_probability'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
axes[0, 0].set_xlabel('Risk Probability')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Risk Probability Distribution')
axes[0, 0].grid(True, alpha=0.3)

# 2. Risk Distribution (Log scale)
risk_nonzero = predictions[predictions['risk_probability'] > 0]['risk_probability'] + 1e-10
axes[0, 1].hist(np.log10(risk_nonzero), bins=50, edgecolor='black', alpha=0.7, color='coral')
axes[0, 1].set_xlabel('Log10(Risk Probability)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Risk Probability Distribution (Log Scale)')
axes[0, 1].grid(True, alpha=0.3)

# 3. Top Risks
sorted_risk = predictions.nlargest(100, 'risk_probability')
axes[1, 0].plot(range(len(sorted_risk)), sorted_risk['risk_probability'].values, 'o-', markersize=4)
axes[1, 0].set_xlabel('Rank')
axes[1, 0].set_ylabel('Risk Probability')
axes[1, 0].set_title('Top 100 Highest Risk Events')
axes[1, 0].grid(True, alpha=0.3)

# 4. Risk Categories Pie Chart
category_data = list(categories.values())
category_labels = [label.split('(')[0].strip() for label in categories.keys()]
colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
axes[1, 1].pie(category_data, labels=category_labels, autopct='%1.1f%%', colors=colors, startangle=90)
axes[1, 1].set_title('Risk Category Distribution')

plt.tight_layout()
output_file = os.path.join(script_dir, 'prediction_analysis.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_file}")

# Save detailed results
print(f"\n💾 Saving detailed analysis...")
detailed_file = os.path.join(script_dir, 'detailed_analysis.txt')
with open(detailed_file, 'w') as f:
    f.write("="*60 + "\n")
    f.write("DETAILED COLLISION RISK ANALYSIS\n")
    f.write("="*60 + "\n\n")
    
    f.write("TOP 50 HIGHEST RISK EVENTS:\n")
    f.write("-"*60 + "\n")
    top_50 = predictions.nlargest(50, 'risk_probability')
    for idx, row in top_50.iterrows():
        f.write(f"Rank {idx+1:2d}: ID={int(row['id']):5d}, Risk={row['risk_probability']:.4%}, High-Risk={'YES' if row['high_risk']==1 else 'NO'}\n")

print(f"✅ Saved: {detailed_file}")

print(f"\n{'='*60}")
print("✅ ANALYSIS COMPLETE!")
print(f"{'='*60}")
print("\nFiles created:")
print(f"  📄 {detailed_file}")
print(f"  📊 {output_file}")