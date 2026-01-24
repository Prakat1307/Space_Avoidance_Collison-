# visualize_high_risk.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

try:
    print("Creating High-Risk Collision Visualizations...")
    print("=" * 60)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

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

    # Get top 50 highest risks
    top_50 = predictions.nlargest(50, 'risk_probability').reset_index(drop=True)

    print(f"✓ Loaded {len(predictions):,} predictions")
    print(f"✓ Analyzing top 50 highest risk events")

    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))

    # 1. TOP 20 HIGHEST RISK EVENTS - BAR CHART
    ax1 = plt.subplot(2, 3, 1)
    top_20 = top_50.head(20)
    colors = ['red' if x > 95 else 'orange' if x > 90 else 'yellow' for x in top_20['risk_probability']]
    bars = ax1.barh(range(len(top_20)), top_20['risk_probability'], color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(top_20)))
    ax1.set_yticklabels([f"ID: {int(id)}" for id in top_20['id']], fontsize=8)
    ax1.set_xlabel('Risk Probability (%)', fontsize=10, fontweight='bold')
    ax1.set_title('🔴 TOP 20 HIGHEST RISK COLLISIONS', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    # Add value labels
    for i, (idx, row) in enumerate(top_20.iterrows()):
        ax1.text(row['risk_probability'] + 1, i, f"{row['risk_probability']:.1f}%", 
                 va='center', fontsize=7, fontweight='bold')

    # 2. RISK DISTRIBUTION - ALL EVENTS
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(predictions['risk_probability'] * 100, bins=100, color='skyblue', 
             edgecolor='black', alpha=0.7, linewidth=0.5)
    ax2.axvline(x=50, color='red', linestyle='--', linewidth=2, label='50% Threshold')
    ax2.set_xlabel('Risk Probability (%)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Number of Events', fontsize=10, fontweight='bold')
    ax2.set_title('📊 OVERALL RISK DISTRIBUTION', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. RISK CATEGORIES - PIE CHART
    ax3 = plt.subplot(2, 3, 3)
    categories = {
        'Very Low\n(0-1%)': (predictions['risk_probability'] < 0.01).sum(),
        'Low\n(1-5%)': ((predictions['risk_probability'] >= 0.01) & 
                         (predictions['risk_probability'] < 0.05)).sum(),
        'Medium\n(5-20%)': ((predictions['risk_probability'] >= 0.05) & 
                             (predictions['risk_probability'] < 0.20)).sum(),
        'High\n(20-50%)': ((predictions['risk_probability'] >= 0.20) & 
                            (predictions['risk_probability'] < 0.50)).sum(),
        'Very High\n(50%+)': (predictions['risk_probability'] >= 0.50).sum(),
    }
    colors_pie = ['green', 'lightgreen', 'yellow', 'orange', 'red']
    explode = (0, 0, 0.05, 0.1, 0.2)
    wedges, texts, autotexts = ax3.pie(categories.values(), labels=categories.keys(), 
                                         autopct='%1.1f%%', colors=colors_pie, 
                                         explode=explode, startangle=90, textprops={'fontsize': 9})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax3.set_title('⚠️ RISK CATEGORY BREAKDOWN', fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('high_risk_dashboard.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: high_risk_dashboard.png")

except Exception as e:
    print(f"❌ An error occurred: {e}")