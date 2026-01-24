# visualize_report.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

try:
    print("Creating Report Visualization Dashboard...")
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

    # Report data
    total_samples = len(predictions)
    validation_accuracy = 99.75
    high_risk_count = predictions['high_risk'].sum()
    high_risk_percentage = predictions['high_risk'].mean() * 100

    print(f"✓ Loaded {len(predictions):,} predictions")
    print(f"✓ Creating comprehensive dashboard...")

    # Create main dashboard
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # ==================== ROW 1 ====================

    # 1. SYSTEM PERFORMANCE GAUGE
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')

    performance_text = f"""
    ╔══════════════════════════════════════╗
    ║   SYSTEM PERFORMANCE METRICS         ║
    ╠══════════════════════════════════════╣
    ║                                      ║
    ║  🎯 Model Type:                      ║
    ║     Random Forest Classifier         ║
    ║                                      ║
    ║  ✅ Validation Accuracy:             ║
    ║     {validation_accuracy}%                           ║
    ║                                      ║
    ║  📊 Features Used:                   ║
    ║     101 numeric features             ║
    ║                                      ║
    ║  🕒 Report Generated:                ║
    ║     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}              ║
    ║                                      ║
    ╚══════════════════════════════════════╝
    """

    ax1.text(0.5, 0.5, performance_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='darkblue', linewidth=2))

    # 2. DATASET OVERVIEW
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    dataset_text = f"""
    ╔══════════════════════════════════════╗
    ║      DATASET INFORMATION             ║
    ╠══════════════════════════════════════╣
    ║                                      ║
    ║  📦 Total Test Samples:              ║
    ║     {total_samples:,}                        ║
    ║                                      ║
    ║  🔴 High-Risk Events:                ║
    ║     {high_risk_count:,} ({high_risk_percentage:.2f}%)                  ║
    ║                                      ║
    ║  🟢 Low-Risk Events:                 ║
    ║     {total_samples - high_risk_count:,} ({100 - high_risk_percentage:.2f}%)              ║
    ║                                      ║
    ║  📈 Risk Range:                      ║
    ║     {predictions['risk_probability'].min()*100:.2f}% - {predictions['risk_probability'].max()*100:.2f}%                   ║
    ║                                      ║
    ╚══════════════════════════════════════╝
    """

    ax2.text(0.5, 0.5, dataset_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, edgecolor='darkgreen', linewidth=2))

    # 3. ACCURACY GAUGE CHART
    ax3 = fig.add_subplot(gs[0, 2])
    categories = ['Validation\nAccuracy', 'Training\nAccuracy']
    values = [99.75, 99.78]
    colors = ['#2ecc71', '#27ae60']
    bars = ax3.bar(categories, values, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
    ax3.set_ylim(99, 100)
    ax3.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax3.set_title('🎯 MODEL ACCURACY', fontsize=13, fontweight='bold', pad=10)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # ==================== ROW 2 ====================

    # 4. RISK PROBABILITY DISTRIBUTION
    ax4 = fig.add_subplot(gs[1, :])
    risk_probs = predictions['risk_probability'] * 100
    ax4.hist(risk_probs, bins=100, color='skyblue', edgecolor='black', alpha=0.7, linewidth=0.5)
    ax4.axvline(x=risk_probs.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {risk_probs.mean():.2f}%')
    ax4.axvline(x=risk_probs.median(), color='orange', linestyle='--', linewidth=2, 
                label=f'Median: {risk_probs.median():.2f}%')
    ax4.set_xlabel('Risk Probability (%)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Number of Events', fontsize=12, fontweight='bold')
    ax4.set_title('📊 RISK PROBABILITY DISTRIBUTION ACROSS ALL EVENTS', fontsize=14, fontweight='bold', pad=15)
    ax4.legend(fontsize=11, loc='upper right')
    ax4.grid(alpha=0.3)
    # Add statistics box
    stats_text = f'Min: {risk_probs.min():.2f}%\nMax: {risk_probs.max():.2f}%\nStd: {risk_probs.std():.2f}%'
    ax4.text(0.98, 0.97, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # ==================== ROW 3 ====================

    # 5. HIGH-RISK vs LOW-RISK PIE CHART
    ax5 = fig.add_subplot(gs[2, 0])
    sizes = [high_risk_count, total_samples - high_risk_count]
    labels = [f'High-Risk\n{high_risk_count:,}\n({high_risk_percentage:.2f}%)', 
              f'Low-Risk\n{total_samples - high_risk_count:,}\n({100 - high_risk_percentage:.2f}%)']
    colors_pie = ['#e74c3c', '#2ecc71']
    explode = (0.1, 0)
    wedges, texts, autotexts = ax5.pie(sizes, labels=labels, autopct='',
                                         colors=colors_pie, explode=explode, 
                                         startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax5.set_title('⚠️ RISK CLASSIFICATION', fontsize=13, fontweight='bold', pad=10)

    # 6. RISK CATEGORIES BREAKDOWN
    ax6 = fig.add_subplot(gs[2, 1])
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
    cat_colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']
    bars = ax6.bar(range(len(categories)), list(categories.values()), 
                   color=cat_colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax6.set_xticks(range(len(categories)))
    ax6.set_xticklabels(list(categories.keys()), fontsize=9, fontweight='bold')
    ax6.set_ylabel('Number of Events', fontsize=11, fontweight='bold')
    ax6.set_title('📈 DETAILED RISK CATEGORIES', fontsize=13, fontweight='bold', pad=10)
    ax6.grid(axis='y', alpha=0.3)
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 7. STATISTICS SUMMARY TABLE
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    stats_summary = f"""
    ╔══════════════════════════════════════╗
    ║    STATISTICAL SUMMARY               ║
    ╠══════════════════════════════════════╣
    ║                                      ║
    ║  Risk Probability:                   ║
    ║  ────────────────                    ║
    ║   • Mean:    {predictions['risk_probability'].mean()*100:6.2f}%          ║
    ║   • Median:  {predictions['risk_probability'].median()*100:6.2f}%          ║
    ║   • Min:     {predictions['risk_probability'].min()*100:6.2f}%          ║
    ║   • Max:     {predictions['risk_probability'].max()*100:6.2f}%         ║
    ║   • Std Dev: {predictions['risk_probability'].std()*100:6.2f}%          ║
    ║                                      ║
    ║  Event ID Range:                     ║
    ║  ────────────────                    ║
    ║   • First:   {int(predictions['id'].min()):5d}                  ║
    ║   • Last:    {int(predictions['id'].max()):5d}                 ║
    ║                                      ║
    ╚══════════════════════════════════════╝
    """

    ax7.text(0.5, 0.5, stats_summary, transform=ax7.transAxes,
             fontsize=9, verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='orange', linewidth=2))

    # ==================== ROW 4 ====================

    # 8. TOP 20 HIGHEST RISKS
    ax8 = fig.add_subplot(gs[3, :2])
    top_20 = predictions.nlargest(20, 'risk_probability')
    colors_bars = ['#c0392b' if x > 0.9 else '#e74c3c' if x > 0.5 else '#e67e22' 
                   for x in top_20['risk_probability']]
    bars = ax8.barh(range(len(top_20)), top_20['risk_probability'] * 100, 
                    color=colors_bars, edgecolor='black', linewidth=0.5, alpha=0.9)
    ax8.set_yticks(range(len(top_20)))
    ax8.set_yticklabels([f"#{i+1}: ID {int(id)}" for i, id in enumerate(top_20['id'])], fontsize=9)
    ax8.set_xlabel('Risk Probability (%)', fontsize=11, fontweight='bold')
    ax8.set_title('🔴 TOP 20 HIGHEST RISK COLLISION EVENTS', fontsize=13, fontweight='bold', pad=10)
    ax8.invert_yaxis()
    ax8.grid(axis='x', alpha=0.3)
    # Add value labels
    for i, (idx, row) in enumerate(top_20.iterrows()):
        ax8.text(row['risk_probability'] * 100 + 1, i, 
                 f"{row['risk_probability']*100:.1f}%", 
                 va='center', fontsize=8, fontweight='bold')

    # 9. NEXT STEPS & RECOMMENDATIONS
    ax9 = fig.add_subplot(gs[3, 2])
    ax9.axis('off')

    next_steps = f"""
    ╔══════════════════════════════════════╗
    ║       NEXT STEPS                     ║
    ╠══════════════════════════════════════╣
    ║                                      ║
    ║  ✓ System Status:                    ║
    ║    Operational ✅                     ║
    ║                                      ║
    ║  📋 Recommended Actions:             ║
    ║                                      ║
    ║  1️⃣  Review {high_risk_count:,} high-risk          ║
    ║     collision predictions            ║
    ║                                      ║
    ║  2️⃣  Submit results to               ║
    ║     Kaggle competition               ║
    ║                                      ║
    ║  3️⃣  Monitor top {(predictions['risk_probability'] > 0.9).sum()} critical        ║
    ║     events (>90% risk)               ║
    ║                                      ║
    ║  4️⃣  Fine-tune model for             ║
    ║     improved accuracy                ║
    ║                                      ║
    ╚══════════════════════════════════════╝
    """

    ax9.text(0.5, 0.5, next_steps, transform=ax9.transAxes,
             fontsize=9, verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8, edgecolor='darkred', linewidth=2))

    # Main title
    fig.suptitle('🛰️ SPACE COLLISION AVOIDANCE SYSTEM - COMPREHENSIVE REPORT DASHBOARD', 
                 fontsize=18, fontweight='bold', y=0.995)

    # ==================== CREATE SIMPLIFIED SUMMARY ====================

    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig2.patch.set_facecolor('white')

    # Panel 1: Key Metrics
    axes[0, 0].axis('off')
    key_metrics = f"""
    ╔════════════════════════════════════════════╗
    ║          KEY PERFORMANCE METRICS           ║
    ╠════════════════════════════════════════════╣
    ║                                            ║
    ║  📊 Total Test Samples:    {total_samples:,}         ║
    ║                                            ║
    ║  🎯 Model Accuracy:        {validation_accuracy}%            ║
    ║                                            ║
    ║  🔴 High-Risk Events:      {high_risk_count:,} ({high_risk_percentage:.2f}%)    ║
    ║                                            ║
    ╚════════════════════════════════════════════╝
    """

    axes[0, 0].text(0.5, 0.5, key_metrics, transform=axes[0, 0].transAxes,
                    fontsize=10, verticalalignment='center', horizontalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8, edgecolor='black', linewidth=2))

    # Panel 2: Risk Distribution
    ax = axes[0, 1]
    risk_probs = predictions['risk_probability'] * 100
    ax.hist(risk_probs, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Risk Probability (%)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of Events', fontsize=10, fontweight='bold')
    ax.set_title('📊 RISK PROBABILITY DISTRIBUTION', fontsize=12, fontweight='bold', pad=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Panel 3: Risk Classification Pie Chart
    ax = axes[1, 0]
    sizes = [high_risk_count, total_samples - high_risk_count]
    labels = [f'High-Risk\n{high_risk_count:,}\n({high_risk_percentage:.2f}%)', 
              f'Low-Risk\n{total_samples - high_risk_count:,}\n({100 - high_risk_percentage:.2f}%)']
    colors_pie = ['#e74c3c', '#2ecc71']
    explode = (0.1, 0)
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='',
                                         colors=colors_pie, explode=explode, 
                                         startangle=90, textprops={'fontsize': 9, 'fontweight': 'bold'})
    ax.set_title('⚠️ RISK CLASSIFICATION', fontsize=12, fontweight='bold', pad=10)

    # Panel 4: Top 20 Highest Risks
    ax = axes[1, 1]
    top_20 = predictions.nlargest(20, 'risk_probability')
    ax.barh(range(len(top_20)), top_20['risk_probability'] * 100, color='darkred', edgecolor='black', alpha=0.9)
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels([f"ID {int(id)}" for id in top_20['id']], fontsize=9)
    ax.set_xlabel('Risk Probability (%)', fontsize=10, fontweight='bold')
    ax.set_title('🔴 TOP 20 HIGHEST RISK EVENTS', fontsize=12, fontweight='bold', pad=10)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('simplified_report_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ Saved: simplified_report_summary.png")

except Exception as e:
    print(f"❌ An error occurred: {e}")