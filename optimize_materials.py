"""
optimize_materials.py – Testing the hypothesis about optimal material selection

This script analyzes how different material selection strategies affect the expected
outage costs for submarine cables. We use both importance tiers (from betweenness centrality)
and risk tiers (from disaster analysis) to guide material allocation.

The main goals are to:
- Test our hypothesis that strategic material allocation can reduce expected outage cost by 30%
- Compare different strategies for material assignment
- Visualize the cost benefits of optimization

Requirements: pandas, numpy, matplotlib, seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load the cable importance and risk data
importance_df = pd.read_csv("output_analysis/cable_importance_tiers.csv", index_col=0)
risk_df = pd.read_csv("output_analysis/cable_risk_tiers.csv")

# Rename columns in importance_df to avoid confusion
importance_df = importance_df.rename(columns={'score': 'importance_score', 'tier': 'importance_tier'})

# Merge the two datasets
cables_df = risk_df.merge(importance_df, left_on='cable_key', right_index=True, how='inner')
print(f"Combined dataset: {len(cables_df)} cables")

# Define the materials and their properties
materials = {
    'CFRP': {'reliability_tier': 10, 'cost_multiplier': 5.0},
    'Titanium_alloys': {'reliability_tier': 9, 'cost_multiplier': 4.0},
    'Nickel_superalloys': {'reliability_tier': 8, 'cost_multiplier': 3.0},
    'GFRP': {'reliability_tier': 7, 'cost_multiplier': 2.0},
    'Stainless_steel': {'reliability_tier': 6, 'cost_multiplier': 1.0},
    'Titanium_pure': {'reliability_tier': 5, 'cost_multiplier': 0.9},
    'Magnesium': {'reliability_tier': 4, 'cost_multiplier': 0.7},
    'Zirconia': {'reliability_tier': 3, 'cost_multiplier': 0.6},
    'Silica_glass': {'reliability_tier': 2, 'cost_multiplier': 0.5},
    'Bronze_Brass': {'reliability_tier': 1, 'cost_multiplier': 0.4},
}

# Simplify our model: assume a base cost proportional to landing points (as a proxy for cable length)
cables_df['base_cost'] = cables_df['landing_points'] * 1_000_000  # $1M per landing point as base cost

# STRATEGY 1: Baseline - All cables use stainless steel (reliability tier 6)
cables_df['baseline_material'] = 'Stainless_steel'
cables_df['baseline_reliability_tier'] = 6
cables_df['baseline_failure_factor'] = (11 - cables_df['baseline_reliability_tier']) / 10  # Higher tier = lower failure
cables_df['baseline_expected_cost'] = (
    cables_df['base_cost'] * 
    cables_df['baseline_failure_factor'] * 
    cables_df['risk_tier'] / 10  # Scale risk tier to 0-1 range
)

# Calculate baseline total cost (needed for all percentage reductions)
total_baseline_cost = cables_df['baseline_expected_cost'].sum()

# STRATEGY 2: Importance-Only - Assign materials based solely on importance tier
def assign_material_by_importance(importance_tier):
    if importance_tier >= 9:
        return 'CFRP'
    elif importance_tier >= 7:
        return 'Titanium_alloys'
    elif importance_tier >= 5:
        return 'Nickel_superalloys'
    elif importance_tier >= 3:
        return 'GFRP'
    else:
        return 'Stainless_steel'

cables_df['importance_material'] = cables_df['importance_tier'].apply(assign_material_by_importance)
cables_df['importance_reliability_tier'] = cables_df['importance_material'].map(lambda m: materials[m]['reliability_tier'])
cables_df['importance_failure_factor'] = (11 - cables_df['importance_reliability_tier']) / 10
cables_df['importance_expected_cost'] = (
    cables_df['base_cost'] * 
    cables_df['importance_failure_factor'] * 
    cables_df['risk_tier'] / 10
)

# Calculate importance-only total cost
total_importance_cost = cables_df['importance_expected_cost'].sum()
importance_reduction = ((total_baseline_cost - total_importance_cost) / total_baseline_cost) * 100

# Define function for material assignment based on weighted importance and risk
def assign_material_combined(row, imp_weight, risk_weight):
    combined_score = (imp_weight * row['importance_tier'] + 
                      risk_weight * row['risk_tier'])
    
    if combined_score >= 9:
        return 'CFRP'
    elif combined_score >= 7:
        return 'Titanium_alloys'
    elif combined_score >= 5:
        return 'Nickel_superalloys'
    elif combined_score >= 3:
        return 'GFRP'
    else:
        return 'Stainless_steel'

# SENSITIVITY ANALYSIS - How do different weightings affect the results?
print("\nSensitivity Analysis:\n-------------------")
weight_results = []

# Use finer increments for a more detailed sensitivity analysis
for importance_weight in np.linspace(0, 1, 21):  # 0 to 1 in steps of 0.05
    risk_weight = 1 - importance_weight
    
    cables_df['sensitivity_material'] = cables_df.apply(
        lambda row: assign_material_combined(row, importance_weight, risk_weight), 
        axis=1
    )
    cables_df['sensitivity_reliability_tier'] = cables_df['sensitivity_material'].map(
        lambda m: materials[m]['reliability_tier']
    )
    cables_df['sensitivity_failure_factor'] = (11 - cables_df['sensitivity_reliability_tier']) / 10
    cables_df['sensitivity_expected_cost'] = (
        cables_df['base_cost'] * 
        cables_df['sensitivity_failure_factor'] * 
        cables_df['risk_tier'] / 10
    )
    
    total_sensitivity_cost = cables_df['sensitivity_expected_cost'].sum()
    sensitivity_reduction = ((total_baseline_cost - total_sensitivity_cost) / total_baseline_cost) * 100
    
    weight_results.append({
        'importance_weight': importance_weight,
        'risk_weight': risk_weight,
        'cost_reduction': sensitivity_reduction,
        'total_cost': total_sensitivity_cost
    })
    
    # Only print a subset of results to keep output clean
    if importance_weight % 0.1 < 0.01 or importance_weight == 0.05:
        print(f"Importance {importance_weight:.2f} / Risk {risk_weight:.2f}: {sensitivity_reduction:.1f}% reduction")

# Find optimal weights
optimal_weights = max(weight_results, key=lambda x: x['cost_reduction'])
print(f"\nOptimal weights: Importance {optimal_weights['importance_weight']:.2f} / Risk {optimal_weights['risk_weight']:.2f}")
print(f"Maximum reduction: {optimal_weights['cost_reduction']:.1f}%")

# STRATEGY 3: Combined (Optimal) - Use the weights identified from sensitivity analysis
optimal_importance_weight = optimal_weights['importance_weight']
optimal_risk_weight = optimal_weights['risk_weight']

# Apply the optimal weights to create the combined strategy
cables_df['combined_material'] = cables_df.apply(
    lambda row: assign_material_combined(row, optimal_importance_weight, optimal_risk_weight),
    axis=1
)
cables_df['combined_reliability_tier'] = cables_df['combined_material'].map(lambda m: materials[m]['reliability_tier'])
cables_df['combined_failure_factor'] = (11 - cables_df['combined_reliability_tier']) / 10
cables_df['combined_expected_cost'] = (
    cables_df['base_cost'] * 
    cables_df['combined_failure_factor'] * 
    cables_df['risk_tier'] / 10
)

# Calculate combined strategy total cost
total_combined_cost = cables_df['combined_expected_cost'].sum()
combined_reduction = ((total_baseline_cost - total_combined_cost) / total_baseline_cost) * 100

print("\nResults:\n--------")
print(f"Baseline Strategy (All Stainless Steel): ${total_baseline_cost:,.2f}")
print(f"Importance-Only Strategy: ${total_importance_cost:,.2f} ({importance_reduction:.1f}% reduction)")
print(f"Combined Strategy (Optimal {optimal_importance_weight:.2f}/{optimal_risk_weight:.2f}): ${total_combined_cost:,.2f} ({combined_reduction:.1f}% reduction)")

# Check if our hypothesis is supported
hypothesis_confirmed = combined_reduction >= 30
print(f"\nHypothesis {'CONFIRMED' if hypothesis_confirmed else 'NOT CONFIRMED'}: ")
print(f"Combined strategy achieved {combined_reduction:.1f}% reduction vs. 30% hypothesized")

# Track material usage in each strategy
material_counts = {
    'baseline': cables_df['baseline_material'].value_counts(),
    'importance': cables_df['importance_material'].value_counts(),
    'combined': cables_df['combined_material'].value_counts()
}

print("\nMaterial Distribution:\n--------------------")
print("Baseline Strategy:")
print(material_counts['baseline'])
print("\nImportance-Only Strategy:")
print(material_counts['importance'])
print("\nCombined Strategy (Optimal):")
print(material_counts['combined'])

# Visualize the results
plt.figure(figsize=(12, 8))

# Cost comparison
plt.subplot(2, 2, 1)
costs = [total_baseline_cost/1e9, total_importance_cost/1e9, total_combined_cost/1e9]
labels = ['Baseline', 'Importance-Only', f'Combined\n({optimal_importance_weight:.2f}/{optimal_risk_weight:.2f})']
colors = ['gray', 'skyblue', 'darkgreen']
plt.bar(labels, costs, color=colors)
plt.ylabel('Expected Outage Cost (Billions $)')
plt.title('Cost Comparison by Strategy')
# Set y limit with 20% headroom
max_cost = max(costs)
plt.ylim(0, max_cost * 1.2)
for i, cost in enumerate(costs):
    plt.text(i, cost + (max_cost * 0.03), f'${cost:.2f}B', ha='center')

# Reduction percentages
plt.subplot(2, 2, 2)
reductions = [0, importance_reduction, combined_reduction]
plt.bar(labels, reductions, color=colors)
plt.ylabel('Cost Reduction (%)')
plt.title('Cost Reduction vs. Baseline')
# Set y limit with adequate headroom
max_reduction = max(reductions)
plt.ylim(0, max(50, max_reduction * 1.3))
plt.axhline(y=30, color='red', linestyle='--', label='Hypothesis Threshold (30%)')
plt.legend(prop={'size': 8})
for i, red in enumerate(reductions):
    if i > 0:
        plt.text(i, red + 2, f'{red:.1f}%', ha='center')

# Sensitivity analysis - improved with full data range
plt.subplot(2, 2, 3)
weights = [w['importance_weight'] for w in weight_results]
reductions = [w['cost_reduction'] for w in weight_results]
plt.plot(weights, reductions, 'o-', color='purple', markersize=3)
plt.xlabel('Importance Weight')
plt.ylabel('Cost Reduction (%)')
plt.title('Sensitivity to Importance/Risk Weights')
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.ylim(0, max(reductions) * 1.1)
plt.axhline(y=30, color='red', linestyle='--', label='30% Threshold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(prop={'size': 8})

# Add marker for optimal point
optimal_idx = weights.index(optimal_weights['importance_weight'])
plt.scatter([weights[optimal_idx]], [reductions[optimal_idx]], 
            color='red', s=80, zorder=5, label='Optimal', marker='*')

# Material distribution
plt.subplot(2, 2, 4)
materials_list = ['CFRP', 'Titanium_alloys', 'Nickel_superalloys', 'GFRP', 'Stainless_steel']
counts_baseline = [material_counts['baseline'].get(m, 0) for m in materials_list]
counts_importance = [material_counts['importance'].get(m, 0) for m in materials_list]
counts_combined = [material_counts['combined'].get(m, 0) for m in materials_list]

x = np.arange(len(materials_list))
width = 0.25

plt.bar(x - width, counts_baseline, width, label='Baseline', color='gray')
plt.bar(x, counts_importance, width, label='Importance-Only', color='skyblue')
plt.bar(x + width, counts_combined, width, label=f'Combined ({optimal_importance_weight:.2f}/{optimal_risk_weight:.2f})', color='darkgreen')
plt.xlabel('Material')
plt.ylabel('Number of Cables')
plt.title('Material Distribution by Strategy')
plt.xticks(x, materials_list, rotation=45)
# Set y limit with headroom
max_count = max(max(counts_baseline), max(counts_importance), max(counts_combined))
plt.ylim(0, max_count * 1.2)
plt.legend(prop={'size': 8})

plt.tight_layout()
Path("output_analysis").mkdir(exist_ok=True)
plt.savefig("output_analysis/optimization_results.png", dpi=300)

# Create a more detailed plot of just the sensitivity analysis
plt.figure(figsize=(10, 6))
plt.plot(weights, reductions, 'o-', color='purple', linewidth=2)
plt.xlabel('Importance Weight (Risk Weight = 1 - Importance)')
plt.ylabel('Cost Reduction (%)')
plt.title('Detailed Sensitivity Analysis')
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(y=30, color='red', linestyle='--', label='30% Hypothesis Threshold')

# Highlight the optimal point
plt.scatter([weights[optimal_idx]], [reductions[optimal_idx]], 
            color='red', s=100, zorder=5, label=f'Optimal: {optimal_weights["importance_weight"]:.2f}/{optimal_weights["risk_weight"]:.2f}')

# Add annotations for key points
plt.annotate(f'Maximum: {optimal_weights["cost_reduction"]:.1f}%', 
             xy=(weights[optimal_idx], reductions[optimal_idx]),
             xytext=(weights[optimal_idx] + 0.1, reductions[optimal_idx]),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=10)

plt.legend()
plt.tight_layout()
plt.savefig("output_analysis/sensitivity_detailed.png", dpi=300)

# Save the optimized cable data
cables_df.to_csv("output_analysis/optimization_results.csv", index=False)
# Also save the sensitivity analysis results
pd.DataFrame(weight_results).to_csv("output_analysis/sensitivity_analysis.csv", index=False)

print("\nResults saved to:")
print("  - output_analysis/optimization_results.png")
print("  - output_analysis/sensitivity_detailed.png")
print("  - output_analysis/optimization_results.csv")
print("  - output_analysis/sensitivity_analysis.csv")

print("\nCONCLUSION:")
if hypothesis_confirmed:
    print("Our hypothesis is CONFIRMED. Strategic material allocation based on both")
    print("importance (betweenness centrality) and risk factors significantly reduces")
    print(f"expected outage costs by {combined_reduction:.1f}%, exceeding our 30% target.")
else:
    print("Our hypothesis is NOT CONFIRMED. While strategic material allocation does")
    print(f"reduce costs by {combined_reduction:.1f}%, it falls short of our 30% target.")
print(f"\nThe optimal strategy gives weight of {optimal_weights['importance_weight']:.2f} to importance")
print(f"and {optimal_weights['risk_weight']:.2f} to risk, achieving a maximum reduction of {optimal_weights['cost_reduction']:.1f}%.")