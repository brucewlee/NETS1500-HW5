import pathlib
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ---------------------------------------------------------------------------
# 1. Project‑relative paths
# ---------------------------------------------------------------------------

ROOT               = pathlib.Path(__file__).resolve().parent
MAP_DIR            = ROOT / "output_cable_map"
ANALYSIS_DIR       = ROOT / "output_analysis"
FIGURES_DIR        = ANALYSIS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

CABLE_FILE         = MAP_DIR      / "cables_with_landing_points.csv"
IMPORTANCE_FILE    = ANALYSIS_DIR / "cable_importance_tiers.csv"
RISK_FILE          = ANALYSIS_DIR / "cable_risk_tiers.csv"
OUT_FILE           = ANALYSIS_DIR / "optimized_material_assignment.csv"
STATS_FILE         = ANALYSIS_DIR / "material_optimization_stats.csv"

# ---------------------------------------------------------------------------
# 2. Material metadata
#    —– Prices are $/km estimated from various sources
# ---------------------------------------------------------------------------

MATERIALS = [
    ("CFRP, epoxy matrix (isotropic)", 10, 45000),
    ("Titanium alloys",                9, 27000),
    ("Nickel‑based superalloys",       8, 22000),
    ("GFRP, epoxy matrix (isotropic)", 7, 14000),
    ("Stainless steel (duplex grades)",6,  8000),
    ("Commercially pure titanium",     5, 19000),
    ("Cast magnesium alloys",          4,  6000),
    ("Zirconia",                       3, 25000),
    ("Silica glass",                   2,  4000),
    ("Bronze / Brass",                 1,  5500),
]
mat_df = pd.DataFrame(MATERIALS, columns=["material",
                                          "reliability_tier",
                                          "price_per_km"]).set_index("material")

# Convenience lookup dictionaries
MAT_BY_TIER = {row.reliability_tier: m for m, row in mat_df.iterrows()}
TIER_BY_MAT = mat_df["reliability_tier"].to_dict()

# ---------------------------------------------------------------------------
# 3. Load and merge cable‑level data
# ---------------------------------------------------------------------------

cables     = pd.read_csv(CABLE_FILE,     low_memory=False)
importance = (pd.read_csv(IMPORTANCE_FILE, low_memory=False)
              .rename(columns={'Unnamed: 0': 'cable_key',
                               'tier':        'importance_tier'}))
risk       = pd.read_csv(RISK_FILE,      low_memory=False)

KEY = 'cable_key'

df = (cables[[KEY] + [c for c in cables.columns if "length" in c.lower()] ]
      .drop_duplicates(KEY)
      .merge(importance[[KEY, "importance_tier"]], on=KEY)
      .merge(risk[[KEY, "risk_tier"]],        on=KEY))

# If length column missing, assume each cable roughly 1 km so results are still relative
if not any(col for col in df.columns if "length" in col.lower()):
    df["length_km"] = 1.0
else:
    # rename the first length‑like column to length_km for consistency
    length_col = next(col for col in df.columns if "length" in col.lower())
    df = df.rename(columns={length_col: "length_km"})
    clean_len = (df["length_km"].astype(str)                 # guarantees string
                         .str.replace(r"[^0-9.]", "", regex=True))  # strip ", km" etc.

    df["length_km"] = pd.to_numeric(clean_len, errors="coerce")  # '' → NaN, safe cast
    df["length_km"].fillna(df["length_km"].median(), inplace=True)

# ---------------------------------------------------------------------------
# 4. Assign materials by priority rule
# ---------------------------------------------------------------------------

def pick_material(importance_tier: int, risk_tier: int) -> str:
    priority = importance_tier + risk_tier
    if   priority >= 18: return MAT_BY_TIER[10]
    elif priority >= 16: return MAT_BY_TIER[9]
    elif priority >= 14: return MAT_BY_TIER[8]
    elif priority >= 12: return MAT_BY_TIER[7]
    elif priority >= 10: return MAT_BY_TIER[6]
    elif priority >=  8: return MAT_BY_TIER[5]
    elif priority >=  6: return MAT_BY_TIER[4]
    elif priority >=  4: return MAT_BY_TIER[3]
    elif priority >=  2: return MAT_BY_TIER[2]
    else:               return MAT_BY_TIER[1]

df["material"] = df.apply(lambda r: pick_material(r.importance_tier, r.risk_tier), axis=1)
df["reliability_tier"] = df["material"].map(TIER_BY_MAT)

# Calculate material costs
df["material_cost_per_km"] = df["material"].map(mat_df["price_per_km"].to_dict())
df["total_material_cost"] = df["length_km"] * df["material_cost_per_km"] / 1000  # Cost in thousands $

# ---------------------------------------------------------------------------
# 5. Expected outage‑cost model
#    – All factors scaled 0‑1 so results are unitless; only the %‑change matters.
# ---------------------------------------------------------------------------

def expected_cost(risk_tier, importance_tier, reliability_tier, length_km):
    risk_factor        = risk_tier        / 10      # 0.1 – 1
    importance_factor  = importance_tier  / 10
    reliability_factor = min (0.9, reliability_tier / 10)
    return (1 - reliability_factor) * risk_factor * importance_factor * length_km

# Baseline: stainless steel everywhere (tier 6)
df["baseline_cost"]   = df.apply(
    lambda r: expected_cost(r.risk_tier, r.importance_tier, 6, r.length_km), axis=1)

# Optimizd: material we just assigned
df["optimized_cost"]  = df.apply(
    lambda r: expected_cost(r.risk_tier, r.importance_tier, r.reliability_tier, r.length_km), axis=1)

# Calculate cost reduction per cable
df["cost_reduction"] = df["baseline_cost"] - df["optimized_cost"]
df["cost_reduction_pct"] = 100 * df["cost_reduction"] / df["baseline_cost"]

# ---------------------------------------------------------------------------
# 6. Generate detailed statistics
# ---------------------------------------------------------------------------

# Overall stats
baseline_total   = df["baseline_cost"].sum()
optimized_total  = df["optimized_cost"].sum()
pct_reduction    = 100 * (baseline_total - optimized_total) / baseline_total
total_cable_length = df["length_km"].sum()
total_material_cost = df["total_material_cost"].sum()

# Material usage stats
material_counts = Counter(df["material"])
material_length = df.groupby("material")["length_km"].sum()
material_cost = df.groupby("material")["total_material_cost"].sum()

# Cable category analysis
df["priority"] = df["importance_tier"] + df["risk_tier"]
df["category"] = pd.cut(df["priority"], 
                        bins=[0, 5, 10, 15, 20], 
                        labels=["Low", "Medium", "High", "Critical"])

category_stats = df.groupby("category").agg({
    "cable_key": "count",
    "length_km": "sum",
    "total_material_cost": "sum",
    "baseline_cost": "sum",
    "optimized_cost": "sum",
    "cost_reduction": "sum",
})
category_stats["cost_reduction_pct"] = 100 * category_stats["cost_reduction"] / category_stats["baseline_cost"]

# Tier analysis - breakdown by reliability tier
tier_stats = df.groupby("reliability_tier").agg({
    "cable_key": "count",
    "length_km": "sum",
    "total_material_cost": "sum",
    "baseline_cost": "sum",
    "optimized_cost": "sum",
    "cost_reduction": "sum",
})
tier_stats["cost_reduction_pct"] = 100 * tier_stats["cost_reduction"] / tier_stats["baseline_cost"]

# Top 10 cables by cost reduction
top_cables = df.sort_values("cost_reduction", ascending=False).head(10)

# Create a detailed stats dataframe
detailed_stats = pd.DataFrame([
    {"Metric": "Total Cable Length (km)", "Value": f"{total_cable_length:,.2f}"},
    {"Metric": "Total Material Cost (million $)", "Value": f"{total_material_cost:,.2f}"},
    {"Metric": "Baseline Expected Outage Cost", "Value": f"{baseline_total:,.4f}"},
    {"Metric": "Optimized Expected Outage Cost", "Value": f"{optimized_total:,.4f}"},
    {"Metric": "Absolute Cost Reduction", "Value": f"{baseline_total - optimized_total:,.4f}"},
    {"Metric": "Percentage Cost Reduction", "Value": f"{pct_reduction:,.2f}%"},
    {"Metric": "Number of Cables", "Value": f"{len(df)}"},
    {"Metric": "Average Length per Cable (km)", "Value": f"{df['length_km'].mean():,.2f}"},
    {"Metric": "Average Material Cost per Cable (thousand $)", "Value": f"{df['total_material_cost'].mean():,.2f}"},
    {"Metric": "Average Cost Reduction per Cable (%)", "Value": f"{df['cost_reduction_pct'].mean():,.2f}%"},
])

# ---------------------------------------------------------------------------
# 7. Generate visualizations
# ---------------------------------------------------------------------------

# Set style for plots
plt.style.use('seaborn-v0_8')

# Figure 1: Material Distribution
plt.figure(figsize=(12, 8))

# Sort materials by reliability tier for better visualization
sorted_materials = [m[0] for m in sorted(MATERIALS, key=lambda x: x[1], reverse=True)]
material_counts_sorted = [material_counts[mat] for mat in sorted_materials]
material_lengths_sorted = [material_length.get(mat, 0) for mat in sorted_materials]

# Create bar plot of material distribution (by count)
ax1 = plt.subplot(121)
bars = ax1.bar(range(len(sorted_materials)), material_counts_sorted, color='steelblue')
ax1.set_xticks(range(len(sorted_materials)))
ax1.set_xticklabels([m.split(',')[0] for m in sorted_materials], rotation=45, ha='right')
ax1.set_ylabel('Number of Cables')
ax1.set_title('Material Distribution by Cable Count')
ax1.grid(False)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.0f}', ha='center', va='bottom')

# Create bar plot of material distribution (by length)
ax2 = plt.subplot(122)
bars = ax2.bar(range(len(sorted_materials)), 
              [length/1000 for length in material_lengths_sorted], 
              color='darkgreen')
ax2.set_xticks(range(len(sorted_materials)))
ax2.set_xticklabels([m.split(',')[0] for m in sorted_materials], rotation=45, ha='right')
ax2.set_ylabel('Total Length (1000 km)')
ax2.set_title('Material Distribution by Cable Length')
ax2.grid(False)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
            f'{height:.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'material_distribution.png', dpi=300)
plt.close()

# Figure 2: Optimization Results
plt.figure(figsize=(12, 6))
plt.style.use('seaborn-v0_8')

# Plot 1: Category-wise cost comparison
ax1 = plt.subplot(121)
category_stats[['baseline_cost', 'optimized_cost']].plot(kind='bar', ax=ax1, 
                                                       color=['steelblue', 'darkgreen'])
ax1.set_title('Outage Cost by Priority Category', fontsize=14)
ax1.set_ylabel('Expected Outage Cost (relative units)', fontsize=12)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
ax1.grid(False)

# Plot 2: Before/After comparison (overall)
ax2 = plt.subplot(122)
labels = ['Baseline', 'Optimized']
heights = [baseline_total, optimized_total]
bars = ax2.bar(labels, heights, color=['steelblue', 'darkgreen'])
ax2.set_title('Overall Expected Outage Cost', fontsize=14)
ax2.set_ylabel('Expected Outage Cost (relative units)', fontsize=12)
ax2.grid(False)

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 5000,
            f'{height:,.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'optimization_results.png', dpi=300)
plt.close()

# ---------------------------------------------------------------------------
# 8. Save results and print headline figure
# ---------------------------------------------------------------------------

# Save the main results
df_out_cols = [KEY, "length_km", "importance_tier", "risk_tier",
               "material", "reliability_tier",
               "baseline_cost", "optimized_cost"]
df[df_out_cols].to_csv(OUT_FILE, index=False)

# Save detailed statistics to a separate file
detailed_stats.to_csv(STATS_FILE, index=False)

# Print summary
print(f"Baseline expected outage cost : {baseline_total:,.4f}  (relative units)")
print(f"Optimized expected outage cost: {optimized_total:,.4f}")
print(f"Reduction                    : {pct_reduction:5.2f} %")

print("\n--- Material Usage Statistics ---")
for material, count in material_counts.most_common():
    pct_cables = 100 * count / len(df)
    print(f"{material:<30}: {count:>4} cables ({pct_cables:5.1f}%), {material_length[material]:>10,.2f} km")

print("\n--- Top 10 Cables by Cost Reduction ---")
for _, row in top_cables.iterrows():
    print(f"{row['cable_key']:<30}: {row['cost_reduction']:>8,.2f} units ({row['cost_reduction_pct']:>6.2f}%)")

print("\n--- Cost Reduction by Priority Category ---")
print(category_stats[["cable_key", "cost_reduction", "cost_reduction_pct"]].to_string())

print("\n--- Cost Reduction by Reliability Tier ---")
print(tier_stats[["cable_key", "cost_reduction", "cost_reduction_pct"]].to_string())

if pct_reduction >= 30:
    print("\n Hypothesis PASSED – outage cost cut by ≥ 30 %.")
else:
    print("\n Hypothesis NOT met; consider tightening the rule or adding budget constraints.")

print(f"\nVisualization files generated:")
print(f"1. {FIGURES_DIR / 'material_distribution.png'}")
print(f"2. {FIGURES_DIR / 'optimization_results.png'}")

# ---------------------------------------------------------------------------
# End of script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not OUT_FILE.exists():
        sys.exit("Something went wrong; output file missing.")