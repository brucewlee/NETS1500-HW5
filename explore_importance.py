#!/usr/bin/env python3
"""
explore_cables.py – Analyzing the global submarine cable network

This script helps us understand the importance of different submarine cables and landing points
by calculating network metrics and visualizing the results on a world map.

The main goals are to:
- Identify critical landing points using betweenness centrality
- Classify cables and landing points into Importance Tiers (1-10)
- Create an interactive world map showing the entire network

Requirements: pandas, networkx, geopy, plotly, tqdm (pip install these if needed)
"""

import sys, warnings, math
from pathlib import Path
import pandas as pd
import networkx as nx
from geopy.distance import great_circle
from tqdm import tqdm
import plotly.graph_objects as go
import numpy as np

# Ignore those annoying pandas version warnings
warnings.filterwarnings("ignore", message="Pandas requires version")

# Where to find our data and how to configure the analysis
DATA_CSV   = "output_cable_map/cables_with_landing_points.csv"
TOP_N_SHOW = 20          # How many items to show in our ranked lists

# Function to assign importance tier (1-10) based on percentile
def assign_tier(value, all_values):
    """
    Assign an importance tier from 1-10 based on percentile ranking.
    Tier 10 = most important, Tier 1 = least important
    """
    if len(all_values) <= 1:
        return 5  # Default to middle tier if we only have one value
        
    # Calculate percentile of the value
    percentile = 100 * (np.searchsorted(np.sort(all_values), value) / len(all_values))
    
    # Map percentile to tier (1-10)
    # We use a simple formula: tier = ceil(percentile/10)
    # This ensures a reasonable distribution with more important items in higher tiers
    tier = min(10, max(1, math.ceil(percentile/10)))
    return tier


# Load the cable and landing point data
df = pd.read_csv(DATA_CSV)
print(f"{len(df):,} cable–landing-point rows loaded.")

# Create a lookup dictionary with metadata for each landing point
lp_meta = (df.groupby("lp_id")
             .agg(lp_name=("lp_name","first"),
                  country=("country","first"),
                  lat=("latitude","first"),
                  lon=("longitude","first"))
             .to_dict(orient="index"))


# Build a network graph connecting landing points
G = nx.Graph()
for lp_id, attrs in lp_meta.items():
    G.add_node(lp_id, **attrs)

# Now let's connect landing points via submarine cables
print("Adding edges between landing points...")
for cable_key, grp in tqdm(df.groupby("cable_key"),
                           total=df["cable_key"].nunique()):
    lps = list(grp["lp_id"].unique())
    for i, u in enumerate(lps):
        for v in lps[i+1:]:
            # Calculate the great-circle distance between points
            dist = great_circle((lp_meta[u]["lat"], lp_meta[u]["lon"]),
                                (lp_meta[v]["lat"], lp_meta[v]["lon"])).km
            # If we already have an edge between these points, 
            # keep the shortest distance and add this cable to the set
            if G.has_edge(u, v):
                if dist < G[u][v]["length_km"]:
                    G[u][v]["length_km"] = dist
                    G[u][v]["cables"].add(cable_key)
            else:
                G.add_edge(u, v, length_km=dist, cables={cable_key})

print(f"Graph: {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")


# Find the most critical landing points using betweenness centrality
#   Betweenness measures how often a node appears on shortest paths between other nodes
#   Higher values = more critical to the network
bet = nx.betweenness_centrality(G, normalized=True, weight="length_km")
nx.set_node_attributes(G, bet, "betweenness")

# Assign importance tiers (1-10) to landing points
bet_values = list(bet.values())
node_tiers = {node: assign_tier(score, bet_values) for node, score in bet.items()}
nx.set_node_attributes(G, node_tiers, "importance_tier")

# Let's see which landing points are the most critical to the network
print(f"\nTop {TOP_N_SHOW} landing-points by importance:")
for lp_id, score in sorted(bet.items(), key=lambda x: -x[1])[:TOP_N_SHOW]:
    info = lp_meta[lp_id]
    tier = node_tiers[lp_id]
    print(f"  {info['lp_name']:<35} {info['country']:<25}  Tier {tier} (score: {score:.4f})")


# Calculate cable importance and assign tiers
#   The importance of a cable is the average importance of its landing points
node_bt = pd.Series(bet, name="node_bt")
node_tier = pd.Series(node_tiers, name="node_tier")

# Calculate average betweenness for each cable
cab_bt = (df[["cable_key", "lp_id"]]
          .merge(node_bt, left_on="lp_id", right_index=True)
          .groupby("cable_key")["node_bt"]
          .mean()
          .sort_values(ascending=False))

# Assign importance tiers to cables
cab_tier_values = list(cab_bt.values)
cab_tiers = {cable: assign_tier(score, cab_tier_values) 
             for cable, score in cab_bt.items()}
cable_tiers = pd.DataFrame({
    "score": cab_bt,
    "tier": pd.Series(cab_tiers)
})

# Show the distribution of cables across tiers
tier_counts = cable_tiers["tier"].value_counts().sort_index()
print("\nCable importance tier distribution:")
for tier, count in tier_counts.items():
    print(f"  Tier {tier}: {count} cables")

# Helper function to display a portion of a sorted series
def list_some(label, series, tiers):
    print(f"\n{label}:")
    for c, s in series.head(TOP_N_SHOW).items():
        tier = tiers[c]
        print(f"  {c:<40}  Tier {tier} (score: {s:.4f})")

# Show some examples from highest and lowest tiers
list_some("► Highest-importance cables", cab_bt, cab_tiers)
list_some("► Sample lowest-importance cables", 
          cab_bt.sort_values().head(TOP_N_SHOW), cab_tiers)


# Create an interactive world map visualization
#   Define a colormap for our tiers
import plotly.express as px
colorscale = px.colors.sequential.Plasma_r  # Reversed so higher tiers = warmer colors

# Function to get color based on tier
def get_tier_color(tier):
    # Map tier (1-10) to color from our colorscale
    idx = (tier - 1) / 9  # normalize to 0-1 range
    return colorscale[min(9, max(0, int(idx * 10)))]  # Map to our color array (0-9 index)

# Generate tier colors dictionary for easy lookup
tier_colors = {i: get_tier_color(i) for i in range(1, 11)}

print("\nRendering world map...")

# We'll create two sets of edges:
# 1. Colored by tier
# 2. Plain grey (as an alternative view)
tier_edges = []
grey_edges = []

for u, v, attr in G.edges(data=True):
    # Find which cables connect these landing points
    cables = attr["cables"]
    
    # Find the tier of the most important cable on this edge
    if not cables:  # Skip if no cables (shouldn't happen)
        continue
        
    edge_tiers = [cable_tiers.loc[c]["tier"] for c in cables if c in cable_tiers.index]
    if not edge_tiers:  # Skip if no valid cable tiers
        continue
        
    # Use the highest tier (most important) for this edge
    edge_tier = max(edge_tiers)
    colour = tier_colors[edge_tier]

    # Get the geographic coordinates for the edge
    lat = [lp_meta[u]["lat"], lp_meta[v]["lat"]]
    lon = [lp_meta[u]["lon"], lp_meta[v]["lon"]]

    # Add the colored version of this edge
    tier_edges.append(
        go.Scattergeo(
            lat=lat, lon=lon, mode="lines",
            line=dict(width=1, color=colour), opacity=0.45,
            hoverinfo="skip", showlegend=False, visible=True
        )
    )
    # Add the grey version of the same edge (initially hidden)
    grey_edges.append(
        go.Scattergeo(
            lat=lat, lon=lon, mode="lines",
            line=dict(width=0.7, color="grey"), opacity=0.3,
            hoverinfo="skip", showlegend=False, visible=False
        )
    )

# Create legend entries (one per tier)
legend_traces = [
    go.Scattergeo(
        lat=[None], lon=[None], mode="lines",
        line=dict(width=6, color=tier_colors[t]),
        name=f"Tier {t}", showlegend=True, hoverinfo="skip", visible=True
    )
    for t in range(10, 0, -1)  # Descending order in legend
]

# Create markers for landing points - size reflects importance
crit_nodes = pd.DataFrame(lp_meta).T
crit_nodes["bt"] = crit_nodes.index.map(bet)
crit_nodes["tier"] = crit_nodes.index.map(node_tiers)

node_trace = go.Scattergeo(
    lat=crit_nodes["lat"], lon=crit_nodes["lon"],
    text=(crit_nodes["lp_name"] + 
          "<br>Country: " + crit_nodes["country"] +
          "<br>Importance: Tier " + crit_nodes["tier"].astype(str)),
    hovertemplate="%{text}<extra></extra>",
    mode="markers",
    marker=dict(
        size=crit_nodes["tier"] * 2 + 3,  # Size based on tier (1-10)
        color=crit_nodes["tier"].apply(lambda t: get_tier_color(t)),
        opacity=0.75
    ),
    name="Landing point", showlegend=False
)

# Combine all our visualization elements
fig = go.Figure(data=tier_edges + grey_edges + legend_traces + [node_trace])

# Set up visibility masks for our toggle button
n_tier = len(tier_edges)
n_grey = len(grey_edges)
n_leg = len(legend_traces)
n_total = n_tier + n_grey + n_leg + 1  # +1 for node_trace

vis_coloured = [True]*n_tier + [False]*n_grey + [True]*n_leg + [True]
vis_grey = [False]*n_tier + [True]*n_grey + [False]*n_leg + [True]

# Add toggle buttons and finalize the map layout
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            x=0.5, xanchor="center", y=0, yanchor="top",
            showactive=True,
            buttons=[
                dict(label="Tier colours",
                     method="restyle",
                     args=["visible", vis_coloured]),
                dict(label="Plain grey",
                     method="restyle",
                     args=["visible", vis_grey]),
            ]
        )
    ],
    legend=dict(
        title="Cable Importance", 
        orientation="h",
        y=1.05, yanchor="bottom", x=0.02
    ),
    title=(
        "Global Submarine-Cable Network by Importance Tier (1-10) "
        "<span style='font-size:0.6em'>(use buttons to switch views)</span>"
    ),
    height=650,
    geo=dict(showland=True, landcolor="rgb(230,230,230)")
)

# Save the interactive map as an HTML file
Path("output_analysis").mkdir(exist_ok=True)
fig.write_html("output_analysis/submarine_cable_map.html", include_plotlyjs=True)


# Save the results to CSV files for later analysis
Path("output_analysis").mkdir(exist_ok=True)

# Save the cable importance scores and tiers
cable_tiers.to_csv("output_analysis/cable_importance_tiers.csv")

# Save the landing point betweenness values and tiers
node_data = pd.DataFrame({
    "betweenness": pd.Series(bet),
    "importance_tier": pd.Series(node_tiers)
})
node_data.to_csv("output_analysis/node_importance_tiers.csv")

print("\nResults saved to output_output_analysis/ (cable_importance_tiers.csv, node_importance_tiers.csv)")