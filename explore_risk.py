#!/usr/bin/env python3
"""
explore_risk.py – Submarine Cable Risk Analysis Based on Natural Disasters

This script analyzes the risk to submarine cables based on their proximity to historical
natural disasters. We use coordinates to identify landing points that might be vulnerable
to different types of disasters and assign risk tiers to cables accordingly.

The main goals are to:
- Identify at-risk landing points based on nearby natural disasters
- Calculate risk scores for each cable based on its landing points
- Classify cables into Risk Tiers (1-10) where Tier 10 = highest risk
- Create an interactive world map showing the network with risk coloring

Requirements: pandas, networkx, geopy, plotly, tqdm (pip install these if needed)
"""

import sys, warnings, math, json
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
from geopy.distance import great_circle
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px

# Ignore those annoying pandas version warnings
warnings.filterwarnings("ignore", message="Pandas requires version")

# Where to find our data and how to configure the analysis
CABLE_CSV = "output_cable_map/cables_with_landing_points.csv"
DISASTER_CSV = "output_disaster/emdat_with_coords.csv"
TOP_N_SHOW = 20                  # How many items to show in our ranked lists
DISASTER_RADIUS_KM = 100         # Consider disasters within this distance of landing points
MAX_DISASTER_AGE_YEARS = 50      # How far back in history to consider disasters
MIN_DISASTER_YEAR = 1970         # Don't consider disasters before this year

# Disaster type risk weights - which disasters are most threatening to cables?
DISASTER_WEIGHTS = {
    # Geophysical disasters have highest impact on undersea infrastructure
    'Earthquake': 5.0,      # Undersea earthquakes can directly break cables
    'Volcanic activity': 3.0,  # Undersea volcanic activity can damage cables
    'Mass movement (dry)': 2.0,  # Landslides can impact coastal infrastructure
    
    # Meteorological disasters have moderate impact
    'Storm': 2.5,           # Storms affect shallow water sections and landing stations
    'Extreme temperature': 1.0,
    
    # Hydrological disasters also impact coastal areas
    'Flood': 2.0,           # Coastal flooding affects landing stations
    'Landslide': 3.0,       # Underwater landslides can break cables
    'Wave action': 4.0,     # Tsunamis can severely damage coastal infrastructure
    
    # Default weight for any other disaster types
    'DEFAULT': 1.0  
}


# Load the cable/landing point data and disaster data
df_cables = pd.read_csv(CABLE_CSV)
print(f"{len(df_cables):,} cable–landing-point rows loaded.")

# Create a lookup dictionary with metadata for each landing point
lp_meta = (df_cables.groupby("lp_id")
             .agg(lp_name=("lp_name","first"),
                  country=("country","first"),
                  lat=("latitude","first"),
                  lon=("longitude","first"))
             .to_dict(orient="index"))

# Load disaster data
df_disasters = pd.read_csv(DISASTER_CSV)

# Clean up disaster data - handle missing coordinates
df_disasters = df_disasters.dropna(subset=['Latitude', 'Longitude'])
print(f"{len(df_disasters):,} disaster events with valid coordinates loaded.")

# Filter to only include natural disasters (not technological/complex)
df_disasters = df_disasters[df_disasters['Disaster Group'] == 'Natural']
print(f"{len(df_disasters):,} natural disaster events after filtering.")

# Filter by year to focus on more recent/relevant disasters
current_year = 2025  # Using today's year
min_year = max(MIN_DISASTER_YEAR, current_year - MAX_DISASTER_AGE_YEARS)
df_disasters = df_disasters[df_disasters['Start Year'] >= min_year]
print(f"{len(df_disasters):,} disasters since {min_year}.")


# Calculate disaster risk for each landing point
print("Calculating landing point risk scores...")

# Dictionary to store risk scores for each landing point
lp_risk_scores = {}
lp_nearby_disasters = {}

# For each landing point, find nearby disasters
for lp_id, attrs in tqdm(lp_meta.items(), desc="Analyzing landing points"):
    lp_lat = attrs["lat"]
    lp_lon = attrs["lon"]
    
    if pd.isna(lp_lat) or pd.isna(lp_lon):
        # Skip landing points with missing coordinates
        lp_risk_scores[lp_id] = 0
        lp_nearby_disasters[lp_id] = []
        continue
    
    # Find all disasters within the specified radius
    nearby_disasters = []
    
    for _, disaster in df_disasters.iterrows():
        disaster_lat = disaster['Latitude']
        disaster_lon = disaster['Longitude']
        
        # Skip disasters with missing coordinates
        if pd.isna(disaster_lat) or pd.isna(disaster_lon):
            continue
        
        # Calculate distance between landing point and disaster location
        distance_km = great_circle(
            (lp_lat, lp_lon), 
            (disaster_lat, disaster_lon)
        ).km
        
        # If within radius, add to nearby disasters
        if distance_km <= DISASTER_RADIUS_KM:
            nearby_disasters.append({
                'disaster_id': disaster['DisNo.'],
                'distance_km': distance_km,
                'type': disaster['Disaster Type'],
                'subtype': disaster['Disaster Subtype'],
                'year': disaster['Start Year'],
                'total_deaths': disaster['Total Deaths'] if not pd.isna(disaster['Total Deaths']) else 0,
                'total_damage': disaster['Total Damage (\'000 US$)'] if not pd.isna(disaster['Total Damage (\'000 US$)']) else 0
            })
    
    # Calculate risk score based on nearby disasters
    risk_score = 0
    for disaster in nearby_disasters:
        # Get the weight for this disaster type
        disaster_type = disaster['type']
        weight = DISASTER_WEIGHTS.get(disaster_type, DISASTER_WEIGHTS['DEFAULT'])
        
        # Account for recency of disaster (more recent = higher weight)
        year_factor = 1.0
        if disaster['year'] is not None:
            years_ago = current_year - disaster['year']
            year_factor = math.exp(-0.05 * years_ago)  # Exponential decay with time
        
        # Impact factor based on deaths and damage
        impact_factor = 1.0
        if disaster['total_deaths'] > 0:
            # Log scale for deaths to avoid extreme values dominating
            impact_factor += min(3.0, math.log10(1 + disaster['total_deaths']))
        
        if disaster['total_damage'] > 0:
            # Log scale for damage with cap
            impact_factor += min(3.0, math.log10(1 + disaster['total_damage']))
        
        # Distance factor - closer disasters have more impact
        distance_factor = 1.0 - (disaster['distance_km'] / DISASTER_RADIUS_KM)
        
        # Calculate total weight for this disaster
        disaster_risk = weight * year_factor * impact_factor * distance_factor
        risk_score += disaster_risk
    
    lp_risk_scores[lp_id] = risk_score
    lp_nearby_disasters[lp_id] = nearby_disasters

# Assign risk tiers to landing points (1-10 scale)
lp_risk_values = list(lp_risk_scores.values())
if len(lp_risk_values) > 0 and max(lp_risk_values) > 0:
    # Function to assign tier based on percentile
    def assign_tier(value, all_values):
        """Assign a risk tier from 1-10 based on percentile."""
        if max(all_values) == 0:
            return 1  # If no risks, assign lowest tier
            
        # Calculate percentile
        percentile = 100 * (np.searchsorted(np.sort(all_values), value) / len(all_values))
        
        # Map percentile to tier (1-10)
        tier = min(10, max(1, math.ceil(percentile/10)))
        return tier
    
    lp_risk_tiers = {lp_id: assign_tier(score, lp_risk_values) 
                      for lp_id, score in lp_risk_scores.items()}
else:
    # If no risks found, assign all to tier 1
    lp_risk_tiers = {lp_id: 1 for lp_id in lp_risk_scores.keys()}


# Calculate cable risk based on landing points
print("Calculating cable risk scores...")

# Dictionary to store risk scores and tiers for each cable
cable_risks = {}

# For each cable, calculate the risk score based on its landing points
for cable_key, group in tqdm(df_cables.groupby("cable_key"), desc="Analyzing cables"):
    # Get unique landing points for this cable
    landing_points = group["lp_id"].unique()
    
    # Sum the risk scores of all landing points
    cable_risk = sum(lp_risk_scores.get(lp_id, 0) for lp_id in landing_points)
    
    # Also calculate the maximum landing point risk (alternative approach)
    max_lp_risk = max((lp_risk_scores.get(lp_id, 0) for lp_id in landing_points), default=0)
    
    # Store both metrics
    cable_risks[cable_key] = {
        'risk_score': cable_risk,
        'max_lp_risk': max_lp_risk,
        'landing_points': len(landing_points)
    }

# Get list of risk scores for percentile calculation
cable_risk_values = [data['risk_score'] for data in cable_risks.values()]

# Assign risk tiers to cables (1-10 scale)
if len(cable_risk_values) > 0 and max(cable_risk_values) > 0:
    cable_risk_tiers = {
        cable_key: assign_tier(data['risk_score'], cable_risk_values)
        for cable_key, data in cable_risks.items()
    }
else:
    # If no risks found, assign all to tier 1
    cable_risk_tiers = {cable_key: 1 for cable_key in cable_risks.keys()}

# Create DataFrame for easier handling
cable_risk_df = pd.DataFrame({
    'cable_key': list(cable_risks.keys()),
    'risk_score': [data['risk_score'] for data in cable_risks.values()],
    'max_lp_risk': [data['max_lp_risk'] for data in cable_risks.values()],
    'landing_points': [data['landing_points'] for data in cable_risks.values()],
    'risk_tier': [cable_risk_tiers[key] for key in cable_risks.keys()]
}).sort_values('risk_score', ascending=False)

# Print top N highest risk cables
print(f"\nTop {TOP_N_SHOW} cables by risk score:")
for i, (_, row) in enumerate(cable_risk_df.head(TOP_N_SHOW).iterrows()):
    print(f"{i+1}. Cable: {row['cable_key']:<40} Risk Tier: {row['risk_tier']} (Score: {row['risk_score']:.2f})")

# Print risk tier distribution
tier_counts = cable_risk_df['risk_tier'].value_counts().sort_index()
print("\nCable risk tier distribution:")
for tier, count in tier_counts.items():
    print(f"  Tier {tier}: {count} cables")


# Build a network graph connecting landing points
G = nx.Graph()
for lp_id, attrs in lp_meta.items():
    # Add risk information to node attributes
    node_attrs = attrs.copy()
    node_attrs['risk_score'] = lp_risk_scores.get(lp_id, 0)
    node_attrs['risk_tier'] = lp_risk_tiers.get(lp_id, 1)
    node_attrs['nearby_disasters'] = len(lp_nearby_disasters.get(lp_id, []))
    G.add_node(lp_id, **node_attrs)

# Connect landing points via submarine cables
print("Building network graph...")
for cable_key, grp in tqdm(df_cables.groupby("cable_key"), desc="Adding edges"):
    lps = list(grp["lp_id"].unique())
    for i, u in enumerate(lps):
        for v in lps[i+1:]:
            # Skip if either landing point is missing coordinates
            if (pd.isna(lp_meta[u]["lat"]) or pd.isna(lp_meta[u]["lon"]) or
                pd.isna(lp_meta[v]["lat"]) or pd.isna(lp_meta[v]["lon"])):
                continue
                
            # Calculate the great-circle distance between points
            dist = great_circle((lp_meta[u]["lat"], lp_meta[u]["lon"]),
                                (lp_meta[v]["lat"], lp_meta[v]["lon"])).km
            
            # If we already have an edge between these points, 
            # keep the edge data and add this cable to the set
            if G.has_edge(u, v):
                G[u][v]["cables"].add(cable_key)
            else:
                G.add_edge(u, v, length_km=dist, cables={cable_key})

print(f"Graph: {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")


# Create an interactive world map visualization
print("Rendering world map...")

# Color scheme for risk tiers - using a manual color list to avoid index issues
# Define risk colors manually - from light to dark red
risk_colors = [
    '#fff5f0',  # Tier 1 - lightest
    '#fee0d2',
    '#fcbba1',
    '#fc9272',
    '#fb6a4a',
    '#ef3b2c',
    '#cb181d',
    '#a50f15',
    '#67000d',
    '#4a000b'   # Tier 10 - darkest
]

# Function to get color based on tier
def get_tier_color(tier):
    # Ensure tier is in range 1-10
    tier_idx = max(1, min(10, tier)) - 1  # Convert 1-10 to 0-9 index
    return risk_colors[tier_idx]

# Generate tier colors dictionary for easy lookup
tier_colors = {i: get_tier_color(i) for i in range(1, 11)}

# Create edges colored by risk tier
risk_edges = []
grey_edges = []

for u, v, attr in G.edges(data=True):
    # Find which cables connect these landing points
    cables = attr.get("cables", set())
    
    # Find the tier of the highest risk cable on this edge
    if not cables:  # Skip if no cables (shouldn't happen)
        continue
        
    edge_tiers = [cable_risk_tiers.get(c, 1) for c in cables]
    if not edge_tiers:  # Skip if no valid cable tiers
        continue
        
    # Use the highest tier (highest risk) for this edge
    edge_tier = max(edge_tiers)
    colour = tier_colors[edge_tier]

    # Get the geographic coordinates for the edge
    lat = [lp_meta[u]["lat"], lp_meta[v]["lat"]]
    lon = [lp_meta[u]["lon"], lp_meta[v]["lon"]]

    # Add the colored version of this edge
    risk_edges.append(
        go.Scattergeo(
            lat=lat, lon=lon, mode="lines",
            line=dict(width=1, color=colour), opacity=0.6,
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
        name=f"Risk Tier {t}", showlegend=True, hoverinfo="skip", visible=True
    )
    for t in range(10, 0, -1)  # Descending order in legend
]

# Create markers for landing points - size reflects risk tier
lp_data = pd.DataFrame(lp_meta).T
lp_data["risk_score"] = lp_data.index.map(lambda x: lp_risk_scores.get(x, 0))
lp_data["risk_tier"] = lp_data.index.map(lambda x: lp_risk_tiers.get(x, 1))
lp_data["nearby_disasters"] = lp_data.index.map(lambda x: len(lp_nearby_disasters.get(x, [])))

node_trace = go.Scattergeo(
    lat=lp_data["lat"], lon=lp_data["lon"],
    text=(lp_data["lp_name"] + 
          "<br>Country: " + lp_data["country"] +
          "<br>Risk Tier: " + lp_data["risk_tier"].astype(str) + 
          "<br>Nearby disasters: " + lp_data["nearby_disasters"].astype(str)),
    hovertemplate="%{text}<extra></extra>",
    mode="markers",
    marker=dict(
        size=lp_data["risk_tier"] * 2 + 3,  # Size based on tier (1-10)
        color=lp_data["risk_tier"].apply(lambda t: get_tier_color(t)),
        opacity=0.75
    ),
    name="Landing point", showlegend=False
)

# Create markers for disaster events (optional - for context)
show_disasters = True
if show_disasters:
    # Only show a sample of disasters to avoid overwhelming the map
    max_disasters_on_map = 300
    # Prioritize more severe disasters for display
    sample_disasters = df_disasters.sample(min(max_disasters_on_map, len(df_disasters)))
    
    disaster_trace = go.Scattergeo(
        lat=sample_disasters["Latitude"], 
        lon=sample_disasters["Longitude"],
        text=(sample_disasters["Disaster Type"] + " (" + 
              sample_disasters["Start Year"].astype(str) + ")<br>" +
              sample_disasters["Country"]),
        hovertemplate="%{text}<extra></extra>",
        mode="markers",
        marker=dict(
            symbol="circle",
            size=4,
            color="rgba(255, 0, 0, 0.3)",
            line=dict(width=0.5, color="rgba(255, 0, 0, 0.8)")
        ),
        name="Disaster events", 
        showlegend=True,
        visible="legendonly"  # Hidden by default, can be toggled on
    )
else:
    disaster_trace = None

# Combine all our visualization elements
all_traces = risk_edges + grey_edges + legend_traces + [node_trace]
if disaster_trace is not None:
    all_traces.append(disaster_trace)

fig = go.Figure(data=all_traces)

# Set up visibility masks for our toggle button
n_risk = len(risk_edges)
n_grey = len(grey_edges)
n_leg = len(legend_traces)
n_other = 1 + (1 if disaster_trace is not None else 0)  # node_trace + optional disaster_trace

vis_coloured = [True]*n_risk + [False]*n_grey + [True]*n_leg + [True]*n_other
vis_grey = [False]*n_risk + [True]*n_grey + [False]*n_leg + [True]*n_other

# Add toggle buttons and finalize the map layout
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            x=0.5, xanchor="center", y=0, yanchor="top",
            showactive=True,
            buttons=[
                dict(label="Risk colours",
                     method="restyle",
                     args=["visible", vis_coloured]),
                dict(label="Plain grey",
                     method="restyle",
                     args=["visible", vis_grey]),
            ]
        )
    ],
    legend=dict(
        title="Risk Levels", 
        orientation="h",
        y=1.05, yanchor="bottom", x=0.02
    ),
    title=(
        "Submarine-Cable Network Disaster Risk Assessment "
        "<span style='font-size:0.6em'>"
        f"(disasters within {DISASTER_RADIUS_KM}km since {min_year})"
        "</span>"
    ),
    height=650,
    geo=dict(showland=True, landcolor="rgb(230,230,230)")
)

# Save the interactive map as an HTML file
Path("output_analysis").mkdir(exist_ok=True)
fig.write_html("output_analysis/submarine_cable_risk_map.html", include_plotlyjs=True)


# Save the results to CSV files for later analysis
cable_risk_df.to_csv("output_analysis/cable_risk_tiers.csv", index=False)

# Save landing point risk data
lp_risk_df = pd.DataFrame({
    'lp_id': list(lp_risk_scores.keys()),
    'lp_name': [lp_meta[lp_id]['lp_name'] for lp_id in lp_risk_scores.keys()],
    'country': [lp_meta[lp_id]['country'] for lp_id in lp_risk_scores.keys()],
    'risk_score': list(lp_risk_scores.values()),
    'risk_tier': [lp_risk_tiers[lp_id] for lp_id in lp_risk_scores.keys()],
    'nearby_disasters': [len(lp_nearby_disasters[lp_id]) for lp_id in lp_risk_scores.keys()]
}).sort_values('risk_score', ascending=False)

lp_risk_df.to_csv("output_analysis/landing_point_risk_tiers.csv", index=False)

print("\nResults saved to output directory:")
print("  - submarine_cable_risk_map.html")
print("  - cable_risk_tiers.csv")
print("  - landing_point_risk_tiers.csv")

# Print a summary of the highest risk landing points
print(f"\nTop {TOP_N_SHOW} landing points by risk score:")
for i, (_, row) in enumerate(lp_risk_df.head(TOP_N_SHOW).iterrows()):
    print(f"{i+1}. {row['lp_name']:<35} ({row['country']}) - Risk Tier: {row['risk_tier']} (Nearby disasters: {row['nearby_disasters']})")