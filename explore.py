#!/usr/bin/env python3
"""
explore_cables.py – betweenness, Tier-1 vs Tier-other, and world map
requirements:  pip install pandas networkx geopy plotly tqdm
"""

import sys, warnings, math
from pathlib import Path
import pandas as pd
import networkx as nx
from geopy.distance import great_circle
from tqdm import tqdm
import plotly.graph_objects as go

warnings.filterwarnings("ignore", message="Pandas requires version")

DATA_CSV   = "output_cable_map/cables_with_landing_points.csv"
TOP_N_SHOW = 20          # rows to print in each list
PCTL_T1    = 0.90        # top-10 % → Tier-1

# ── 1. Load data ────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_CSV)
print(f"{len(df):,} cable–landing-point rows loaded.")

lp_meta = (df.groupby("lp_id")
             .agg(lp_name=("lp_name","first"),
                  country=("country","first"),
                  lat=("latitude","first"),
                  lon=("longitude","first"))
             .to_dict(orient="index"))

# ── 2. Build graph ──────────────────────────────────────────────────────────
G = nx.Graph()
for lp_id, attrs in lp_meta.items():
    G.add_node(lp_id, **attrs)

print("Adding edges …")
for cable_key, grp in tqdm(df.groupby("cable_key"),
                           total=df["cable_key"].nunique()):
    lps = list(grp["lp_id"].unique())
    for i, u in enumerate(lps):
        for v in lps[i+1:]:
            dist = great_circle((lp_meta[u]["lat"], lp_meta[u]["lon"]),
                                (lp_meta[v]["lat"], lp_meta[v]["lon"])).km
            if G.has_edge(u, v):
                if dist < G[u][v]["length_km"]:
                    G[u][v]["length_km"] = dist
                    G[u][v]["cables"].add(cable_key)
            else:
                G.add_edge(u, v, length_km=dist, cables={cable_key})

print(f"Graph: {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")

# ── 3. Betweenness centrality ───────────────────────────────────────────────
bet = nx.betweenness_centrality(G, normalized=True, weight="length_km")
nx.set_node_attributes(G, bet, "betweenness")

print(f"\nTop {TOP_N_SHOW} landing-points by betweenness:")
for lp_id, score in sorted(bet.items(), key=lambda x: -x[1])[:TOP_N_SHOW]:
    info = lp_meta[lp_id]
    print(f"  {info['lp_name']:<35} {info['country']:<25}  {score:8.4f}")

# ── 4. Cable-level importance & tiering (Tier-1 vs other) ───────────────────
node_bt = pd.Series(bet, name="node_bt")
cab_bt  = (df[["cable_key", "lp_id"]]
           .merge(node_bt, left_on="lp_id", right_index=True)
           .groupby("cable_key")["node_bt"]
           .mean()
           .sort_values(ascending=False))

threshold_t1 = cab_bt.quantile(PCTL_T1)
tiers = cab_bt.apply(lambda s: "Tier-1" if s >= threshold_t1 else "Tier-other")
cable_tiers = pd.DataFrame({"score": cab_bt, "tier": tiers})

print(f"\nTier-1 threshold (≥ {PCTL_T1:.0%} percentile): {threshold_t1:.4f}")
print(f"Tier-1 cables: {sum(tiers == 'Tier-1'):,}")
print(f"Tier-other  : {sum(tiers == 'Tier-other'):,}")

def list_some(label, series):
    print(f"\n{label}:")
    for c, s in series.head(TOP_N_SHOW).items():
        print(f"  {c:<40}  {s:8.4f}")

list_some("► Highest-importance cables (Tier-1)", cab_bt[tiers == "Tier-1"])
list_some("► Sample lowest-importance cables (Tier-other)",
          cab_bt[tiers == "Tier-other"].tail(TOP_N_SHOW))

# ── 5. World map (unchanged except colour dict) ─────────────────────────────
tier_colour = {"Tier-1": "red", "Tier-other": "steelblue"}

# ──────────────────────────────────────────────────────────────────────────
# 5.  World map – full network with working toggle & legend
# ──────────────────────────────────────────────────────────────────────────
print("\nRendering world map …")

tier_edges  = []   # coloured by tier
grey_edges  = []   # plain grey duplicate

for u, v, attr in G.edges(data=True):
    # ─── inside the map-building loop, replace the min( … ) block ───
    edge_tier = min(
        (cable_tiers.loc[c]["tier"] for c in attr["cables"]),
        key=lambda t: {"Tier-1": 0, "Tier-other": 1}[t]     # ← updated
    )
    colour = tier_colour[edge_tier]


    lat = [lp_meta[u]["lat"], lp_meta[v]["lat"]]
    lon = [lp_meta[u]["lon"], lp_meta[v]["lon"]]

    tier_edges.append(
        go.Scattergeo(
            lat=lat, lon=lon, mode="lines",
            line=dict(width=1, color=colour), opacity=0.45,
            hoverinfo="skip", showlegend=False, visible=True
        )
    )
    grey_edges.append(
        go.Scattergeo(
            lat=lat, lon=lon, mode="lines",
            line=dict(width=0.7, color="grey"), opacity=0.3,
            hoverinfo="skip", showlegend=False, visible=False
        )
    )

# legend handles (one dummy trace per tier)
legend_traces = [
    go.Scattergeo(
        lat=[None], lon=[None], mode="lines",
        line=dict(width=6, color=tier_colour[t]),
        name=t, showlegend=True, hoverinfo="skip", visible=True
    )
    for t in ["Tier-1", "Tier-other"]
]

# landing-points (size ∝ true betweenness)
crit_nodes = pd.DataFrame(lp_meta).T
crit_nodes["bt"] = crit_nodes.index.map(bet)

node_trace = go.Scattergeo(
    lat=crit_nodes["lat"], lon=crit_nodes["lon"],
    text=(crit_nodes["lp_name"] +
          "<br>betweenness = " + crit_nodes["bt"].round(4).astype(str)),
    hovertemplate="%{text}<extra></extra>",
    mode="markers",
    marker=dict(size=crit_nodes["bt"] * 40 + 3,
                color="black", opacity=0.75),
    name="Landing point", showlegend=False
)

fig = go.Figure(data=tier_edges + grey_edges + legend_traces + [node_trace])

# helper to build visibility masks
n_tier  = len(tier_edges)
n_grey  = len(grey_edges)
n_leg   = len(legend_traces)
n_total = n_tier + n_grey + n_leg + 1  # +1 for node_trace

vis_coloured = [True]*n_tier + [False]*n_grey + [True]*n_leg + [True]
vis_grey     = [False]*n_tier + [True]*n_grey + [False]*n_leg + [True]

# toggle buttons
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            x=0.5, xanchor="center", y=1.10, yanchor="top",
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
    legend=dict(title="Cable tier", orientation="h",
                y=1.05, yanchor="bottom", x=0.02),
    title=("Global Submarine-Cable Network "
           "<span style='font-size:0.6em'>(use buttons to switch views)</span>"),
    height=650,
    geo=dict(showland=True, landcolor="rgb(230,230,230)")
)

fig.write_html("output/submarine_cable_map.html", include_plotlyjs=True)


# ────────────────────────────────────────────────────────────────────────────
# 6.  Save CSVs
# ────────────────────────────────────────────────────────────────────────────
Path("output").mkdir(exist_ok=True)
cable_tiers.to_csv("output/cable_criticality.csv")
pd.Series(bet, name="node_betweenness").to_csv("output/node_betweenness.csv")
print("\n✓ Results saved to output/ (cable_criticality.csv, node_betweenness.csv)")