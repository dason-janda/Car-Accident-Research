#!/usr/bin/env python3
"""
Traffic-normalized version of crash_heatmap (same style, same map extent).

Each crash is weighted by 100,000,000 / VMT of the road segment it happened
on, instead of counting as 1. A segment's weights add up to its crash rate
per 100 million vehicle miles traveled, so hotspots show where crashes are
high *relative to traffic*, not just where traffic is heavy.

Only crashes on AADT-counted segments can be normalized (~78% of St. George
crashes; local streets and ramps have no traffic counts). Segments flagged
as having a suspect AADT count are left out. Run crashRates.py first.

Outputs (written next to this script):
    crash_heatmap_normalized.html
    crash_heatmap_normalized.png
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RATES_CSV = ROOT / "crashRatesBySegment.csv"
STATE_CACHE = ROOT / "data" / "StatewideCrashes.pkl"

# Leaflet's heat layer scales color linearly to the hottest point, so a handful of
# crashes on very low-volume segments would wash out everything else. Clip the
# interactive weights at this percentile; the static map uses a log scale instead.
INTERACTIVE_WEIGHT_CLIP_PCT = 95


# --------------------------------------------------------------------------
# Data prep
# --------------------------------------------------------------------------

def load_weighted_crashes():
    """St. George crashes with a normalization weight, plus all St. George
    crash coordinates (used to match the raw heatmap's extent)."""
    if not RATES_CSV.exists():
        raise SystemExit(f"{RATES_CSV} not found; run crashRates.py first.")
    segments = pd.read_csv(RATES_CSV)
    segments = segments[~segments["check_aadt"]]
    segments["weight"] = 1e8 / segments["VMT"]

    state = pd.read_pickle(STATE_CACHE)
    sg = state[state["city_geo"] == "St. George"].copy()
    sg["Latitude"] = pd.to_numeric(sg["latitude"], errors="coerce")
    sg["Longitude"] = pd.to_numeric(sg["longitude"], errors="coerce")
    sg = sg[(sg["Latitude"].abs() > 0.001) & (sg["Longitude"].abs() > 0.001)]
    all_points = sg[["Latitude", "Longitude"]]

    mainline = sg[sg["type"] == "M"].copy()
    mainline["route"] = pd.to_numeric(mainline["route"], errors="coerce")
    mainline["milePoint"] = pd.to_numeric(mainline["milePoint"], errors="coerce")
    mainline = mainline.dropna(subset=["route", "milePoint"])
    mainline["route"] = mainline["route"].astype(int)

    # Same route + milepoint matching as crashRates.py.
    matched = pd.merge_asof(
        mainline.sort_values("milePoint"),
        segments[["route", "BeginPoint", "EndPoint", "weight", "rate", "AADT", "road"]].sort_values("BeginPoint"),
        left_on="milePoint", right_on="BeginPoint", by="route", direction="backward",
    )
    matched = matched[matched["weight"].notna() & (matched["milePoint"] <= matched["EndPoint"])]
    return matched, all_points


# --------------------------------------------------------------------------
# Interactive map
# --------------------------------------------------------------------------

def build_popup(row: pd.Series) -> str:
    fields = {
        "Crash ID": row["Crash_ID"],
        "Date": row["crash_date"],
        "Severity": row["crashSeverity_label"],
        "Road": str(row["road"]).title(),
        "Segment AADT": f"{row['AADT']:,.0f}",
        "Segment crash rate": f"{row['rate']:,.1f} per 100 MVMT",
    }
    rows = "".join(
        f"<tr><td style='padding-right:8px;color:#666'>{k}</td><td><b>{v}</b></td></tr>"
        for k, v in fields.items()
    )
    return f"<table style='font-size:12px;font-family:sans-serif'>{rows}</table>"


def make_interactive_map(crashes: pd.DataFrame, all_points: pd.DataFrame, outpath: Path) -> None:
    import folium
    from folium.plugins import HeatMap, MarkerCluster

    center = [all_points["Latitude"].mean(), all_points["Longitude"].mean()]
    m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")

    weights = crashes["weight"].clip(upper=np.percentile(crashes["weight"], INTERACTIVE_WEIGHT_CLIP_PCT))
    HeatMap(
        np.column_stack([crashes["Latitude"], crashes["Longitude"], weights]).tolist(),
        name=f"Crash rate density ({len(crashes):,} crashes on counted roads)",
        radius=10,
        blur=14,
        min_opacity=0.3,
    ).add_to(m)

    cluster = MarkerCluster(name="Individual crashes", show=False)
    for _, row in crashes.iterrows():
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=folium.Popup(build_popup(row), max_width=350),
            tooltip=f"{row['crashSeverity_label']} — segment rate {row['rate']:,.0f}",
        ).add_to(cluster)
    cluster.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(outpath))
    print(f"  wrote {outpath}")


# --------------------------------------------------------------------------
# Static map
# --------------------------------------------------------------------------

def make_static_map(crashes: pd.DataFrame, all_points: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 10))

    # Use the raw heatmap's extent so both figures share the same hex grid.
    extent = (all_points["Longitude"].min(), all_points["Longitude"].max(),
              all_points["Latitude"].min(), all_points["Latitude"].max())
    hb = ax.hexbin(
        crashes["Longitude"], crashes["Latitude"], C=crashes["weight"], reduce_C_function=np.sum,
        gridsize=70, extent=extent, mincnt=1, cmap="inferno_r", norm=LogNorm(), linewidths=0.2,
    )
    fig.colorbar(hb, ax=ax, shrink=0.7, label="Crashes per 100 million VMT per cell (log scale)")

    # One degree of longitude is shorter than one of latitude by cos(lat).
    # At 37°N that is a ~20% distortion if ignored.
    mean_lat = all_points["Latitude"].mean()
    ax.set_aspect(1 / math.cos(math.radians(mean_lat)))

    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Crash density normalized by traffic volume\n"
        f"St. George, Utah — n = {len(crashes):,} crashes on AADT-counted roads",
        fontsize=13, pad=12,
    )
    ax.grid(alpha=0.25, linewidth=0.5, zorder=0)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"  wrote {outpath}")


# --------------------------------------------------------------------------

def main():
    crashes, all_points = load_weighted_crashes()
    print(f"Normalizing {len(crashes):,} of {len(all_points):,} St. George crashes "
          f"({len(crashes) / len(all_points):.0%}); the rest have no traffic count")
    make_interactive_map(crashes, all_points, HERE / "crash_heatmap_normalized.html")
    make_static_map(crashes, all_points, HERE / "crash_heatmap_normalized.png")


if __name__ == "__main__":
    main()
