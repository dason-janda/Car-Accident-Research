#!/usr/bin/env python3
"""
Map of St. George road segments colored by crash rate per 100 million
vehicle miles traveled (MVMT), so busy roads aren't flagged just for
carrying more traffic.

Reads crashRatesBySegment.csv (run crashRates.py first) and draws each AADT
segment by cutting UDOT's route centerlines (UGRC Open SGID) at the
segment's begin/end milepoints.

Outputs (written next to this script):
    crash_rate_map.html   interactive map: crash rate layer + anomaly layer
    crash_rate_map.png    static figure for print/poster use
"""

from bisect import bisect_right
from pathlib import Path

import geopandas as gpd
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, to_hex
from matplotlib.lines import Line2D
from shapely.geometry import MultiLineString
from shapely.ops import substring
from sqlalchemy import create_engine, text

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RATES_CSV = ROOT / "normalized_crash_rates" / "crashRatesBySegment.csv"
CITY_CRASHES = ROOT / "data" / "AccidentData21-26.csv.xlsx"
SGID = "postgresql://agrc:agrc@opensgid.ugrc.utah.gov:5432/opensgid"
METERS_PER_MILE = 1609.344

RATE_BINS = [0, 50, 100, 200, 300, 500, 1000]
COLORS = [to_hex(c) for c in plt.get_cmap("RdYlGn_r")([i / (len(RATE_BINS) - 1) for i in range(len(RATE_BINS))])]

TOOLTIP_FIELDS = {
    "road": "Road",
    "RouteID": "Route",
    "BeginPoint": "Begin MP",
    "EndPoint": "End MP",
    "AADT": "AADT",
    "crashes": "Crashes (5 yr)",
    "ka_crashes": "Serious/fatal",
    "rate": "Rate per 100 MVMT",
    "critical_rate": "Critical rate",
    "rate_ratio": "Rate / critical",
    "check_aadt": "AADT suspect",
}


def rate_color(rate: float) -> str:
    return COLORS[bisect_right(RATE_BINS, rate) - 1]


# --------------------------------------------------------------------------
# Segment geometry
# --------------------------------------------------------------------------

def load_route_lines(route_ids) -> gpd.GeoSeries:
    """UDOT ALRS route centerlines, in UTM 12N meters."""
    engine = create_engine(SGID)
    query = text("SELECT route_id, shape FROM transportation.udot_routes_alrs WHERE route_id = ANY(:ids)")
    with engine.connect() as conn:
        routes = gpd.read_postgis(query, conn, geom_col="shape", params={"ids": list(route_ids)})
    return routes.set_index("route_id")["shape"]


def cut_segment(route_line, begin_mi: float, end_mi: float):
    """On UDOT's ALRS a milepoint is the distance along the route line, so a segment
    is the stretch of line between those two distances. Multi-part routes are
    measured part by part in stored order."""
    parts = list(route_line.geoms) if isinstance(route_line, MultiLineString) else [route_line]
    begin, end = begin_mi * METERS_PER_MILE, end_mi * METERS_PER_MILE
    pieces, offset = [], 0.0
    for part in parts:
        lo, hi = max(begin, offset), min(end, offset + part.length)
        if hi > lo:
            pieces.append(substring(part, lo - offset, hi - offset))
        offset += part.length
    if not pieces:
        return None
    return pieces[0] if len(pieces) == 1 else MultiLineString(pieces)


def build_segments() -> gpd.GeoDataFrame:
    if not RATES_CSV.exists():
        raise SystemExit(f"{RATES_CSV} not found; run crashRates.py first.")
    rates = pd.read_csv(RATES_CSV)
    rates = rates[rates["st_george_crashes"] > 0].copy()

    lines = load_route_lines(rates["RouteID"].unique())
    rates["geometry"] = [
        cut_segment(lines[r.RouteID], r.BeginPoint, r.EndPoint) for r in rates.itertuples()
    ]
    rates = rates.dropna(subset=["geometry"])

    rates["AADT"] = rates["AADT"].round(0).astype(int)
    rates[["rate", "critical_rate"]] = rates[["rate", "critical_rate"]].round(1)
    rates["rate_ratio"] = rates["rate_ratio"].round(2)
    rates["road"] = rates["road"].fillna("").str.title()
    return gpd.GeoDataFrame(rates, geometry="geometry", crs=lines.crs)


# --------------------------------------------------------------------------
# Interactive map
# --------------------------------------------------------------------------

def legend_html() -> str:
    labels = [f"{lo:,}–{hi:,}" for lo, hi in zip(RATE_BINS, RATE_BINS[1:])] + [f"{RATE_BINS[-1]:,}+"]
    rows = "".join(
        f"<div style='margin:2px 0'><span style='display:inline-block;width:22px;height:5px;"
        f"background:{color};margin-right:7px;vertical-align:middle'></span>{label}</div>"
        for color, label in zip(COLORS, labels)
    )
    return (
        "<div style='position:fixed;bottom:28px;left:14px;z-index:9999;background:white;"
        "padding:11px 14px;border:1px solid #bbb;border-radius:5px;font-family:sans-serif;"
        "font-size:12px;box-shadow:0 1px 4px rgba(0,0,0,.2)'>"
        "<div style='font-weight:600;margin-bottom:6px'>Crashes per 100 million<br>vehicle miles traveled</div>"
        + rows
        + "<div style='margin-top:7px;padding-top:6px;border-top:1px solid #ddd;color:#555'>"
        "Dashed = AADT count looks wrong;<br>treat that rate with caution</div></div>"
    )


def make_interactive_map(segments: gpd.GeoDataFrame, outpath: Path) -> None:
    import folium

    web = segments.to_crs(4326)[list(TOOLTIP_FIELDS) + ["anomaly", "geometry"]]
    bounds = web.total_bounds
    m = folium.Map(location=[(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2],
                   zoom_start=13, tiles="OpenStreetMap")

    tooltip = folium.GeoJsonTooltip(fields=list(TOOLTIP_FIELDS), aliases=list(TOOLTIP_FIELDS.values()))

    folium.GeoJson(
        web,
        name="Crash rate per 100 MVMT",
        style_function=lambda f: {
            "color": rate_color(f["properties"]["rate"]),
            "weight": 6,
            "opacity": 0.9,
            "dashArray": "6 6" if f["properties"]["check_aadt"] else None,
        },
        highlight_function=lambda f: {"weight": 10},
        tooltip=tooltip,
    ).add_to(m)

    folium.GeoJson(
        web,
        name="Above critical rate (statistically high)",
        show=False,
        style_function=lambda f: {
            "color": "#d62728" if f["properties"]["anomaly"] else "#9e9e9e",
            "weight": 4 + 2 * min(f["properties"]["rate_ratio"], 4) if f["properties"]["anomaly"] else 2,
            "opacity": 0.9,
            "dashArray": "6 6" if f["properties"]["check_aadt"] else None,
        },
        tooltip=folium.GeoJsonTooltip(fields=list(TOOLTIP_FIELDS), aliases=list(TOOLTIP_FIELDS.values())),
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(legend_html()))
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    m.save(str(outpath))
    print(f"  wrote {outpath}")


# --------------------------------------------------------------------------
# Static map
# --------------------------------------------------------------------------

def make_static_map(segments: gpd.GeoDataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 11))

    # All St. George crashes as faint dots: gives street-grid context and shows
    # how many crashes sit on local streets with no traffic counts.
    city = pd.read_excel(CITY_CRASHES, usecols=["Latitude", "Longitude"])
    crash_points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(city["Longitude"], city["Latitude"]), crs=4326
    ).to_crs(segments.crs)
    crash_points.plot(ax=ax, color="#888888", markersize=1, alpha=0.25, zorder=1)

    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(RATE_BINS + [max(segments["rate"].max(), RATE_BINS[-1]) + 1], cmap.N)
    trusted = segments[~segments["check_aadt"]]
    trusted.plot(ax=ax, column="rate", cmap=cmap, norm=norm, linewidth=3, zorder=2, legend=True,
                 legend_kwds={"label": "Crashes per 100 million vehicle miles traveled", "shrink": 0.6})
    suspect = segments[segments["check_aadt"]]
    if not suspect.empty:
        suspect.plot(ax=ax, color="#555555", linewidth=2, linestyle="--", zorder=2)

    ax.legend(
        handles=[
            Line2D([0], [0], color="#555555", linestyle="--", lw=2, label="AADT count looks wrong"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888", markersize=5,
                   label="Individual crashes (all roads)"),
        ],
        loc="lower left", fontsize=9, frameon=True,
    )
    ax.set_axis_off()
    ax.set_title(
        "Crash rate by road segment, normalized by traffic volume\n"
        f"St. George, Utah, Jun 2021 – May 2026 — {len(segments)} segments",
        fontsize=13, pad=12,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"  wrote {outpath}")


# --------------------------------------------------------------------------

def main():
    print(f"Reading {RATES_CSV.name} and fetching UDOT route lines from Open SGID...")
    segments = build_segments()
    print(f"  {len(segments)} St. George segments")
    make_interactive_map(segments, HERE / "crash_rate_map.html")
    make_static_map(segments, HERE / "crash_rate_map.png")


if __name__ == "__main__":
    main()
