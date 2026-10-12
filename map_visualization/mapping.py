#!/usr/bin/env python3
"""
Heatmap of crash density in St. George, UT.

Outputs:
    crash_heatmap.html   interactive Leaflet heatmap, plus a toggleable
                         layer of individual crashes with detail popups
    crash_heatmap.png    static hexbin density figure (for print/poster use)

Usage:
    python mapping.py
    python mapping.py --input path/to/Crash_View.xlsx --outdir figures/
    python mapping.py --severity "Fatal" "Suspected Serious Injury"   # subset only
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

POPUP_FIELDS = [
    "Crash ID",
    "Crash Date Time",
    "Crash Severity",
    "Full Route Name",
    "Milepoint",
    "Roadway Junction Type",
    "Manner of Collision",
    "Light Condition",
    "Weather Condition",
    "Roadway Surface Condition",
]


# --------------------------------------------------------------------------
# Data prep
# --------------------------------------------------------------------------

def load_crashes(path: Path) -> pd.DataFrame:
    """Read the crash extract and normalize column names."""
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # The city extract ships several headers with trailing spaces
    # ("Roadway Junction Type ", "Most Harmful Event "). Strip them so
    # downstream references don't depend on invisible whitespace.
    df.columns = [c.strip() for c in df.columns]
    return df


def prepare(df: pd.DataFrame, severity_levels=None) -> pd.DataFrame:
    """Optionally filter by severity and drop crashes without usable coordinates."""
    missing = [c for c in ("Latitude", "Longitude", "Crash Severity") if c not in df.columns]
    if missing:
        raise KeyError(f"Required column(s) missing from input: {missing}")

    if severity_levels:
        df = df[df["Crash Severity"].isin(severity_levels)]

    # Drop nulls and the 0,0 null-island placeholder that crash geocoders
    # emit on a failed match.
    coords_ok = (
        df["Latitude"].notna()
        & df["Longitude"].notna()
        & (df["Latitude"].abs() > 0.001)
        & (df["Longitude"].abs() > 0.001)
    )
    dropped = (~coords_ok).sum()
    if dropped:
        print(f"  dropped {dropped} crash(es) with unusable coordinates")
    return df[coords_ok].copy()


# --------------------------------------------------------------------------
# Interactive map
# --------------------------------------------------------------------------

def build_popup(row: pd.Series) -> str:
    rows = []
    for field in POPUP_FIELDS:
        if field in row.index and pd.notna(row[field]):
            rows.append(
                f"<tr><td style='padding-right:8px;color:#666'>{field}</td>"
                f"<td><b>{row[field]}</b></td></tr>"
            )
    return f"<table style='font-size:12px;font-family:sans-serif'>{''.join(rows)}</table>"


def make_interactive_map(crashes: pd.DataFrame, outpath: Path) -> None:
    import folium
    from folium.plugins import HeatMap, MarkerCluster

    center = [crashes["Latitude"].mean(), crashes["Longitude"].mean()]
    m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")

    HeatMap(
        crashes[["Latitude", "Longitude"]].values.tolist(),
        name=f"Crash density ({len(crashes):,} crashes)",
        radius=10,
        blur=14,
        min_opacity=0.3,
    ).add_to(m)

    # Off by default: lets you click into a hotspot and see what's happening there.
    cluster = MarkerCluster(name="Individual crashes", show=False)
    for _, row in crashes.iterrows():
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=folium.Popup(build_popup(row), max_width=350),
            tooltip=str(row["Crash Severity"]),
        ).add_to(cluster)
    cluster.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(outpath))
    print(f"  wrote {outpath}")


# --------------------------------------------------------------------------
# Static map
# --------------------------------------------------------------------------

def make_static_map(crashes: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 10))

    # Log scale: a few intersections have far more crashes than anywhere else,
    # and a linear scale would wash out every other hotspot.
    hb = ax.hexbin(
        crashes["Longitude"], crashes["Latitude"],
        gridsize=70, mincnt=1, cmap="inferno_r", norm=LogNorm(), linewidths=0.2,
    )
    fig.colorbar(hb, ax=ax, shrink=0.7, label="Crashes per cell (log scale)")

    # One degree of longitude is shorter than one of latitude by cos(lat).
    # At 37°N that is a ~20% distortion if ignored.
    mean_lat = crashes["Latitude"].mean()
    ax.set_aspect(1 / math.cos(math.radians(mean_lat)))

    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Crash density\nSt. George, Utah — n = {len(crashes):,}",
        fontsize=13, pad=12,
    )
    ax.grid(alpha=0.25, linewidth=0.5, zorder=0)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"  wrote {outpath}")


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="./data/AccidentData21-26.csv.xlsx",
                   help="path to the crash extract (.xlsx or .csv)")
    p.add_argument("--outdir", default=".", help="directory for output files")
    p.add_argument("--severity", nargs="+", default=None,
                   help="only map these severity levels (default: all crashes)")
    p.add_argument("--no-html", action="store_true", help="skip the interactive map")
    p.add_argument("--no-png", action="store_true", help="skip the static map")
    args = p.parse_args()

    inpath = Path(args.input)
    if not inpath.exists():
        sys.exit(f"Input file not found: {inpath}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {inpath}")
    crashes = prepare(load_crashes(inpath), args.severity)
    print(f"\nMapping {len(crashes):,} crashes"
          + (f" at severity {args.severity}" if args.severity else ""))

    if not args.no_html:
        make_interactive_map(crashes, outdir / "crash_heatmap.html")
    if not args.no_png:
        make_static_map(crashes, outdir / "crash_heatmap.png")


if __name__ == "__main__":
    main()
