#!/usr/bin/env python3
"""
Map severe-injury and fatal (KA) crashes in St. George, UT, colored by road user type.

Categories (mutually exclusive in this dataset -- verified, no crash carries
more than one vulnerable-road-user flag):
    Pedestrian   red
    Cyclist      orange
    Motorcyclist purple
    Other        blue

Outputs:
    ka_crash_map.html   interactive Leaflet map with per-category layer toggles
    ka_crash_map.png    static figure, no basemap (for print/poster use)

Usage:
    python map_ka_crashes.py
    python map_ka_crashes.py --input path/to/Crash_View.xlsx --outdir figures/
    python map_ka_crashes.py --severity "Fatal"          # fatals only
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

KA_LEVELS = ["Suspected Serious Injury", "Fatal"]

# Order matters: the first matching flag wins. Kept explicit so the precedence
# is auditable if a future extract does allow overlapping flags.
VRU_RULES = [
    ("Pedestrian", "Pedestrian Involved"),
    ("Cyclist", "Bicycle Involved"),
    ("Motorcyclist", "Motorcycle Involved"),
]

COLORS = {
    "Pedestrian": "#d62728",    # red
    "Cyclist": "#ff7f0e",       # orange
    "Motorcyclist": "#9467bd",  # purple
    "Other": "#1f77b4",         # blue
}

PLOT_ORDER = ["Other", "Motorcyclist", "Cyclist", "Pedestrian"]  # blue drawn first

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


def categorize(df: pd.DataFrame) -> pd.Series:
    """Assign each crash to a road user category."""
    cat = pd.Series("Other", index=df.index, dtype="object")
    assigned = pd.Series(False, index=df.index)

    for label, col in VRU_RULES:
        if col not in df.columns:
            print(f"  warning: column '{col}' not found, skipping {label}",
                  file=sys.stderr)
            continue
        hit = df[col].astype("string").str.strip().str.upper().eq("Y").fillna(False)
        cat[hit & ~assigned] = label
        assigned |= hit

    return cat


def prepare(df: pd.DataFrame, severity_levels) -> pd.DataFrame:
    """Filter to the target severity levels and attach categories."""
    missing = [c for c in ("Latitude", "Longitude", "Crash Severity") if c not in df.columns]
    if missing:
        raise KeyError(f"Required column(s) missing from input: {missing}")

    ka = df[df["Crash Severity"].isin(severity_levels)].copy()

    # Drop unusable coordinates: nulls and the 0,0 null-island placeholder
    # that crash geocoders emit on a failed match.
    coords_ok = (
        ka["Latitude"].notna()
        & ka["Longitude"].notna()
        & (ka["Latitude"].abs() > 0.001)
        & (ka["Longitude"].abs() > 0.001)
    )
    dropped = (~coords_ok).sum()
    if dropped:
        print(f"  dropped {dropped} crash(es) with unusable coordinates")
    ka = ka[coords_ok]

    ka["user_category"] = categorize(ka)
    return ka


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


def make_interactive_map(ka: pd.DataFrame, outpath: Path) -> None:
    import folium

    center = [ka["Latitude"].mean(), ka["Longitude"].mean()]
    # OpenStreetMap needs no API key. CartoDB's Positron basemap is cleaner for
    # point data but now requires a key, so it's offered only as an opt-in:
    # pass one via --carto-key to enable it.
    m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")

    for label in PLOT_ORDER:
        subset = ka[ka["user_category"] == label]
        if subset.empty:
            continue

        layer = folium.FeatureGroup(name=f"{label} ({len(subset)})", show=True)

        for _, row in subset.iterrows():
            is_fatal = row["Crash Severity"] == "Fatal"
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=7 if is_fatal else 5,
                color="#000000" if is_fatal else COLORS[label],
                weight=2 if is_fatal else 1,
                fill=True,
                fill_color=COLORS[label],
                fill_opacity=0.85 if is_fatal else 0.6,
                popup=folium.Popup(build_popup(row), max_width=350),
                tooltip=f"{label} — {row['Crash Severity']}",
            ).add_to(layer)

        layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(legend_html(ka)))
    m.save(str(outpath))
    print(f"  wrote {outpath}")


def legend_html(ka: pd.DataFrame) -> str:
    counts = ka["user_category"].value_counts()
    items = []
    for label in PLOT_ORDER[::-1]:
        n = int(counts.get(label, 0))
        items.append(
            f"<div style='margin:3px 0'>"
            f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;"
            f"background:{COLORS[label]};margin-right:7px'></span>{label} ({n})</div>"
        )
    n_fatal = int((ka["Crash Severity"] == "Fatal").sum())
    items.append(
        "<div style='margin-top:7px;padding-top:6px;border-top:1px solid #ddd;"
        "font-size:11px;color:#555'>Black outline = fatal "
        f"({n_fatal})</div>"
    )
    return (
        "<div style='position:fixed;bottom:28px;left:14px;z-index:9999;background:white;"
        "padding:11px 14px;border:1px solid #bbb;border-radius:5px;font-family:sans-serif;"
        "font-size:13px;box-shadow:0 1px 4px rgba(0,0,0,.2)'>"
        "<div style='font-weight:600;margin-bottom:6px'>Severe & fatal crashes</div>"
        + "".join(items)
        + "</div>"
    )


# --------------------------------------------------------------------------
# Static map
# --------------------------------------------------------------------------

def make_static_map(ka: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 10))

    for label in PLOT_ORDER:
        subset = ka[ka["user_category"] == label]
        if subset.empty:
            continue
        fatal = subset["Crash Severity"] == "Fatal"
        ax.scatter(
            subset.loc[~fatal, "Longitude"], subset.loc[~fatal, "Latitude"],
            c=COLORS[label], s=34, alpha=0.75, linewidths=0.4, edgecolors="white",
            zorder=2,
        )
        ax.scatter(
            subset.loc[fatal, "Longitude"], subset.loc[fatal, "Latitude"],
            c=COLORS[label], s=90, alpha=0.95, linewidths=1.3, edgecolors="black",
            marker="o", zorder=3,
        )

    counts = ka["user_category"].value_counts()
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[lbl],
               markersize=9, label=f"{lbl} ({int(counts.get(lbl, 0))})")
        for lbl in PLOT_ORDER[::-1]
    ]
    handles.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#999999",
               markeredgecolor="black", markeredgewidth=1.3, markersize=12,
               label=f"Fatal ({int((ka['Crash Severity'] == 'Fatal').sum())})")
    )
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=10)

    # Equal-area-ish aspect: one degree of longitude is shorter than one of
    # latitude by cos(lat). At 37°N that is a ~20% distortion if ignored.
    mean_lat = ka["Latitude"].mean()
    ax.set_aspect(1 / math.cos(math.radians(mean_lat)))

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Severe injury and fatal crashes by road user type\n"
        f"St. George, Utah — n = {len(ka)}",
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
    p.add_argument("--severity", nargs="+", default=KA_LEVELS,
                   help='severity levels to map (default: the two KA levels)')
    p.add_argument("--no-html", action="store_true", help="skip the interactive map")
    p.add_argument("--no-png", action="store_true", help="skip the static map")
    args = p.parse_args()

    inpath = Path(args.input)
    if not inpath.exists():
        sys.exit(f"Input file not found: {inpath}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {inpath}")
    df = load_crashes(inpath)
    ka = prepare(df, args.severity)

    print(f"\n{len(ka)} crashes at severity {args.severity}")
    counts = ka["user_category"].value_counts()
    for label in PLOT_ORDER[::-1]:
        n = int(counts.get(label, 0))
        print(f"  {label:<13} {n:>4}  ({n / len(ka):.1%})")
    vru = int(len(ka) - counts.get("Other", 0))
    print(f"  {'VRU subtotal':<13} {vru:>4}  ({vru / len(ka):.1%})\n")

    if not args.no_html:
        make_interactive_map(ka, outdir / "ka_crash_map.html")
    if not args.no_png:
        make_static_map(ka, outdir / "ka_crash_map.png")


if __name__ == "__main__":
    main()