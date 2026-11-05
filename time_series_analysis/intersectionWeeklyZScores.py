"""
Weekly crash z-scores per intersection in St. George: which intersections
stand out overall, and which weeks stand out at each intersection.

Intersections: built from UGRC's road network (Open SGID). Any point where 3+
road lines meet is an intersection; points within 40 m are merged so divided
roads and roundabouts count once. Each crash coded at an intersection
junction type is assigned to the nearest intersection within 250 ft (the
Highway Safety Manual's intersection influence area).

Two questions:
  1. Which intersections stand out overall?
       crashes per week, z-scored across all intersections with crashes, and
       crash rate per million entering vehicles (MEV), z-scored across the
       intersections where every road meeting there has a UDOT traffic count:
           MEV rate = crashes * 1,000,000 / (entering vehicles/day * 365 * years)
           entering vehicles/day = sum over roads of AADT * (legs at node / 2)
  2. Which weeks stand out at an intersection?
       z = (crashes that week - intersection's average week) / sqrt(average week)
     Weekly counts at one intersection are rare events (mostly 0 or 1), so the
     spread is the Poisson one, sqrt(mean), and each week also gets the
     probability of seeing that many crashes by chance. With thousands of
     intersection-weeks some low probabilities happen by luck; the script
     reports how many to expect.

Requires normalized_crash_rates/crashRatesBySegment.csv (run crashRates.py).

Outputs (written next to this script):
    intersectionWeeklyZScores.xlsx   Intersections, Stand-out Weeks sheets
    intersectionWeeklyZScores.png
"""

import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import poisson
from shapely.geometry import box
from sqlalchemy import create_engine, text

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "map_visualization"))
from crashRateMap import SGID, cut_segment, load_route_lines  # noqa: E402
from crashZScores import END, START, holidays  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 50)

RATES_CSV = ROOT / "normalized_crash_rates" / "crashRatesBySegment.csv"
CITY_CRASHES = ROOT / "data" / "AccidentData21-26.csv.xlsx"

INTERSECTION_JUNCTIONS = {
    "4-Leg Intersection", "T-Intersection", "Y-Intersection", "5-Leg or More Intersection",
    "Roundabout/Traffic Circle", "Ramp Intersection With Crossroad",
}
NON_ROADS = {"12", "13", "14", "15", "16", "17", "18"}  # UGRC CARTOCODEs: other, driveway, proposed, 4WD, service...
KA_LEVELS = ["Suspected Serious Injury", "Fatal"]
CLEAR_WEATHER = {"Clear", "Cloudy", "Unknown"}

NODE_MERGE_M = 40           # road endpoints this close are one intersection
SNAP_M = 76                 # 250 ft
AADT_SEARCH_M = 30          # AADT segments this close to an intersection feed its entering volume
ROUTE_END_M = 40            # a route ending this close to an intersection has 1 leg there, else 2
MIN_CRASHES_FOR_WEEKS = 10  # intersections with fewer crashes are too sparse for weekly z-scores
TOP_N = 20


# --------------------------------------------------------------------------
# Crashes
# --------------------------------------------------------------------------

def load_crashes() -> gpd.GeoDataFrame:
    df = pd.read_excel(CITY_CRASHES)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Roadway Junction Type"].isin(INTERSECTION_JUNCTIONS)].copy()
    df["time"] = pd.to_datetime(df["Crash Date Time"], errors="coerce")
    df["week_starting"] = df["time"].dt.normalize() - pd.to_timedelta(df["time"].dt.dayofweek, unit="D")
    df["ka"] = df["Crash Severity"].isin(KA_LEVELS)
    df["bad_weather"] = df["Weather Condition"].notna() & ~df["Weather Condition"].isin(CLEAR_WEATHER)
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]), crs=4326).to_crs(26912)


# --------------------------------------------------------------------------
# Intersections from the road network
# --------------------------------------------------------------------------

def load_roads(area) -> gpd.GeoDataFrame:
    query = text(
        "SELECT fullname, dot_rtname, cartocode, shape FROM transportation.roads "
        "WHERE ST_Intersects(shape, ST_MakeEnvelope(:xmin, :ymin, :xmax, :ymax, 26912))"
    )
    xmin, ymin, xmax, ymax = area.bounds
    with create_engine(SGID).connect() as conn:
        roads = gpd.read_postgis(query, conn, geom_col="shape",
                                 params={"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
    roads = roads[~roads["cartocode"].isin(NON_ROADS)].reset_index(drop=True)
    roads["route"] = roads["dot_rtname"].where(roads["dot_rtname"].fillna("").str.fullmatch(r"\d{4}[PN]")).str[:4]
    return roads.set_crs(26912, allow_override=True)


def short_name(name: str) -> str:
    """'I-15 SB X8 OFF ST GEORGE BLVD RAMP' -> 'I-15 EXIT 8 RAMPS'."""
    exit_ramp = re.match(r"^(I-15|SR \d+)\b.*?\bX(\d+)\b.*RAMP", name)
    if exit_ramp:
        return f"{exit_ramp.group(1)} EXIT {exit_ramp.group(2)} RAMPS"
    return name


def build_intersections(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    ends = []
    for road_id, geom in roads.geometry.items():
        for part in getattr(geom, "geoms", [geom]):
            for xy in (part.coords[0], part.coords[-1]):
                ends.append({"road_id": road_id, "x": xy[0], "y": xy[1]})
    ends = pd.DataFrame(ends)
    pts = gpd.GeoDataFrame(ends, geometry=gpd.points_from_xy(ends["x"], ends["y"]), crs=26912)

    blobs = gpd.GeoDataFrame(geometry=list(pts.buffer(NODE_MERGE_M / 2).union_all().geoms), crs=26912)
    blobs["node_id"] = range(len(blobs))
    pts = gpd.sjoin(pts, blobs, predicate="within").drop(columns="index_right")
    pts = pts.join(roads[["fullname", "route", "cartocode"]], on="road_id")

    def describe(g):
        names = (g.dropna(subset=["fullname"])
                 .assign(rank=lambda d: pd.to_numeric(d["cartocode"], errors="coerce"))
                 .sort_values("rank")["fullname"].str.strip().map(short_name).drop_duplicates())
        return pd.Series({
            "road_ends": len(g),
            "name": " & ".join(names.head(3)) or "(unnamed)",
            "routes_at_node": sorted(set(g["route"].dropna())),
            "has_uncounted_leg": bool(g["route"].isna().any()),
            "x": g["x"].mean(), "y": g["y"].mean(),
        })

    nodes = pts.groupby("node_id").apply(describe)
    nodes = nodes[nodes["road_ends"] >= 3]
    return gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes["x"], nodes["y"]), crs=26912)


def entering_volume(nodes: gpd.GeoDataFrame) -> pd.Series:
    """Entering vehicles/day where every road at the intersection is on an AADT-counted route."""
    seg = pd.read_csv(RATES_CSV)
    seg = seg[~seg["check_aadt"]]
    lines = load_route_lines(seg["RouteID"].unique())
    seg["geometry"] = [cut_segment(lines[r.RouteID], r.BeginPoint, r.EndPoint) for r in seg.itertuples()]
    seg = gpd.GeoDataFrame(seg.dropna(subset=["geometry"]), geometry="geometry", crs=lines.crs)
    seg["route4"] = seg["RouteID"].str[:4]
    route_ends = {rid: [pt for part in getattr(g, "geoms", [g]) for pt in (part.boundary.geoms)]
                  for rid, g in lines.items()}

    zones = nodes[["geometry"]].copy()
    zones["geometry"] = zones.buffer(AADT_SEARCH_M)
    near = gpd.sjoin(seg[["route4", "RouteID", "AADT", "geometry"]], zones, predicate="intersects")

    volumes = {}
    for node_id, g in near.groupby("node_id"):
        node = nodes.loc[node_id]
        if node["has_uncounted_leg"] or not set(node["routes_at_node"]) <= set(g["route4"]):
            continue
        total = 0.0
        for rid, rg in g.groupby("RouteID"):
            legs = 1 if any(node.geometry.distance(p) <= ROUTE_END_M for p in route_ends[rid]) else 2
            total += rg["AADT"].mean() * legs / 2
        volumes[node_id] = total
    return pd.Series(volumes, name="entering_vehicles_per_day")


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def expected_by_luck(weekly_means: pd.Series, n_weeks: int, threshold: float) -> float:
    """Counts are whole numbers, so for each intersection find the smallest weekly
    count that would fall below the threshold and how often that happens by chance."""
    total = 0.0
    for lam in weekly_means:
        k = 1
        while poisson.sf(k - 1, lam) >= threshold:
            k += 1
        total += n_weeks * poisson.sf(k - 1, lam)
    return total

def main():
    crashes = load_crashes()
    area = box(*crashes.total_bounds).buffer(500)
    print("Building intersections from UGRC's road network...")
    nodes = build_intersections(load_roads(area))

    crashes = gpd.sjoin_nearest(crashes, nodes[["geometry"]], max_distance=SNAP_M, how="inner",
                                distance_col="snap_m")

    week_index = pd.date_range(START + pd.Timedelta(days=(7 - START.dayofweek) % 7),
                               END - pd.Timedelta(days=6), freq="W-MON")
    crashes = crashes[crashes["week_starting"].isin(week_index)]
    n_weeks, years = len(week_index), len(week_index) * 7 / 365.25
    print(f"  {len(nodes):,} intersections; {len(crashes):,} intersection crashes assigned "
          f"over {n_weeks} full weeks\n")

    # ---- 1. Intersections overall -------------------------------------
    per_node = crashes.groupby("node_id").agg(crashes=("node_id", "size"), ka=("ka", "sum"))
    inter = nodes.loc[per_node.index, ["name", "road_ends"]].join(per_node)
    inter["crashes_per_week"] = inter["crashes"] / n_weeks
    inter["count_z"] = (inter["crashes_per_week"] - inter["crashes_per_week"].mean()) / inter["crashes_per_week"].std()

    inter = inter.join(entering_volume(nodes))
    has_vol = inter["entering_vehicles_per_day"].notna()
    inter["rate_per_mev"] = inter["crashes"] * 1e6 / (inter["entering_vehicles_per_day"] * 365.25 * years)
    rates = inter.loc[has_vol, "rate_per_mev"]
    inter["rate_z"] = (inter["rate_per_mev"] - rates.mean()) / rates.std()

    print(f"Intersections with the highest crash rate per million entering vehicles "
          f"({has_vol.sum()} intersections where every road has a traffic count):")
    cols = ["name", "crashes", "ka", "entering_vehicles_per_day", "rate_per_mev", "rate_z", "count_z"]
    print(inter[has_vol].nlargest(TOP_N, "rate_z")[cols].round(2).to_string(index=False))

    print(f"\nIntersections with the most crashes per week (all {len(inter)} intersections with crashes):")
    cols = ["name", "crashes", "ka", "crashes_per_week", "count_z", "rate_per_mev"]
    print(inter.nlargest(TOP_N, "count_z")[cols].round(2).to_string(index=False))

    # ---- 2. Weeks at each intersection --------------------------------
    eligible = inter[inter["crashes"] >= MIN_CRASHES_FOR_WEEKS]
    hol = holidays()
    week_rows = []
    for node_id, g in crashes[crashes["node_id"].isin(eligible.index)].groupby(["node_id", "week_starting"]):
        node_id, week = node_id
        lam = eligible.loc[node_id, "crashes_per_week"]
        k = len(g)
        week_rows.append({
            "node_id": node_id,
            "intersection": eligible.loc[node_id, "name"],
            "week_starting": week.date(),
            "crashes": k,
            "usual_per_week": round(lam, 3),
            "z": (k - lam) / np.sqrt(lam),
            "chance_probability": poisson.sf(k - 1, lam),
            "days": ", ".join(g["time"].dt.strftime("%a").sort_values()),
            "ka": int(g["ka"].sum()),
            "bad_weather": int(g["bad_weather"].sum()),
            "manner_of_collision": "; ".join(g["Manner of Collision"].dropna().value_counts().index[:2]),
            "holiday": ", ".join(hol[(hol.index >= week) & (hol.index <= week + pd.Timedelta(days=6))]),
        })
    weeks = pd.DataFrame(week_rows).sort_values(["chance_probability", "z"], ascending=[True, False])

    print(f"\nIf nothing unusual ever happened, how many intersection-weeks would look this extreme "
          f"by luck alone? ({len(eligible) * n_weeks:,} intersection-weeks tested)")
    for threshold in [0.01, 0.001, 0.0001]:
        print(f"  chance probability < {threshold}: {(weeks['chance_probability'] < threshold).sum():>4} observed, "
              f"{expected_by_luck(eligible['crashes_per_week'], n_weeks, threshold):6.1f} expected by luck")
    print(f"\n\nWeeks that stand out most at an intersection "
          f"({len(eligible)} intersections with ≥ {MIN_CRASHES_FOR_WEEKS} crashes):")
    show = weeks.head(TOP_N).copy()
    show["chance_probability"] = show["chance_probability"].map("{:.1e}".format)
    print(show.drop(columns="node_id").round(2).to_string(index=False))

    # ---- Output ---------------------------------------------------------
    with pd.ExcelWriter(HERE / "intersectionWeeklyZScores.xlsx", engine="openpyxl") as writer:
        out = inter.sort_values("count_z", ascending=False).round(3)
        out.to_excel(writer, sheet_name="Intersections")
        weeks.round(4).to_excel(writer, sheet_name="Stand-out Weeks", index=False)
        for sheet in writer.sheets.values():
            sheet.freeze_panes = "B2"

    plot(inter, crashes, week_index, weeks)
    print("\nSaved intersectionWeeklyZScores.xlsx and intersectionWeeklyZScores.png")


def plot(inter, crashes, week_index, weeks):
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [1, 1.1]})

    ax = axes[0]
    top = inter[inter["rate_z"].notna()].nlargest(15, "rate_z").iloc[::-1]
    ax.barh(top["name"].str.title(), top["rate_per_mev"], color="#d62728")
    for y, (rate, n) in enumerate(zip(top["rate_per_mev"], top["crashes"])):
        ax.text(rate, y, f"  {n} crashes", va="center", fontsize=8)
    ax.set_xlabel("Crashes per million entering vehicles")
    ax.set_title("Intersections with the highest crash rate (traffic-normalized)")

    ax = axes[1]
    top_nodes = inter.nlargest(15, "count_z").index
    grid = (crashes[crashes["node_id"].isin(top_nodes)]
            .groupby(["node_id", "week_starting"]).size().unstack(fill_value=0)
            .reindex(index=top_nodes, columns=week_index, fill_value=0))
    im = ax.imshow(grid.values, aspect="auto", cmap="Reds", interpolation="nearest")
    ax.set_yticks(range(len(top_nodes)), inter.loc[top_nodes, "name"].str.title(), fontsize=8)
    yearly = [i for i, w in enumerate(week_index) if w.month == 1 and w.day <= 7]
    ax.set_xticks(yearly, [week_index[i].year for i in yearly])
    flagged = weeks[(weeks["chance_probability"] < 0.001) & weeks["node_id"].isin(top_nodes)]
    for r in flagged.itertuples():
        ax.scatter(week_index.get_loc(pd.Timestamp(r.week_starting)), list(top_nodes).index(r.node_id),
                   s=60, facecolors="none", edgecolors="black", linewidths=1.2)
    fig.colorbar(im, ax=ax, label="Crashes that week")
    ax.set_title("Weekly crashes at the 15 busiest crash intersections (circled = chance probability < 0.001)")

    fig.tight_layout()
    fig.savefig(HERE / "intersectionWeeklyZScores.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
