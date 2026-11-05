"""
Which road features predict crash rate? A safety performance function (SPF)
for Washington County surface streets, following the Highway Safety Manual.

Each row is one AADT road segment. The model is a negative binomial
regression of the segment's 5-year crash count:

    ln(expected crashes) = b0 + b1*ln(AADT) + b2*feature2 + ... + ln(miles * years)

The ln(miles * years) offset turns the model into a crash-rate model, and
ln(AADT) is estimated rather than fixed so the model can show whether
crashes grow faster or slower than traffic. Negative binomial (not Poisson)
because crash counts vary more than a Poisson model allows.

Features (all from data that's reliable for every segment):
    ln(AADT)                 traffic volume
    City road                city-owned federal-aid road vs. UDOT state route
    Intersections per mile   cross-street connections, from UGRC's road network
    Curvature (deg/mile)     total change in road direction per mile
    Bike facility            bike lane or path along at least half the segment
    Urban                    at least half the segment inside a city

Left out: speed limit (UGRC's values look like 25 mph defaults on city
roads, and 93% of crash reports leave posted speed blank) and lane counts
(not in UGRC's data). Interstate segments are left out because freeways
need their own model, and the city doesn't control them.

Requires normalized_crash_rates/crashRatesBySegment.csv (run crashRates.py).

Outputs (written next to this script):
    segmentRoadFeatures.csv     features, observed and predicted crashes per segment
    modelSummary.txt            full statsmodels output
    roadFeatureEffects.png      effect of each feature on crash rate
"""

import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from shapely.geometry import MultiPoint
from sqlalchemy import create_engine, text

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "map_visualization"))
from crashRateMap import SGID, cut_segment, load_route_lines  # noqa: E402

pd.set_option("display.width", 200)

RATES_CSV = ROOT / "normalized_crash_rates" / "crashRatesBySegment.csv"
YEARS = 5.0

ROUTE_MATCH_BUFFER_M = 15      # how close a UGRC road must be to count as the segment itself
CROSS_STREET_SNAP_M = 12       # cross-street endpoint must be this close to the segment
INTERSECTION_MERGE_M = 25      # endpoints closer than this are one intersection
CURVE_SIMPLIFY_M = 5           # smooths digitizing jitter before measuring curvature
NON_ROADS = {"12", "13", "14", "15", "16", "17", "18"}  # UGRC CARTOCODEs: other, driveway, proposed, 4WD, service...

FEATURES = {
    "log_aadt": "ln(AADT)",
    "city_road": "City road (vs. state route)",
    "intersections_per_mile": "Intersections per mile",
    "curvature_deg_per_mile": "Curvature (degrees per mile)",
    "bike_facility": "Bike lane or path",
    "urban": "Urban (inside a city)",
}
BINARY = {"city_road", "bike_facility", "urban"}


# --------------------------------------------------------------------------
# Segment features
# --------------------------------------------------------------------------

def load_segments() -> gpd.GeoDataFrame:
    if not RATES_CSV.exists():
        raise SystemExit(f"{RATES_CSV} not found; run normalized_crash_rates/crashRates.py first.")
    seg = pd.read_csv(RATES_CSV)
    seg = seg[(seg["facility"] != "Interstate") & ~seg["check_aadt"] & (seg["Section_Length"] > 0)]
    lines = load_route_lines(seg["RouteID"].unique())
    seg["geometry"] = [cut_segment(lines[r.RouteID], r.BeginPoint, r.EndPoint) for r in seg.itertuples()]
    return gpd.GeoDataFrame(seg.dropna(subset=["geometry"]), geometry="geometry", crs=lines.crs)


def load_roads(crs) -> gpd.GeoDataFrame:
    query = text(
        "SELECT dot_rtname, cartocode, bike_l, bike_r, incmuni_l, shape "
        "FROM transportation.roads WHERE county_l = '49053'"
    )
    with create_engine(SGID).connect() as conn:
        roads = gpd.read_postgis(query, conn, geom_col="shape")
    roads = roads[~roads["cartocode"].isin(NON_ROADS)].to_crs(crs)
    roads["route"] = pd.to_numeric(roads["dot_rtname"].str[:4], errors="coerce")
    roads["is_mainline"] = roads["dot_rtname"].fillna("").str.fullmatch(r"\d{4}[PN]")
    return roads


def attribute_shares(seg: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> pd.DataFrame:
    """Length-weighted share of each segment with a bike facility / inside a city,
    from the UGRC road lines that make up the segment."""
    own = roads[roads["is_mainline"]].copy()
    # UGRC bike codes: 1x = separated path, 2x = bike lane, 3x = shared roadway / signed route.
    own["bike"] = own["bike_l"].fillna("").str[:1].isin(["1", "2"]) | own["bike_r"].fillna("").str[:1].isin(["1", "2"])
    own["in_city"] = own["incmuni_l"].notna() & (own["incmuni_l"].str.strip() != "")

    zones = seg[["segment_id", "route", "geometry"]].copy()
    zones["geometry"] = zones.buffer(ROUTE_MATCH_BUFFER_M)
    hits = gpd.sjoin(own, zones, predicate="intersects")
    hits = hits[hits["route_left"] == hits["route_right"]]
    zone_geom = gpd.GeoSeries(zones.set_index("segment_id").loc[hits["segment_id"], "geometry"].values,
                              index=hits.index, crs=zones.crs)
    hits["w"] = hits.geometry.intersection(zone_geom).length

    def share(col):
        return (hits[col] * hits["w"]).groupby(hits["segment_id"]).sum() / hits.groupby("segment_id")["w"].sum()

    return pd.DataFrame({"bike_share": share("bike"), "urban_share": share("in_city")})


def count_intersections(seg: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> pd.Series:
    """Cross streets are UGRC road lines on a different route whose endpoint touches the
    segment. A street that crosses straight through is split into two lines sharing an
    endpoint, so nearby endpoints are merged into one intersection."""
    ends = []
    for road in roads.itertuples():
        geoms = getattr(road.shape, "geoms", [road.shape])
        for g in geoms:
            ends.append((road.route, g.coords[0]))
            ends.append((road.route, g.coords[-1]))
    endpoints = gpd.GeoDataFrame(
        {"road_route": [r for r, _ in ends]},
        geometry=gpd.points_from_xy([c[0] for _, c in ends], [c[1] for _, c in ends]),
        crs=roads.crs,
    )

    zones = seg[["segment_id", "route", "geometry"]].copy()
    zones["geometry"] = zones.buffer(CROSS_STREET_SNAP_M)
    near = gpd.sjoin(endpoints, zones, predicate="within")
    near = near[near["road_route"] != near["route"]]

    counts = {}
    for seg_id, pts in near.groupby("segment_id"):
        merged = MultiPoint(list(pts.geometry)).buffer(INTERSECTION_MERGE_M / 2)
        counts[seg_id] = len(getattr(merged, "geoms", [merged]))
    return pd.Series(counts, dtype=float)


def curvature_deg_per_mile(geom) -> float:
    total, length = 0.0, 0.0
    for part in getattr(geom, "geoms", [geom]):
        coords = np.asarray(part.simplify(CURVE_SIMPLIFY_M).coords)
        length += part.length
        if len(coords) < 3:
            continue
        headings = np.degrees(np.arctan2(np.diff(coords[:, 1]), np.diff(coords[:, 0])))
        turns = (np.diff(headings) + 180) % 360 - 180
        total += np.abs(turns).sum()
    return total / (length / 1609.344) if length else 0.0


def build_features() -> pd.DataFrame:
    seg = load_segments()
    roads = load_roads(seg.crs)

    shares = attribute_shares(seg, roads)
    seg = seg.merge(shares, left_on="segment_id", right_index=True, how="left")
    seg["intersections"] = seg["segment_id"].map(count_intersections(seg, roads)).fillna(0)

    seg["log_aadt"] = np.log(seg["AADT"])
    seg["city_road"] = (seg["facility"] == "Federal-aid city road").astype(int)
    seg["intersections_per_mile"] = seg["intersections"] / seg["Section_Length"]
    seg["curvature_deg_per_mile"] = seg.geometry.apply(curvature_deg_per_mile)
    seg["bike_facility"] = (seg["bike_share"].fillna(0) >= 0.5).astype(int)
    seg["urban"] = (seg["urban_share"].fillna(0) >= 0.5).astype(int)
    return pd.DataFrame(seg.drop(columns="geometry"))


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def fit(data: pd.DataFrame, features):
    X = sm.add_constant(data[list(features)], has_constant="add")
    model = sm.NegativeBinomial(data["crashes"], X, loglike_method="nb2",
                                exposure=data["Section_Length"] * YEARS)
    # The optimizer probes extreme alpha values on the way to the answer, which
    # triggers harmless log(0) warnings; convergence is checked instead.
    with np.errstate(divide="ignore", invalid="ignore"):
        result = model.fit(disp=False, maxiter=500)
    if not result.mle_retvals["converged"]:
        print(f"  warning: model with {list(features)} did not converge")
    return result


def main():
    print("Building road features from UGRC Open SGID...")
    data = build_features()
    print(f"  {len(data)} surface-street segments in Washington County "
          f"({int(data['crashes'].sum()):,} crashes)\n")

    full = fit(data, FEATURES)
    traffic_only = fit(data, ["log_aadt"])

    # Effect of a typical change: 25th -> 75th percentile for continuous features,
    # no -> yes for binary ones. Puts every feature on a comparable scale.
    rows = []
    ci = full.conf_int()
    for col, label in FEATURES.items():
        if col in BINARY:
            step, step_desc = 1.0, "no → yes"
        else:
            lo, hi = data[col].quantile([0.25, 0.75])
            step = hi - lo
            step_desc = (f"{math.exp(lo):,.0f} → {math.exp(hi):,.0f} vehicles/day" if col == "log_aadt"
                         else f"{lo:,.1f} → {hi:,.1f}")
        # Traffic is in both the crash count and the rate's denominator, so its effect on
        # the rate per vehicle mile is (b - 1); every other feature holds traffic fixed.
        shift = 1.0 if col == "log_aadt" else 0.0
        reduced = fit(data, [c for c in FEATURES if c != col])
        lr = 2 * (full.llf - reduced.llf)
        rows.append({
            "Feature": label,
            "Typical change": step_desc,
            "Crash rate change": math.exp((full.params[col] - shift) * step) - 1,
            "CI low": math.exp((ci.loc[col, 0] - shift) * step) - 1,
            "CI high": math.exp((ci.loc[col, 1] - shift) * step) - 1,
            "p-value": full.pvalues[col],
            "LR chi2 (drop-one)": lr,
            "LR p-value": chi2.sf(lr, 1),
        })
    effects = pd.DataFrame(rows).sort_values("LR chi2 (drop-one)", ascending=False)

    lr_all = 2 * (full.llf - traffic_only.llf)
    print("Do road features add anything beyond traffic volume?")
    print(f"  Likelihood ratio test, full model vs. AADT-only: chi2 = {lr_all:.1f}, "
          f"df = {len(FEATURES) - 1}, p = {chi2.sf(lr_all, len(FEATURES) - 1):.2g}")
    print(f"  Pseudo R²: AADT-only {traffic_only.prsquared:.3f} → full {full.prsquared:.3f}")
    print(f"  Overdispersion alpha = {full.params['alpha']:.3f} "
          f"(> 0 means negative binomial was the right choice over Poisson)")
    b_aadt = full.params["log_aadt"]
    print(f"  ln(AADT) coefficient = {b_aadt:.2f}: doubling traffic multiplies crashes by "
          f"{2 ** b_aadt:.2f}, so the crash *rate* per vehicle mile "
          f"{'falls' if b_aadt < 1 else 'rises'} as traffic grows\n")

    print("Feature effects, ranked by importance (drop-one likelihood ratio):\n")
    show = effects.copy()
    for c in ["Crash rate change", "CI low", "CI high"]:
        show[c] = (show[c] * 100).map("{:+.0f}%".format)
    show["p-value"] = show["p-value"].map("{:.3g}".format)
    show["LR chi2 (drop-one)"] = show["LR chi2 (drop-one)"].map("{:.1f}".format)
    show["LR p-value"] = show["LR p-value"].map("{:.3g}".format)
    print(show.to_string(index=False))
    print("\nCrash rate change = % change in expected crashes per vehicle mile for the typical")
    print("change shown, holding the other features constant. CI = 95% confidence interval.")

    print("\nIntersection counts for a few known segments (sanity check):")
    check = data[data["road"].isin(["ST GEORGE BLVD", "MALL DR", "RIVER RD", "RED HILLS PKWY"])]
    print(check[["road", "DESC", "Section_Length", "intersections", "intersections_per_mile"]]
          .round(2).head(10).to_string(index=False))

    data["predicted_crashes"] = full.predict()
    data["observed_minus_predicted"] = data["crashes"] - data["predicted_crashes"]
    keep = ["segment_id", "road", "RouteID", "BeginPoint", "EndPoint", "DESC", "Section_Length", "AADT",
            "crashes", "predicted_crashes", "observed_minus_predicted", "st_george_crashes",
            "intersections", *FEATURES]
    data[keep].sort_values("observed_minus_predicted", ascending=False).to_csv(
        HERE / "segmentRoadFeatures.csv", index=False)
    (HERE / "modelSummary.txt").write_text(str(full.summary()))

    plot = effects.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(plot))
    colors = ["#d62728" if p < 0.05 and v > 0 else "#2ca02c" if p < 0.05 else "#9e9e9e"
              for p, v in zip(plot["p-value"], plot["Crash rate change"])]
    ax.errorbar(plot["Crash rate change"] * 100, y,
                xerr=[(plot["Crash rate change"] - plot["CI low"]) * 100,
                      (plot["CI high"] - plot["Crash rate change"]) * 100],
                fmt="none", ecolor="#555555", capsize=4, zorder=1)
    ax.scatter(plot["Crash rate change"] * 100, y, c=colors, s=70, zorder=2)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, [f"{f}\n({c})" for f, c in zip(plot["Feature"], plot["Typical change"])], fontsize=9)
    ax.set_xlabel("% change in crash rate per vehicle mile (95% CI)")
    ax.set_title("Road features that predict crash rate, Washington County surface streets\n"
                 "Red = raises rate, green = lowers rate (p < 0.05); gray = not significant")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(HERE / "roadFeatureEffects.png", dpi=150)
    plt.close(fig)

    print("\nSaved segmentRoadFeatures.csv, modelSummary.txt, roadFeatureEffects.png")


if __name__ == "__main__":
    main()
