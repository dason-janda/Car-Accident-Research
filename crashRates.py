"""
Crash rate per 100 million vehicle miles traveled (MVMT) for each AADT
roadway segment in Washington County, with St. George segments flagged
using the critical crash rate method (Highway Safety Manual).

    Rate = crashes * 100,000,000 / VMT
    VMT  = segment length (mi) * AADT * 365 * years of crash data

Crashes come from the statewide extract (not the city extract) because
many segments run past St. George city limits, and the city extract only
has the crashes inside the city. Matching is by route number + milepoint.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 60)

AADT_PATH = "data/AADTHistory 2024.xlsx"
STATE_CACHE = "data/StatewideCrashes.pkl"
WASHINGTON_COUNTY_STATION_PREFIX = "053"

# Crash data runs Jun 1 2021 - May 31 2026.
YEARS = 5.0
# Months of crash data falling in each calendar year. There are no AADT counts
# past 2024 yet, so 2024 volumes stand in for 2025 and the first 5 months of 2026.
AADT_YEAR_WEIGHTS = {"AADT2021": 7, "AADT2022": 12, "AADT2023": 12, "AADT2024": 12 + 12 + 5}

CONFIDENCE_K = 1.645  # 95% one-sided

# --------------------------------------------------------------------------
# Segments
# --------------------------------------------------------------------------

aadt = pd.read_excel(AADT_PATH, sheet_name="UnroundedAADT2024")
segments = aadt[
    aadt["STATION"].str.startswith(WASHINGTON_COUNTY_STATION_PREFIX)
    & ~aadt["DESC"].str.contains("unconstructed", case=False, na=False)
].copy()
# STATION isn't unique (some stations cover several segments), so key on the row instead.
segments = segments.reset_index(drop=True)
segments["segment_id"] = segments.index
segments["route"] = segments["RouteID"].str[:4].astype(int)

weights = pd.Series(AADT_YEAR_WEIGHTS)
segments["AADT"] = (segments[weights.index] * weights).sum(axis=1) / weights.sum()
segments["VMT"] = segments["Section_Length"] * segments["AADT"] * 365 * YEARS

segments["facility"] = "Federal-aid city road"
segments.loc[segments["route"] < 1000, "facility"] = "State route"
segments.loc[segments["route"] == 15, "facility"] = "Interstate"

# Flag AADT values under a quarter of the neighboring segments' on the same route;
# these are more likely count errors than real drops in traffic, and they inflate the rate.
segments = segments.sort_values(["route", "BeginPoint"])
neighbor_aadt = pd.concat(
    [segments.groupby("route")["AADT"].shift(1), segments.groupby("route")["AADT"].shift(-1)], axis=1
).min(axis=1)
segments["check_aadt"] = segments["AADT"] < 0.25 * neighbor_aadt

# --------------------------------------------------------------------------
# Crashes -> segments
# --------------------------------------------------------------------------

state = pd.read_pickle(STATE_CACHE)
crashes = state[(state["countyName"] == "Washington") & (state["type"] == "M")].copy()
crashes["route"] = pd.to_numeric(crashes["route"], errors="coerce")
crashes["milePoint"] = pd.to_numeric(crashes["milePoint"], errors="coerce")
crashes = crashes.dropna(subset=["route", "milePoint"])
crashes["route"] = crashes["route"].astype(int)
crashes["is_ka"] = crashes["crashSeverity_label"].isin(["Suspected Serious Injury", "Fatal"])
crashes["in_st_george"] = crashes["city_geo"] == "St. George"

# Each crash goes to the last segment on its route that begins at or before its
# milepoint; crashes that land in a gap between segments are dropped below.
matched = pd.merge_asof(
    crashes.sort_values("milePoint"),
    segments[["segment_id", "route", "BeginPoint", "EndPoint"]].sort_values("BeginPoint"),
    left_on="milePoint", right_on="BeginPoint", by="route", direction="backward",
)
matched = matched[matched["segment_id"].notna() & (matched["milePoint"] <= matched["EndPoint"])]

counts = matched.groupby("segment_id").agg(
    crashes=("Crash_ID", "size"),
    ka_crashes=("is_ka", "sum"),
    st_george_crashes=("in_st_george", "sum"),
)
segments = segments.merge(counts, left_on="segment_id", right_index=True, how="left")
segments[["crashes", "ka_crashes", "st_george_crashes"]] = (
    segments[["crashes", "ka_crashes", "st_george_crashes"]].fillna(0).astype(int)
)

road_names = matched.groupby("route")["roadName"].agg(lambda s: s.str.upper().str.strip().mode().iat[0])
segments["road"] = segments["route"].map(road_names).fillna("")

sg_total = int(state["city_geo"].eq("St. George").sum())
sg_matched = int(matched["in_st_george"].sum())
print(f"Washington County mainline crashes matched to a segment: {len(matched):,} of {len(crashes):,}")
print(f"St. George crashes matched to a segment: {sg_matched:,} of {sg_total:,} ({sg_matched / sg_total:.0%})")
print("  (the rest are on local streets or ramps, which have no AADT counts)\n")

# --------------------------------------------------------------------------
# Crash rate and critical rate
# --------------------------------------------------------------------------

segments["rate"] = segments["crashes"] * 1e8 / segments["VMT"]

# Average rate for each facility type across all of Washington County.
totals = segments.groupby("facility")[["crashes", "VMT"]].sum()
avg_rate = totals["crashes"] * 1e8 / totals["VMT"]
segments["avg_rate"] = segments["facility"].map(avg_rate)

exposure = segments["VMT"] / 1e8
segments["critical_rate"] = (
    segments["avg_rate"]
    + CONFIDENCE_K * (segments["avg_rate"] / exposure).pow(0.5)
    + 1 / (2 * exposure)
)
segments["rate_ratio"] = segments["rate"] / segments["critical_rate"]
segments["anomaly"] = segments["rate"] > segments["critical_rate"]

print("Average crash rate per 100 MVMT by facility type (Washington County):")
for facility, rate in avg_rate.items():
    n = (segments["facility"] == facility).sum()
    print(f"  {facility:<22} {rate:7.1f}   ({n} segments)")

# --------------------------------------------------------------------------
# St. George results
# --------------------------------------------------------------------------

sg_segments = segments[segments["st_george_crashes"] > 0].sort_values("rate_ratio", ascending=False)

columns = {
    "road": "Road",
    "route": "Rt",
    "BeginPoint": "Begin MP",
    "EndPoint": "End MP",
    "Section_Length": "Miles",
    "AADT": "AADT",
    "crashes": "Crashes",
    "ka_crashes": "KA",
    "rate": "Rate",
    "critical_rate": "Critical",
    "rate_ratio": "Ratio",
    "facility": "Facility",
    "check_aadt": "Check AADT",
    "DESC": "Segment",
}
report = sg_segments[list(columns)].rename(columns=columns)
report["AADT"] = report["AADT"].round(0).astype(int)
report[["Rate", "Critical"]] = report[["Rate", "Critical"]].round(1)
report["Ratio"] = report["Ratio"].round(2)

flagged = report[report["Ratio"] > 1]
print(f"\n{len(sg_segments)} segments carry St. George crashes; "
      f"{len(flagged)} have a crash rate above their critical rate (ratio > 1):\n")
print(flagged.to_string(index=False))
print("\n'Check AADT' = True means the segment's AADT is under 1/4 of its neighbors' on the same")
print("route; verify that count before treating the segment as a real anomaly.")

print("\nHighest raw crash counts, for comparison (count alone ignores traffic volume):\n")
print(report.sort_values("Crashes", ascending=False).head(10).to_string(index=False))

segments.sort_values("rate_ratio", ascending=False).to_csv("crashRatesBySegment.csv", index=False)

# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------

top = sg_segments.head(15).iloc[::-1]
labels = [
    f"{r.road.title()} (Rt {r.route}) MP {r.BeginPoint:.2f}-{r.EndPoint:.2f}" for r in top.itertuples()
]
fig, ax = plt.subplots(figsize=(11, 7))
ax.barh(labels, top["rate"], color=["#d62728" if a else "#1f77b4" for a in top["anomaly"]], label="Crash rate")
ax.scatter(top["critical_rate"], labels, color="black", marker="|", s=300, linewidths=2.5, zorder=3,
           label="Critical rate (95%)")
ax.set_xlabel("Crashes per 100 million vehicle miles traveled")
ax.set_title("St. George Road Segments Ranked by Crash Rate / Critical Rate\n"
             "Red = statistically higher than similar roads in Washington County")
ax.legend(loc="lower right")
plt.tight_layout()
fig.savefig("segmentCrashRates.png", dpi=150)
plt.close(fig)

print("\nSaved crashRatesBySegment.csv and segmentCrashRates.png")
