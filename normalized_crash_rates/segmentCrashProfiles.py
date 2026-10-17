"""
Crash profiles for the 10 St. George segments whose crash rate is furthest
above their critical rate, for matching against FHWA's Proven Safety
Countermeasures.

For each segment: most common crash types, manners of collision, junction
types, light conditions and crash attributes, each compared with St. George
as a whole so over-represented patterns stand out. Run crashRates.py first.

Output: segmentCrashProfiles.xlsx (Summary + Details sheets)
"""

from pathlib import Path

import pandas as pd

pd.set_option("display.width", 200)

HERE = Path(__file__).resolve().parent
RATES_CSV = HERE / "crashRatesBySegment.csv"
CITY_CRASHES = HERE.parent / "data" / "AccidentData21-26.csv.xlsx"
OUTPUT = HERE / "segmentCrashProfiles.xlsx"

TOP_N = 10
# AADTs this low on city roads are usually old counts rolled forward, which
# inflates the rate; leave them out of the ranking along with flagged counts.
MIN_AADT = 1000
# Skip segments that are mostly in a neighboring city: the city extract only
# has St. George crashes, and the fix wouldn't be St. George's to make.
MIN_ST_GEORGE_SHARE = 0.5
KA_LEVELS = ["Suspected Serious Injury", "Fatal"]
CATEGORIES = ["Crash Type", "Manner of Collision", "Roadway Junction Type", "Light Condition", "Crash Attributes"]

# An attribute "stands out" if it's in at least this share of the segment's
# crashes and this many times more common than citywide.
STANDOUT_MIN_PCT = 10
STANDOUT_MIN_RATIO = 1.5


def explode_attributes(series: pd.Series) -> pd.Series:
    """Crash Attributes cells look like '["Intersection Related", "Left U Turn Involved"]'.
    Split them into one row per attribute with the brackets/quotes stripped."""
    tokens = series.dropna().apply(lambda cell: [tok.strip(' []"') for tok in str(cell).split(",")])
    return tokens.explode().str.strip().replace("", pd.NA).dropna()


def category_values(df: pd.DataFrame, category: str) -> pd.Series:
    if category == "Crash Attributes":
        return explode_attributes(df[category])
    return df[category].dropna().astype(str).str.strip()


# --------------------------------------------------------------------------
# Top segments and their crashes
# --------------------------------------------------------------------------

segments = pd.read_csv(RATES_CSV)
top = segments[
    (segments["st_george_crashes"] >= MIN_ST_GEORGE_SHARE * segments["crashes"])
    & (segments["st_george_crashes"] > 0)
    & segments["anomaly"]
    & ~segments["check_aadt"]
    & (segments["AADT"] >= MIN_AADT)
].nlargest(TOP_N, "rate_ratio")

city = pd.read_excel(CITY_CRASHES)
city.columns = [c.strip() for c in city.columns]

# Mainline route names look like "3196P" / "0015N"; ramps ("0015PR00409") and
# local streets ("0000") aren't on an AADT segment.
route_name = city["Full Route Name"].astype(str).str.strip()
mainline = city[route_name.str.fullmatch(r"\d{4}[PN]") & city["Milepoint"].notna()].copy()
mainline["route"] = mainline["Full Route Name"].str[:4].astype(int)

matched = pd.merge_asof(
    mainline.sort_values("Milepoint"),
    segments[["segment_id", "route", "BeginPoint", "EndPoint"]].sort_values("BeginPoint"),
    left_on="Milepoint", right_on="BeginPoint", by="route", direction="backward",
)
matched = matched[matched["segment_id"].notna() & (matched["Milepoint"] <= matched["EndPoint"])]

city_share = {
    cat: category_values(city, cat).value_counts() / len(city) * 100 for cat in CATEGORIES
}

# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------

summary_rows, detail_rows = [], []

for rank, seg in enumerate(top.itertuples(), start=1):
    crashes = matched[matched["segment_id"] == seg.segment_id]
    n = len(crashes)
    label = f"{str(seg.road).title()} MP {seg.BeginPoint:.3f}-{seg.EndPoint:.3f}"

    top_values = {}
    for cat in CATEGORIES:
        counts = category_values(crashes, cat).value_counts()
        rows = []
        for value, count in counts.items():
            pct = count / n * 100
            base = city_share[cat].get(value, 0.0)
            rows.append((value, count, pct, base, pct / base if base else float("inf")))
            detail_rows.append({
                "Rank": rank, "Segment": label, "Category": cat, "Value": value,
                "Crashes": count, "% of segment crashes": round(pct, 1),
                "% of St. George crashes": round(base, 1),
                "Times more common than citywide": round(pct / base, 2) if base else None,
            })
        top_values[cat] = rows

    def fmt(cat, k):
        return "; ".join(f"{v} ({p:.0f}%)" for v, _, p, _, _ in top_values[cat][:k])

    standouts = "; ".join(
        f"{v} ({p:.0f}%, {r:.1f}x city)"
        for v, _, p, _, r in top_values["Crash Attributes"]
        if p >= STANDOUT_MIN_PCT and r >= STANDOUT_MIN_RATIO
    )

    summary_rows.append({
        "Rank": rank,
        "Road": str(seg.road).title(),
        "Route": seg.RouteID,
        "Milepoints": f"{seg.BeginPoint:.3f}-{seg.EndPoint:.3f}",
        "Segment": seg.DESC,
        "AADT": round(seg.AADT),
        "Crash rate (per 100 MVMT)": round(seg.rate, 1),
        "Critical rate": round(seg.critical_rate, 1),
        "Rate / critical": round(seg.rate_ratio, 2),
        "Crashes profiled": n,
        "Serious/fatal (KA)": int(crashes["Crash Severity"].isin(KA_LEVELS).sum()),
        "Most common crash type": fmt("Crash Type", 1),
        "Top manners of collision": fmt("Manner of Collision", 3),
        "Top junction types": fmt("Roadway Junction Type", 2),
        "Light condition": fmt("Light Condition", 2),
        "Top crash attributes": fmt("Crash Attributes", 5),
        "Stands out vs. city": standouts or "—",
    })

summary = pd.DataFrame(summary_rows)
details = pd.DataFrame(detail_rows)

# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

print(f"Top {len(summary)} St. George segments by crash rate / critical rate "
      f"(excluding suspect or <{MIN_AADT:,} AADT counts and segments mostly outside St. George)\n")
for row in summary_rows:
    print(f"#{row['Rank']}  {row['Road']} ({row['Route']} MP {row['Milepoints']})  —  {row['Segment']}")
    print(f"    AADT {row['AADT']:,}   rate {row['Crash rate (per 100 MVMT)']}   "
          f"{row['Rate / critical']}x critical   {row['Crashes profiled']} crashes, {row['Serious/fatal (KA)']} KA")
    for key in ["Most common crash type", "Top manners of collision", "Top junction types",
                "Light condition", "Top crash attributes", "Stands out vs. city"]:
        print(f"    {key + ':':<26} {row[key]}")
    print()

with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
    for name, frame in [("Summary", summary), ("Details", details)]:
        frame.to_excel(writer, sheet_name=name, index=False)
        sheet = writer.sheets[name]
        for col_cells in sheet.columns:
            longest = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
            sheet.column_dimensions[col_cells[0].column_letter].width = min(longest + 2, 60)
        sheet.freeze_panes = "A2"

print(f"Saved {OUTPUT.name}")
