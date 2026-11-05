"""
Compare St. George crashes to the rest of Utah (Jun 2021 - May 2026).

St. George's rows in the statewide extract match the city extract exactly
(same 9,728 Crash IDs), so everything here comes from the statewide file.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
XLSX_PATH = DATA / "StatewideAccidentData21-26.xlsx"
CACHE_PATH = DATA / "StatewideCrashes.pkl"

# 2020 Census counts. Swap in estimates for another year if you prefer;
# the per-capita result is sensitive to this choice (St. George has grown fast).
POP_ST_GEORGE = 95_342
POP_UTAH = 3_271_616
YEARS_COVERED = 5

KA_LEVELS = ["Suspected Serious Injury", "Fatal"]
SEVERITY_ORDER = ["No injury", "Possible injury", "Suspected Minor Injury", "Suspected Serious Injury", "Fatal"]

# --------------------------------------------------------------------------
# Load (the xlsx takes ~2 minutes, so cache it after the first read)
# --------------------------------------------------------------------------

if CACHE_PATH.exists():
    state = pd.read_pickle(CACHE_PATH)
else:
    print("Reading statewide xlsx (first run only, ~2 min)...")
    state = pd.read_excel(XLSX_PATH, sheet_name="Crashes", dtype=str)
    state.columns = [c.strip() for c in state.columns]
    state.to_pickle(CACHE_PATH)

state["is_ka"] = state["crashSeverity_label"].isin(KA_LEVELS)

sg = state[state["city_geo"] == "St. George"]
rest = state[state["city_geo"] != "St. George"]
other_cities = rest[rest["city_geo"] != "Outside city limits"]

groups = {"St. George": sg, "Rest of Utah": rest, "Other Utah cities": other_cities}

# --------------------------------------------------------------------------
# 1. Crash volume
# --------------------------------------------------------------------------

print("=" * 70)
print("CRASH VOLUME")
print("=" * 70)
print(f"Statewide crashes: {len(state):,}")
print(f"St. George crashes: {len(sg):,} ({len(sg) / len(state):.1%} of the state)")
print(f"St. George share of Utah population (2020): {POP_ST_GEORGE / POP_UTAH:.1%}")

sg_rate = len(sg) / YEARS_COVERED / POP_ST_GEORGE * 1000
rest_rate = len(rest) / YEARS_COVERED / (POP_UTAH - POP_ST_GEORGE) * 1000
print(f"\nCrashes per 1,000 residents per year:")
print(f"  St. George:   {sg_rate:.1f}")
print(f"  Rest of Utah: {rest_rate:.1f}")
print("  (St. George's rate is inflated by tourists and I-15 through-traffic,")
print("   who crash there but aren't counted as residents.)")

city_counts = state.loc[state["city_geo"] != "Outside city limits", "city_geo"].value_counts()
print(f"\nSt. George ranks #{city_counts.index.get_loc('St. George') + 1} of {len(city_counts)} "
      f"Utah cities by total crashes.")

# --------------------------------------------------------------------------
# 2. Severity: when a crash happens, how bad is it?
# --------------------------------------------------------------------------

print("\n" + "=" * 70)
print("SEVERITY (% of each group's crashes)")
print("=" * 70)

severity_pct = pd.DataFrame({
    name: g["crashSeverity_label"].value_counts(normalize=True).reindex(SEVERITY_ORDER).fillna(0) * 100
    for name, g in groups.items()
}).round(2)
severity_pct.loc["KA (Serious + Fatal)"] = [g["is_ka"].mean() * 100 for g in groups.values()]
print(severity_pct.round(2))

for label, comparison in [("rest of Utah", rest), ("other Utah cities", other_cities)]:
    table = [[sg["is_ka"].sum(), (~sg["is_ka"]).sum()],
             [comparison["is_ka"].sum(), (~comparison["is_ka"]).sum()]]
    _, p, _, _ = chi2_contingency(table)
    print(f"KA share, St. George vs. {label}: p = {p:.4f}"
          f" ({'significant' if p < 0.05 else 'not significant'} at 0.05)")

# --------------------------------------------------------------------------
# 3. Intersection / junction type
# --------------------------------------------------------------------------

print("\n" + "=" * 70)
print("ROADWAY JUNCTION TYPE")
print("=" * 70)

junction_pct = pd.DataFrame({
    name: g["roadwayJunction_label"].value_counts(normalize=True) * 100
    for name, g in groups.items()
}).fillna(0)
junction_pct["St. George - Rest (pts)"] = junction_pct["St. George"] - junction_pct["Rest of Utah"]
junction_pct = junction_pct.sort_values("St. George", ascending=False)
print("\n% of crashes at each junction type:")
print(junction_pct.round(2))

contingency = pd.crosstab(
    state["city_geo"] == "St. George", state["roadwayJunction_label"]
)
_, p, _, _ = chi2_contingency(contingency)
print(f"\nJunction mix, St. George vs. rest of Utah: chi-square p {'< 0.001' if p < 0.001 else f'= {p:.3f}'}")

print("\nKA rate by junction type (types with >= 30 St. George crashes):")
ka_by_junction = pd.DataFrame({
    "St. George crashes": sg["roadwayJunction_label"].value_counts(),
    "St. George KA %": sg.groupby("roadwayJunction_label")["is_ka"].mean() * 100,
    "Rest of Utah KA %": rest.groupby("roadwayJunction_label")["is_ka"].mean() * 100,
})
ka_by_junction = ka_by_junction[ka_by_junction["St. George crashes"] >= 30]
print(ka_by_junction.sort_values("St. George crashes", ascending=False).round(2))

# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

top_junctions = junction_pct.head(10).index
fig, ax = plt.subplots(figsize=(10, 6))
junction_pct.loc[top_junctions, ["St. George", "Rest of Utah", "Other Utah cities"]].plot(kind="barh", ax=ax)
ax.invert_yaxis()
ax.set_title("Share of Crashes by Junction Type: St. George vs. Utah")
ax.set_xlabel("% of crashes")
ax.set_ylabel("")
plt.tight_layout()
fig.savefig(HERE / "junctionTypeComparison.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
severity_pct.loc[SEVERITY_ORDER[1:]].plot(kind="bar", ax=ax)
ax.set_title("Crash Severity Mix (excluding No Injury): St. George vs. Utah")
ax.set_ylabel("% of crashes")
ax.set_xlabel("")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
fig.savefig(HERE / "severityComparison.png", dpi=150)
plt.close(fig)

print("\nSaved junctionTypeComparison.png and severityComparison.png")
