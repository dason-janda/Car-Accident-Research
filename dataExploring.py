"""
Initial exploratory analysis of the St. George crash data.

Looks at overall trends across the full dataset, then narrows in on
crashes with a "Suspected Serious Injury" or "Fatal" severity (the KA
crashes) to see how those differ from the dataset as a whole.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.width", 120)

KA_LEVELS = ["Suspected Serious Injury", "Fatal"]
VRU_COLS = ["Pedestrian Involved", "Bicycle Involved", "Motorcycle Involved"]

# --------------------------------------------------------------------------
# Load & clean
# --------------------------------------------------------------------------

df = pd.read_excel("./data/AccidentData21-26.csv.xlsx")
df.columns = [c.strip() for c in df.columns]

if "Crash Date Time" in df.columns:
    df["Crash Date Time"] = pd.to_datetime(df["Crash Date Time"], errors="coerce")
    df["Crash Year"] = df["Crash Date Time"].dt.year
    df["Crash Month"] = df["Crash Date Time"].dt.month


def explode_attributes(series: pd.Series) -> pd.Series:
    """Crash Attributes cells look like '["Intersection Related", "Left U Turn Involved"]'.
    Split them into one row per attribute with the brackets/quotes stripped."""
    tokens = series.dropna().apply(lambda cell: [tok.strip(' []"') for tok in str(cell).split(",")])
    return tokens.explode().str.strip().replace("", pd.NA).dropna()


print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(f"Rows: {len(df)}   Columns: {len(df.columns)}")

missing = df.isna().mean().sort_values(ascending=False)
print("\nColumns with the most missing data:")
print((missing[missing > 0] * 100).round(1).head(10).astype(str) + "%")

# --------------------------------------------------------------------------
# Overall trends
# --------------------------------------------------------------------------

print("\n" + "=" * 70)
print("OVERALL TRENDS")
print("=" * 70)

print("\nCrash Severity breakdown:")
severity_counts = df["Crash Severity"].value_counts(dropna=False)
print(severity_counts)

if "Crash Year" in df.columns:
    print("\nCrashes per year:")
    crashes_per_year = df["Crash Year"].value_counts().sort_index()
    print(crashes_per_year)

for col in ["Light Condition", "Weather Condition", "Roadway Surface Condition", "Manner of Collision"]:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).head(10))

if "Crash Attributes" in df.columns:
    overall_attrs = explode_attributes(df["Crash Attributes"])

# --------------------------------------------------------------------------
# Suspected Serious Injury / Fatal (KA) crashes
# --------------------------------------------------------------------------

print("\n" + "=" * 70)
print("SUSPECTED SERIOUS INJURY & FATAL (KA) CRASHES")
print("=" * 70)

ka = df[df["Crash Severity"].isin(KA_LEVELS)].copy()
print(f"\n{len(ka)} of {len(df)} crashes ({len(ka) / len(df):.1%}) are KA crashes")
print(ka["Crash Severity"].value_counts())

if "Crash Year" in ka.columns:
    print("\nKA crashes per year (with share of that year's total crashes):")
    ka_per_year = ka["Crash Year"].value_counts().sort_index()
    share_per_year = (ka_per_year / crashes_per_year * 100).round(1)
    trend = pd.DataFrame({"KA crashes": ka_per_year, "% of year's crashes": share_per_year})
    print(trend)

print("\nVulnerable road user involvement (KA crashes vs. overall dataset):")
for col in VRU_COLS:
    if col in df.columns:
        overall_rate = df[col].astype("string").str.strip().str.upper().eq("Y").mean()
        ka_rate = ka[col].astype("string").str.strip().str.upper().eq("Y").mean()
        print(f"  {col:<20} overall {overall_rate:.1%}   KA {ka_rate:.1%}")

for col in ["Light Condition", "Weather Condition", "Roadway Surface Condition", "Manner of Collision"]:
    if col in ka.columns:
        print(f"\n{col} (KA crashes only):")
        print(ka[col].value_counts(dropna=False).head(10))

if "Crash Attributes" in ka.columns:
    ka_attrs = explode_attributes(ka["Crash Attributes"])

    overall_attr_pct = (overall_attrs.value_counts() / len(df) * 100).round(2)
    ka_attr_pct = (ka_attrs.value_counts() / len(ka) * 100).round(2)

    attr_comparison = pd.DataFrame({
        "% of all crashes": overall_attr_pct,
        "% of KA crashes": ka_attr_pct,
    }).fillna(0.0)
    attr_comparison["difference (KA - overall), pts"] = (
        attr_comparison["% of KA crashes"] - attr_comparison["% of all crashes"]
    ).round(2)
    attr_comparison = attr_comparison.sort_values("% of KA crashes", ascending=False)

    print("\nCrash Attributes: % of all crashes vs. % of KA crashes:")
    print(attr_comparison.head(20))

# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))
severity_counts.plot(kind="bar", ax=ax)
ax.set_title("Crashes by Severity")
ax.set_ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
fig.savefig("SeverityBreakdown.png")
plt.close(fig)

if "Crash Attributes" in ka.columns:
    fig, ax = plt.subplots(figsize=(9, 6))
    ka_attrs.value_counts().head(15).plot(kind="barh", ax=ax)
    ax.invert_yaxis()
    ax.set_title("Top Crash Attributes in KA Crashes")
    ax.set_xlabel("Count")
    plt.tight_layout()
    fig.savefig("KACrashAttributes.png")
    plt.close(fig)

print("\nSaved SeverityBreakdown.png, KACrashAttributes.png")
