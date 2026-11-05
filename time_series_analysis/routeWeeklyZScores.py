"""
Weekly crash z-scores per road (route number) in St. George: which roads had
unusual weeks, where on the road those crashes happened, and which roads have
seasonal surges.

Each route (e.g. 3196 = River Rd, 0034 = St. George Blvd) is one unit, so a road
gets several crashes in a typical week and its z-scores are meaningful:

    z = (crashes on the road that week - road's average week) / road's weekly std

Traffic on a road is roughly constant week to week (AADT is annual), so the
z-score of the weekly count is also the z-score of its weekly crash rate.
Each unusual week also gets the Poisson probability of that many crashes by
chance, and the script reports how many such weeks luck alone would produce.

Only crashes on a route's mainline (not ramps or local streets) are counted.

Outputs (written next to this script):
    routeWeeklyZScores.xlsx    Roads, Stand-out Weeks, Monthly Pattern sheets
    routeWeeklyZScores.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from crashZScores import END, START, holidays  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 55)

CITY_CRASHES = ROOT / "data" / "AccidentData21-26.csv.xlsx"
RATES_CSV = ROOT / "normalized_crash_rates" / "crashRatesBySegment.csv"

MIN_CRASHES = 100   # roads with fewer crashes average < 0.4/week, too sparse for weekly z-scores
TOP_N = 20
KA_LEVELS = ["Suspected Serious Injury", "Fatal"]
CLEAR_WEATHER = {"Clear", "Cloudy", "Unknown"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load_crashes(week_index) -> pd.DataFrame:
    df = pd.read_excel(CITY_CRASHES)
    df.columns = [c.strip() for c in df.columns]
    route_name = df["Full Route Name"].astype(str).str.strip()
    df = df[route_name.str.fullmatch(r"\d{4}[PN]")].copy()
    df["route"] = df["Full Route Name"].str[:4].astype(int)
    df["time"] = pd.to_datetime(df["Crash Date Time"], errors="coerce")
    df["week_starting"] = df["time"].dt.normalize() - pd.to_timedelta(df["time"].dt.dayofweek, unit="D")
    df["ka"] = df["Crash Severity"].isin(KA_LEVELS)
    df["bad_weather"] = df["Weather Condition"].notna() & ~df["Weather Condition"].isin(CLEAR_WEATHER)
    return df[df["week_starting"].isin(week_index)]


def attach_sections(crashes: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """Label each crash with the AADT section of the road it happened on."""
    with_mp = crashes.dropna(subset=["Milepoint"]).sort_values("Milepoint")
    matched = pd.merge_asof(
        with_mp, segments[["route", "BeginPoint", "EndPoint", "DESC"]].sort_values("BeginPoint"),
        left_on="Milepoint", right_on="BeginPoint", by="route", direction="backward",
    )
    in_seg = matched["EndPoint"].notna() & (matched["Milepoint"] <= matched["EndPoint"])
    matched["section"] = np.where(
        in_seg,
        matched["DESC"].fillna("") + " (MP " + matched["BeginPoint"].round(2).astype(str)
        + "–" + matched["EndPoint"].round(2).astype(str) + ")",
        "MP " + matched["Milepoint"].round(2).astype(str),
    )
    return pd.concat([matched, crashes[crashes["Milepoint"].isna()].assign(section="unknown milepoint")])


def expected_by_luck(weekly_means: pd.Series, n_weeks: int, threshold: float) -> float:
    """Counts are whole numbers, so for each road find the smallest weekly count
    that falls below the threshold and how often that happens by chance."""
    total = 0.0
    for lam in weekly_means:
        k = 1
        while poisson.sf(k - 1, lam) >= threshold:
            k += 1
        total += n_weeks * poisson.sf(k - 1, lam)
    return total


def main():
    week_index = pd.date_range(START + pd.Timedelta(days=(7 - START.dayofweek) % 7),
                               END - pd.Timedelta(days=6), freq="W-MON")
    n_weeks = len(week_index)

    segments = pd.read_csv(RATES_CSV)
    road_names = segments.dropna(subset=["road"]).groupby("route")["road"].agg(lambda s: s.mode().iat[0])
    crashes = attach_sections(load_crashes(week_index), segments)

    counts = crashes.groupby(["route", "week_starting"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=week_index, fill_value=0)
    counts = counts[counts.sum(axis=1) >= MIN_CRASHES]

    mean, std = counts.mean(axis=1), counts.std(axis=1)
    z = counts.sub(mean, axis=0).div(std, axis=0)

    roads = pd.DataFrame({
        "road": [str(road_names.get(r, f"Route {r}")).title() for r in counts.index],
        "crashes": counts.sum(axis=1),
        "avg_per_week": mean.round(2),
        "weekly_std": std.round(2),
        "max_week": counts.max(axis=1),
        "weeks_z_over_2": (z >= 2).sum(axis=1),
    }, index=counts.index)
    roads.index.name = "route"
    print(f"{len(roads)} St. George roads with ≥ {MIN_CRASHES} mainline crashes, {n_weeks} full weeks\n")
    print(roads.sort_values("crashes", ascending=False).to_string())

    # ---- Stand-out weeks ----------------------------------------------
    hol = holidays()
    rows = []
    for (route, week), zval in z.stack().items():
        if zval < 2:
            continue
        g = crashes[(crashes["route"] == route) & (crashes["week_starting"] == week)]
        top_section = g["section"].value_counts()
        rows.append({
            "route": route,
            "road": roads.loc[route, "road"],
            "week_starting": week.date(),
            "crashes": counts.loc[route, week],
            "avg_per_week": roads.loc[route, "avg_per_week"],
            "z": round(zval, 2),
            "chance_probability": poisson.sf(counts.loc[route, week] - 1, mean[route]),
            "busiest_section": f"{top_section.index[0]}: {top_section.iat[0]} of {len(g)}",
            "ka": int(g["ka"].sum()),
            "bad_weather": int(g["bad_weather"].sum()),
            "top_manner": "; ".join(g["Manner of Collision"].dropna().value_counts().index[:2]),
            "holiday": ", ".join(hol[(hol.index >= week) & (hol.index <= week + pd.Timedelta(days=6))]),
        })
    weeks = pd.DataFrame(rows).sort_values("z", ascending=False)

    print(f"\nHow many road-weeks would look this extreme by luck alone? "
          f"({len(roads) * n_weeks:,} road-weeks tested)")
    for threshold in [0.01, 0.001, 0.0001]:
        observed = (poisson.sf(counts.values - 1, mean.values[:, None]) < threshold).sum()
        print(f"  chance probability < {threshold}: {observed:>4} observed, "
              f"{expected_by_luck(mean, n_weeks, threshold):6.1f} expected by luck")

    print(f"\nWeeks that stand out most on a road (z ≥ 2: {len(weeks)} road-weeks):")
    show = weeks.head(TOP_N).copy()
    show["chance_probability"] = show["chance_probability"].map("{:.1e}".format)
    print(show.drop(columns="route").to_string(index=False))

    # ---- Seasonal pattern per road --------------------------------------
    monthly = z.T.groupby(z.columns.month).mean().T
    monthly.columns = MONTHS
    monthly.index = roads["road"] + " (" + roads.index.astype(str) + ")"
    print("\nAverage weekly z-score by month (positive = more crashes than usual for that road):")
    print(monthly.round(2).to_string())

    with pd.ExcelWriter(HERE / "routeWeeklyZScores.xlsx", engine="openpyxl") as writer:
        roads.sort_values("crashes", ascending=False).to_excel(writer, sheet_name="Roads")
        weeks.round(4).to_excel(writer, sheet_name="Stand-out Weeks", index=False)
        monthly.round(2).to_excel(writer, sheet_name="Monthly Pattern")
        for sheet in writer.sheets.values():
            sheet.freeze_panes = "B2"

    plot(z, roads, monthly, week_index)
    print("\nSaved routeWeeklyZScores.xlsx and routeWeeklyZScores.png")


def plot(z, roads, monthly, week_index):
    order = roads.sort_values("crashes", ascending=False).index
    labels = [f"{roads.loc[r, 'road']} ({r})" for r in order]
    fig, axes = plt.subplots(2, 1, figsize=(15, 13), gridspec_kw={"height_ratios": [1.2, 1]})

    ax = axes[0]
    im = ax.imshow(z.loc[order].values, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest")
    ax.set_yticks(range(len(order)), labels, fontsize=8)
    yearly = [i for i, w in enumerate(week_index) if w.month == 1 and w.day <= 7]
    ax.set_xticks(yearly, [week_index[i].year for i in yearly])
    fig.colorbar(im, ax=ax, label="Weekly z-score")
    ax.set_title("Weekly crash z-score on each road (red = more crashes than usual for that road)")

    ax = axes[1]
    m = monthly.loc[[f"{roads.loc[r, 'road']} ({r})" for r in order]]
    im = ax.imshow(m.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_yticks(range(len(m)), m.index, fontsize=8)
    ax.set_xticks(range(12), MONTHS)
    for i in range(m.shape[0]):
        for j in range(12):
            ax.text(j, i, f"{m.iat[i, j]:+.1f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, label="Average weekly z-score")
    ax.set_title("Seasonal pattern: average weekly z-score by month on each road")

    fig.tight_layout()
    fig.savefig(HERE / "routeWeeklyZScores.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
