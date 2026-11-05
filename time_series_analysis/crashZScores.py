"""
Z-scores of St. George crash counts by day, week and month, to find time
periods with unusually many crashes (events, holidays, tourist seasons,
storms) where safety efforts could be focused.

    z = (period's crashes - average period) / standard deviation of periods

A z of 2 or more happens by chance about 2% of the time; 3 or more about 0.1%.

Adjustments, so the rankings show unusual periods rather than built-in patterns:
    Days     Weekdays average ~6 crashes and Sundays ~2, so a plain z mostly ranks
             weekdays. The day-of-week-adjusted z compares each day only with the
             same weekday (Fridays vs. Fridays, ...). Both are reported.
    Weeks    Only full Monday-Sunday weeks are scored.
    Months   Scored on crashes per day, so 31-day months aren't favored.

Each top period is annotated with holidays, the share of crashes in bad
weather (rain/snow storms cause spikes unrelated to events) and serious or
fatal crash counts. A week-of-year table shows which weeks are high year after
year, which points to recurring events or seasons rather than one-off spikes.

Outputs (written next to this script):
    crashZScores.xlsx      Days, Weeks, Months and Week of Year sheets
    crashZScores.png       z-scores over time plus the recurring weekly pattern
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dateutil.easter import easter
from pandas.tseries.holiday import USFederalHolidayCalendar

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)

HERE = Path(__file__).resolve().parent
CITY_CRASHES = HERE.parent / "data" / "AccidentData21-26.csv.xlsx"

# Full extract window (from the crash portal's filter), so crash-free days count as zeros.
START, END = pd.Timestamp("2021-06-01"), pd.Timestamp("2026-05-31")
TOP_N = 10
KA_LEVELS = ["Suspected Serious Injury", "Fatal"]
CLEAR_WEATHER = {"Clear", "Cloudy", "Unknown"}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_daily() -> pd.DataFrame:
    df = pd.read_excel(CITY_CRASHES)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["Crash Date Time"], errors="coerce").dt.normalize()
    df["bad_weather"] = df["Weather Condition"].notna() & ~df["Weather Condition"].isin(CLEAR_WEATHER)
    df["ka"] = df["Crash Severity"].isin(KA_LEVELS)

    daily = df.groupby("date").agg(crashes=("date", "size"), bad_weather=("bad_weather", "sum"), ka=("ka", "sum"))
    return daily.reindex(pd.date_range(START, END, freq="D"), fill_value=0)


def holidays() -> pd.Series:
    names = USFederalHolidayCalendar().holidays(START, END, return_name=True)
    easters = pd.Series("Easter", index=pd.to_datetime([easter(y) for y in range(START.year, END.year + 1)]))
    return pd.concat([names, easters]).sort_index()


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


# --------------------------------------------------------------------------
# Periods
# --------------------------------------------------------------------------

def score_days(daily: pd.DataFrame, hol: pd.Series) -> pd.DataFrame:
    days = daily.copy()
    days["weekday"] = [WEEKDAYS[d] for d in days.index.dayofweek]
    days["z"] = zscore(days["crashes"])
    by_weekday = days.groupby("weekday")["crashes"]
    days["z_weekday_adjusted"] = (days["crashes"] - by_weekday.transform("mean")) / by_weekday.transform("std")

    def nearby_holiday(day):
        for offset, tag in [(0, ""), (-1, " (day before)"), (1, " (day after)")]:
            hit = hol.get(day - pd.Timedelta(days=offset))
            if hit is not None:
                return (hit if isinstance(hit, str) else ", ".join(hit)) + tag
        return ""

    days["holiday"] = [nearby_holiday(d) for d in days.index]
    days.index.name = "date"
    return days


def score_weeks(daily: pd.DataFrame, hol: pd.Series) -> pd.DataFrame:
    weeks = daily.resample("W-SUN").agg({"crashes": "sum", "bad_weather": "sum", "ka": "sum"})
    weeks["days"] = daily["crashes"].resample("W-SUN").size()
    weeks = weeks[weeks["days"] == 7].drop(columns="days")
    weeks.index = weeks.index - pd.Timedelta(days=6)  # label by the Monday
    weeks.index.name = "week_starting"
    weeks["z"] = zscore(weeks["crashes"])
    weeks["holiday"] = [", ".join(hol[(hol.index >= w) & (hol.index <= w + pd.Timedelta(days=6))]) for w in weeks.index]
    return weeks


def score_months(daily: pd.DataFrame, hol: pd.Series) -> pd.DataFrame:
    months = daily.resample("MS").agg({"crashes": "sum", "bad_weather": "sum", "ka": "sum"})
    months["crashes_per_day"] = months["crashes"] / months.index.days_in_month
    months.index.name = "month"
    months["z"] = zscore(months["crashes_per_day"])
    months["holiday"] = [", ".join(hol[(hol.index.year == m.year) & (hol.index.month == m.month)]) for m in months.index]
    return months


def recurring_weeks(weeks: pd.DataFrame) -> pd.DataFrame:
    """Same week of the year across all five years."""
    w = weeks.copy()
    w["week_of_year"] = w.index.isocalendar().week.values
    w["day_of_year"] = w.index.dayofyear
    pattern = w.groupby("week_of_year").agg(
        avg_crashes=("crashes", "mean"),
        years=("crashes", "size"),
        years_with_z_over_1=("z", lambda z: int((z > 1).sum())),
        typical_day_of_year=("day_of_year", "median"),
    )
    pattern = pattern[pattern["years"] >= 3]
    pattern["z"] = zscore(pattern["avg_crashes"])
    start = pd.Timestamp("2025-01-01") + pd.to_timedelta(pattern["typical_day_of_year"] - 1, unit="D")
    pattern["typical_week"] = [f"{s:%b %d} – {s + pd.Timedelta(days=6):%b %d}" for s in start]
    return pattern.drop(columns="typical_day_of_year")


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def with_weather_share(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["% bad weather"] = (out["bad_weather"] / out["crashes"].where(out["crashes"] > 0) * 100).round(0)
    return out.drop(columns="bad_weather")


def print_top(title, frame, z_col, columns, date_fmt):
    top = with_weather_share(frame).nlargest(TOP_N, z_col)
    top.index = top.index.strftime(date_fmt)
    print(f"\n{title}")
    print(top[columns].round(2).to_string())


def plot(days, weeks, months, pattern, outpath):
    fig, axes = plt.subplots(4, 1, figsize=(13, 15))

    ax = axes[0]
    hi = days["z_weekday_adjusted"] >= 3
    ax.scatter(days.index[~hi], days.loc[~hi, "z_weekday_adjusted"], s=4, color="#9e9e9e")
    ax.scatter(days.index[hi], days.loc[hi, "z_weekday_adjusted"], s=14, color="#d62728", label="z ≥ 3")
    ax.set_title("Daily crashes, z-score vs. the same weekday")
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.plot(weeks.index, weeks["z"], color="#1f77b4", linewidth=1)
    hi = weeks["z"] >= 2
    ax.scatter(weeks.index[hi], weeks.loc[hi, "z"], s=18, color="#d62728", zorder=3, label="z ≥ 2")
    ax.set_title("Weekly crashes, z-score")
    ax.legend(loc="upper right")

    ax = axes[2]
    ax.bar(months.index, months["z"], width=25, color=["#d62728" if z >= 1 else "#9e9e9e" for z in months["z"]])
    ax.set_title("Monthly crashes per day, z-score (red = z ≥ 1)")

    ax = axes[3]
    ax.bar(pattern.index, pattern["z"], color=["#d62728" if z >= 1 else "#9e9e9e" for z in pattern["z"]])
    ticks = pattern.index[::4]
    ax.set_xticks(ticks, [pattern.loc[t, "typical_week"].split(" – ")[0] for t in ticks], rotation=45, ha="right")
    ax.set_title("Recurring pattern: average crashes in each week of the year (2021–2026), z-score")

    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylabel("z-score")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main():
    daily = load_daily()
    hol = holidays()
    days = score_days(daily, hol)
    weeks = score_weeks(daily, hol)
    months = score_months(daily, hol)
    pattern = recurring_weeks(weeks)

    print(f"{int(daily['crashes'].sum()):,} crashes, {START:%b %d %Y} – {END:%b %d %Y}")
    print(f"Average: {daily['crashes'].mean():.1f} per day, {weeks['crashes'].mean():.1f} per week, "
          f"{months['crashes'].mean():.0f} per month")

    print_top("Highest days (day-of-week adjusted):", days, "z_weekday_adjusted",
              ["weekday", "crashes", "z_weekday_adjusted", "z", "ka", "% bad weather", "holiday"], "%Y-%m-%d")
    print_top("Highest weeks:", weeks, "z", ["crashes", "z", "ka", "% bad weather", "holiday"], "%Y-%m-%d")
    print_top("Highest months:", months, "z",
              ["crashes", "crashes_per_day", "z", "ka", "% bad weather", "holiday"], "%Y-%m")

    print("\nWeeks of the year that are high every year (recurring events / seasons):")
    print(pattern.nlargest(TOP_N, "z")[["typical_week", "avg_crashes", "z", "years_with_z_over_1", "years"]]
          .round(2).to_string())

    with pd.ExcelWriter(HERE / "crashZScores.xlsx", engine="openpyxl") as writer:
        for name, frame in [("Days", with_weather_share(days)), ("Weeks", with_weather_share(weeks)),
                            ("Months", with_weather_share(months)), ("Week of Year", pattern)]:
            out = frame.round(2)
            if isinstance(out.index, pd.DatetimeIndex):
                out.index = out.index.date
            out.to_excel(writer, sheet_name=name)
            writer.sheets[name].freeze_panes = "B2"
    plot(days, weeks, months, pattern, HERE / "crashZScores.png")
    print("\nSaved crashZScores.xlsx and crashZScores.png")


if __name__ == "__main__":
    main()
