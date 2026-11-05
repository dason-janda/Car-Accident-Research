"""
How does daily average temperature relate to weekly crash z-scores in St. George?

Weather: Open-Meteo historical archive (daily mean temperature, precipitation),
cached to data/stGeorgeDailyWeather.csv after the first download.

Two analyses:
  1. Week of year: each calendar week's average temperature (2021-2026) vs. its
     average crash z-score (the recurring pattern from crashZScores.py).
     Temperature and season move together, so this mixes the effect of heat with
     everything else seasonal (tourists, snowbirds, school breaks, events).
  2. Same week, different years: for each week, how much warmer or colder than
     usual it was for that time of year vs. how many more or fewer crashes it had
     than usual for that time of year. Comparing a week only with the same week in
     other years removes the seasonal pattern, so what's left is closer to the
     effect of temperature itself. Rainy days are controlled for.

Outputs (written next to this script):
    temperatureVsCrashes.png
    temperatureVsCrashes.xlsx     weekly and week-of-year tables
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr

from crashZScores import END, START, holidays, load_daily, recurring_weeks, score_weeks, zscore

pd.set_option("display.width", 200)

HERE = Path(__file__).resolve().parent
WEATHER_CACHE = HERE.parent / "data" / "stGeorgeDailyWeather.csv"
ST_GEORGE = {"latitude": 37.0965, "longitude": -113.5684}
RAINY_DAY_INCHES = 0.05


# --------------------------------------------------------------------------
# Weather
# --------------------------------------------------------------------------

def load_weather() -> pd.DataFrame:
    if WEATHER_CACHE.exists():
        return pd.read_csv(WEATHER_CACHE, index_col="date", parse_dates=["date"])
    print("Downloading daily weather from Open-Meteo...")
    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            **ST_GEORGE,
            "start_date": f"{START:%Y-%m-%d}",
            "end_date": f"{END:%Y-%m-%d}",
            "daily": "temperature_2m_mean,temperature_2m_max,precipitation_sum",
            "temperature_unit": "fahrenheit",
            "precipitation_unit": "inch",
            "timezone": "America/Denver",
        },
        timeout=60,
    )
    resp.raise_for_status()
    daily = resp.json()["daily"]
    weather = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "temp_mean_f": daily["temperature_2m_mean"],
        "temp_max_f": daily["temperature_2m_max"],
        "precip_in": daily["precipitation_sum"],
    }).set_index("date")
    weather.to_csv(WEATHER_CACHE)
    return weather


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def weekly_table(weather: pd.DataFrame) -> pd.DataFrame:
    weeks = score_weeks(load_daily(), holidays())
    weather = weather.assign(rainy=(weather["precip_in"] >= RAINY_DAY_INCHES).astype(int))
    by_week = weather.resample("W-SUN").agg({"temp_mean_f": "mean", "temp_max_f": "mean", "rainy": "sum"})
    by_week.index = by_week.index - pd.Timedelta(days=6)
    weeks = weeks.join(by_week.rename(columns={"rainy": "rainy_days"}))
    weeks["week_of_year"] = weeks.index.isocalendar().week.values

    # Departures from what's normal for that calendar week.
    normal = weeks.groupby("week_of_year")[["crashes", "temp_mean_f", "rainy_days"]].transform("mean")
    weeks["crashes_vs_normal"] = weeks["crashes"] - normal["crashes"]
    weeks["temp_vs_normal_f"] = weeks["temp_mean_f"] - normal["temp_mean_f"]
    weeks["rainy_days_vs_normal"] = weeks["rainy_days"] - normal["rainy_days"]
    return weeks


def week_of_year_table(weeks: pd.DataFrame) -> pd.DataFrame:
    pattern = recurring_weeks(weeks)
    temps = weeks.groupby("week_of_year")[["temp_mean_f", "rainy_days"]].mean()
    return pattern.join(temps, how="left")


def describe_corr(x, y):
    r, p = pearsonr(x, y)
    rho, p_rho = spearmanr(x, y)
    return f"Pearson r = {r:+.2f} (p = {p:.2g}), Spearman rho = {rho:+.2f} (p = {p_rho:.2g})"


def main():
    weather = load_weather()
    weeks = weekly_table(weather)
    pattern = week_of_year_table(weeks)

    print(f"{len(weeks)} full weeks, {len(pattern)} weeks of the year\n")

    print("1. Week of year: average temperature vs. average crash z-score")
    print("   " + describe_corr(pattern["temp_mean_f"], pattern["z"]))
    bands = pd.cut(pattern["temp_mean_f"], bins=[0, 45, 55, 65, 75, 85, 120],
                   labels=["< 45°F", "45–55°F", "55–65°F", "65–75°F", "75–85°F", "85°F+"])
    band_table = pattern.groupby(bands, observed=True).agg(
        weeks=("z", "size"), avg_crash_z=("z", "mean"), avg_crashes=("avg_crashes", "mean"))
    print(band_table.round(2).to_string())
    print("   Note: this mixes temperature with everything else seasonal.\n")

    print("2. Same calendar week in different years (seasonal pattern removed)")
    print("   " + describe_corr(weeks["temp_vs_normal_f"], weeks["crashes_vs_normal"]))
    model = smf.ols("crashes_vs_normal ~ temp_vs_normal_f + rainy_days_vs_normal", data=weeks).fit()
    for term, label, unit in [("temp_vs_normal_f", "Each 1°F warmer than normal", "°F"),
                              ("rainy_days_vs_normal", "Each extra rainy day", "day")]:
        lo, hi = model.conf_int().loc[term]
        print(f"   {label}: {model.params[term]:+.2f} crashes that week "
              f"(95% CI {lo:+.2f} to {hi:+.2f}, p = {model.pvalues[term]:.2g})")
    print(f"   R² = {model.rsquared:.3f}")

    with pd.ExcelWriter(HERE / "temperatureVsCrashes.xlsx", engine="openpyxl") as writer:
        out = weeks.round(2)
        out.index = out.index.date
        out.to_excel(writer, sheet_name="Weeks")
        pattern.round(2).to_excel(writer, sheet_name="Week of Year")

    plot(weeks, pattern, model)
    print("\nSaved temperatureVsCrashes.png and temperatureVsCrashes.xlsx")


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------

def plot(weeks, pattern, model):
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    ax = axes[0]
    ax.bar(pattern.index, pattern["z"], color=["#d62728" if z >= 1 else "#9e9e9e" for z in pattern["z"]])
    ax.set_ylabel("Crash z-score (bars)")
    ax.axhline(0, color="black", linewidth=0.6)
    ax2 = ax.twinx()
    ax2.plot(pattern.index, pattern["temp_mean_f"], color="#ff7f0e", linewidth=2.5)
    ax2.set_ylabel("Average temperature °F (line)", color="#ff7f0e")
    ticks = pattern.index[::4]
    ax.set_xticks(ticks, [pattern.loc[t, "typical_week"].split(" – ")[0] for t in ticks], rotation=45, ha="right")
    ax.set_title("Through the year: average weekly crash z-score and temperature (2021–2026)")

    ax = axes[1]
    sc = ax.scatter(pattern["temp_mean_f"], pattern["z"], c=pattern.index, cmap="twilight", s=45)
    fit = np.polyfit(pattern["temp_mean_f"], pattern["z"], 2)
    xs = np.linspace(pattern["temp_mean_f"].min(), pattern["temp_mean_f"].max(), 100)
    ax.plot(xs, np.polyval(fit, xs), color="black", linewidth=1.5, label="quadratic fit")
    for wk in pattern.nlargest(4, "z").index:
        ax.annotate(pattern.loc[wk, "typical_week"].split(" – ")[0],
                    (pattern.loc[wk, "temp_mean_f"], pattern.loc[wk, "z"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    fig.colorbar(sc, ax=ax, label="Week of year")
    ax.set_xlabel("Average temperature that week of the year (°F)")
    ax.set_ylabel("Average crash z-score")
    ax.set_title("Week of year: temperature vs. crash z-score (includes all seasonal effects)")
    ax.legend()

    ax = axes[2]
    ax.scatter(weeks["temp_vs_normal_f"], weeks["crashes_vs_normal"], s=14, color="#1f77b4", alpha=0.6)
    xs = np.linspace(weeks["temp_vs_normal_f"].min(), weeks["temp_vs_normal_f"].max(), 50)
    ax.plot(xs, model.params["Intercept"] + model.params["temp_vs_normal_f"] * xs, color="black")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Temperature vs. normal for that week of the year (°F)")
    ax.set_ylabel("Crashes vs. normal for that week")
    ax.set_title(f"Same week, different years: warmer-than-usual weeks "
                 f"({model.params['temp_vs_normal_f']:+.2f} crashes per °F, p = {model.pvalues['temp_vs_normal_f']:.2g})")

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(HERE / "temperatureVsCrashes.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
