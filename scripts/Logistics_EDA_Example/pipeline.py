"""Reproducible logistics EDA. Run: python pipeline.py

Core: numpy, pandas, scipy, matplotlib, seaborn.
Optional: statsmodels for adjusted inference; plotly for offline interactive HTML.
"""

# %% 1. Imports
from pathlib import Path
import json
import platform
import sqlite3
import importlib.metadata
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()  # Technicality: to also allow for execution in a Notebook
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42
SNAPSHOT = pd.Timestamp("2026-04-02 00:00:00", tz="UTC")  # WHEN the data is analyzed; all orders released before this time are included, but only those completed by this time have known outcomes.
sns.set_theme(style="whitegrid", palette="colorblind", context="notebook")

# %% 2. Simulate source systems, including realistic quality problems
def simulate():
    rng = np.random.default_rng(SEED)
    records = []
    order_id = 0
    # IGNORE: this code simply generates a synthetic dataset for demonstration purposes. It is not a model of any real logistics system.
    for day in pd.date_range("2026-01-01", periods=90, tz="UTC"):
        volume = int(rng.poisson(65 + 15 * (day.dayofweek == 0)))
        shared_day_delay = rng.normal(0, 2)
        for _ in range(volume):
            order_id += 1
            lane = rng.choice(["Local", "Regional"])
            carrier = "A" if rng.random() < (0.85 if lane == "Regional" else 0.15) else "B"
            distance = max(10, rng.normal(320 if lane == "Regional" else 55, 12))
            units = int(rng.integers(1, 9))
            released = day + pd.Timedelta(minutes=int(rng.integers(0, 1440)))
            lead_h = max(1, (49 if lane == "Regional" else 19)
                         + (0 if carrier == "A" else 3)
                         + 0.018 * (distance - (320 if lane == "Regional" else 55))
                         + 0.15 * units + 0.07 * (volume - 65)
                         + shared_day_delay + rng.normal(0, 3))
            if rng.random() < 0.015:
                lead_h += rng.uniform(15, 35)
            delivered = released + pd.Timedelta(hours=lead_h)
            status = "delivered" if delivered <= SNAPSHOT else "in_transit"
            records.append([f"O{order_id:06d}", released, lane, carrier,
                            distance, units, delivered if status == "delivered" else pd.NaT, status])
    raw = pd.DataFrame(records, columns=["order_id", "released_at", "lane", "carrier",
                                        "distance_km", "units", "delivered_at", "status"])
    # Artificially mess up data to illustrate cleaning and validation
    done = raw.index[raw.status.eq("delivered")].to_numpy()
    bad = rng.choice(done, 37, replace=False)
    raw.loc[bad[:25], "delivered_at"] = pd.NaT  # like NaN but for dates and times
    raw.loc[bad[25:], "delivered_at"] = raw.loc[bad[25:], "released_at"] - pd.Timedelta(hours=2)
    raw.loc[rng.choice(raw.index, 80, replace=False), "distance_km"] = np.nan
    raw.loc[rng.choice(raw.index, 60, replace=False), "carrier"] = " a "
    original = pd.Series([r[3] for r in records], index=raw.index)
    changed = raw.carrier.eq(" a ")
    raw.loc[changed, "carrier"] = original[changed].str.lower().map(lambda x: f" {x} ")
    raw = pd.concat([raw, raw.sample(30, random_state=SEED)], ignore_index=True)
    # make data snapshot available as .CSV
    raw.to_csv(OUT / "orders_raw.csv", index=False)
    # make lane reference available as SQLite table
    lanes = pd.DataFrame({"lane": ["Local", "Regional"], "sla_h": [24, 48]})  # SLA = Service Level Agreement, the maximum time allowed for delivery
    with sqlite3.connect(OUT / "reference.sqlite") as con:
        lanes.to_sql("lanes", con, if_exists="replace", index=False)

simulate()

# %% 3. Read, profile, validate grain, clean, and join
raw = pd.read_csv(OUT / "orders_raw.csv", dtype={"order_id": "string"})
print("\n" + "-" * 80)
print(f"Raw data: {len(raw):,} rows, {raw.columns.size:,} columns")
raw.info(verbose=True)
print(raw.describe())
print(raw.head(10))
print("-" * 80)
audit = {"raw_rows": len(raw), "exact_duplicate_rows": int(raw.duplicated().sum())}  # count duplicate data
raw.isna().mean().rename("raw_missing_fraction").to_csv(OUT / "raw_missingness.csv")  # count missing data for each column and store as new column in new CSV
d = raw.drop_duplicates().copy()

# Remove leading/trailing whitespace and standardize carrier codes to uppercase
d["carrier"] = d.carrier.str.strip().str.upper()
# Convert columns to appropriate types, coerce errors to NaT or NaN
for col in ["released_at", "delivered_at"]:
    d[col] = pd.to_datetime(d[col], utc=True, errors="coerce", format="mixed")
for col in ["distance_km", "units"]:
    d[col] = pd.to_numeric(d[col], errors="coerce")
print("\n" + "-" * 80)
print(f"Cleaned data: {len(d):,} rows, {d.columns.size:,} columns")
d.info(verbose=True)
print("-" * 80)

# Data validation: 'assert' statements will raise an AssertionError if the condition is not met
assert d.released_at.notna().all()
assert d.carrier.isin(["A", "B"]).all()
assert d.status.isin(["delivered", "in_transit"]).all()
assert d.units.ge(1).all()

# Audit: keep track of data quality issues and store in audit dictionary
bad_time = d.delivered_at.lt(d.released_at) | d.delivered_at.gt(SNAPSHOT)  # Find data that does not make sense: e.g. delivery before release, or delivered after snapshot
audit["invalid_delivery_times"] = int(bad_time.sum())
d["invalid_delivery_time"] = bad_time
# Fix it with NaTs and NaNs
d.loc[bad_time, "delivered_at"] = pd.NaT
d.loc[d.distance_km.le(0), "distance_km"] = np.nan
d["distance_missing"] = d.distance_km.isna()
audit["missing_distance"] = int(d.distance_missing.sum())
print("\n" + "-" * 80)
print("Audit summary:")
for k in audit:
    print(f"{k}: {audit[k]}")
print("-" * 80)

# Join with lane reference table to get SLA hours for each lane
with sqlite3.connect(OUT / "reference.sqlite") as con:
    lanes = pd.read_sql_query("SELECT lane, sla_h FROM lanes", con)
d = d.merge(lanes, on="lane", how="left", validate="many_to_one", indicator=True)  # 'validate' checks that each lane in d matches exactly one lane in lanes; 'indicator' adds a column showing whether the merge was successful
assert d["_merge"].eq("both").all(), "Unknown lane"  # checks whether all rows in d matched a lane in lanes; if not, raises an AssertionError
d = d.drop(columns="_merge")
print("\n" + "-" * 80)
print(f"Merged data: {len(d):,} rows, {d.columns.size:,} columns")
d.info(verbose=True)
print("-" * 80)

# %% 4. Define the additional, derived data you need for your EDA: metrics, eligibility, missing outcomes, and time features; essentially everything needed for our KPIs
d["due_at"] = d.released_at + pd.to_timedelta(d.sla_h, unit="h")
d["lead_h"] = (d.delivered_at - d.released_at).dt.total_seconds() / 3600  # lead time in hours
d["day"] = d.released_at.dt.floor("D")
d["weekday"] = d.released_at.dt.day_name()
d["day_index"] = (d.day - d.day.min()).dt.days
d["daily_orders"] = d.groupby("day").order_id.transform("size")  # groupby("day") groups rows by release day, and transform("size") counts rows in each group while returning a result aligned with the original rows. So every order released on the same day gets the same count.

# Distinguish between orders that are past due date (matured) and those that are not yet due. Only matured orders can be evaluated for on-time performance.
d["matured"] = d.due_at.le(SNAPSHOT)  # "matured" orders are past due date
known_delivery = d.status.eq("delivered") & d.delivered_at.notna()
known_open_late = d.status.eq("in_transit") & d.matured
d["on_time"] = pd.Series(pd.NA, index=d.index, dtype="boolean")
eligible_known = d.matured & known_delivery
d.loc[eligible_known, "on_time"] = d.loc[eligible_known, "delivered_at"].le(d.loc[eligible_known, "due_at"])
d.loc[known_open_late, "on_time"] = False
d["late"] = ~d.on_time  # "~" is the NOT logical operator: late orders are those that are NOT on time :)
eligible = d[d.on_time.notna()].copy()
eligible["on_time"] = eligible.on_time.astype(bool)
eligible["late"] = eligible.late.astype(bool)
completed = d[d.lead_h.notna()].copy()
print(f"Final data for EDA: {len(completed):,} rows, {completed.columns.size:,} columns")
completed.info(verbose=True)
print(completed.describe())
print(completed.head(10))
print("-" * 80)

audit.update(unique_orders=len(d), matured_orders=int(d.matured.sum()),
             known_matured_outcomes=len(eligible),
             unknown_matured_outcomes=int((d.matured & d.on_time.isna()).sum()),
             not_yet_due=int((~d.matured).sum()),
             valid_completed_lead_times=len(completed))
pd.Series(audit, name="count").to_csv(OUT / "quality_audit.csv")
d.to_csv(OUT / "orders_clean.csv", index=False)
print("Updated Audit:")
for k in audit:
    print(f"{k}: {audit[k]}")
print("-" * 80)

# %% 5. Summary statistics and correct denominators
# group completed orders by carrier and lane, and compute some statistics for lead time (size=number of orders)
summary = completed.groupby(["carrier", "lane"]).lead_h.agg(
    n="size", mean_h="mean", median_h="median", p90_h=lambda x: x.quantile(.90), sd_h="std")
summary.to_csv(OUT / "lead_time_summary.csv")
# group eligible orders by carrier and lane, and compute on-time performance metrics
kpi = eligible.groupby(["carrier", "lane"]).agg(
    n=("order_id", "size"), on_time=("on_time", "sum"), late=("late", "sum"))
kpi["on_time_rate"] = kpi.on_time / kpi.n
kpi.to_csv(OUT / "service_by_lane.csv")

overall_rate = eligible.on_time.mean()
# Sensitivity bounds: "How low or high could the on-time rate be among orders whose SLA deadline has passed?”
# • The lower bound assumes every unknown order was late.
# • The upper bound assumes every unknown order was on time.
matured_n = int(d.matured.sum())
unknown_n = audit["unknown_matured_outcomes"]
success_n = int(eligible.on_time.sum())
bounds = (success_n / matured_n, (success_n + unknown_n) / matured_n)

# %% 6. Daily trends and uncertainty: resample days, not individual orders
daily = eligible.groupby("day").agg(n=("order_id", "size"), success=("on_time", "sum"))
daily = daily.reindex(pd.date_range(d.day.min(), d.day.max(), freq="D"))
daily["rate"] = daily.success / daily.n
daily["rolling_7d_rate"] = daily.success.rolling(7, min_periods=7).sum() / daily.n.rolling(7, min_periods=7).sum()
daily.index.name = "day"
daily.to_csv(OUT / "daily_service.csv")
valid_days = daily.dropna(subset=["n", "success"])
rng = np.random.default_rng(SEED + 1)
indices = rng.integers(0, len(valid_days), size=(2000, len(valid_days)))
boot_rates = valid_days.success.to_numpy()[indices].sum(axis=1) / valid_days.n.to_numpy()[indices].sum(axis=1)
ci = np.quantile(boot_rates, [.025, .975])
# Days are the resampling units. Serial dependence across days would require
# moving blocks or a different dependence model. This is an illustrative CI
# for a broader process, not uncertainty about the observed finite census.

# %% 7. Distributions and outliers: flag for investigation, retain valid tails
q1 = completed.groupby("lane").lead_h.transform(lambda x: x.quantile(.25))
q3 = completed.groupby("lane").lead_h.transform(lambda x: x.quantile(.75))
completed["iqr_flag"] = (completed.lead_h < q1 - 1.5 * (q3-q1)) | (completed.lead_h > q3 + 1.5 * (q3-q1))
completed.loc[completed.iqr_flag].to_csv(OUT / "lead_time_flags.csv", index=False)

# %% 8. Relationships and confounding: raw versus lane-standardized rates
raw_carrier = eligible.groupby("carrier").on_time.mean()
stratified = eligible.groupby(["carrier", "lane"]).on_time.mean().unstack()
weights = eligible.lane.value_counts(normalize=True).reindex(stratified.columns)
assert stratified.notna().all().all(), "Standardization requires overlap"
standardized = stratified.mul(weights, axis=1).sum(axis=1)
comparison = pd.DataFrame({"raw_on_time_rate": raw_carrier,
                           "common_lane_mix_on_time_rate": standardized})
comparison.to_csv(OUT / "carrier_comparison.csv")
pair = completed[["distance_km", "lead_h"]].dropna()
pearson = stats.pearsonr(pair.distance_km, pair.lead_h).statistic
spearman = stats.spearmanr(pair.distance_km, pair.lead_h).statistic
within_lane = completed.groupby("lane")[["distance_km", "lead_h"]].corr(method="spearman")
within_lane.to_csv(OUT / "within_lane_correlations.csv")
# P-values from a row-wise correlation test would ignore shared daily shocks.

# %% 9. Optional adjusted regression with day-clustered standard errors
optional = {}
try:
    import statsmodels.formula.api as smf
    model_data = completed.dropna(subset=["distance_km"]).copy()
    model = smf.ols("lead_h ~ C(carrier) + C(lane) + distance_km + units + daily_orders + C(weekday) + day_index",
                    data=model_data).fit(cov_type="cluster", cov_kwds={"groups": model_data.day})
    (OUT / "adjusted_model.txt").write_text(model.summary().as_text(), encoding="utf-8")
    optional["statsmodels"] = "adjusted_model.txt written"
except ImportError:
    optional["statsmodels"] = "not installed; adjusted regression skipped"
# Adjustment is descriptive. It does not establish causation; unmeasured
# confounding and completed-order selection remain possible.

# %% 10. Purposeful static plots: comparisons, distributions, trends, relationships
fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
sns.ecdfplot(data=completed, x="lead_h", hue="lane", ax=axes[0, 0])
axes[0, 0].set(title="Completed orders: lead-time distribution", xlabel="Lead time (hours)", ylabel="Cumulative share")
sns.boxplot(data=completed, x="lane", y="lead_h", hue="carrier", ax=axes[0, 1], showfliers=False)
axes[0, 1].set(title="Carrier comparisons within each lane", xlabel="Lane", ylabel="Lead time (hours)")
# showfliers=False hides points visually only; data and summaries retain them.
axes[1, 0].plot(daily.index, daily.rate * 100, alpha=.35, label="Daily")
axes[1, 0].plot(daily.index, daily.rolling_7d_rate * 100, label="7-day pooled rate")
axes[1, 0].set(title="Service by release cohort (known, matured)", xlabel="Release date (UTC)", ylabel="On-time deliveries (%)", ylim=(0, 100))
axes[1, 0].legend()
date_locator = mdates.AutoDateLocator(minticks=4, maxticks=6)
axes[1, 0].xaxis.set_major_locator(date_locator)
axes[1, 0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(date_locator))
comparison.mul(100).rename(columns={"raw_on_time_rate": "Raw", "common_lane_mix_on_time_rate": "Common lane mix"}).plot.bar(ax=axes[1, 1], rot=0)
axes[1, 1].set(title="Carrier ranking depends on work mix", xlabel="Carrier", ylabel="On-time deliveries (%)", ylim=(0, 100))
sns.despine(fig=fig)
fig.savefig(OUT / "overview.png", dpi=170)
plt.close(fig)
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
sns.scatterplot(data=completed.sample(min(1500, len(completed)), random_state=SEED), x="distance_km", y="lead_h", hue="lane", alpha=.35, s=15, ax=ax)
ax.set(title="Distance and lead time: route mix is a confounder", xlabel="Distance (km)", ylabel="Lead time (hours)")
fig.savefig(OUT / "relationship.png", dpi=170)
plt.close(fig)

# %% 11. Reporting, reproducibility, and optional offline interactive output
metrics = {"simulated": True, "snapshot_utc": str(SNAPSHOT), "seed": SEED,
           "unique_orders": len(d), "known_matured_orders": len(eligible),
           "on_time_rate": float(overall_rate), "day_bootstrap_95_interval": ci.tolist(),
           "missing_outcome_bounds": list(bounds), "pearson_distance_lead": float(pearson),
           "spearman_distance_lead": float(spearman), "optional_modules": optional}
html = """<!doctype html><html><head><meta charset='utf-8'><title>Simulated logistics EDA</title>
<style>body{font:17px system-ui;max-width:1150px;margin:40px auto;color:#172b42}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccd6df}img{max-width:100%}</style></head><body>
<h1>Simulated logistics EDA</h1><p>One row per order. Snapshot: 2 April 2026, 00:00 UTC. All data are simulated.</p>"""
html += f"<p>Known matured orders: {len(eligible):,}. On-time rate: {overall_rate:.1%}. Illustrative day-bootstrap 95% interval: {ci[0]:.1%}–{ci[1]:.1%}.</p>"
html += f"<p>Missing-outcome sensitivity bounds: {bounds[0]:.1%}–{bounds[1]:.1%}. Not-yet-due orders are excluded from service-rate denominators.</p>"
html += "<h2>Data quality</h2>" + pd.Series(audit, name="Count").to_frame().to_html()
html += "<h2>Raw versus common lane mix</h2>" + comparison.to_html(float_format=lambda x: f"{x:.1%}")
html += "<p>Carrier A receives more regional work. Compare within lanes or at a common lane mix before drawing performance conclusions. Standardization alone does not prove causality.</p>"
html += "<img src='overview.png' alt='Lead-time distributions, within-lane comparisons, daily trends and standardized carrier performance'>"
html += "<h2>Lead times for completed orders</h2>" + summary.to_html(float_format=lambda x: f"{x:.2f}")
html += "<p>Lead-time summaries exclude open and missing-timestamp orders and can therefore underrepresent slow orders. Keep true long delays; investigate flags separately. Consider survival analysis for censored transit times.</p>"
try:
    import plotly.express as px
    interactive = px.scatter(completed, x="distance_km", y="lead_h", color="lane", symbol="carrier",
                             hover_data=["order_id"], opacity=.35, template="plotly_white",
                             labels={"distance_km":"Distance (km)", "lead_h":"Lead time (hours)"},
                             title="Explore simulated order-level relationships")
    html += interactive.to_html(full_html=False, include_plotlyjs=True)
    optional["plotly"] = "interactive chart embedded; no CDN needed"
except ImportError:
    optional["plotly"] = "not installed; static report written"
html += "<h2>Decision and next experiment</h2><p>Review late orders by lane and release cohort; audit unknown outcomes; compare carriers on comparable assignments. Test operational changes prospectively with randomized or defensible quasi-experimental allocation. These synthetic results are teaching examples.</p></body></html>"
(OUT / "report.html").write_text(html, encoding="utf-8")
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
versions = {"python": platform.python_version()}
for package in ["numpy", "pandas", "scipy", "matplotlib", "seaborn", "statsmodels", "plotly", "streamlit"]:
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = "not installed"
(OUT / "versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))
print(comparison.round(4))
print("Outputs:", OUT)
