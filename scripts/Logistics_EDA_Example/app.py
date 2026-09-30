"""Run after pipeline.py: streamlit run app.py"""
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Logistics EDA", layout="wide")
st.title("Logistics EDA — simulated data")
st.caption("Snapshot: 2 April 2026, 00:00 UTC. Filters change denominators; rankings are unadjusted.")
path = Path(__file__).resolve().parent / "outputs" / "orders_clean.csv"
if not path.exists():
    st.error("Run python pipeline.py first.")
    st.stop()

@st.cache_data
def load_data(path_string, modified_ns):
    return pd.read_csv(path_string, parse_dates=["released_at", "day"],
                       dtype={"on_time": "boolean", "matured": "boolean"})

d = load_data(str(path), path.stat().st_mtime_ns)
lanes = st.sidebar.multiselect("Lane", sorted(d.lane.unique()), default=sorted(d.lane.unique()))
carriers = st.sidebar.multiselect("Carrier", sorted(d.carrier.unique()), default=sorted(d.carrier.unique()))
dates = st.sidebar.date_input("Release dates (UTC)", value=(d.day.min().date(), d.day.max().date()))
if len(dates) != 2:
    st.info("Select a start and end date.")
    st.stop()
f = d[d.lane.isin(lanes) & d.carrier.isin(carriers) & d.day.dt.date.between(*dates)].copy()
if f.empty:
    st.info("No orders match the filters.")
    st.stop()
known = f[f.on_time.notna()].copy()
complete = f.dropna(subset=["lead_h"])
c1, c2, c3, c4 = st.columns(4)
c1.metric("Orders", f"{len(f):,}")
c2.metric("Known matured outcomes", f"{len(known):,}")
c3.metric("On-time rate", f"{known.on_time.mean():.1%}" if len(known) else "N/A")
c4.metric("Unknown matured outcomes", int((f.matured & f.on_time.isna()).sum()))
if not known.empty:
    daily = known.groupby("day").agg(n=("order_id", "size"), successes=("on_time", "sum")).reset_index()
    daily["on_time_pct"] = 100 * daily.successes / daily.n
    fig = px.line(daily, x="day", y="on_time_pct", markers=True, hover_data=["n"],
                  labels={"day":"Release date (UTC)", "on_time_pct":"On time (%)"}, template="plotly_white")
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig)
if not complete.empty:
    st.plotly_chart(px.ecdf(complete, x="lead_h", color="carrier", facet_col="lane",
                           labels={"lead_h":"Lead time (hours)"}, template="plotly_white"))
st.caption("Lead times use completed orders with valid timestamps; open orders are censored. Missing outcomes are not imputed.")
st.dataframe(f)
st.download_button("Download filtered orders", f.to_csv(index=False).encode("utf-8"),
                   file_name="filtered_orders.csv", mime="text/csv")

