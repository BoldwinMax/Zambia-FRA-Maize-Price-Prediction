"""
Zambia FRA Maize Floor Price — Interactive Dashboard
====================================================
A Streamlit app that loads the same data as the analysis notebook (live World
Bank + FRED, embedded fallback), trains the Ridge model, and lets the user
explore how the 2026 prediction responds to different assumptions.

Run:
    pip install streamlit pandas numpy scikit-learn matplotlib requests
    streamlit run dashboard.py
"""

import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error

# --------------------------------------------------------------------------- #
#  CONFIG & DATA
# --------------------------------------------------------------------------- #
ACCENT, RED, GREY, GREEN = "#1f77b4", "#d62728", "#9e9e9e", "#2ca02c"
ELECTION_YEARS = {2006, 2011, 2016, 2021, 2026}
BASE_FEATURES = ["prev_price", "cpi_inflation_avg", "zmw_usd_avg",
                 "maize_harvest_mt", "election_year"]

_EMBEDDED = """year,fra_price,cpi_inflation_avg,zmw_usd_avg,maize_harvest_mt,global_maize_usd_mt,fertiliser_dap_usd_mt
2002,40,22.23,4.40,602000,99.3,168
2003,30,21.40,4.73,1158000,105.3,188
2004,36,17.97,4.78,1214000,111.8,222
2005,36,18.32,4.46,866000,98.7,247
2006,38,9.02,3.60,1424000,121.9,261
2007,38,10.66,4.00,1366000,163.7,433
2008,55,12.45,3.75,1211000,223.1,967
2009,65,13.40,5.05,1890000,165.5,323
2010,65,8.50,4.80,2800000,185.9,501
2011,65,6.43,4.86,3020000,291.7,619
2012,65,6.58,5.14,2852000,298.4,540
2013,65,6.98,5.40,2533000,259.4,445
2014,70,7.81,6.15,3350000,192.9,472
2015,75,10.11,8.63,2618000,169.8,459
2016,85,17.87,10.31,2873000,159.2,342
2017,60,6.58,9.52,2900000,154.5,344
2018,70,7.49,10.46,2395000,164.4,395
2019,110,9.15,12.89,2004000,170.1,306
2020,110,15.73,18.30,3387000,165.5,312
2021,150,22.02,20.02,3600000,259.5,601
2022,180,10.99,16.93,2700000,318.8,772
2023,280,10.88,20.17,3200000,252.7,537
2024,330,14.99,25.80,1511143,190.6,480
2025,340,14.99,27.00,3655646,195.0,600
2026,,14.99,28.00,4937605,195.0,600"""


def _fetch_wb(indicator):
    try:
        url = f"https://api.worldbank.org/v2/country/ZMB/indicator/{indicator}"
        r = requests.get(url, params={"format": "json", "date": "2002:2026",
                                      "per_page": 100}, timeout=20)
        r.raise_for_status()
        p = r.json()
        if len(p) < 2 or p[1] is None:
            return {}
        return {int(d["date"]): round(float(d["value"]), 2)
                for d in p[1] if d["value"] is not None}
    except Exception:
        return {}


def _fetch_fred(series_id):
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        raw = pd.read_csv(url)
        dc, vc = raw.columns[0], raw.columns[1]
        raw["year"] = pd.to_datetime(raw[dc]).dt.year
        raw[vc] = pd.to_numeric(raw[vc], errors="coerce")
        return {int(y): round(float(v), 1)
                for y, v in zip(raw["year"], raw[vc]) if pd.notna(v)}
    except Exception:
        return {}


@st.cache_data(show_spinner="Loading data (World Bank + FRED, with fallback)...")
def load_data():
    """Embedded baseline, overridden by live values where available."""
    df = pd.read_csv(io.StringIO(_EMBEDDED))
    sources = {"inflation": "embedded", "exchange rate": "embedded",
               "global maize": "embedded"}
    for col, code, key in [("cpi_inflation_avg", "FP.CPI.TOTL.ZG", "inflation"),
                           ("zmw_usd_avg", "PA.NUS.FCRF", "exchange rate")]:
        live = _fetch_wb(code)
        if live:
            for yr, val in live.items():
                if yr in df["year"].values:
                    df.loc[df["year"] == yr, col] = val
            sources[key] = f"World Bank (live, {min(live)}-{max(live)})"
    maize = _fetch_fred("PMAIZMTUSDA")
    if maize:
        for yr, val in maize.items():
            if yr in df["year"].values:
                df.loc[df["year"] == yr, "global_maize_usd_mt"] = val
        sources["global maize"] = f"FRED (live, {min(maize)}-{max(maize)})"

    df["election_year"] = df["year"].isin(ELECTION_YEARS).astype(int)
    df["prev_price"] = df["fra_price"].shift(1)
    return df, sources


@st.cache_resource(show_spinner="Training model...")
def train_model(df):
    hist = df[df["fra_price"].notna()].dropna(subset=["prev_price"])
    X, y = hist[BASE_FEATURES].values, hist["fra_price"].values
    scaler = StandardScaler().fit(X)
    model = Ridge(alpha=1.0).fit(scaler.transform(X), y)
    # leave-one-out MAE for the uncertainty band
    preds, actuals = [], []
    for tr, te in LeaveOneOut().split(X):
        s = StandardScaler().fit(X[tr])
        m = Ridge(alpha=1.0).fit(s.transform(X[tr]), y[tr])
        preds.append(m.predict(s.transform(X[te]))[0]); actuals.append(y[te][0])
    mae = mean_absolute_error(actuals, preds)
    return model, scaler, mae


# --------------------------------------------------------------------------- #
#  APP
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Zambia FRA Maize Price Predictor",
                   page_icon="🌽", layout="wide")

st.title("🌽 Zambia FRA Maize Floor Price - 2026 Predictor")
st.caption("Predicting the Food Reserve Agency's annual maize floor price "
           "(Kwacha per 50 kg bag) from macroeconomic and commodity data.")

df, sources = load_data()
model, scaler, mae = train_model(df)

hist = df[df["fra_price"].notna()].copy()
last_price = hist["fra_price"].iloc[-1]                 # 2025 price
defaults = df[df["year"] == 2026].iloc[0]

# ---- Sidebar: 2026 assumptions ----
st.sidebar.header("2026 assumptions")
st.sidebar.caption("Adjust the drivers and watch the prediction update.")
fx = st.sidebar.slider("Exchange rate (ZMW per US$)", 10.0, 40.0,
                       float(defaults["zmw_usd_avg"]), 0.5)
harvest_m = st.sidebar.slider("Maize harvest (million tonnes)", 1.0, 5.0,
                              float(defaults["maize_harvest_mt"]) / 1e6, 0.1)
infl = st.sidebar.slider("Inflation (annual %)", 5.0, 25.0,
                         float(defaults["cpi_inflation_avg"]), 0.5)
st.sidebar.markdown("*2026 is an election year (flag fixed on).*")

# ---- Predict with the chosen assumptions ----
X_2026 = pd.DataFrame([{
    "prev_price": last_price,
    "cpi_inflation_avg": infl,
    "zmw_usd_avg": fx,
    "maize_harvest_mt": harvest_m * 1e6,
    "election_year": 1,
}])[BASE_FEATURES].values
pred = float(model.predict(scaler.transform(X_2026))[0])
delta = pred - last_price

# ---- Headline metrics ----
c1, c2, c3 = st.columns(3)
c1.metric("Predicted 2026 price", f"K{pred:,.0f}", f"{delta:+.0f} vs 2025")
c2.metric("Indicative range", f"K{pred-mae:,.0f} – K{pred+mae:,.0f}")
c3.metric("2025 price (baseline)", f"K{last_price:,.0f}")

# ---- History + prediction chart ----
left, right = st.columns([3, 2])
with left:
    st.subheader("History & prediction")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hist["year"], hist["fra_price"], "o-", color=ACCENT, lw=2,
            label="Actual FRA price")
    ax.scatter([2026], [pred], color=RED, s=90, zorder=5,
               label=f"2026 prediction: K{pred:.0f}")
    ax.errorbar([2026], [pred], yerr=mae, color=RED, capsize=6)
    for i, yr in enumerate(sorted(ELECTION_YEARS)):
        ax.axvspan(yr - 0.4, yr + 0.4, color=GREY, alpha=0.30,
                   label="Election year" if i == 0 else None)
    ax.set_xlabel("Year"); ax.set_ylabel("Kwacha per 50 kg bag")
    ax.grid(alpha=0.3); ax.legend()
    st.pyplot(fig)

with right:
    st.subheader("What moves the price")
    coefs = pd.Series(model.coef_, index=BASE_FEATURES).sort_values()
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    ax2.barh(coefs.index, coefs.values,
             color=[RED if c < 0 else ACCENT for c in coefs.values])
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("Standardised coefficient (blue raises, red lowers)")
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

# ---- Data + provenance ----
with st.expander("View the data and sources"):
    st.write("**Live data sources this session:**")
    for k, v in sources.items():
        st.write(f"- {k.title()}: {v}")
    st.write("- FRA price, maize harvest, fertiliser: embedded "
             "(no reliable public API)")
    show = df.copy()
    show["maize_harvest_mt"] = show["maize_harvest_mt"].map(lambda v: f"{v:,.0f}")
    st.dataframe(show, use_container_width=True, hide_index=True)

st.caption("Model: Ridge regression (5 features), leave-one-out cross-validated. "
           "Small annual sample — treat the prediction as indicative, not exact. "
           "The FRA price is ultimately a policy decision.")
