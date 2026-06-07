# 🌽 Zambia FRA Maize Floor Price — Predicting the 2026 Announcement

Predicting the **Food Reserve Agency (FRA)** maize floor price (Kwacha per 50 kg
bag) ahead of its annual announcement, using Zambian and global economic data
from 2002–2025.

**Headline result:** the model points to a 2026 floor price in the **mid-K350s**
(indicative range ~K341–K371) — a modest, single-digit-percentage rise on the
2025 price of K340.

> ⚠️ A small-sample, indicative model — see the caveats below. The FRA price is
> ultimately a policy decision.

## 🔗 Live dashboard

**[➡️ Try the interactive dashboard](https://YOUR-APP-NAME.streamlit.app)**

Move the sliders for the 2026 exchange rate, harvest, and inflation, and watch
the predicted price update live.

## 📂 What's in this repo

| File | What it is |
|------|------------|
| `Zambia-FRA-Maize-Price-Prediction.ipynb` | The full analysis — EDA, modelling, and verdict (renders directly on GitHub) |
| `dashboard.py` | Interactive Streamlit dashboard (the "what-if" tool) |
| `requirements.txt` | Python dependencies |

## 📊 Data sources

| Feature | Source | Access |
|---------|--------|--------|
| Inflation (annual %) | World Bank API (`FP.CPI.TOTL.ZG`) | live |
| Exchange rate (ZMW/US$) | World Bank API (`PA.NUS.FCRF`) | live |
| Global maize price ($/t) | FRED API (`PMAIZMTUSDA`) | live |
| FRA floor price | FRA announcements / academic records | embedded |
| Maize harvest | ZamStats Crop Forecast Survey | embedded |
| Fertiliser (DAP) | World Bank Pink Sheet | embedded |

Live data is fetched at runtime, with embedded values as a fallback so the
project always runs offline. All prices are in today's rebased kwacha (pre-2013
figures divided by 1,000).

## 🔬 Method

A **Ridge regression** (regularised linear regression) on five features —
previous year's price, exchange rate, inflation, maize harvest, and an
election-year flag — evaluated with **leave-one-out cross-validation** (≈24
annual observations). The previous year's price and the exchange rate dominate;
the FRA effectively anchors on last year's figure and adjusts upward.

## ▶️ Run it locally

```bash
pip install -r requirements.txt

# the dashboard
streamlit run dashboard.py

# or open the notebook
jupyter notebook Zambia-FRA-Maize-Price-Prediction.ipynb
```

## ⚠️ Caveats

- Small annual sample; the high R² partly reflects the price's strong
  year-to-year autocorrelation.
- The price is a political decision — sudden cuts or jumps can't be modelled.
- Recent harvest (2017–2025) and macro data are from official sources; some
  earlier years are best-available estimates.
- Treat the forecast as **directional**, not exact.

## 👤 Author

**Boldwin Mweemba** — Data & Financial Analyst, Lusaka, Zambia
[LinkedIn](https://www.linkedin.com/in/boldwin-mweemba) ·
[GitHub](https://github.com/BoldwinMax)
