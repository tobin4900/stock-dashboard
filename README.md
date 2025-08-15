# StockInsight Pro — Black & Gold

A sleek stock market dashboard with a left-panel company list and a rich, interactive price chart. It serves a FastAPI backend that fetches live data via `yfinance`, caches per-day results in SQLite, and exposes REST endpoints consumed by a static frontend (Chart.js).

## Live Demo
**▶ [https://stock-dashboard-7oxn.onrender.com](https://stock-dashboard-7oxn.onrender.com)**


## Features
- Left panel with at least 10 tickers (AAPL, GOOGL, MSFT, AMZN, TSLA, META, NVDA, JPM, NFLX, BTC-USD).
- Interactive Chart.js line chart with hover and click highlight.
- Key analytics: 52-week high/low, average volume, volatility, 7-day MA, and a simple next-day price prediction (linear trend).
- Daily caching to SQLite to reduce API hits and speed up responses.
- Fully responsive “black & gold” UI.

## Tech Stack
**Frontend:** HTML, CSS, JavaScript, Chart.js  
**Backend:** FastAPI, Uvicorn, SQLAlchemy, SQLite, Pandas, NumPy, yfinance  
**Hosting:** Render (Free tier)

## Run Locally
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv && source venv/bin/activate  # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
