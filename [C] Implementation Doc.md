# Market Analyzer — Implementation Doc

## What It Does
Interactive web app where a user enters a stock ticker or crypto name/symbol and gets:
- Current price + recent performance
- Technical analysis (RSI, Moving Average crossover, momentum)
- Fundamental analysis (stocks only: P/E, EPS growth, 52w position)
- Market cap rank signal (crypto only)
- Composite score (0–100) → Buy / Hold / Sell verdict
- 90-day price chart

## Scoring System

Each of 4 signals contributes 0–25 points → total 0–100:

| Score | Verdict |
|---|---|
| 70–100 | BUY |
| 45–69 | HOLD |
| 0–44 | SELL |

### Signals
1. **RSI (14-period)** — oversold/overbought momentum indicator
2. **MA Crossover** — price vs 50MA and 200MA (golden/death cross)
3. **Momentum** — 7d + 30d price change average
4. **Fundamentals** (stocks) / **Market Cap Rank** (crypto)

## Data Sources

| Asset | Source | Cost |
|---|---|---|
| US Stocks | Yahoo Finance (`yfinance`) | Free |
| Indian Stocks (NSE/BSE) | Yahoo Finance (`.NS` / `.BO` suffix) | Free |
| Crypto | CoinGecko API (v3) | Free, no key needed |

## Files

```
app.py              — Streamlit UI
analyzer.py         — analysis engine (all scoring logic)
requirements.txt    — Python dependencies
.streamlit/
  config.toml       — dark theme
```

## How to Run Locally

```bash
cd "02 Projects/Crypto and Share Analysis"
pip3 install -r requirements.txt
/Users/mukesh/Library/Python/3.10/bin/streamlit run app.py
```

## How to Deploy to Streamlit Cloud

1. Push this folder to a GitHub repo (e.g. `muralmu/market-analyzer`)
2. Go to share.streamlit.io → "New app"
3. Connect your GitHub, select the repo, set `app.py` as the main file
4. Click Deploy — done. Free hosting, always-on.

## Ticker Format Guide

| Asset | Format | Example |
|---|---|---|
| US Stock | Plain ticker | `AAPL`, `TSLA`, `MSFT` |
| Indian Stock (NSE) | Ticker + `.NS` | `RELIANCE.NS`, `TCS.NS` |
| Indian Stock (BSE) | Ticker + `.BO` | `RELIANCE.BO` |
| Crypto | Symbol or name | `BTC`, `ETH`, `bitcoin` |

## Known Limitations / Future Work
- Crypto 200MA not available (only 90 days of OHLC from free CoinGecko tier)
- No news/sentiment analysis yet (would need a news API)
- No portfolio mode (analyse multiple at once)
- No historical verdict tracking
- CoinGecko free tier rate limits: ~10–30 req/min
