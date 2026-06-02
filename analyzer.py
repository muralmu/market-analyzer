"""
Analysis engine — Financial Modeling Prep (stocks) + CoinGecko (crypto).
No Yahoo Finance dependency.
"""

import requests
import pandas as pd
import numpy as np
import time
import os

# ---------------------------------------------------------------------------
# API key — from Streamlit secrets or env var
# ---------------------------------------------------------------------------

def _get_fmp_key() -> str:
    try:
        import streamlit as st
        return st.secrets["FMP_API_KEY"]
    except Exception:
        return os.environ.get("FMP_API_KEY", "")


FMP_BASE  = "https://financialmodelingprep.com/api/v3"   # legacy (not used)
FMP_STABLE = "https://financialmodelingprep.com/stable"   # new free endpoints
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
})

# ---------------------------------------------------------------------------
# Sector PE benchmarks
# ---------------------------------------------------------------------------

SECTOR_PE = {
    "Technology": 28, "Healthcare": 22, "Financial Services": 14,
    "Consumer Cyclical": 20, "Consumer Defensive": 18, "Industrials": 18,
    "Energy": 12, "Basic Materials": 14, "Real Estate": 30,
    "Communication Services": 18, "Utilities": 16,
}

# ---------------------------------------------------------------------------
# FMP helpers
# ---------------------------------------------------------------------------

def _fmp_stable(path: str, params: dict = {}) -> list | dict | None:
    """Call the new FMP /stable/ endpoints."""
    key = _get_fmp_key()
    if not key:
        return None
    try:
        r = _SESSION.get(
            f"{FMP_STABLE}/{path}",
            params={"apikey": key, **params},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and data.get("Error Message"):
            return None
        return data
    except Exception:
        return None


def _fmp_history(ticker: str) -> pd.Series | None:
    """Returns daily close prices as a Series, last 365 days.
    Tries FMP first, falls back to yfinance for tickers not in FMP free tier."""
    data = _fmp_stable("historical-price-eod/full", {"symbol": ticker, "limit": 365})
    if data and isinstance(data, list) and len(data) >= 20:
        data_sorted = sorted(data, key=lambda x: x["date"])
        closes = pd.Series(
            [d["close"] for d in data_sorted],
            index=pd.to_datetime([d["date"] for d in data_sorted]),
            dtype=float,
        ).dropna()
        if len(closes) >= 20:
            return closes

    # Fallback: yfinance (works for less popular tickers not in FMP free tier)
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
        if not hist.empty and len(hist) >= 20:
            closes = hist["Close"].dropna()
            if isinstance(closes, pd.DataFrame):
                closes = closes.iloc[:, 0]
            return closes
    except Exception:
        pass
    return None


def _fmp_quote(ticker: str) -> dict:
    data = _fmp_stable("quote", {"symbol": ticker})
    if data and isinstance(data, list) and data:
        return data[0]
    return {}


def _fmp_profile(ticker: str) -> dict:
    data = _fmp_stable("profile", {"symbol": ticker})
    if data and isinstance(data, list) and data:
        return data[0]
    elif isinstance(data, dict):
        return data
    return {}


def _fmp_income(ticker: str, period: str = "quarter", limit: int = 8) -> list:
    data = _fmp_stable("income-statement", {"symbol": ticker, "period": period, "limit": limit})
    return data if isinstance(data, list) else []


def _fmp_ratios(ticker: str) -> dict:
    data = _fmp_stable("ratios", {"symbol": ticker, "limit": 1})
    if data and isinstance(data, list) and data:
        return data[0]
    return {}


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def _rsi(prices: pd.Series, period: int = 14) -> float:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    val = (100 - (100 / (1 + gain / loss))).iloc[-1]
    return round(float(val), 1) if pd.notna(val) else 50.0


def _ma(prices: pd.Series, window: int) -> float:
    val = prices.rolling(window).mean().iloc[-1]
    return round(float(val), 4) if pd.notna(val) else float(prices.iloc[-1])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_rsi(rsi):
    if rsi < 30:   return 20, "Oversold (bullish)"
    elif rsi < 45: return 17, "Approaching oversold"
    elif rsi < 55: return 12, "Neutral"
    elif rsi < 70: return 8,  "Approaching overbought"
    else:          return 3,  "Overbought (bearish)"


def _score_ma_crossover(price, ma50, ma200):
    a50, a200, golden = price > ma50, price > ma200, ma50 > ma200
    if a50 and a200 and golden: return 23, "Golden cross — strong uptrend"
    elif a50 and a200:          return 18, "Above both MAs"
    elif a200 and not a50:      return 13, "Above 200MA, below 50MA"
    elif a50 and not a200:      return 10, "Above 50MA, below 200MA"
    else:                       return 4,  "Below both MAs — downtrend"


def _score_momentum(c7, c30):
    avg = (c7 + c30) / 2
    if avg > 10:    return 22, f"Strong upward momentum (+{avg:.1f}% avg)"
    elif avg > 3:   return 17, f"Positive momentum (+{avg:.1f}% avg)"
    elif avg > -3:  return 12, f"Flat momentum ({avg:.1f}% avg)"
    elif avg > -10: return 7,  f"Negative momentum ({avg:.1f}% avg)"
    else:           return 2,  f"Sharp decline ({avg:.1f}% avg)"


def _score_fundamentals_fmp(quote: dict, ratios: dict, km: dict, sector: str):
    score = 12
    notes = []
    sector_pe = SECTOR_PE.get(sector)

    pe = ratios.get("priceToEarningsRatio") or km.get("peRatio")
    if pe and isinstance(pe, (int, float)) and 0 < pe < 1000:
        if sector_pe:
            ratio = pe / sector_pe
            if ratio < 0.8:
                score += 6; notes.append(f"P/E {pe:.1f} vs sector {sector_pe} — undervalued")
            elif ratio < 1.1:
                score += 3; notes.append(f"P/E {pe:.1f} vs sector {sector_pe} — fairly valued")
            else:
                score -= 3; notes.append(f"P/E {pe:.1f} vs sector {sector_pe} — premium")
        else:
            if pe < 15:   score += 5; notes.append(f"Low P/E ({pe:.1f})")
            elif pe < 30: score += 2; notes.append(f"P/E {pe:.1f} — fair")
            else:         score -= 4; notes.append(f"High P/E ({pe:.1f})")

    # ROE from key-metrics
    roe = km.get("returnOnEquity")
    if roe is not None:
        if roe > 0.20:   score += 3; notes.append(f"Strong ROE ({roe*100:.1f}%)")
        elif roe > 0.10: score += 1; notes.append(f"ROE {roe*100:.1f}%")
        else:            score -= 2; notes.append(f"Weak ROE ({roe*100:.1f}%)")

    return max(0, min(25, score)), " | ".join(notes) if notes else "Limited fundamental data"


def _verdict(total):
    if total >= 70:   return "BUY",  "green"
    elif total >= 45: return "HOLD", "orange"
    else:             return "SELL", "red"


# ---------------------------------------------------------------------------
# Financials formatter
# ---------------------------------------------------------------------------

def _build_quarterly(income_list: list) -> list:
    rows = []
    for s in income_list[:8]:
        rows.append({
            "Period":     s.get("date", "")[:7],
            "Revenue":    s.get("revenue"),
            "Net Income": s.get("netIncome") or s.get("bottomLineNetIncome"),
            "EPS":        s.get("epsDiluted") or s.get("eps"),
        })
    return rows


def _build_annual(income_list: list) -> list:
    rows = []
    for s in income_list[:4]:
        rows.append({
            "Year":       s.get("date", "")[:4],
            "Revenue":    s.get("revenue"),
            "Net Income": s.get("netIncome") or s.get("bottomLineNetIncome"),
            "EPS":        s.get("epsDiluted") or s.get("eps"),
        })
    return rows


# ---------------------------------------------------------------------------
# News via yfinance (separate, lighter endpoint)
# ---------------------------------------------------------------------------

def get_news(ticker: str, asset_type: str = "stock", coin_symbol: str = "") -> list[dict]:
    import yfinance as yf
    news = []
    yf_ticker = f"{coin_symbol}-USD" if asset_type == "crypto" and coin_symbol else ticker
    try:
        raw = yf.Ticker(yf_ticker).news or []
        for item in raw[:8]:
            content = item.get("content", {})
            title = content.get("title") or item.get("title", "")
            url = ""
            ct = content.get("clickThroughUrl") or {}
            if isinstance(ct, dict): url = ct.get("url", "")
            if not url:
                cu = content.get("canonicalUrl") or {}
                if isinstance(cu, dict): url = cu.get("url", "")
            pub = content.get("provider") or {}
            publisher = pub.get("displayName", "") if isinstance(pub, dict) else ""
            pub_time = content.get("pubDate", "") or ""
            if title:
                news.append({"title": title, "url": url,
                             "publisher": publisher, "time": pub_time})
    except Exception:
        pass
    return news


# ---------------------------------------------------------------------------
# Stock analysis
# ---------------------------------------------------------------------------

def analyze_stock(ticker: str) -> dict:
    ticker = ticker.upper().strip()
    try:
        # Price history
        closes = _fmp_history(ticker)
        if closes is None or len(closes) < 50:
            return {"error": f"No price data found for '{ticker}'. "
                             f"Check the ticker symbol, or this stock may not be supported on the free data tier. "
                             f"Try major US stocks (AAPL, MSFT, TSLA) or add '.NS' for Indian stocks (RELIANCE.NS)."}

        price_now  = float(closes.iloc[-1])
        price_7d   = float(closes.iloc[-6])  if len(closes) >= 6  else price_now
        price_30d  = float(closes.iloc[-21]) if len(closes) >= 21 else price_now
        change_7d  = (price_now - price_7d)  / price_7d  * 100
        change_30d = (price_now - price_30d) / price_30d * 100
        ma50  = _ma(closes, 50)
        ma200 = _ma(closes, 200) if len(closes) >= 200 else ma50
        rsi_val = _rsi(closes)

        # Fundamental data
        quote   = _fmp_quote(ticker)
        profile = _fmp_profile(ticker)
        ratios  = _fmp_ratios(ticker)
        km_data = _fmp_stable("key-metrics", {"symbol": ticker, "limit": 1})
        km      = km_data[0] if km_data and isinstance(km_data, list) else {}

        name     = profile.get("companyName") or profile.get("name") or ticker
        currency = profile.get("currency") or "USD"
        sector   = profile.get("sector") or ""
        industry = profile.get("industry") or ""

        # Scoring
        s_rsi,  l_rsi  = _score_rsi(rsi_val)
        s_ma,   l_ma   = _score_ma_crossover(price_now, ma50, ma200)
        s_mom,  l_mom  = _score_momentum(change_7d, change_30d)
        s_fund, l_fund = _score_fundamentals_fmp(quote, ratios, km, sector)
        total = s_rsi + s_ma + s_mom + s_fund
        verdict, color = _verdict(total)

        # Financials
        q_income = _fmp_income(ticker, "quarter", 4)
        a_income = _fmp_income(ticker, "annual",  4)

        pe      = ratios.get("priceToEarningsRatio") or km.get("peRatio")
        pb      = ratios.get("priceToBookRatio")
        mktcap  = km.get("marketCap") or profile.get("marketCap")
        high_52 = profile.get("range", "").split("-")[-1].strip() if profile.get("range") else None
        low_52  = profile.get("range", "").split("-")[0].strip()  if profile.get("range") else None

        fundamentals = {
            "Market Cap":            mktcap,
            "Sector":                sector or None,
            "Industry":              industry or None,
            "P/E (Trailing)":        pe,
            "Sector Avg P/E":        SECTOR_PE.get(sector),
            "Price/Book":            pb,
            "EPS (TTM)":             ratios.get("netIncomePerShare") or km.get("earningsYield"),
            "Profit Margin":         ratios.get("netProfitMargin"),
            "Operating Margin":      ratios.get("operatingProfitMargin"),
            "ROE":                   km.get("returnOnEquity") or ratios.get("returnOnEquity"),
            "ROA":                   km.get("returnOnAssets") or ratios.get("returnOnAssets"),
            "Debt/Equity":           ratios.get("debtToEquityRatio"),
            "Current Ratio":         ratios.get("currentRatio"),
            "Dividend Yield":        ratios.get("dividendYield"),
            "52w Range":             profile.get("range"),
            "Beta":                  profile.get("beta"),
            "Avg Volume":            profile.get("volAvg"),
        }

        news = get_news(ticker, "stock")

        return {
            "type": "stock", "ticker": ticker, "name": name,
            "currency": currency, "price": price_now,
            "change_7d": change_7d, "change_30d": change_30d,
            "rsi": rsi_val, "ma50": ma50, "ma200": ma200,
            "score": total, "verdict": verdict, "verdict_color": color,
            "signals": {
                "RSI":          {"score": s_rsi,  "max": 25, "label": l_rsi},
                "MA Crossover": {"score": s_ma,   "max": 25, "label": l_ma},
                "Momentum":     {"score": s_mom,  "max": 25, "label": l_mom},
                "Fundamentals": {"score": s_fund, "max": 25, "label": l_fund},
            },
            "fundamentals": fundamentals,
            "quarterly_earnings": _build_quarterly(q_income),
            "annual_financials":  _build_annual(a_income),
            "news": news,
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Crypto — CoinGecko (no key needed, reliable)
# ---------------------------------------------------------------------------

def _resolve_coin_id(query: str) -> str | None:
    q = query.lower().strip()
    r = _SESSION.get(f"{COINGECKO_BASE}/search", params={"query": q}, timeout=10)
    r.raise_for_status()
    coins = r.json().get("coins", [])
    if not coins: return None
    for c in coins:
        if c["symbol"].lower() == q: return c["id"]
    return coins[0]["id"]


def analyze_crypto(query: str) -> dict:
    try:
        coin_id = _resolve_coin_id(query)
        if not coin_id:
            return {"error": f"Could not find crypto '{query}'."}

        r = _SESSION.get(
            f"{COINGECKO_BASE}/coins/{coin_id}",
            params={"localization": "false", "tickers": "false",
                    "community_data": "true", "developer_data": "false"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()

        market     = data.get("market_data", {})
        price      = market.get("current_price", {}).get("usd")
        if not price: return {"error": "Price data unavailable."}

        change_24h = market.get("price_change_percentage_24h") or 0
        change_7d  = market.get("price_change_percentage_7d")  or 0
        change_30d = market.get("price_change_percentage_30d") or 0
        change_1y  = market.get("price_change_percentage_1y")  or 0
        ath        = market.get("ath", {}).get("usd")
        ath_date   = market.get("ath_date", {}).get("usd", "")
        atl        = market.get("atl", {}).get("usd")
        market_cap = market.get("market_cap", {}).get("usd")
        market_cap_rank    = data.get("market_cap_rank")
        volume_24h         = market.get("total_volume", {}).get("usd")
        circulating_supply = market.get("circulating_supply")
        max_supply         = market.get("max_supply")
        symbol             = data.get("symbol", query).upper()
        ath_drawdown       = ((price - ath) / ath * 100) if ath else None

        time.sleep(0.5)
        ohlc_r = _SESSION.get(
            f"{COINGECKO_BASE}/coins/{coin_id}/ohlc",
            params={"vs_currency": "usd", "days": "90"}, timeout=15,
        )
        ohlc_r.raise_for_status()
        closes = pd.Series([row[4] for row in ohlc_r.json()])

        rsi_val = _rsi(closes) if len(closes) >= 15 else 50.0
        ma50    = _ma(closes, 50) if len(closes) >= 50 else float(closes.mean())

        s_rsi, l_rsi = _score_rsi(rsi_val)
        s_ma,  l_ma  = _score_ma_crossover(price, ma50, ma50 * 0.98)
        s_mom, l_mom = _score_momentum(change_7d, change_30d)

        s_rank, l_rank = 12, "Unknown rank"
        if market_cap_rank:
            if market_cap_rank <= 10:    s_rank, l_rank = 22, f"Top-10 coin (rank #{market_cap_rank})"
            elif market_cap_rank <= 50:  s_rank, l_rank = 17, f"Top-50 coin (rank #{market_cap_rank})"
            elif market_cap_rank <= 200: s_rank, l_rank = 12, f"Mid-cap coin (rank #{market_cap_rank})"
            else:                        s_rank, l_rank = 5,  f"Small/micro cap (rank #{market_cap_rank})"

        total = s_rsi + s_ma + s_mom + s_rank
        verdict, color = _verdict(total)
        news = get_news(symbol, "crypto", symbol)

        return {
            "type": "crypto", "ticker": symbol, "coin_id": coin_id,
            "name": data.get("name", query), "currency": "USD",
            "price": price, "change_24h": change_24h, "change_7d": change_7d,
            "change_30d": change_30d, "change_1y": change_1y,
            "rsi": rsi_val, "ma50": ma50,
            "score": total, "verdict": verdict, "verdict_color": color,
            "signals": {
                "RSI":             {"score": s_rsi,  "max": 25, "label": l_rsi},
                "MA Crossover":    {"score": s_ma,   "max": 25, "label": l_ma},
                "Momentum":        {"score": s_mom,  "max": 25, "label": l_mom},
                "Market Cap Rank": {"score": s_rank, "max": 25, "label": l_rank},
            },
            "fundamentals": {
                "Market Cap":         market_cap,
                "Market Cap Rank":    market_cap_rank,
                "24h Volume":         volume_24h,
                "Volume/Mkt Cap":     round(volume_24h / market_cap, 4) if market_cap and volume_24h else None,
                "Circulating Supply": circulating_supply,
                "Max Supply":         max_supply,
                "Supply Used":        f"{circulating_supply/max_supply*100:.1f}%" if max_supply and circulating_supply else "Unlimited",
                "ATH":                ath,
                "ATH Date":           ath_date[:10] if ath_date else None,
                "ATH Drawdown":       f"{ath_drawdown:.1f}%" if ath_drawdown is not None else None,
                "ATL":                atl,
                "24h Change":         f"{change_24h:+.2f}%",
                "7d Change":          f"{change_7d:+.2f}%",
                "30d Change":         f"{change_30d:+.2f}%",
                "1y Change":          f"{change_1y:+.2f}%",
                "Description":        (data.get("description", {}).get("en", "") or "")[:300] or None,
            },
            "quarterly_earnings": [], "annual_financials": [],
            "news": news,
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def analyze(query: str, asset_type: str = "auto") -> dict:
    q = query.strip()
    if asset_type == "crypto": return analyze_crypto(q)
    if asset_type == "stock":  return analyze_stock(q)
    crypto_hints = {"BTC","ETH","SOL","BNB","XRP","ADA","DOGE","AVAX","DOT","MATIC",
                    "LINK","UNI","LTC","SHIB","TRX","ATOM","NEAR","APT","ARB","OP"}
    if q.upper() in crypto_hints: return analyze_crypto(q)
    return analyze_stock(q)
