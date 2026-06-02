"""
Analysis engine for stocks (US + Indian) and crypto.
Calls Yahoo Finance raw API directly (bypasses yfinance rate limiting).
CoinGecko for crypto.
"""

import requests
import pandas as pd
import numpy as np
import time
import yfinance as yf
from datetime import datetime

# ---------------------------------------------------------------------------
# Session with browser-like headers
# ---------------------------------------------------------------------------

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
})

# ---------------------------------------------------------------------------
# Yahoo Finance direct API helpers
# ---------------------------------------------------------------------------

_YF_CRUMB = None
_YF_COOKIES = None

def _get_crumb():
    """Fetch Yahoo Finance crumb + cookies (needed for quoteSummary API)."""
    global _YF_CRUMB, _YF_COOKIES
    if _YF_CRUMB:
        return _YF_CRUMB, _YF_COOKIES
    try:
        r = _SESSION.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            timeout=10
        )
        if r.status_code == 200 and r.text.strip():
            _YF_CRUMB = r.text.strip()
            _YF_COOKIES = r.cookies
            return _YF_CRUMB, _YF_COOKIES
    except Exception:
        pass
    return None, None


def _fetch_chart(ticker: str, period: str = "1y", interval: str = "1d") -> pd.Series | None:
    """Fetch OHLCV from Yahoo Finance v8 chart API. Returns close price Series."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"interval": interval, "range": period, "includePrePost": "false"}
    try:
        r = _SESSION.get(url, params=params, timeout=15)
        if r.status_code == 429:
            time.sleep(3)
            r = _SESSION.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None
        closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
        timestamps = result[0].get("timestamp", [])
        if not closes or not timestamps:
            return None
        idx = pd.to_datetime(timestamps, unit="s")
        s = pd.Series(closes, index=idx, dtype=float).dropna()
        return s if len(s) >= 20 else None
    except Exception:
        return None


def _fetch_quote_summary(ticker: str) -> dict:
    """Fetch fundamental data from Yahoo Finance quoteSummary API."""
    modules = "financialData,defaultKeyStatistics,summaryDetail,assetProfile,price"
    crumb, cookies = _get_crumb()

    for host in ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]:
        try:
            params = {"modules": modules, "crumb": crumb} if crumb else {"modules": modules}
            r = _SESSION.get(
                f"https://{host}/v10/finance/quoteSummary/{ticker}",
                params=params,
                cookies=cookies,
                timeout=15,
            )
            if r.status_code == 429:
                time.sleep(2)
                continue
            if r.status_code != 200:
                continue
            js = r.json().get("quoteSummary", {}).get("result", [])
            if js:
                merged = {}
                for block in js:
                    merged.update(block)
                return merged
        except Exception:
            continue
    return {}


def _safe(d: dict, *keys, default=None):
    """Safely extract nested dict value: d['key1']['raw'] etc."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, {})
    if isinstance(d, dict):
        return d.get("raw", default)
    return d if d is not None else default


def _fetch_financials_direct(ticker: str) -> dict:
    """Fetch quarterly + annual income statement via Yahoo Finance v10."""
    result = {"quarterly_earnings": [], "annual_financials": []}

    for path, key in [
        (f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
         "?modules=incomeStatementHistoryQuarterly", "incomeStatementHistoryQuarterly"),
        (f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
         "?modules=incomeStatementHistory", "incomeStatementHistory"),
    ]:
        try:
            r = _SESSION.get(path, timeout=15)
            if r.status_code != 200:
                continue
            stmts = (r.json().get("quoteSummary", {})
                     .get("result", [{}])[0]
                     .get(key, {})
                     .get("incomeStatementHistory" if "Quarterly" not in key else "incomeStatementHistory", []))
            rows = []
            for s in stmts[:8]:
                end = _safe(s, "endDate", "fmt") or ""
                rev = _safe(s, "totalRevenue")
                net = _safe(s, "netIncome")
                eps = _safe(s, "dilutedEps") or _safe(s, "basicEps")
                if "Quarterly" in key:
                    rows.append({"Period": end[:7], "Revenue": rev, "Net Income": net, "EPS": eps})
                else:
                    rows.append({"Year": end[:4], "Revenue": rev, "Net Income": net, "EPS": eps})
            if "Quarterly" in key:
                result["quarterly_earnings"] = rows
            else:
                result["annual_financials"] = rows
        except Exception:
            continue
    return result


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
# Scoring
# ---------------------------------------------------------------------------

def _rsi(prices: pd.Series, period: int = 14) -> float:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    val = (100 - (100 / (1 + rs))).iloc[-1]
    return round(float(val), 1) if pd.notna(val) else 50.0


def _ma(prices: pd.Series, window: int) -> float:
    val = prices.rolling(window).mean().iloc[-1]
    return round(float(val), 4) if pd.notna(val) else float(prices.iloc[-1])


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


def _score_fundamentals(qs: dict, sector: str):
    score = 12
    notes = []
    sector_pe = SECTOR_PE.get(sector)

    fd = qs.get("financialData", {})
    ks = qs.get("defaultKeyStatistics", {})
    sd = qs.get("summaryDetail", {})

    pe = _safe(sd, "trailingPE") or _safe(ks, "forwardPE")
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

    eps_g = _safe(ks, "earningsQuarterlyGrowth")
    if eps_g is not None:
        if eps_g > 0.15:  score += 5; notes.append(f"Strong EPS growth ({eps_g*100:.0f}%)")
        elif eps_g > 0:   score += 2; notes.append(f"Positive EPS growth ({eps_g*100:.0f}%)")
        else:             score -= 3; notes.append(f"EPS declining ({eps_g*100:.0f}%)")

    roe = _safe(fd, "returnOnEquity")
    if roe is not None:
        if roe > 0.20:   score += 3; notes.append(f"Strong ROE ({roe*100:.1f}%)")
        elif roe > 0.10: score += 1; notes.append(f"ROE {roe*100:.1f}%")
        else:            score -= 2; notes.append(f"Weak ROE ({roe*100:.1f}%)")

    return max(0, min(25, score)), " | ".join(notes) if notes else "Limited data"


def _verdict(total):
    if total >= 70:   return "BUY",  "green"
    elif total >= 45: return "HOLD", "orange"
    else:             return "SELL", "red"


# ---------------------------------------------------------------------------
# News (via yfinance — separate, less restricted endpoint)
# ---------------------------------------------------------------------------

def get_news(ticker: str, asset_type: str = "stock", coin_symbol: str = "") -> list[dict]:
    news = []
    yf_ticker = f"{coin_symbol}-USD" if asset_type == "crypto" and coin_symbol else ticker
    try:
        t = yf.Ticker(yf_ticker)
        raw = t.news or []
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
                news.append({"title": title, "url": url, "publisher": publisher, "time": pub_time})
    except Exception:
        pass
    return news


# ---------------------------------------------------------------------------
# Stock analysis
# ---------------------------------------------------------------------------

def analyze_stock(ticker: str) -> dict:
    ticker = ticker.upper().strip()
    try:
        # 1. Price history
        closes = _fetch_chart(ticker, period="1y", interval="1d")
        if closes is None or len(closes) < 50:
            return {"error": f"No price data found for '{ticker}'. Check the ticker symbol."}

        price_now  = float(closes.iloc[-1])
        price_7d   = float(closes.iloc[-6])  if len(closes) >= 6  else price_now
        price_30d  = float(closes.iloc[-21]) if len(closes) >= 21 else price_now
        change_7d  = (price_now - price_7d)  / price_7d  * 100
        change_30d = (price_now - price_30d) / price_30d * 100
        ma50  = _ma(closes, 50)
        ma200 = _ma(closes, 200) if len(closes) >= 200 else ma50
        rsi_val = _rsi(closes)

        # 2. Fundamental data
        qs = _fetch_quote_summary(ticker)
        ap = qs.get("assetProfile", {})
        ks = qs.get("defaultKeyStatistics", {})
        sd = qs.get("summaryDetail", {})
        fd = qs.get("financialData", {})
        pr = qs.get("price", {})

        name     = _safe(pr, "longName") or _safe(pr, "shortName") or ticker
        currency = _safe(pr, "currency") or "USD"
        sector   = ap.get("sector", "")
        industry = ap.get("industry", "")

        # 3. Scoring
        s_rsi,  l_rsi  = _score_rsi(rsi_val)
        s_ma,   l_ma   = _score_ma_crossover(price_now, ma50, ma200)
        s_mom,  l_mom  = _score_momentum(change_7d, change_30d)
        s_fund, l_fund = _score_fundamentals(qs, sector)
        total = s_rsi + s_ma + s_mom + s_fund
        verdict, color = _verdict(total)

        # 4. Financials
        fin = _fetch_financials_direct(ticker)

        # 5. Key metrics table
        fundamentals = {
            "Market Cap":            _safe(pr, "marketCap"),
            "Sector":                sector or None,
            "Industry":              industry or None,
            "P/E (Trailing)":        _safe(sd, "trailingPE"),
            "P/E (Forward)":         _safe(ks, "forwardPE"),
            "Sector Avg P/E":        SECTOR_PE.get(sector),
            "Price/Book":            _safe(ks, "priceToBook"),
            "EPS (TTM)":             _safe(ks, "trailingEps"),
            "EPS (Forward)":         _safe(ks, "forwardEps"),
            "Revenue Growth (YoY)":  _safe(fd, "revenueGrowth"),
            "Earnings Growth (QoQ)": _safe(ks, "earningsQuarterlyGrowth"),
            "Profit Margin":         _safe(fd, "profitMargins"),
            "Operating Margin":      _safe(fd, "operatingMargins"),
            "ROE":                   _safe(fd, "returnOnEquity"),
            "ROA":                   _safe(fd, "returnOnAssets"),
            "Debt/Equity":           _safe(ks, "debtToEquity"),
            "Current Ratio":         _safe(fd, "currentRatio"),
            "Dividend Yield":        _safe(sd, "dividendYield"),
            "52w High":              _safe(sd, "fiftyTwoWeekHigh"),
            "52w Low":               _safe(sd, "fiftyTwoWeekLow"),
            "Beta":                  _safe(sd, "beta"),
            "Avg Volume":            _safe(sd, "averageVolume"),
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
            "quarterly_earnings": fin["quarterly_earnings"],
            "annual_financials":  fin["annual_financials"],
            "news": news,
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Crypto analysis via CoinGecko
# ---------------------------------------------------------------------------

COINGECKO_BASE = "https://api.coingecko.com/api/v3"


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
        ohlc   = ohlc_r.json()
        closes = pd.Series([row[4] for row in ohlc])

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
                "Market Cap":        market_cap,
                "Market Cap Rank":   market_cap_rank,
                "24h Volume":        volume_24h,
                "Volume/Mkt Cap":    round(volume_24h / market_cap, 4) if market_cap and volume_24h else None,
                "Circulating Supply": circulating_supply,
                "Max Supply":        max_supply,
                "Supply Used":       f"{circulating_supply/max_supply*100:.1f}%" if max_supply and circulating_supply else "Unlimited",
                "ATH":               ath,
                "ATH Date":          ath_date[:10] if ath_date else None,
                "ATH Drawdown":      f"{ath_drawdown:.1f}%" if ath_drawdown is not None else None,
                "ATL":               atl,
                "24h Change":        f"{change_24h:+.2f}%",
                "7d Change":         f"{change_7d:+.2f}%",
                "30d Change":        f"{change_30d:+.2f}%",
                "1y Change":         f"{change_1y:+.2f}%",
                "Description":       (data.get("description", {}).get("en", "") or "")[:300] or None,
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
