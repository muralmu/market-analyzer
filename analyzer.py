"""
Analysis engine for stocks (US + Indian) and crypto.
Returns a scored analysis with Buy / Hold / Sell verdict.
"""

import yfinance as yf
import requests
from datetime import datetime
import pandas as pd
import numpy as np
import time

# ---------------------------------------------------------------------------
# Session — plain requests with browser headers for CoinGecko calls
# ---------------------------------------------------------------------------

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
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
# Technical helpers
# ---------------------------------------------------------------------------

def _rsi(prices: pd.Series, period: int = 14) -> float:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
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
    above_50  = price > ma50
    above_200 = price > ma200
    golden    = ma50 > ma200
    if above_50 and above_200 and golden: return 23, "Golden cross — strong uptrend"
    elif above_50 and above_200:          return 18, "Above both MAs"
    elif above_200 and not above_50:      return 13, "Above 200MA, below 50MA"
    elif above_50 and not above_200:      return 10, "Above 50MA, below 200MA"
    else:                                 return 4,  "Below both MAs — downtrend"


def _score_momentum(change_7d, change_30d):
    avg = (change_7d + change_30d) / 2
    if avg > 10:    return 22, f"Strong upward momentum (+{avg:.1f}% avg)"
    elif avg > 3:   return 17, f"Positive momentum (+{avg:.1f}% avg)"
    elif avg > -3:  return 12, f"Flat momentum ({avg:.1f}% avg)"
    elif avg > -10: return 7,  f"Negative momentum ({avg:.1f}% avg)"
    else:           return 2,  f"Sharp decline ({avg:.1f}% avg)"


def _score_fundamentals(fast, info_data, sector):
    score = 12
    notes = []
    sector_pe = SECTOR_PE.get(sector)

    pe = info_data.get("trailingPE") or info_data.get("forwardPE")
    if pe and isinstance(pe, (int, float)) and pe > 0:
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

    eps_growth = info_data.get("earningsQuarterlyGrowth")
    if eps_growth is not None and isinstance(eps_growth, float):
        if eps_growth > 0.15:   score += 5; notes.append(f"Strong EPS growth ({eps_growth*100:.0f}%)")
        elif eps_growth > 0:    score += 2; notes.append(f"Positive EPS growth ({eps_growth*100:.0f}%)")
        else:                   score -= 3; notes.append(f"EPS declining ({eps_growth*100:.0f}%)")

    roe = info_data.get("returnOnEquity")
    if roe is not None and isinstance(roe, float):
        if roe > 0.20:   score += 3; notes.append(f"Strong ROE ({roe*100:.1f}%)")
        elif roe > 0.10: score += 1; notes.append(f"ROE {roe*100:.1f}%")
        else:            score -= 2; notes.append(f"Weak ROE ({roe*100:.1f}%)")

    # 52w position from fast_info
    try:
        h52 = fast.year_high
        l52 = fast.year_low
        price = fast.last_price
        if h52 and l52 and price and (h52 - l52) > 0:
            pct = (price - l52) / (h52 - l52) * 100
            notes.append(f"52w position: {pct:.0f}% from low")
    except Exception:
        pass

    return max(0, min(25, score)), " | ".join(notes) if notes else "Limited fundamental data"


def _verdict(total):
    if total >= 70:   return "BUY",  "green"
    elif total >= 45: return "HOLD", "orange"
    else:             return "SELL", "red"


# ---------------------------------------------------------------------------
# Fetch stock data using download() + fast_info (avoids rate-limited .info)
# ---------------------------------------------------------------------------

def _fetch_stock_data(ticker: str):
    """
    Returns (hist_df, fast_info, info_dict, financials_dict).
    Uses yf.download for price history and fast_info for live quote.
    Falls back to ticker.info only for fundamental ratios (cached separately).
    """
    # Price history — download() is rarely rate-limited
    hist = yf.download(ticker, period="1y", auto_adjust=True,
                       progress=False, threads=False)
    if hist.empty or len(hist) < 50:
        return None, None, {}, {}

    t = yf.Ticker(ticker)
    fast = t.fast_info

    # Try to get fundamental info — may fail on cloud, that's OK
    info_data = {}
    try:
        info_data = t.info or {}
    except Exception:
        pass

    # Financials
    fin = {"quarterly_earnings": [], "annual_financials": []}
    try:
        qe = t.quarterly_income_stmt
        if qe is not None and not qe.empty:
            rows = []
            for col in list(qe.columns)[:8]:
                period = col.strftime("%b %Y") if hasattr(col, "strftime") else str(col)
                rev = qe.loc["Total Revenue", col] if "Total Revenue" in qe.index else None
                net = qe.loc["Net Income", col] if "Net Income" in qe.index else None
                eps = None
                for k in ["Basic EPS", "Diluted EPS"]:
                    if k in qe.index:
                        eps = qe.loc[k, col]; break
                rows.append({"Period": period, "Revenue": rev, "Net Income": net, "EPS": eps})
            fin["quarterly_earnings"] = rows
    except Exception:
        pass
    try:
        af = t.income_stmt
        if af is not None and not af.empty:
            rows = []
            for col in list(af.columns)[:4]:
                year = col.strftime("%Y") if hasattr(col, "strftime") else str(col)
                rev = af.loc["Total Revenue", col] if "Total Revenue" in af.index else None
                net = af.loc["Net Income", col] if "Net Income" in af.index else None
                eps = None
                for k in ["Basic EPS", "Diluted EPS"]:
                    if k in af.index:
                        eps = af.loc[k, col]; break
                rows.append({"Year": year, "Revenue": rev, "Net Income": net, "EPS": eps})
            fin["annual_financials"] = rows
    except Exception:
        pass

    return hist, fast, info_data, fin


# ---------------------------------------------------------------------------
# News
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
        hist, fast, info_data, fin = _fetch_stock_data(ticker)

        if hist is None:
            return {"error": f"No price data found for '{ticker}'. Check the ticker symbol."}

        # Handle MultiIndex columns from yf.download
        closes = hist["Close"]
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0]
        closes = closes.dropna()

        price_now  = float(closes.iloc[-1])
        price_7d   = float(closes.iloc[-6])  if len(closes) >= 6  else price_now
        price_30d  = float(closes.iloc[-21]) if len(closes) >= 21 else price_now
        change_7d  = (price_now - price_7d)  / price_7d  * 100
        change_30d = (price_now - price_30d) / price_30d * 100

        ma50  = _ma(closes, 50)
        ma200 = _ma(closes, 200) if len(closes) >= 200 else ma50
        rsi_val = _rsi(closes)

        # Currency + name from fast_info
        try:
            currency = fast.currency or "USD"
        except Exception:
            currency = info_data.get("currency", "USD")

        try:
            name = info_data.get("longName") or info_data.get("shortName") or ticker
        except Exception:
            name = ticker

        sector = info_data.get("sector", "")

        s_rsi,  l_rsi  = _score_rsi(rsi_val)
        s_ma,   l_ma   = _score_ma_crossover(price_now, ma50, ma200)
        s_mom,  l_mom  = _score_momentum(change_7d, change_30d)
        s_fund, l_fund = _score_fundamentals(fast, info_data, sector)

        total = s_rsi + s_ma + s_mom + s_fund
        verdict, color = _verdict(total)

        # Safe fast_info reads
        def _fi(attr, default=None):
            try: return getattr(fast, attr)
            except Exception: return default

        fundamentals = {
            "Market Cap":            _fi("market_cap"),
            "Sector":                sector or None,
            "Industry":              info_data.get("industry"),
            "P/E (Trailing)":        info_data.get("trailingPE"),
            "P/E (Forward)":         info_data.get("forwardPE"),
            "Sector Avg P/E":        SECTOR_PE.get(sector),
            "Price/Book":            info_data.get("priceToBook"),
            "EPS (TTM)":             info_data.get("trailingEps"),
            "EPS (Forward)":         info_data.get("forwardEps"),
            "Revenue Growth (YoY)":  info_data.get("revenueGrowth"),
            "Earnings Growth (QoQ)": info_data.get("earningsQuarterlyGrowth"),
            "Profit Margin":         info_data.get("profitMargins"),
            "Operating Margin":      info_data.get("operatingMargins"),
            "ROE":                   info_data.get("returnOnEquity"),
            "ROA":                   info_data.get("returnOnAssets"),
            "Debt/Equity":           info_data.get("debtToEquity"),
            "Current Ratio":         info_data.get("currentRatio"),
            "Dividend Yield":        info_data.get("dividendYield"),
            "52w High":              _fi("year_high"),
            "52w Low":               _fi("year_low"),
            "Avg Volume":            _fi("three_month_average_volume"),
            "Beta":                  info_data.get("beta"),
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

        market = data.get("market_data", {})
        price  = market.get("current_price", {}).get("usd")
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

        time.sleep(0.5)  # small delay between CoinGecko calls

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
            if market_cap_rank <= 10:   s_rank, l_rank = 22, f"Top-10 coin (rank #{market_cap_rank})"
            elif market_cap_rank <= 50: s_rank, l_rank = 17, f"Top-50 coin (rank #{market_cap_rank})"
            elif market_cap_rank <= 200:s_rank, l_rank = 12, f"Mid-cap coin (rank #{market_cap_rank})"
            else:                       s_rank, l_rank = 5,  f"Small/micro cap (rank #{market_cap_rank})"

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
                "RSI":             {"score": s_rsi,   "max": 25, "label": l_rsi},
                "MA Crossover":    {"score": s_ma,    "max": 25, "label": l_ma},
                "Momentum":        {"score": s_mom,   "max": 25, "label": l_mom},
                "Market Cap Rank": {"score": s_rank,  "max": 25, "label": l_rank},
            },
            "fundamentals": {
                "Market Cap":       market_cap,
                "Market Cap Rank":  market_cap_rank,
                "24h Volume":       volume_24h,
                "Volume/Mkt Cap":   round(volume_24h / market_cap, 4) if market_cap and volume_24h else None,
                "Circulating Supply": circulating_supply,
                "Max Supply":       max_supply,
                "Supply Used":      f"{circulating_supply/max_supply*100:.1f}%" if max_supply and circulating_supply else "Unlimited",
                "ATH":              ath,
                "ATH Date":         ath_date[:10] if ath_date else None,
                "ATH Drawdown":     f"{ath_drawdown:.1f}%" if ath_drawdown is not None else None,
                "ATL":              atl,
                "24h Change":       f"{change_24h:+.2f}%",
                "7d Change":        f"{change_7d:+.2f}%",
                "30d Change":       f"{change_30d:+.2f}%",
                "1y Change":        f"{change_1y:+.2f}%",
                "Description":      (data.get("description", {}).get("en", "") or "")[:300] or None,
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
