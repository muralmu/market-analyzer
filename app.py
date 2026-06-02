import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from streamlit_searchbox import st_searchbox
from analyzer import analyze
import yfinance as yf

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Market Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .verdict-box {
        text-align: center;
        padding: 20px;
        border-radius: 12px;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 16px;
    }
    .score-bar-label { font-size: 0.85rem; color: #888; margin-bottom: 2px; }
    .signal-label { font-size: 0.9rem; }
    .stMetric label { font-size: 0.8rem !important; }
    .news-item { padding: 10px 0; border-bottom: 1px solid #2a2a2a; }
    .news-title { font-size: 0.95rem; font-weight: 500; }
    .news-meta { font-size: 0.78rem; color: #888; margin-top: 3px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("📊 Market Analyzer")
st.caption("US Stocks · Indian Stocks (NSE/BSE) · Crypto — Buy / Hold / Sell analysis")

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _search_stocks_and_crypto(query: str, **kwargs) -> list[str]:
    if not query or len(query) < 1:
        return []
    results = []
    try:
        r = requests.get(
            "https://query1.finance.yahoo.com/v1/finance/search",
            params={"q": query, "lang": "en-US", "region": "US", "quotesCount": 8, "newsCount": 0},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5,
        )
        for q in r.json().get("quotes", []):
            sym = q.get("symbol", "")
            name = q.get("longname") or q.get("shortname") or ""
            exch = q.get("exchDisp") or q.get("exchange") or ""
            if sym and name:
                results.append(f"{sym} — {name} ({exch})" if exch else f"{sym} — {name}")
    except Exception:
        pass
    try:
        r = requests.get("https://api.coingecko.com/api/v3/search", params={"query": query}, timeout=5)
        for c in r.json().get("coins", [])[:5]:
            sym = c.get("symbol", "").upper()
            name = c.get("name", "")
            rank = c.get("market_cap_rank")
            rank_str = f" #{rank}" if rank else ""
            results.append(f"{sym} — {name} (Crypto{rank_str})")
    except Exception:
        pass
    return results[:12]


def _extract_ticker(selected: str) -> str:
    return selected.split(" — ")[0].strip() if selected else ""

def _is_crypto(selected: str) -> bool:
    return "Crypto" in (selected or "")

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

CURRENCY_SYMBOL = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}

def csym(currency):
    return CURRENCY_SYMBOL.get(currency, currency + " ")

def fmt_price(price, currency):
    s = csym(currency)
    if price < 0.01:
        return f"{s}{price:,.6f}"
    elif price < 1:
        return f"{s}{price:,.4f}"
    else:
        return f"{s}{price:,.2f}"

def fmt_number(n, currency="USD"):
    if n is None:
        return "—"
    s = csym(currency)
    if n >= 1e12:
        return f"{s}{n/1e12:.2f}T"
    if n >= 1e9:
        return f"{s}{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{s}{n/1e6:.2f}M"
    return f"{s}{n:,.2f}"

def fmt_pct(v):
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    return f"{v*100:.2f}%"

def fmt_val(k, v, currency):
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    pct_keys = {"Revenue Growth (YoY)", "Earnings Growth (QoQ)", "Profit Margin",
                "Operating Margin", "ROE", "ROA", "Dividend Yield"}
    large_keys = {"Market Cap", "24h Volume", "Circulating Supply", "Max Supply", "Avg Volume"}
    if k in pct_keys:
        return fmt_pct(v)
    if k in large_keys:
        return fmt_number(v, currency)
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)

# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def render_verdict(verdict, color, score):
    bg = {"green": "#1a4a2e", "orange": "#4a3a1a", "red": "#4a1a1a"}.get(color, "#222")
    tc = {"green": "#00e676", "orange": "#ffb300", "red": "#ff5252"}.get(color, "#fff")
    st.markdown(
        f'<div class="verdict-box" style="background:{bg}; color:{tc}">'
        f'{verdict}<br><span style="font-size:1rem;font-weight:400;color:#ccc">Score: {score}/100</span>'
        f'</div>', unsafe_allow_html=True)


def render_score_bar(label, score, max_score, signal_label):
    pct = score / max_score * 100
    color = "#00e676" if pct >= 70 else "#ffb300" if pct >= 45 else "#ff5252"
    st.markdown(f'<div class="score-bar-label">{label} — {score}/{max_score}</div>', unsafe_allow_html=True)
    st.progress(int(pct))
    st.markdown(f'<div class="signal-label" style="color:{color};margin-bottom:12px">{signal_label}</div>', unsafe_allow_html=True)


def render_price_chart(ticker, asset_type_val, currency="USD"):
    s = csym(currency)
    try:
        if asset_type_val == "crypto":
            from analyzer import _resolve_coin_id, COINGECKO_BASE
            coin_id = _resolve_coin_id(ticker)
            r = requests.get(f"{COINGECKO_BASE}/coins/{coin_id}/market_chart",
                             params={"vs_currency": "usd", "days": "90"}, timeout=15)
            prices = r.json().get("prices", [])
            df = pd.DataFrame(prices, columns=["ts", "price"])
            df["date"] = pd.to_datetime(df["ts"], unit="ms")
        else:
            hist = yf.Ticker(ticker).history(period="3mo")
            df = hist.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "price"})

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["price"], mode="lines",
            line=dict(color="#00b4d8", width=2),
            fill="tozeroy", fillcolor="rgba(0,180,216,0.08)",
            hovertemplate=f"{s}%{{y:,.2f}}<extra></extra>",
        ))
        fig.update_layout(
            height=260, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, color="#888"),
            yaxis=dict(showgrid=True, gridcolor="#333", color="#888", tickprefix=s),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.caption("Chart unavailable.")


def render_financials_charts(quarterly, annual, currency):
    s = csym(currency)

    if quarterly:
        st.markdown("#### Quarterly Revenue & Net Income")
        df = pd.DataFrame(quarterly).iloc[::-1]  # oldest first
        if not df.empty and "Revenue" in df.columns:
            fig = go.Figure()
            if "Revenue" in df.columns:
                fig.add_trace(go.Bar(name="Revenue", x=df["Period"], y=df["Revenue"],
                                     marker_color="#00b4d8",
                                     hovertemplate=f"{s}%{{y:,.0f}}<extra>Revenue</extra>"))
            if "Net Income" in df.columns:
                fig.add_trace(go.Bar(name="Net Income", x=df["Period"], y=df["Net Income"],
                                     marker_color="#00e676",
                                     hovertemplate=f"{s}%{{y:,.0f}}<extra>Net Income</extra>"))
            fig.update_layout(
                barmode="group", height=260, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(color="#888"), yaxis=dict(color="#888", gridcolor="#333"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                font=dict(color="#ccc"),
            )
            st.plotly_chart(fig, use_container_width=True)

        # EPS table
        eps_cols = [c for c in ["Period", "EPS"] if c in df.columns]
        if "EPS" in df.columns and df["EPS"].notna().any():
            st.caption("Quarterly EPS")
            eps_df = df[eps_cols].copy()
            eps_df["EPS"] = eps_df["EPS"].apply(lambda x: f"{s}{x:.2f}" if pd.notna(x) else "—")
            st.dataframe(eps_df, use_container_width=True, hide_index=True)

    if annual:
        st.markdown("#### Annual Revenue & Net Income (3 Years)")
        df = pd.DataFrame(annual).iloc[::-1]
        if not df.empty:
            fig = go.Figure()
            if "Revenue" in df.columns:
                fig.add_trace(go.Bar(name="Revenue", x=df["Year"], y=df["Revenue"],
                                     marker_color="#00b4d8",
                                     hovertemplate=f"{s}%{{y:,.0f}}<extra>Revenue</extra>"))
            if "Net Income" in df.columns:
                fig.add_trace(go.Bar(name="Net Income", x=df["Year"], y=df["Net Income"],
                                     marker_color="#00e676",
                                     hovertemplate=f"{s}%{{y:,.0f}}<extra>Net Income</extra>"))
            fig.update_layout(
                barmode="group", height=240, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(color="#888"), yaxis=dict(color="#888", gridcolor="#333"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                font=dict(color="#ccc"),
            )
            st.plotly_chart(fig, use_container_width=True)


def render_news(news_items):
    if not news_items:
        st.caption("No recent news found.")
        return
    for item in news_items:
        title = item.get("title", "")
        url = item.get("url", "")
        publisher = item.get("publisher", "")
        pub_time = item.get("time", "")

        # Format time
        time_str = ""
        if pub_time:
            try:
                from datetime import datetime, timezone
                dt = datetime.fromisoformat(pub_time.replace("Z", "+00:00"))
                time_str = dt.strftime("%d %b %Y, %H:%M UTC")
            except Exception:
                time_str = pub_time[:10]

        meta = " · ".join(filter(None, [publisher, time_str]))
        if url:
            st.markdown(
                f'<div class="news-item">'
                f'<div class="news-title"><a href="{url}" target="_blank" style="color:#00b4d8;text-decoration:none;">{title}</a></div>'
                f'<div class="news-meta">{meta}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="news-item">'
                f'<div class="news-title">{title}</div>'
                f'<div class="news-meta">{meta}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Input row
# ---------------------------------------------------------------------------

col1, col2 = st.columns([4, 1])
with col1:
    selected = st_searchbox(
        _search_stocks_and_crypto,
        placeholder="Search stock or crypto — e.g. Apple, Reliance, Bitcoin, BTC…",
        label="Stock or Crypto", label_visibility="collapsed",
        key="ticker_search", debounce=300, clear_on_submit=False,
    )
with col2:
    analyse_btn = st.button("Analyse →", type="primary", use_container_width=True)

st.markdown("**Quick examples:** `AAPL` · `TSLA` · `RELIANCE.NS` · `TCS.NS` · `BTC` · `ETH` · `SOL`")

with st.expander("ℹ️ How to use / ticker formats"):
    st.markdown("""
| What you want | Type | Or search by name |
|---|---|---|
| US Stock | `AAPL`, `TSLA`, `MSFT` | "Apple", "Tesla" |
| Indian Stock (NSE) | `RELIANCE.NS`, `TCS.NS` | "Reliance", "TCS" |
| Indian Stock (BSE) | `RELIANCE.BO`, `TCS.BO` | — |
| Crypto | `BTC`, `ETH`, `SOL` | "Bitcoin", "Ethereum" |

Indian stock prices are shown in **₹ INR**. US stocks and crypto in **$ USD**.
    """)

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

if analyse_btn:
    raw_input = selected or ""
    if not raw_input.strip():
        st.warning("Please search and select a stock or crypto first.")
    else:
        ticker = _extract_ticker(raw_input)
        asset_type = "crypto" if _is_crypto(raw_input) else "auto"

        @st.cache_data(ttl=300, show_spinner=False)
        def _cached_analyze(t, at):
            return analyze(t, at)

        with st.spinner(f"Fetching data for **{ticker}**…"):
            result = _cached_analyze(ticker, asset_type)

        if "error" in result:
            st.error(f"**Error:** {result['error']}")
        else:
            currency = result.get("currency", "USD")

            # ── Top row: price info / verdict / chart ──
            left, mid, right = st.columns([1, 1.2, 1.5])

            with left:
                st.subheader(f"{result['name']} ({result['ticker']})")
                st.metric("Price", fmt_price(result["price"], currency))

                def pct(val): return f"{val:+.1f}%"

                if result["type"] == "crypto":
                    st.metric("24h Change", pct(result.get("change_24h", 0)))
                    st.metric("7d Change",  pct(result.get("change_7d",  0)))
                    st.metric("30d Change", pct(result.get("change_30d", 0)))
                    st.metric("1y Change",  pct(result.get("change_1y",  0)))
                else:
                    st.metric("7d Change",  pct(result.get("change_7d",  0)))
                    st.metric("30d Change", pct(result.get("change_30d", 0)))
                st.metric("RSI (14)", result["rsi"])
                st.metric("MA50",  fmt_price(result["ma50"], currency))
                if result.get("ma200"):
                    st.metric("MA200", fmt_price(result["ma200"], currency))

            with mid:
                render_verdict(result["verdict"], result["verdict_color"], result["score"])
                st.markdown("**Signal Breakdown**")
                for sig_name, sig in result["signals"].items():
                    render_score_bar(sig_name, sig["score"], sig["max"], sig["label"])

            with right:
                st.markdown("**90-day Price Chart**")
                render_price_chart(result["ticker"], result["type"], currency)

            # ── Fundamentals ──
            st.divider()
            st.markdown("### 📋 Fundamentals")
            fund = result.get("fundamentals", {})

            # Description (crypto)
            desc = fund.pop("Description", None)
            if desc:
                st.caption(desc)

            # PE vs Sector highlight (stocks)
            if result["type"] == "stock":
                pe_trail = fund.get("P/E (Trailing)")
                pe_fwd = fund.get("P/E (Forward)")
                sector_pe = fund.get("Sector Avg P/E")
                if pe_trail and sector_pe:
                    ratio = pe_trail / sector_pe
                    color = "#00e676" if ratio < 0.8 else "#ffb300" if ratio < 1.1 else "#ff5252"
                    st.markdown(
                        f'<div style="background:#1a1a2e;border-radius:8px;padding:12px;margin-bottom:12px">'
                        f'<b>P/E vs Sector:</b> Trailing P/E <span style="color:{color};font-weight:700">{pe_trail:.1f}</span>'
                        f' vs sector average <b>{sector_pe}</b> — '
                        f'<span style="color:{color}">{"Undervalued" if ratio < 0.8 else "Fairly valued" if ratio < 1.1 else "At a premium"}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Fundamentals table — always show all rows, — for missing
            fund_items = [(k, fmt_val(k, v, currency) if v is not None else "—")
                          for k, v in fund.items() if k != "Description"]
            half = len(fund_items) // 2 + len(fund_items) % 2
            fc1, fc2 = st.columns(2)
            with fc1:
                st.dataframe(pd.DataFrame(fund_items[:half], columns=["Metric", "Value"]),
                             use_container_width=True, hide_index=True)
            with fc2:
                st.dataframe(pd.DataFrame(fund_items[half:], columns=["Metric", "Value"]),
                             use_container_width=True, hide_index=True)

            # ── Earnings & Revenue charts (stocks only) ──
            quarterly = result.get("quarterly_earnings", [])
            annual = result.get("annual_financials", [])
            if quarterly or annual:
                st.divider()
                st.markdown("### 📈 Earnings & Revenue")
                render_financials_charts(quarterly, annual, currency)

            # ── News ──
            st.divider()
            st.markdown("### 📰 Recent News")
            render_news(result.get("news", []))

            # ── Disclaimer ──
            st.divider()
            st.caption(
                "⚠️ **Disclaimer:** This analysis is generated algorithmically from publicly available data. "
                "It is not financial advice. Always do your own research before investing."
            )
