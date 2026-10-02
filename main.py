import os
import time
from typing import List, Optional

import pandas as pd
import yfinance as yf
from fastapi import Depends, FastAPI, Header, HTTPException, Query

app = FastAPI(
    title="Yahoo Finance API",
    description="Simple REST wrapper around yfinance",
    version="1.0.0",
)

# ---------- Optional API key auth ----------
# Set API_KEY env var to require header:  X-API-Key: <key>
API_KEY = os.getenv("API_KEY")


def check_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------- Tiny in-memory cache (avoids hammering Yahoo) ----------
CACHE_TTL = int(os.getenv("CACHE_TTL", "60"))  # seconds
_cache = {}


def cached(key, fn):
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < CACHE_TTL:
        return hit[1]
    value = fn()
    _cache[key] = (now, value)
    return value


def df_to_records(df: pd.DataFrame, index_name="date"):
    if df is None or df.empty:
        return []
    df = df.copy()
    df.index = df.index.astype(str)
    df = df.reset_index().rename(columns={"index": index_name})
    # NaN -> None so JSON is valid
    return df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")


def fin_to_dict(df: pd.DataFrame):
    """Financial statements: rows = line items, columns = dates."""
    if df is None or df.empty:
        return {}
    df = df.copy()
    df.columns = [str(c.date()) if hasattr(c, "date") else str(c) for c in df.columns]
    df = df.astype(object).where(pd.notnull(df), None)
    return df.to_dict(orient="index")


def valid(symbol: str) -> str:
    s = symbol.strip().upper()
    if not s or len(s) > 15:
        raise HTTPException(status_code=400, detail="Invalid symbol")
    return s


# ---------- Endpoints ----------
@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}


@app.get("/quote/{symbol}", dependencies=[Depends(check_key)])
def quote(symbol: str):
    """Latest price snapshot."""
    s = valid(symbol)

    def fetch():
        fi = yf.Ticker(s).fast_info
        return {
            "symbol": s,
            "currency": fi.get("currency"),
            "last_price": fi.get("lastPrice"),
            "previous_close": fi.get("previousClose"),
            "open": fi.get("open"),
            "day_high": fi.get("dayHigh"),
            "day_low": fi.get("dayLow"),
            "volume": fi.get("lastVolume"),
            "market_cap": fi.get("marketCap"),
            "year_high": fi.get("yearHigh"),
            "year_low": fi.get("yearLow"),
        }

    try:
        data = cached(("quote", s), fetch)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Yahoo error: {e}")
    if data["last_price"] is None:
        raise HTTPException(status_code=404, detail=f"No data for {s}")
    return data


@app.get("/quotes", dependencies=[Depends(check_key)])
def quotes(symbols: str = Query(..., description="Comma separated, e.g. AAPL,MSFT,TSLA")):
    """Latest prices for multiple symbols."""
    out = {}
    for s in [valid(x) for x in symbols.split(",") if x.strip()][:50]:
        try:
            out[s] = quote(s)
        except HTTPException as e:
            out[s] = {"error": e.detail}
    return out


@app.get("/history/{symbol}", dependencies=[Depends(check_key)])
def history(
    symbol: str,
    period: str = Query("1mo", description="1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max"),
    interval: str = Query("1d", description="1m,5m,15m,30m,60m,1d,1wk,1mo"),
    start: Optional[str] = Query(None, description="YYYY-MM-DD (overrides period)"),
    end: Optional[str] = Query(None, description="YYYY-MM-DD"),
):
    """Historical OHLCV candles."""
    s = valid(symbol)

    def fetch():
        t = yf.Ticker(s)
        if start:
            return t.history(start=start, end=end, interval=interval)
        return t.history(period=period, interval=interval)

    try:
        df = cached(("hist", s, period, interval, start, end), fetch)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Yahoo error: {e}")
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No history for {s}")
    return {"symbol": s, "interval": interval, "data": df_to_records(df, "date")}


@app.get("/info/{symbol}", dependencies=[Depends(check_key)])
def info(symbol: str):
    """Company profile and key stats (sector, PE, dividend yield, etc.)."""
    s = valid(symbol)
    try:
        data = cached(("info", s), lambda: yf.Ticker(s).info)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Yahoo error: {e}")
    if not data:
        raise HTTPException(status_code=404, detail=f"No info for {s}")
    return data


@app.get("/financials/{symbol}", dependencies=[Depends(check_key)])
def financials(
    symbol: str,
    statement: str = Query("income", description="income | balance | cashflow"),
    quarterly: bool = False,
):
    """Income statement, balance sheet, or cash flow."""
    s = valid(symbol)
    t = yf.Ticker(s)
    mapping = {
        "income": (t.quarterly_income_stmt if quarterly else t.income_stmt),
        "balance": (t.quarterly_balance_sheet if quarterly else t.balance_sheet),
        "cashflow": (t.quarterly_cashflow if quarterly else t.cashflow),
    }
    if statement not in mapping:
        raise HTTPException(status_code=400, detail="statement must be income, balance or cashflow")
    try:
        df = cached(("fin", s, statement, quarterly), lambda: mapping[statement])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Yahoo error: {e}")
    return {"symbol": s, "statement": statement, "quarterly": quarterly, "data": fin_to_dict(df)}


@app.get("/dividends/{symbol}", dependencies=[Depends(check_key)])
def dividends(symbol: str):
    s = valid(symbol)
    series = cached(("div", s), lambda: yf.Ticker(s).dividends)
    df = series.to_frame("dividend") if not series.empty else pd.DataFrame()
    return {"symbol": s, "data": df_to_records(df, "date")}


@app.get("/news/{symbol}", dependencies=[Depends(check_key)])
def news(symbol: str):
    s = valid(symbol)
    try:
        return {"symbol": s, "news": cached(("news", s), lambda: yf.Ticker(s).news)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Yahoo error: {e}")
