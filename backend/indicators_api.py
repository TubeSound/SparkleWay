# backend/indicators_api.py (JST対応・重複/並び順ケア版)
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import os
from datetime import datetime

app = FastAPI(title="SparkleWay CSV + Indicators API")

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(os.getenv("SPARKLEWAY_DATA_DIR", Path(__file__).parent / "data")).resolve()
DATA_DIR.mkdir(exist_ok=True)

# ---------------- In-memory cache (per symbol/timeframe) ----------------
CACHE: Dict[str, Dict[str, Any]] = {}

def _dataset_key(symbol: str, timeframe: str) -> str:
    return f"{symbol.upper()}::{timeframe.lower()}"

# ---------- utility: safe JSON-able series->[{time,value}] ----------
def _series_to_kv(time_s: pd.Series, val_s: pd.Series, dropna: bool = False) -> List[Dict[str, Any]]:
    """Align time/value safely and make it JSON-safe (no NaN).
    If dropna=True, rows with NaN values are removed. Otherwise value=None for NaN.
    """
    dfv = pd.DataFrame({"time": time_s, "value": val_s}).copy()
    # ensure integer seconds for time
    dfv["time"] = pd.to_numeric(dfv["time"], errors="coerce").astype("Int64")
    if dropna:
        dfv = dfv.loc[dfv["value"].notna()]
    dfv = dfv.loc[dfv["time"].notna()]
    out: List[Dict[str, Any]] = []
    for t, v in zip(dfv["time"].astype(int), dfv["value"]):
        out.append({"time": int(t), "value": (float(v) if pd.notna(v) else None)})
    return out

# ---------------- CSV resolution helpers ----------------
_TIMEFRAME_ALIASES = {
    "1m": ["1m", "m1"],
    "5m": ["5m", "m5"],
    "15m": ["15m", "m15"],
    "1h": ["1h", "h1"],
    "1d": ["1d", "d1", "daily"],
}

def _resolve_csv(symbol: str, timeframe: str) -> Path:
    sym = symbol.lower()
    tf = timeframe.lower()
    tf_keys = _TIMEFRAME_ALIASES.get(tf, [tf])

    candidates: List[Path] = []
    for p in DATA_DIR.glob("*.csv"):
        name = p.name.lower()
        if sym in name and any(t in name for t in tf_keys):
            candidates.append(p)
    if not candidates:
        for p in DATA_DIR.glob("*.csv"):
            if sym in p.name.lower():
                candidates.append(p)
    if not candidates:
        raise HTTPException(status_code=404, detail=f"CSV not found for symbol={symbol}, timeframe={timeframe} in {DATA_DIR}")
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]

# ---------------- Load & transform ----------------
def _jststr2timestamp(df: pd.DataFrame) ->[int]:
    timestamp = []
    for t_str in df['jst']:
        dt = datetime.fromisoformat(t_str)
        ts = int(dt.timestamp())
        timestamp.append(ts)
    return timestamp
    

def _load_ohlc_from_file(csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV read error: {e}")

    need = {"open", "high", "low", "close"}
    if not need.issubset({c.lower() for c in df.columns}):
        raise HTTPException(status_code=400, detail="CSV must contain open, high, low, close columns")

    # normalize column names to lower
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df["time"] = _jststr2timestamp(df)

    # ensure numeric OHLC
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop missing
    df = df.dropna(subset=["time", "open", "high", "low", "close"]).copy()
    df["time"] = df["time"].astype(int)

    # sort asc & drop duplicates (keep last)
    df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)

    return df[["time", "open", "high", "low", "close"]]

# ---------------- Indicators ----------------

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=series.index)

SUPPORTED_INDICATORS = {"ema", "rsi"}

# ---------------- Endpoints ----------------

@app.get("/candles")
def get_candles(
    symbol: str = Query(..., description="シンボル例: JP225, NASDAQ, USDJPY など（ファイル名に含まれている語）"),
    timeframe: str = Query(..., description="1m|5m|15m|1h|1d など"),
    length: int = Query(2000, ge=1, le=200000, description="返すバー数（末尾から）"),
    indicators: Optional[str] = Query(None, description="事前計算するインジ。例: ema:20,ema:50,rsi:14"),
):
    csv_path = _resolve_csv(symbol, timeframe)
    df = _load_ohlc_from_file(csv_path)
    if length:
        df = df.iloc[-length:]

    key = _dataset_key(symbol, timeframe)
    CACHE[key] = CACHE.get(key, {})
    CACHE[key]["candles_df"] = df

    # Precompute indicators if requested
    if indicators:
        for token in indicators.split(','):
            token = token.strip().lower()
            if not token:
                continue
            if token.startswith("ema:"):
                period = int(token.split(":", 1)[1])
                vals = _ema(df["close"], period)
                CACHE[key][f"ema:{period}"] = _series_to_kv(df["time"], vals, dropna=True)
            elif token.startswith("rsi:"):
                period = int(token.split(":", 1)[1])
                vals = _rsi(df["close"], period)
                CACHE[key][f"rsi:{period}"] = _series_to_kv(df["time"], vals, dropna=True)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported indicator token: {token}")

    candles = df.to_dict(orient="records")
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "length": len(candles),
        "candles": candles,
        "precomputed": [k for k in CACHE[key].keys() if k not in ("candles_df")],
    }


@app.get("/indicator")
def get_indicator(
    symbol: str = Query(...),
    timeframe: str = Query(...),
    name: str = Query(..., description="インジ名: ema|rsi"),
    period: Optional[int] = Query(None, ge=1, description="期間: ema/rsi用"),
):
    key = _dataset_key(symbol, timeframe)
    if key not in CACHE or "candles_df" not in CACHE[key]:
        raise HTTPException(status_code=404, detail="Candles not prepared. Call /candles first.")

    df = CACHE[key]["candles_df"]

    token = name.lower()
    if token not in SUPPORTED_INDICATORS:
        raise HTTPException(status_code=400, detail=f"Unsupported indicator: {name}")

    cache_key = f"{token}:{period}" if period else token
    if cache_key in CACHE[key]:
        return {"name": token, "period": period, "values": CACHE[key][cache_key]}

    if token == "ema":
        if period is None:
            raise HTTPException(status_code=400, detail="EMA requires 'period'")
        vals = _ema(df["close"], period)
        data = _series_to_kv(df["time"], vals, dropna=True)
    elif token == "rsi":
        if period is None:
            raise HTTPException(status_code=400, detail="RSI requires 'period'")
        vals = _rsi(df["close"], period)
        data = _series_to_kv(df["time"], vals, dropna=True)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported indicator: {name}")

    CACHE[key][cache_key] = data
    return {"name": token, "period": period, "values": data}

# 起動例:
# uvicorn backend.indicators_api:app --reload --port 8000
