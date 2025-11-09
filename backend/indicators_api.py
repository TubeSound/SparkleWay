# backend/indicators_api.py (CSV/MT5 別実装のインジ計算・DF一体キャッシュ版)
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import os
from datetime import datetime

app = FastAPI(title="SparkleWay CSV + MT5 Indicators API")

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

# 既定で事前計算するインジ（必要なら編集）
DEFAULT_PRECOMPUTE = ["ema:20", "ema:50", "atr:14"]

# ---------- JSON 安全 utility（Series -> [{time,value}]） ----------
def _series_to_kv(time_s: pd.Series, val_s: pd.Series, dropna: bool = False) -> List[Dict[str, Any]]:
    dfv = pd.DataFrame({"time": time_s, "value": val_s}).copy()
    dfv["time"] = pd.to_numeric(dfv["time"], errors="coerce").astype("Int64")
    if dropna:
        dfv = dfv.loc[dfv["value"].notna()]
    dfv = dfv.loc[dfv["time"].notna()]
    out: List[Dict[str, Any]] = []
    for t, v in zip(dfv["time"].astype(int), dfv["value"]):
        out.append({"time": int(t), "value": (float(v) if pd.notna(v) else None)})
    return out

# ---------------- CSV 検出 ----------------
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

# ---------------- 取り込み（CSV / MT5） ----------------

def _jststr2timestamp(df: pd.DataFrame) -> List[int]:
    """CSVの 'jst'（ISO文字列, +09:00付き想定）→ UTC epoch 秒"""
    ts_list: List[int] = []
    for t_str in df["jst"]:
        dt = datetime.fromisoformat(t_str)  # ex. 2025-11-08 06:00:00+09:00
        ts_list.append(int(dt.timestamp()))  # UTC epoch
    return ts_list


def _load_ohlc_from_file(csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV read error: {e}")

    need = {"open", "high", "low", "close"}
    if not need.issubset({c.lower() for c in df.columns}):
        raise HTTPException(status_code=400, detail="CSV must contain open, high, low, close columns")

    df = df.rename(columns={c: c.lower() for c in df.columns})

    # 時間は epoch(UTC 秒)
    if "jst" in df.columns:
        df["time"] = _jststr2timestamp(df)
    elif "time" in df.columns:
        # time が epoch で来ている場合を許容
        df["time"] = pd.to_numeric(df["time"], errors="coerce").astype("Int64")
    else:
        raise HTTPException(status_code=400, detail="CSV must have 'jst' or 'time' column")

    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["time", "open", "high", "low", "close"]).copy()
    df["time"] = df["time"].astype(int)
    df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    return df[["time", "open", "high", "low", "close"]]


def _load_ohlc_from_mt5(symbol: str, timeframe: str, length: int) -> pd.DataFrame:
    """
    ★あなた実装ポイント★
    - MetaTrader5 から `length` 本の TOHLCV を取得して DataFrame を返す
    - 必須列: time(秒, UTC epoch), open, high, low, close
    - 必要ならここで並び順・重複も整えること
    """
    raise NotImplementedError("Implement `_load_ohlc_from_mt5()` to fetch data from MetaTrader5")

# ---------------- インジ計算（共通プリミティブ） ----------------

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

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # True Range
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

SUPPORTED_INDICATORS = {"ema", "rsi", "atr"}

# ---------------- インジ計算（CSV版 / MT5版） ----------------

def _compute_indicators_csv(df: pd.DataFrame, tokens: List[str]) -> pd.DataFrame:
    """CSV 由来データに対するインジ計算（列として df に追加）"""
    for token in tokens:
        t = token.strip().lower()
        if not t:
            continue
        if t.startswith("ema:"):
            p = int(t.split(":", 1)[1])
            df[f"ema_{p}"] = _ema(df["close"], p)
        elif t.startswith("rsi:"):
            p = int(t.split(":", 1)[1])
            df[f"rsi_{p}"] = _rsi(df["close"], p)
        elif t.startswith("atr:"):
            p = int(t.split(":", 1)[1])
            df[f"atr_{p}"] = _atr(df, p)
        else:
            # 未対応はスキップ（必要なら raise）
            continue
    return df


def _compute_indicators_mt5(df: pd.DataFrame, tokens: List[str]) -> pd.DataFrame:
    """MT5 由来データに対するインジ計算（列として df に追加）
    ※ここはあなたが好きなロジック／ライブラリで差し替えてOK。
    今は CSV と同等の計算をデフォルト実装として仮置き。
    """
    # TODO: あなたの MT5 計算ロジックに置換してね
    return _compute_indicators_csv(df, tokens)

# ---------------- 包括 API: データ取得 + インジ計算 + キャッシュ ----------------

def retrieve_data(symbol: str, timeframe: str, length: int, how: str = "csv", precompute: Optional[List[str]] = None) -> pd.DataFrame:
    """
    1) TOHLC を取得（how='csv' or 'mt5'）
    2) precompute のインジを **列として** 追加（データソースに応じた関数で）
    3) 丸ごと DataFrame を CACHE に格納
    4) 末尾 length 本に制限して返却
    """
    pre_tokens = precompute if precompute is not None else DEFAULT_PRECOMPUTE

    # 1) load
    if how == "csv":
        csv_path = _resolve_csv(symbol, timeframe)
        df = _load_ohlc_from_file(csv_path)
    elif how == "mt5":
        df = _load_ohlc_from_mt5(symbol, timeframe, length)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported 'how': {how}")

    # 2) compute indicators (by source)
    if how == "csv":
        df = _compute_indicators_csv(df, pre_tokens)
    elif how == "mt5":
        df = _compute_indicators_mt5(df, pre_tokens)

    # 3) cache whole df (full length before trim)
    key = _dataset_key(symbol, timeframe)
    CACHE[key] = CACHE.get(key, {})
    CACHE[key]["df"] = df.copy()

    # 4) tail trim for response
    if length:
        df = df.iloc[-length:].copy()

    return df

# ---------------- Endpoints ----------------

@app.get("/candles")
def get_candles(
    symbol: str = Query(..., description="シンボル例: JP225, NASDAQ, USDJPY など（ファイル名に含まれている語）"),
    timeframe: str = Query(..., description="1m|5m|15m|1h|1d など"),
    length: int = Query(2000, ge=1, le=200000, description="返すバー数（末尾から）"),
    indicators: Optional[str] = Query(None, description="追加で事前計算するインジ。例: ema:20,ema:50,rsi:14,atr:14"),
    how: str = Query("csv", description="'csv' or 'mt5'"),
):
    # まず retrieve で既定インジを DF 列として付与 → CACHE['df'] に丸ごと保存
    df = retrieve_data(symbol, timeframe, length, how=how)

    # 追加指定があれば上乗せ計算（DF列として追記し、CACHE も更新）
    key = _dataset_key(symbol, timeframe)
    if indicators:
        more = [t for t in indicators.split(',') if t.strip()]
        if how == "csv":
            CACHE[key]["df"] = _compute_indicators_csv(CACHE[key]["df"], more)
        else:
            CACHE[key]["df"] = _compute_indicators_mt5(CACHE[key]["df"], more)
        # レスポンス DF にも反映
        df = CACHE[key]["df"].iloc[-length:].copy()

    # 出力は従来通り OHLC のみ（互換性保持）。必要なら 'columns' メタを返す。
    candles = df[["time", "open", "high", "low", "close"]].to_dict(orient="records")
    indicator_cols = [c for c in df.columns if c not in ("time", "open", "high", "low", "close")]
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "length": len(candles),
        "candles": candles,
        "columns": indicator_cols,  # 追加済みのインジ列名（例: ['ema_20','rsi_14']）
    }


@app.get("/indicator")
def get_indicator(
    symbol: str = Query(...),
    timeframe: str = Query(...),
    name: str = Query(..., description="インジ名: ema|rsi|atr"),
    period: Optional[int] = Query(None, ge=1, description="期間: ema/rsi/atr 用"),
):
    key = _dataset_key(symbol, timeframe)
    if key not in CACHE or "df" not in CACHE[key]:
        raise HTTPException(status_code=404, detail="Candles not prepared. Call /candles first.")

    df = CACHE[key]["df"]
    token = name.lower()
    if token not in SUPPORTED_INDICATORS:
        raise HTTPException(status_code=400, detail=f"Unsupported indicator: {name}")

    # 列名規約: ema_20, rsi_14, atr_14
    if period is None:
        raise HTTPException(status_code=400, detail=f"{token.upper()} requires 'period'")
    col = f"{token}_{period}"

    # なければここで追計算して列追加（CSV を既定とする。必要に応じ how を CACHE に保存して切替可）
    if col not in df.columns:
        # 既存 DF の由来（csv/mt5）を記録したければ、retrieve_data 内で CACHE[key]['source'] を持たせる実装にして切替
        df = _compute_indicators_csv(df, [f"{token}:{period}"])  # デフォルト実装
        CACHE[key]["df"] = df

    series = _series_to_kv(df["time"], df[col], dropna=True)
    return {"name": token, "period": period, "values": series}


def test1():    
    df = retrieve_data('JP225', 'M1', 500, how='csv')
    print(df)

def test2():
    pass

def main():
    pass    


if __name__ == '__main__':
    test1()
# 起動例:
# uvicorn backend.indicators_api:app --reload --port 8000
