# backend/indicators_api.py (トークン直指定版: 'ema20' / 'sma200' / 'atr' など)
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# ---- あなたのライブラリ (MT5 連携) ----
sys.path.append(os.path.join(os.path.dirname(__file__), 'libs'))
from common import Columns  # noqa: F401 (将来利用)
from mt5_trade import Mt5Trade
from montblanc import MontblancParam, Montblanc  # noqa: F401 (将来利用)

mt5 = Mt5Trade(3, 2, 11, 1, 3.0)
mt5.connect()

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

# 既定の事前計算インジ（フロントと合わせて小文字トークンで統一）
DEFAULT_PRECOMPUTE = ["upper", "lower", "atr", "entries", "exits", "counts"]



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
        dt = datetime.fromisoformat(t_str)  # 例: 2025-11-08 06:00:00+09:00
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
    """あなたの MT5 実装に合わせて取得。必須列: time(open/high/low/close)。
    ここでは jst(datetime) を epoch(UTC秒) に変換して 'time' 列にする。"""
    df = mt5.get_rates(symbol, timeframe, length)
    timestamp = []
    for jst in df['jst']:
        timestamp.append(int(jst.timestamp()))
    df['time'] = timestamp
    # 必須列の存在を軽く保証
    need = {"time", "jst", "open", "high", "low", "close"}
    if not need.issubset(df.columns):
        raise HTTPException(status_code=500, detail="MT5 data must contain time/open/high/low/close")
    # ソート＋重複排除
    df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    return df[["time", "jst", "open", "high", "low", "close"]]

# ---------------- インジ計算（共通プリミティブ） ----------------

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()

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
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

# ---------------- トークン解釈 ----------------
# 例: 'ema20' → ("ema", 20) / 'sma200' → ("sma", 200) / 'atr' → ("atr", 14) / 'atr14' → ("atr", 14)

def _parse_token_simple(token: str) -> Tuple[str, Optional[int]]:
    t = token.strip().lower()
    if not t:
        return "", None
    if t.startswith("ema"):
        p = int(t[3:]) if t[3:].isdigit() else 20
        return "ema", p
    if t.startswith("sma"):
        p = int(t[3:]) if t[3:].isdigit() else 20
        return "sma", p
    if t.startswith("rsi"):
        p = int(t[3:]) if t[3:].isdigit() else 14
        return "rsi", p
    if t.startswith("atr"):
        p = int(t[3:]) if t[3:].isdigit() else 14
        return "atr", p
    return t, None

# ---------------- インジ計算（CSV版 / MT5版） ----------------

def _compute_indicators_csv(symbol, df: pd.DataFrame, tokens: List[str]) -> pd.DataFrame:
    """CSV 由来データに対するインジ計算（列として df に **トークン名で** 追加）
    例: 'ema20' 列, 'sma200' 列, 'atr' 列 など。"""
    for token in tokens:
        t = (token or "").strip().lower()
        if not t:
            continue
        name, period = _parse_token_simple(t)
        if name == "ema" and period:
            df[t] = _ema(df["close"], period)
        elif name == "sma" and period:
            df[t] = _sma(df["close"], period)
        elif name == "rsi" and period:
            df[t] = _rsi(df["close"], period)
        elif name == "atr":
            df[t] = _atr(df, period or 14)
        else:
            # 未対応はスキップ（必要なら raise）
            continue
    return df

# マーカー生成
def build_signal_markers(df: pd.DataFrame) -> (List[Dict[str, Any]], List[Dict[str, Any]]):
    """df 末尾2本の close を (time, y, type='dot') で返す"""
    if df is None or df.empty:
        return [], []
    time = df['time'].to_list()
    entries = df['entries'].to_list()
    cl = df['close'].to_list()
    buy: List[Dict[str, Any]] = []
    sell: List[Dict[str, Any]] = []
    for t, c, e in zip(time, cl, entries):
        t = int(t)
        y = float(c)
        if e == 1:
            buy.append({"time": t, "y": y, "type": "dot"})
        elif e == -1:
            sell.append({"time": t, "y": y, "type": "dot"})
    return buy, sell

def load_params(symbol, ver, volume, position_max):
    def array_str2int(s):
        i = s.find('[')
        j = s.find(']')
        v = s[i + 1: j]
        return float(v)
    strategy = 'Montblanc'
    #print( os.getcwd())
    path = f'./param/{strategy}_v{ver}_best_trade_params.xlsx'
    df = pd.read_excel(path)
    df = df[df['symbol'] == symbol]
    params = []
    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        param = MontblancParam.load_from_dic(row)
        param.volume = volume
        param.position_max = position_max
        params.append(param)  
    return params

def _compute_indicators_mt5(symbol: str, df: pd.DataFrame, tokens: List[str]) -> pd.DataFrame:
    #print('indicator ', df.columns, tokens)
    df_out = df.copy()
    params = load_params(symbol, "2", 0, 0)
    montblanc = Montblanc(symbol, params[0])
    montblanc.calc(df_out)    
    for token in tokens:
        token = token.strip().lower()
        if token == 'upper':
            df_out[token] = montblanc.upper_minor
        elif token == 'lower':
            df_out[token] = montblanc.lower_minor
        elif token == 'atr':
            df_out[token] = montblanc.atr
        elif token == "entries":
            df_out[token] = montblanc.entries
        elif token == "exits":
            df_out[token] = montblanc.exits
        elif token == "counts":
            df_out[token] = montblanc.update_counts
    #print(len(df_out), df_out.columns)
    return df_out
   

# ---------------- 包括 API: データ取得 + インジ計算 + キャッシュ ----------------

def retrieve_data(symbol: str, timeframe: str, length: int, how: str = "csv", precompute: Optional[List[str]] = None) -> pd.DataFrame:
    """
    1) TOHLC を取得（how='csv' or 'mt5'）
    2) precompute のトークン（'ema20', 'sma200', 'atr' など）を **列として** 追加
    3) 丸ごと DataFrame を CACHE に格納
    4) 末尾 length 本に制限して返却
    """
    pre_tokens = [t.strip().lower() for t in (precompute if precompute is not None else DEFAULT_PRECOMPUTE) if t.strip()]

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
        df = _compute_indicators_csv(symbol, df, pre_tokens)
    elif how == "mt5":
        df = _compute_indicators_mt5(symbol, df, pre_tokens)

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
    symbol: str = Query(..., description="シンボル例: JP225, XAUUSD, USDJPY など"),
    timeframe: str = Query(..., description="1m|5m|15m|1h|1d / M1 など"),
    length: int = Query(2000, ge=1, le=200000, description="返すバー数（末尾から）"),
    indicators: Optional[str] = Query(None, description="追加で事前計算するインジ。例: ema20,ema200,atr,rsi14"),
    how: str = Query("mt5", description="'csv' or 'mt5'"),
):
    # retrieve で既定インジを DF 列として付与 → CACHE['df'] に丸ごと保存
    df = retrieve_data(symbol, timeframe, length, how=how)

    # 追加指定があれば上乗せ計算（トークンで列追記し、CACHE も更新）
    key = _dataset_key(symbol, timeframe)
    if indicators:
        more = [t.strip().lower() for t in indicators.split(',') if t.strip()]
        if how == "csv":
            CACHE[key]["df"] = _compute_indicators_csv(symbol, CACHE[key]["df"], more)
        else:
            CACHE[key]["df"] = _compute_indicators_mt5(symbol, CACHE[key]["df"], more)
        df = CACHE[key]["df"].iloc[-length:].copy()
        
    # --- 最後2本の終値マーカーを生成してキャッシュ ---
    key = _dataset_key(symbol, timeframe)  # 既存で使っているキー
    buy, sell = build_signal_markers(df)
    CACHE[key] = CACHE.get(key, {})
    CACHE[key].setdefault("marker_points", {})
    CACHE[key]["marker_points"]["buy"] = buy
    CACHE[key]["marker_points"]["sell"] = sell
    

    candles = df[["time", "open", "high", "low", "close"]].to_dict(orient="records")
    indicator_cols = [c for c in df.columns if c not in ("time", "open", "high", "low", "close")]
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "length": len(candles),
        "candles": candles,
        "columns": indicator_cols,  # 例: ['ema20','ema200','atr']
    }


@app.get("/indicator")
def get_indicator(
    symbol: str = Query(...),
    timeframe: str = Query(...),
    name: str = Query(..., description="インジトークン: ema20|sma200|rsi14|atr など。period パラメータは不要。"),
):
    key = _dataset_key(symbol, timeframe)
    if key not in CACHE or "df" not in CACHE[key]:
        raise HTTPException(status_code=404, detail="Candles not prepared. Call /candles first.")

    df = CACHE[key]["df"]
    token = (name or "").strip().lower()
    if not token:
        raise HTTPException(status_code=400, detail="'name' is required")

    if token not in df.columns:
        # 未計算ならここで追計算（CSV版を既定。必要なら CACHE[key]['source'] を保持して出し分け）
        df = _compute_indicators_csv(symbol, df, [token])
        CACHE[key]["df"] = df
        if token not in df.columns:
            raise HTTPException(status_code=400, detail=f"Unsupported indicator token: {name}")

    series = _series_to_kv(df["time"], df[token], dropna=True)
    return {"name": token, "values": series}

@app.get("/markers")
def get_markers(
    symbol: str = Query(...),
    timeframe: str = Query(...),
    dataset: Optional[str] = Query("buy", description="マーカーデータ名"),
):
    key = _dataset_key(symbol, timeframe)
    if key not in CACHE or "marker_points" not in CACHE[key]:
        return {"dataset": dataset, "points": []}
    points = CACHE[key]["marker_points"].get(dataset or "buy", [])
    # 形式: [{time: epochSec, y: number, type: 'dot'|...}]
    return {"dataset": dataset or "buy", "points": points}


# 起動例:
# uvicorn backend.indicators_api:app --reload --port 8000


def test():
    symbol = 'JP225'
    df = retrieve_data(symbol, 'M1', 1000, how='mt5')
    print(df.columns, len(df))
    df_c = _compute_indicators_mt5(symbol, df, ['upper', 'lower', 'atr', 'counts'])
    print(df_c.columns, len(df_c))
    

if __name__ == "__main__":
    test()