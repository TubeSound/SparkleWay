# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Literal, Optional
from pathlib import Path
import pandas as pd

app = FastAPI(title="SparkleWay CSV API")

# ▼ フロント（Next.js Dev）のオリジンを許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).parent / "data"  # CSV置き場（backend/data 配下）
DATA_DIR.mkdir(exist_ok=True)


def _safe_path(filename: str) -> Path:
    """dataディレクトリ配下のファイルに解決。ディレクトリトラバーサルを防ぐ。"""
    p = (DATA_DIR / filename).resolve()
    try:
        p.relative_to(DATA_DIR.resolve())
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="CSV not found")
    return p


def _load_candles_from_csv(csv_path: Path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV read error: {e}")

    # 必須列チェック（time または jst のどちらか + OHLC）
    needed = {"open", "high", "low", "close"}
    if not needed.issubset(set(df.columns)):
        raise HTTPException(status_code=400, detail="CSV must contain open,high,low,close columns")

    time_col: Optional[str] = None
    if "time" in df.columns:
        time_col = "time"  # UTC推奨
    elif "jst" in df.columns:
        time_col = "jst"   # JST(+09:00)でもOK
    else:
        raise HTTPException(status_code=400, detail="CSV must contain time or jst column")

    # pandasで時刻をUTCエポック秒へ（Lightweight Chartsの推奨形式）
    ts = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    if ts.isna().all():
        raise HTTPException(status_code=400, detail=f"Cannot parse datetime column: {time_col}")

    df = df.loc[~ts.isna()].copy()
    df["epoch"] = ts[~ts.isna()].astype("int64") // 10**9  # 秒

    records = (
        df[["epoch", "open", "high", "low", "close"]]
        .rename(columns={"epoch": "time"})
        .to_dict(orient="records")
    )
    return records


@app.get("/candles")
def get_candles(
    file: str,  # 例: NSDQ_M1_2025-07-01-4.csv（backend/data に置く）
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    limit: Optional[int] = 2000,
):
    """
    指定CSVからOHLCを読み込み、Lightweight Charts 互換の配列を返す。
    返却: [{ time: 1719816000, open:..., high:..., low:..., close:... }, ...]
    """
    csv_path = _safe_path(file)
    candles = _load_candles_from_csv(csv_path)
    if limit and isinstance(limit, int) and limit > 0:
        candles = candles[-limit:]
    return {
        "file": file,
        "symbol": symbol,
        "timeframe": timeframe,
        "count": len(candles),
        "candles": candles,
    }


# 実行例: uvicorn backend.main:app --reload --port 8000
