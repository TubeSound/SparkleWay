// app/page.tsx
'use client';

import { useRef, useState, useEffect } from 'react';
import type {
  IChartApi,
  ISeriesApi,
  CandlestickData,
  UTCTimestamp,
  DeepPartial,
  ChartOptions,
  CandlestickSeriesPartialOptions,
} from 'lightweight-charts';
import { createChart, CandlestickSeries } from 'lightweight-charts';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

// === ここを必要に応じて変更 ===
const BACKEND_BASE = 'http://localhost:8000';
const DEFAULT_BACKEND_FILE = 'NIKKEI_H1_2025_10.csv';

const SYMBOLS: Record<string, string> = {
  USDJPY: 'USD/JPY',
  EURUSD: 'EUR/USD',
  GBPUSD: 'GBP/USD',
  AUDUSD: 'AUD/USD',
};

const TIMEFRAMES: Record<string, string> = {
  '1m': '1分足',
  '5m': '5分足',
  '15m': '15分足',
  '1h': '1時間足',
  '1d': '日足',
};

export default function Home() {
  const chartContainerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);

  const [symbol, setSymbol] = useState<string>('');
  const [timeframe, setTimeframe] = useState<string>('');

  // バックエンドから読むファイル名（テキスト指定）
  const [backendFile, setBackendFile] = useState<string>(DEFAULT_BACKEND_FILE);

  // チャートを作り直す共通関数（あなたの動いていた版に忠実）
  const createOrResetChart = () => {
    if (!chartContainerRef.current) return;

    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
      seriesRef.current = null;
    }

    const chartOptions: DeepPartial<ChartOptions> = {
      width: chartContainerRef.current.clientWidth,
      height: chartContainerRef.current.clientHeight,
      layout: {
        background: { color: '#ffffff' },
        textColor: '#333',
      },
      grid: {
        vertLines: { color: '#e2e8f0' },
        horzLines: { color: '#e2e8f0' },
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
      },
    };

    const chart = createChart(chartContainerRef.current, chartOptions);

    const seriesOptions: CandlestickSeriesPartialOptions = {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderDownColor: '#ef4444',
      borderUpColor: '#22c55e',
      wickDownColor: '#ef4444',
      wickUpColor: '#22c55e',
    };

    const candleSeries = chart.addSeries(CandlestickSeries, seriesOptions);

    chartRef.current = chart;
    seriesRef.current = candleSeries;
  };

  // バックエンドから取得 → チャートにセット
  const loadFromBackend = async () => {
    if (!backendFile) return;

    const url = `${BACKEND_BASE}/candles?file=${encodeURIComponent(
      backendFile,
    )}&limit=2000`;

    const res = await fetch(url);
    if (!res.ok) {
      console.error('Backend error:', await res.text());
      return;
    }
    const json = await res.json();

    const candles: CandlestickData[] = (json.candles ?? []).map((d: any) => ({
      time: Number(d.time) as UTCTimestamp, // FastAPIはepoch秒で返す設計
      open: Number(d.open),
      high: Number(d.high),
      low: Number(d.low),
      close: Number(d.close),
    }));

    if (!candles.length) {
      console.warn('No candles from backend');
      return;
    }

    if (!seriesRef.current) {
      createOrResetChart();
    }
    seriesRef.current?.setData(candles);
  };

  // リサイズ対応（あなたの実装に合わせて）
  useEffect(() => {
    const handleResize = () => {
      if (chartRef.current && chartContainerRef.current) {
        chartRef.current.resize(
          chartContainerRef.current.clientWidth,
          chartContainerRef.current.clientHeight
        );
      }
    };

    // 初回にチャート作成（空のシリーズ）
    createOrResetChart();

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <main className="flex min-h-screen items-stretch justify-start p-6 bg-gray-50 text-gray-800">
      {/* 左カラム：GUI */}
      <section className="flex flex-col gap-4 p-6 bg-white rounded-lg shadow-md w-[280px]">
        <h2 className="text-lg font-semibold border-b pb-2">設定</h2>

        <div className="flex flex-col gap-1.5">
          <label className="text-sm font-medium text-gray-600">通貨ペア</label>
          <Select value={symbol} onValueChange={setSymbol}>
            <SelectTrigger>
              <SelectValue placeholder="選択してください" />
            </SelectTrigger>
            <SelectContent>
              {Object.entries(SYMBOLS).map(([key, label]) => (
                <SelectItem key={key} value={key}>
                  {label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex flex-col gap-1.5">
          <label className="text-sm font-medium text-gray-600">時間足</label>
          <Select value={timeframe} onValueChange={setTimeframe}>
            <SelectTrigger>
              <SelectValue placeholder="選択してください" />
            </SelectTrigger>
            <SelectContent>
              {Object.entries(TIMEFRAMES).map(([key, label]) => (
                <SelectItem key={key} value={key}>
                  {label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* バックエンドCSV指定 */}
        <div className="flex flex-col gap-1.5">
          <label className="text-sm font-medium text-gray-600">CSV（backend/data）</label>
          <div className="flex gap-2">
            <Input
              type="text"
              value={backendFile}
              onChange={(e) => setBackendFile(e.target.value)}
              placeholder="NSDQ_M1_2025-07-01-4.csv"
            />
            <Button onClick={loadFromBackend}>表示</Button>
          </div>
          <p className="text-xs text-gray-500">
            FastAPI 側: <code>/candles?file=&lt;ファイル名&gt;</code> に対応
          </p>
        </div>
      </section>

      {/* 右カラム：チャート */}
      <section className="flex-1 ml-6 bg-white rounded-lg shadow-md p-2">
        <div ref={chartContainerRef} className="w-full h-full min-h-[500px]" />
      </section>
    </main>
  );
}
