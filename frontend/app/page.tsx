// app/page.tsx
'use client';

import { useEffect, useRef, useState } from 'react';
import type {
  IChartApi,
  ISeriesApi,
  UTCTimestamp,
  CandlestickData,
  LineData,
} from 'lightweight-charts';
import {
  createChart,
  CandlestickSeries,
  LineSeries,
} from 'lightweight-charts';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

// ---------- UI 設定 ----------
const SYMBOLS: Record<string, string> = {
  JP225: 'JP225',
  NASDAQ: 'NASDAQ',
  USDJPY: 'USD/JPY',
};
const TIMEFRAMES: Record<string, string> = {
  '1m': '1分',
  '5m': '5分',
  '15m': '15分',
  '1h': '1時間',
  '1d': '日足',
};

// 表示するインジ名と色（バックエンド側は name/period で応答）
const INDICATORS = {
  ema20: { name: 'ema', period: 20, color: '#0ea5e9' },
  ema50: { name: 'ema', period: 50, color: '#a855f7' },
  rsi14: { name: 'rsi', period: 14, color: '#111827' },
} as const;
type IndicatorKey = keyof typeof INDICATORS;

export default function Home() {
  const [symbol, setSymbol] = useState<string>('JP225');
  const [timeframe, setTimeframe] = useState<string>('1m');
  const [length, setLength] = useState<number>(2000);
  const [enabled, setEnabled] = useState<Record<IndicatorKey, boolean>>({
    ema20: true,
    ema50: true,
    rsi14: false,
  });

  // ---- チャート参照 ----
  const chartContainerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const lineSeriesRef = useRef<Partial<Record<IndicatorKey, ISeriesApi<'Line'>>>>({});
  const rsiContainerRef = useRef<HTMLDivElement | null>(null);
  const rsiChartRef = useRef<IChartApi | null>(null);
  const rsiSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);

  // 1) 先頭の方でフォーマッタを用意
  const jstFormatter = new Intl.DateTimeFormat('ja-JP', {
    timeZone: 'Asia/Tokyo',
    year: '2-digit',
    month: 'numeric',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });

  // BusinessDay or UTCTimestampの両方を受けてJST文字列にする関数
  function formatTimeJST(t: import('lightweight-charts').Time): string {
    if (typeof t === 'number') {
      // UTCTimestamp（秒） -> Date -> JST表示
      return jstFormatter.format(new Date(t * 1000));
    } else {
      // BusinessDay -> UTC midnight -> JST表示
      const ms = Date.UTC(t.year, t.month - 1, t.day);
      return jstFormatter.format(new Date(ms));
    }
  }


  // ---- 初期化 ----
  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      height: 520,
      layout: { background: { color: 'white' }, textColor: '#111827' },
      timeScale: { timeVisible: true, secondsVisible: true },
      rightPriceScale: { borderVisible: false },
      grid: { vertLines: { visible: false }, horzLines: { visible: true } },
      localization: {
        locale: 'ja-JP',
        timeFormatter: formatTimeJST, // ★ここがポイント（JST表示）
      },
    });
    chartRef.current = chart;

    // ★ v5: addSeries(CandlestickSeries, options)
    const candle = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderUpColor: '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    });
    candleSeriesRef.current = candle;

    // ★ v5: addSeries(LineSeries, options)
    lineSeriesRef.current.ema20 = chart.addSeries(LineSeries, {
      color: INDICATORS.ema20.color,
      lineWidth: 2,
    });
    lineSeriesRef.current.ema50 = chart.addSeries(LineSeries, {
      color: INDICATORS.ema50.color,
      lineWidth: 2,
    });

    // RSI用のサブパネル
    if (rsiContainerRef.current) {
      const rsiChart = createChart(rsiContainerRef.current, {
        height: 160,
        layout: { background: { color: 'white' }, textColor: '#374151' },
        timeScale: { timeVisible: true, secondsVisible: true },
        rightPriceScale: { borderVisible: false },
        grid: { vertLines: { visible: false }, horzLines: { visible: true } },
        localization: {
          locale: 'ja-JP',
          timeFormatter: formatTimeJST, // ★ここがポイント（JST表示）
        },
      });
      rsiChartRef.current = rsiChart;
      rsiSeriesRef.current = rsiChart.addSeries(LineSeries, {
        color: INDICATORS.rsi14.color,
        lineWidth: 1,
      });
    }

    const handleResize = () => {
      chart.applyOptions({ width: chartContainerRef.current!.clientWidth });
      if (rsiChartRef.current && rsiContainerRef.current) {
        rsiChartRef.current.applyOptions({
          width: rsiContainerRef.current.clientWidth,
        });
      }
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      rsiChartRef.current?.remove();
    };
  }, []);

  // ---- データ取得 ----
  async function fetchCandles() {
    const url = new URL('http://localhost:8000/candles');
    url.searchParams.set('symbol', symbol);
    url.searchParams.set('timeframe', timeframe);
    url.searchParams.set('length', String(length));
    // 事前計算（サーバ側キャッシュ用）
    const pre = [INDICATORS.ema20, INDICATORS.ema50, INDICATORS.rsi14]
      .map((i) => `${i.name}:${i.period}`)
      .join(',');
    url.searchParams.set('indicators', pre);

    const res = await fetch(url.toString());
    if (!res.ok) throw new Error(await res.text());
    const json = await res.json();

    const candles: CandlestickData[] = (json.candles ?? []).map((d: any) => ({
      time: Number(d.time) as UTCTimestamp,
      open: Number(d.open),
      high: Number(d.high),
      low: Number(d.low),
      close: Number(d.close),
    }));
    candleSeriesRef.current?.setData(candles);
  }

  async function fetchIndicator(key: IndicatorKey) {
    if (!enabled[key]) {
      // 消去
      lineSeriesRef.current[key]?.setData([]);
      if (key === 'rsi14') rsiSeriesRef.current?.setData([]);
      return;
    }
    const cfg = INDICATORS[key];
    const url = new URL('http://localhost:8000/indicator');
    url.searchParams.set('symbol', symbol);
    url.searchParams.set('timeframe', timeframe);
    url.searchParams.set('name', cfg.name);
    if (cfg.period) url.searchParams.set('period', String(cfg.period));

    const res = await fetch(url.toString());
    if (!res.ok) throw new Error(await res.text());
    const json = await res.json();
    const series = (json.values ?? []) as LineData[];

    if (key === 'rsi14') {
      rsiSeriesRef.current?.setData(series);
    } else {
      lineSeriesRef.current[key]?.setData(series);
    }
  }

  async function loadAll() {
    await fetchCandles();
    const tasks: Promise<any>[] = [];
    (Object.keys(INDICATORS) as IndicatorKey[]).forEach((k) =>
      tasks.push(fetchIndicator(k)),
    );
    await Promise.allSettled(tasks);
  }

  return (
    <main className="flex min-h-screen items-start justify-start p-6 bg-gray-50">
      {/* GUI */}
      <section className="flex flex-col gap-4 p-4 bg-white rounded-lg shadow-md w-[280px]">
        <h1 className="text-xl font-semibold text-gray-800">SparkleWay</h1>

        {/* Symbol */}
        <div className="flex flex-col gap-1">
          <label className="text-sm text-gray-600">Symbol</label>
          <Select value={symbol} onValueChange={setSymbol}>
            <SelectTrigger>
              <SelectValue placeholder="Symbol" />
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

        {/* Timeframe */}
        <div className="flex flex-col gap-1">
          <label className="text-sm text-gray-600">Timeframe</label>
          <Select value={timeframe} onValueChange={setTimeframe}>
            <SelectTrigger>
              <SelectValue placeholder="TF" />
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

        {/* Length */}
        <div className="flex flex-col gap-1">
          <label className="text-sm text-gray-600">Length</label>
          <Input
            type="number"
            value={length}
            onChange={(e) => setLength(Number(e.target.value || 0))}
          />
        </div>

        {/* Toggles */}
        <div className="mt-2 space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor="ema20">EMA 20</Label>
            <Switch
              id="ema20"
              checked={enabled.ema20}
              onCheckedChange={(v) =>
                setEnabled((p) => ({ ...p, ema20: v }))
              }
            />
          </div>
          <div className="flex items-center justify-between">
            <Label htmlFor="ema50">EMA 50</Label>
            <Switch
              id="ema50"
              checked={enabled.ema50}
              onCheckedChange={(v) =>
                setEnabled((p) => ({ ...p, ema50: v }))
              }
            />
          </div>
          <div className="flex items-center justify-between">
            <Label htmlFor="rsi14">RSI 14</Label>
            <Switch
              id="rsi14"
              checked={enabled.rsi14}
              onCheckedChange={(v) =>
                setEnabled((p) => ({ ...p, rsi14: v }))
              }
            />
          </div>
        </div>

        <Button onClick={loadAll}>Load</Button>
      </section>

      {/* Chart */}
      <section className="flex-1 ml-6 bg-white rounded-lg shadow-md p-4">
        <div ref={chartContainerRef} className="w-full h-[520px]" />
        <div className="mt-2" />
        <div ref={rsiContainerRef} className="w-full h-[160px]" />
      </section>
    </main>
  );
}
