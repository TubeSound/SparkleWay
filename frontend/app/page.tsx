'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import type {
  IChartApi,
  ISeriesApi,
  UTCTimestamp,
  CandlestickData,
  LineData,
  Time,
  BusinessDay,
} from 'lightweight-charts';
import {
  createChart,
  CandlestickSeries,
  LineSeries,
  createSeriesMarkers,
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
  US100: 'US100',
  US330: 'US30',
  XAUUSD: 'XAUUSD',
  USDJPY: 'USDJPY',
};
const TIMEFRAMES: Record<string, string> = {
  M1: '1分',
  M5: '5分',
  M15: '15分',
  H1: '1時間',
  D1: '日足',
};

// ========= インディケータ（点表示） =========
// chart: 0=Main, 1=Sub1, 2=Sub2
const INDICATORS = {
  upper: { name: 'upper', color: '#eb9ae7ff', chart: 0, radius: 3 },
  lower: { name: 'lower', color: '#88e9ecff', chart: 0, radius: 3 },
  atr: { name: 'atr', color: '#044ed6ff', chart: 1, radius: 2 }, // 既存のサブ1
  counts: { name: 'counts', color: '#e01191ff', chart: 2, radius: 2 }, // 新しくサブ2へ（バックエンドで values を返す前提）
} as const;

type IndicatorKey = keyof typeof INDICATORS;

// ========= マーカー設定（v5: createSeriesMarkers を使用） =========
// dataset: バックエンドのデータ名
// chart: 0=メイン(ローソクへ), 1=サブ1(不可視ラインへ), 2=サブ2(不可視ラインへ)
// shape: 'arrowUp'|'arrowDown'|'circle'|'square'
// position: 'aboveBar'|'belowBar'|'inBar'
const MARKERS = {
  buy: { dataset: 'buy', color: '#3b82f6', shape: 'arrowUp', position: 'belowBar', chart: 0 },
  sell: { dataset: 'sell', color: '#f0141fff', shape: 'arrowDown', position: 'aboveBar', chart: 0 },
} as const;

type MarkerKey = keyof typeof MARKERS;

// --- 型 ---
type RawPoint = { time: number; y?: number; type?: string };

type MarkerItem = {
  time: UTCTimestamp;
  color?: string;
  shape?: any;
  position?: any;
  text?: string;
  size?: number;
};

export default function Page() {
  const [symbol, setSymbol] = useState<string>('JP225');
  const [timeframe, setTimeframe] = useState<string>('M1');
  const [length, setLength] = useState<number>(500);

  const [enabledIndi, setEnabledIndi] = useState<Record<IndicatorKey, boolean>>({ upper: true, lower: true, atr: true, rsi14: true });
  const [enabledMarker, setEnabledMarker] = useState<Record<MarkerKey, boolean>>({ buy: true, sell: true });

  const enabledIndiJSON = useMemo(() => JSON.stringify(enabledIndi), [enabledIndi]);
  const enabledMarkerJSON = useMemo(() => JSON.stringify(enabledMarker), [enabledMarker]);

  // ---- DOM コンテナ ----
  const mainEl = useRef<HTMLDivElement | null>(null);
  const sub1El = useRef<HTMLDivElement | null>(null);
  const sub2El = useRef<HTMLDivElement | null>(null);

  // ---- Chart API 参照（0=Main, 1=Sub1, 2=Sub2）
  const chartsRef = useRef<Array<IChartApi | null>>([null, null, null]);

  const candleSeries = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const indiSeries = useRef<Partial<Record<IndicatorKey, ISeriesApi<'Line'>>>>({});
  const markerHostSeries = useRef<Partial<Record<MarkerKey, ISeriesApi<'Line'>>>>({});

  // markers primitive 保持（Main は1つ、Sub はキー毎）
  const mainMarkersPrimitive = useRef<ReturnType<typeof createSeriesMarkers> | null>(null);
  const subMarkersPrimitive = useRef<Partial<Record<MarkerKey, ReturnType<typeof createSeriesMarkers>>>>({});

  // --- JST フォーマット ---
  const jstTickFmt = useMemo(
    () => new Intl.DateTimeFormat('ja-JP', { timeZone: 'Asia/Tokyo', hour12: false, day: '2-digit', hour: '2-digit', minute: '2-digit' }),
    [],
  );
  const jstFullFmt = useMemo(
    () =>
      new Intl.DateTimeFormat('ja-JP', {
        timeZone: 'Asia/Tokyo',
        hour12: false,
        year: '2-digit',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      }),
    [],
  );

  const timeToDate = (t: Time): Date => {
    if (typeof t === 'number') return new Date(t * 1000);
    const bd = t as BusinessDay;
    return new Date(Date.UTC(bd.year, (bd as any).month - 1, bd.day));
  };
  const formatTimeJST = (t: Time) => jstFullFmt.format(timeToDate(t));

  // ---- 初期化 ----
  useEffect(() => {
    if (!mainEl.current) return;

    const makeChart = (el: HTMLDivElement, height: number, textColor: string) => {
      const c = createChart(el, {
        height,
        layout: { background: { color: 'white' }, textColor },
        timeScale: { timeVisible: true, secondsVisible: true },
        rightPriceScale: { borderVisible: false },
        grid: { vertLines: { visible: false }, horzLines: { visible: true } },
        localization: { locale: 'ja-JP', timeFormatter: formatTimeJST },
      });
      c.applyOptions({
        timeScale: { timeVisible: true, secondsVisible: true, tickMarkFormatter: (t) => jstTickFmt.format(timeToDate(t)) },
        localization: { timeFormatter: (t) => jstFullFmt.format(timeToDate(t)) },
      });
      return c;
    };

    // main
    const main = makeChart(mainEl.current, 520, '#111827');
    chartsRef.current[0] = main;

    const candle = main.addSeries(CandlestickSeries, {
      upColor: '#22c55e', downColor: '#ef4444', borderUpColor: '#22c55e', borderDownColor: '#ef4444', wickUpColor: '#22c55e', wickDownColor: '#ef4444',
    });
    candleSeries.current = candle;

    // sub1
    if (sub1El.current) chartsRef.current[1] = makeChart(sub1El.current, 160, '#374151');
    // sub2
    if (sub2El.current) chartsRef.current[2] = makeChart(sub2El.current, 160, '#374151');

    const handleResize = () => {
      const w0 = mainEl.current?.clientWidth ?? undefined;
      const w1 = sub1El.current?.clientWidth ?? undefined;
      const w2 = sub2El.current?.clientWidth ?? undefined;
      if (chartsRef.current[0] && w0) chartsRef.current[0]!.applyOptions({ width: w0 });
      if (chartsRef.current[1] && w1) chartsRef.current[1]!.applyOptions({ width: w1 });
      if (chartsRef.current[2] && w2) chartsRef.current[2]!.applyOptions({ width: w2 });
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      mainMarkersPrimitive.current?.setMarkers?.([] as any);
      Object.values(subMarkersPrimitive.current).forEach((p) => p?.setMarkers?.([] as any));
      chartsRef.current.forEach((c) => c?.remove());
      chartsRef.current = [null, null, null];
    };
  }, []);

  const getChart = (index: number) => chartsRef.current[index] ?? null;

  const ensureDots = (key: IndicatorKey): ISeriesApi<'Line'> | null => {
    if (indiSeries.current[key]) return indiSeries.current[key]!;
    const cfg = INDICATORS[key];
    const chart = getChart(cfg.chart);
    if (!chart) return null;
    const s = chart.addSeries(LineSeries, {
      color: cfg.color,
      lineWidth: 0,
      priceLineVisible: false,
      lastValueVisible: false,
      pointMarkersVisible: true,
      pointMarkersRadius: cfg.radius ?? 2,
    });
    indiSeries.current[key] = s;
    return s;
  };

  const ensureMarkerHost = (key: MarkerKey): ISeriesApi<'Line'> | null => {
    if (markerHostSeries.current[key]) return markerHostSeries.current[key]!;
    const cfg = MARKERS[key];
    const chart = getChart(cfg.chart);
    if (!chart) return null;
    const s = chart.addSeries(LineSeries, { color: '#00000000', lineWidth: 1, visible: false });
    markerHostSeries.current[key] = s;
    return s;
  };

  // ---- 取得 ----
  async function fetchCandles() {
    const url = new URL('http://localhost:8000/candles');
    url.searchParams.set('symbol', symbol);
    url.searchParams.set('timeframe', timeframe);
    url.searchParams.set('length', String(length));
    const preTokens = (Object.keys(INDICATORS) as IndicatorKey[])
      .filter((k) => enabledIndi[k])
      .map((k) => INDICATORS[k].name)
      .join(',');
    if (preTokens) url.searchParams.set('indicators', preTokens);

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
    candleSeries.current?.setData(candles);
    getChart(0)?.timeScale().scrollToPosition(0, false);
    getChart(1)?.timeScale().scrollToPosition(0, false);
    getChart(2)?.timeScale().scrollToPosition(0, false);
  }

  const asDotsData = (seriesData: Array<{ time: number; value: number | null }>): LineData[] =>
    seriesData.map((d) => ({ time: d.time as UTCTimestamp, value: d.value == null || Number.isNaN(d.value as number) ? (undefined as any) : (d.value as number) }));

  async function fetchIndicator(k: IndicatorKey) {
    if (!enabledIndi[k]) {
      indiSeries.current[k]?.setData([]);
      return;
    }
    const cfg = INDICATORS[k];
    const url = new URL('http://localhost:8000/indicator');
    url.searchParams.set('symbol', symbol);
    url.searchParams.set('timeframe', timeframe);
    url.searchParams.set('name', cfg.name);
    const res = await fetch(url.toString());
    if (!res.ok) return;
    const json = await res.json();
    const raw = (json.values ?? []) as Array<{ time: number; value: number | null }>;
    const s = ensureDots(k);
    s?.setData(asDotsData(raw));
  }

  async function fetchMarkerDataset(k: MarkerKey) {
    if (!enabledMarker[k]) {
      if (k in subMarkersPrimitive.current) subMarkersPrimitive.current[k]?.setMarkers?.([] as any);
      if (k === 'buy' || k === 'sell') mainMarkersPrimitive.current?.setMarkers?.([] as any);
      return [] as MarkerItem[];
    }

    const cfg = MARKERS[k];
    const url = new URL('http://localhost:8000/markers');
    url.searchParams.set('symbol', symbol);
    url.searchParams.set('timeframe', timeframe);
    url.searchParams.set('dataset', cfg.dataset);
    const res = await fetch(url.toString());
    if (!res.ok) return [] as MarkerItem[];
    const json = await res.json();
    const pts = (json.points ?? []) as RawPoint[];

    const items: MarkerItem[] = pts.map((p) => ({
      time: p.time as UTCTimestamp,
      color: cfg.color,
      shape: (cfg as any).shape,
      position: (cfg as any).position,
      size: 3,
    }));

    if (cfg.chart === 1 || cfg.chart === 2) {
      const host = ensureMarkerHost(k);
      if (host) {
        if (!subMarkersPrimitive.current[k]) subMarkersPrimitive.current[k] = createSeriesMarkers(host, [] as any);
        subMarkersPrimitive.current[k]?.setMarkers?.(items as any);
      }
    }

    return items;
  }

  async function loadAll() {
    await fetchCandles();
    await Promise.allSettled((Object.keys(INDICATORS) as IndicatorKey[]).map((k) => fetchIndicator(k)));

    const all = await Promise.all((Object.keys(MARKERS) as MarkerKey[]).map((k) => fetchMarkerDataset(k)));
    const mergedForMain = all.flat();

    if (candleSeries.current) {
      if (!mainMarkersPrimitive.current) mainMarkersPrimitive.current = createSeriesMarkers(candleSeries.current, [] as any);
      mainMarkersPrimitive.current.setMarkers?.(mergedForMain as any);
    }
  }

  // ===== ポーリング =====
  const pollMs = 5_000;
  const busy = useRef(false);
  const timer = useRef<number | null>(null);
  useEffect(() => {
    const run = async () => {
      if (busy.current) return;
      busy.current = true;
      try {
        await loadAll();
      } finally {
        busy.current = false;
      }
    };
    run();
    if (timer.current) window.clearInterval(timer.current);
    timer.current = window.setInterval(run, pollMs) as any;
    return () => {
      if (timer.current) window.clearInterval(timer.current);
      timer.current = null;
    };
  }, [symbol, timeframe, length, enabledIndiJSON, enabledMarkerJSON]);

  const labelOf = (k: IndicatorKey) => INDICATORS[k].name.toUpperCase();

  return (
    <main className="flex min-h-screen items-start justify-start p-6 bg-gray-50">
      <section className="flex flex-col gap-4 p-4 bg-white rounded-lg shadow-md w-[300px]">
        <h1 className="text-xl font-semibold text-gray-800">Montblanc</h1>

        <div className="flex flex-col gap-1">
          <label className="text-sm text-gray-600">Symbol</label>
          <Select value={symbol} onValueChange={setSymbol}>
            <SelectTrigger><SelectValue placeholder="Symbol" /></SelectTrigger>
            <SelectContent>
              {Object.entries(SYMBOLS).map(([k, v]) => (<SelectItem key={k} value={k}>{v}</SelectItem>))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-sm text-gray-600">Timeframe</label>
          <Select value={timeframe} onValueChange={setTimeframe}>
            <SelectTrigger><SelectValue placeholder="TF" /></SelectTrigger>
            <SelectContent>
              {Object.entries(TIMEFRAMES).map(([k, v]) => (<SelectItem key={k} value={k}>{v}</SelectItem>))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-sm text-gray-600">Length</label>
          <Input type="number" value={length} onChange={(e) => setLength(Number(e.target.value || 0))} />
        </div>

        <div className="mt-2 space-y-2">
          <div className="text-sm font-medium text-gray-700">Indicators</div>
          {(Object.keys(INDICATORS) as IndicatorKey[]).map((k) => (
            <div key={k} className="flex items-center justify-between">
              <Label htmlFor={k}>{labelOf(k)}<span className="ml-2 text-xs text-gray-400">{['Main', 'Sub1', 'Sub2'][INDICATORS[k].chart]}</span></Label>
              <Switch id={k} checked={enabledIndi[k]} onCheckedChange={(v) => setEnabledIndi((p) => ({ ...p, [k]: v }))} />
            </div>
          ))}
        </div>

        <div className="mt-2 space-y-2">
          <div className="text-sm font-medium text-gray-700">Markers</div>
          {(Object.keys(MARKERS) as MarkerKey[]).map((k) => (
            <div key={k} className="flex items-center justify-between">
              <Label htmlFor={`m-${k}`}>{k.toUpperCase()}<span className="ml-2 text-xs text-gray-400">{['Main', 'Sub1', 'Sub2'][MARKERS[k].chart]}</span></Label>
              <Switch id={`m-${k}`} checked={enabledMarker[k]} onCheckedChange={(v) => setEnabledMarker((p) => ({ ...p, [k]: v }))} />
            </div>
          ))}
        </div>

        <Button onClick={loadAll}>Load</Button>
      </section>

      <section className="flex-1 ml-6 bg-white rounded-lg shadow-md p-4">
        <div ref={mainEl} className="w-full h-[520px]" />
        <div className="mt-2" />
        <div ref={sub1El} className="w-full h-[160px]" />
        <div className="mt-2" />
        <div ref={sub2El} className="w-full h-[160px]" />
      </section>
    </main>
  );
}
