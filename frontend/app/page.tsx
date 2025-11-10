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

// インディケータ（点のみを描画したいので lineWidth:0 + pointMarkersVisible:true で描画）
const INDICATORS = {
  upper: { name: 'upper', color: '#e11d48', chart: 0, radius: 3 },
  lower: { name: 'lower', color: '#10b981', chart: 0, radius: 3 },
  atr: { name: 'atr', color: '#111827', chart: 1, radius: 2 }, // ※未実装ならサーバ側で404→握りつぶし
} as const;

type IndicatorKey = keyof typeof INDICATORS;
type IndicatorCfg = (typeof INDICATORS)[IndicatorKey];

export default function Home() {
  const [symbol, setSymbol] = useState<string>('JP225');
  const [timeframe, setTimeframe] = useState<string>('M1');
  const [length, setLength] = useState<number>(1000);

  // トグル初期値（chart=0はtrue、chart=1はfalse開始）
  const [enabled, setEnabled] = useState<Record<IndicatorKey, boolean>>(() => {
    const init = {} as Record<IndicatorKey, boolean>;
    (Object.keys(INDICATORS) as IndicatorKey[]).forEach((k) => {
      init[k] = INDICATORS[k].chart === 0;
    });
    return init;
  });
  const enabledJSON = useMemo(() => JSON.stringify(enabled), [enabled]);

  // ---- チャート参照（0=メイン, 1=サブ）----
  const mainContainerRef = useRef<HTMLDivElement | null>(null);
  const subContainerRef = useRef<HTMLDivElement | null>(null);

  const mainChartRef = useRef<IChartApi | null>(null);
  const subChartRef = useRef<IChartApi | null>(null);

  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const lineSeriesByKey = useRef<Partial<Record<IndicatorKey, ISeriesApi<'Line'>>>>({});

  // --- JSTのフォーマッタ群 ---
  const jstTickFmt = useMemo(
    () =>
      new Intl.DateTimeFormat('ja-JP', {
        timeZone: 'Asia/Tokyo',
        hour12: false,
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
      }),
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
    // メインチャート
    if (!mainContainerRef.current) return;
    const main = createChart(mainContainerRef.current, {
      height: 520,
      layout: { background: { color: 'white' }, textColor: '#111827' },
      timeScale: { timeVisible: true, secondsVisible: true },
      rightPriceScale: { borderVisible: false },
      grid: { vertLines: { visible: false }, horzLines: { visible: true } },
      localization: { locale: 'ja-JP', timeFormatter: formatTimeJST },
    });
    // 目盛のラベル
    main.applyOptions({
      timeScale: {
        timeVisible: true,
        secondsVisible: true,
        tickMarkFormatter: (t) => jstTickFmt.format(timeToDate(t)),
      },
      localization: { timeFormatter: (t) => jstFullFmt.format(timeToDate(t)) },
    });
    mainChartRef.current = main;

    // ローソク
    const candle = main.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderUpColor: '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    });
    candleSeriesRef.current = candle;

    // サブチャート（必要なら）
    if (subContainerRef.current) {
      const sub = createChart(subContainerRef.current, {
        height: 160,
        layout: { background: { color: 'white' }, textColor: '#374151' },
        timeScale: { timeVisible: true, secondsVisible: true },
        rightPriceScale: { borderVisible: false },
        grid: { vertLines: { visible: false }, horzLines: { visible: true } },
        localization: { locale: 'ja-JP', timeFormatter: formatTimeJST },
      });
      sub.applyOptions({
        timeScale: {
          timeVisible: true,
          secondsVisible: true,
          tickMarkFormatter: (t) => jstTickFmt.format(timeToDate(t)),
        },
        localization: { timeFormatter: (t) => jstFullFmt.format(timeToDate(t)) },
      });
      subChartRef.current = sub;
    }

    // 画面サイズ対応
    const handleResize = () => {
      if (mainContainerRef.current && mainChartRef.current) {
        mainChartRef.current.applyOptions({
          width: mainContainerRef.current.clientWidth,
        });
      }
      if (subContainerRef.current && subChartRef.current) {
        subChartRef.current.applyOptions({
          width: subContainerRef.current.clientWidth,
        });
      }
    };
    handleResize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      subChartRef.current?.remove();
      main.remove();
    };
  }, []);

  // 指定チャートに“点だけシリーズ”を用意（なければ作る）
  const ensureDotsSeries = (key: IndicatorKey): ISeriesApi<'Line'> | null => {
    if (lineSeriesByKey.current[key]) return lineSeriesByKey.current[key]!;
    const cfg: IndicatorCfg = INDICATORS[key];
    const targetChart = cfg.chart === 0 ? mainChartRef.current : subChartRef.current;
    if (!targetChart) return null;
    const s = targetChart.addSeries(LineSeries, {
      color: cfg.color,
      lineWidth: 0, // 線は描かない
      priceLineVisible: false,
      lastValueVisible: false,
      pointMarkersVisible: true, // 点のみ
      pointMarkersRadius: cfg.radius ?? 2,
    });
    lineSeriesByKey.current[key] = s;
    return s;
  };

  // ---- データ取得 ----
  async function fetchCandles() {
    const url = new URL('http://localhost:8000/candles');
    url.searchParams.set('symbol', symbol);
    url.searchParams.set('timeframe', timeframe);
    url.searchParams.set('length', String(length));

    // 事前計算（サーバキャッシュ・任意）
    const preTokens = (Object.keys(INDICATORS) as IndicatorKey[])
      .filter((k) => enabled[k])
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

    // 描画 & 右端寄せ（アニメなし）
    candleSeriesRef.current?.setData(candles);
    mainChartRef.current?.timeScale().scrollToPosition(0, false);
    subChartRef.current?.timeScale().scrollToPosition(0, false);
  }

  // NaN/null を undefined に置換して線の連結を完全に断つ
  const asDotsData = (seriesData: Array<{ time: number; value: number | null }>): LineData[] => {
    return seriesData.map((d) => ({
      time: d.time as UTCTimestamp,
      value:
        d.value === null || Number.isNaN(d.value as number)
          ? (undefined as unknown as number)
          : (d.value as number),
    }));
  };

  async function fetchIndicator(key: IndicatorKey) {
    // トグルOFF → 既存線を空に
    if (!enabled[key]) {
      lineSeriesByKey.current[key]?.setData([]);
      return;
    }

    const cfg = INDICATORS[key];
    const url = new URL('http://localhost:8000/indicator');
    url.searchParams.set('symbol', symbol);
    url.searchParams.set('timeframe', timeframe);
    url.searchParams.set('name', cfg.name);

    const res = await fetch(url.toString());
    if (!res.ok) {
      console.warn('Indicator fetch failed:', key, await res.text());
      return;
    }
    const json = await res.json();
    const raw = (json.values ?? []) as Array<{ time: number; value: number | null }>;

    const s = ensureDotsSeries(key);
    s?.setData(asDotsData(raw));
    (cfg.chart === 0 ? mainChartRef.current : subChartRef.current)
      ?.timeScale()
      .scrollToPosition(0, false);
  }

  async function loadAll() {
    await fetchCandles();
    const tasks: Promise<any>[] = [];
    (Object.keys(INDICATORS) as IndicatorKey[]).forEach((k) =>
      tasks.push(fetchIndicator(k)),
    );
    await Promise.allSettled(tasks);
  }

  // ===== 自動更新（10秒ポーリング） =====
  const pollMs = 10_000; // 10sec
  const loadingRef = useRef(false);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    const runOnce = async () => {
      if (loadingRef.current) return;
      loadingRef.current = true;
      try {
        await loadAll();
      } catch (e) {
        console.error(e);
      } finally {
        loadingRef.current = false;
      }
    };

    // 即時→周期実行
    runOnce();
    if (timerRef.current) window.clearInterval(timerRef.current);
    timerRef.current = window.setInterval(runOnce, pollMs) as unknown as number;

    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current);
      timerRef.current = null;
    };
  }, [symbol, timeframe, length, enabledJSON]);

  const labelFor = (k: IndicatorKey) => INDICATORS[k].name.toUpperCase();

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

        {/* Toggles（INDICATORS から自動生成） */}
        <div className="mt-2 space-y-2">
          {(Object.keys(INDICATORS) as IndicatorKey[]).map((k) => (
            <div key={k} className="flex items-center justify-between">
              <Label htmlFor={k}>
                {labelFor(k)}
                <span className="ml-2 text-xs text-gray-400">
                  {INDICATORS[k].chart === 0 ? 'Main' : 'Sub'}
                </span>
              </Label>
              <Switch
                id={k}
                checked={enabled[k]}
                onCheckedChange={(v) => setEnabled((p) => ({ ...p, [k]: v }))}
              />
            </div>
          ))}
        </div>

        <Button onClick={loadAll}>Load</Button>
      </section>

      {/* Charts */}
      <section className="flex-1 ml-6 bg-white rounded-lg shadow-md p-4">
        <div ref={mainContainerRef} className="w-full h-[520px]" />
        <div className="mt-2" />
        <div ref={subContainerRef} className="w-full h-[160px]" />
      </section>
    </main>
  );
}
