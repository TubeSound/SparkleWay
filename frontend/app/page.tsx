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
import { parseCsvToCandles } from '@/lib/parseCsv';


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

  const [file, setFile] = useState<File | null>(null);
  const [symbol, setSymbol] = useState<string>('');
  const [timeframe, setTimeframe] = useState<string>('');

  const handleCSVRead = async () => {
    if (!file || !chartContainerRef.current) return;

    const text = await file.text();
    const candlesticks = parseCsvToCandles(text);

    if (candlesticks.length === 0) {
      console.error("No valid candlestick data parsed from CSV.");
      return;
    }

    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
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
    candleSeries.setData(candlesticks);

    chartRef.current = chart;
    seriesRef.current = candleSeries;
  };

  useEffect(() => {
    const handleResize = () => {
      if (chartRef.current && chartContainerRef.current) {
        chartRef.current.resize(
          chartContainerRef.current.clientWidth,
          chartContainerRef.current.clientHeight
        );
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <main className="flex min-h-screen items-stretch justify-start p-6 bg-gray-50 text-gray-800">
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
        <div className="flex flex-col gap-1.5">
          <label className="text-sm font-medium text-gray-600">CSVファイル</label>
          <div className="flex flex-col gap-2">
            <Input
              type="file"
              accept=".csv"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
            <Button onClick={handleCSVRead}>
              チャート表示
            </Button>
          </div>
        </div>
      </section>
      <section className="flex-1 ml-6 bg-white rounded-lg shadow-md p-2">
        <div ref={chartContainerRef} className="w-full h-full min-h-[500px]" />
      </section>
    </main>
  );
}
