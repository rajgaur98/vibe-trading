"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Landmark, TrendingUp, TrendingDown } from "lucide-react";

interface EquityPoint {
  timestamp: string;
  balance: number;
}

interface MetricsData {
  total_trades: number;
  win_rate: number;
  total_pnl: number;
  avg_return: number;
  profit_factor: number;
  balance: number;
  peak_balance: number;
  drawdown: number;
  equity_curve: EquityPoint[];
}

export default function EquityChart({ metrics }: { metrics: MetricsData | null }) {
  const renderChart = () => {
    if (!metrics || !metrics.equity_curve || metrics.equity_curve.length === 0) {
      return (
        <div className="h-[300px] flex items-center justify-center text-slate-500 text-sm border border-dashed border-slate-900 rounded-lg">
          No equity data available.
        </div>
      );
    }

    const curve = metrics.equity_curve;
    
    // If only one point exists (e.g. startup), add a leading point at the SAME balance so
    // we can render a (flat) line — never a fabricated $10k starting value.
    const chartPoints = [...curve];
    if (chartPoints.length === 1) {
      const firstPointDate = new Date(chartPoints[0].timestamp);
      const startDateTime = new Date(firstPointDate.getTime() - 24 * 60 * 60 * 1000); // 1 day before
      chartPoints.unshift({
        timestamp: startDateTime.toISOString(),
        balance: chartPoints[0].balance,
      });
    }

    const balances = chartPoints.map((p) => p.balance);
    const minVal = Math.min(...balances) * 0.99; // 1% padding
    const maxVal = Math.max(...balances) * 1.01; // 1% padding
    const range = maxVal - minVal === 0 ? 100 : maxVal - minVal;

    const width = 1000;
    const height = 300;
    const paddingLeft = 60;
    const paddingRight = 30;
    const paddingTop = 20;
    const paddingBottom = 40;

    const chartWidth = width - paddingLeft - paddingRight;
    const chartHeight = height - paddingTop - paddingBottom;

    // Map points to SVG coordinates
    const coords = chartPoints.map((p, idx) => {
      const x = paddingLeft + (idx / (chartPoints.length - 1)) * chartWidth;
      const y = paddingTop + chartHeight - ((p.balance - minVal) / range) * chartHeight;
      return { x, y };
    });

    const linePath = coords.map((p) => `${p.x},${p.y}`).join(" ");
    const fillPath = `${coords[0].x},${paddingTop + chartHeight} ` + linePath + ` ${coords[coords.length - 1].x},${paddingTop + chartHeight}`;

    // Create horizontal grid lines & labels
    const gridCount = 4;
    const gridLines = [];
    for (let i = 0; i <= gridCount; i++) {
      const ratio = i / gridCount;
      const price = minVal + ratio * range;
      const y = paddingTop + chartHeight - ratio * chartHeight;
      gridLines.push({ y, price });
    }

    const isPnLPositive = metrics.total_pnl >= 0;

    return (
      <div className="w-full overflow-hidden">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-80 overflow-visible">
          <defs>
            <linearGradient id="equity-chart-gradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={isPnLPositive ? "#10b981" : "#ef4444"} stopOpacity="0.2" />
              <stop offset="100%" stopColor={isPnLPositive ? "#10b981" : "#ef4444"} stopOpacity="0.0" />
            </linearGradient>
          </defs>

          {/* Grid lines & Y axis labels */}
          {gridLines.map((line, idx) => (
            <g key={idx}>
              <line
                x1={paddingLeft}
                y1={line.y}
                x2={width - paddingRight}
                y2={line.y}
                stroke="#1e293b"
                strokeWidth="1"
                strokeDasharray={idx === 0 || idx === gridCount ? "none" : "3,3"}
              />
              <text
                x={paddingLeft - 10}
                y={line.y + 4}
                textAnchor="end"
                className="fill-slate-500 font-semibold font-mono text-[10px]"
              >
                ${line.price.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </text>
            </g>
          ))}

          {/* Shaded Area Under Line */}
          <polygon points={fillPath} fill="url(#equity-chart-gradient)" />

          {/* Main Line */}
          <polyline
            points={linePath}
            fill="none"
            stroke={isPnLPositive ? "#10b981" : "#ef4444"}
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="shadow-[0_0_15px_rgba(16,185,129,0.2)]"
          />

          {/* Highlight Circles for Points */}
          {coords.map((p, idx) => (
            <circle
              key={idx}
              cx={p.x}
              cy={p.y}
              r="4.5"
              className={`fill-slate-950 stroke-2 cursor-pointer transition-all ${
                isPnLPositive ? "stroke-emerald-400" : "stroke-rose-400"
              } hover:r-6`}
            />
          ))}
        </svg>

        {/* X axis dates */}
        <div className="flex justify-between text-[10px] text-slate-500 font-bold px-12 pt-1 border-t border-slate-900/60">
          <span>{new Date(chartPoints[0].timestamp).toLocaleDateString()}</span>
          <span>{new Date(chartPoints[chartPoints.length - 1].timestamp).toLocaleDateString()}</span>
        </div>
      </div>
    );
  };

  const isPnLPositive = metrics ? metrics.total_pnl >= 0 : true;

  return (
    <Card className="bg-slate-900/40 border-slate-900/60 backdrop-blur-sm shadow-xl col-span-1 lg:col-span-3 flex flex-col justify-between">
      <CardHeader className="flex flex-row items-center justify-between pb-4 border-b border-slate-900/60">
        <div className="flex items-center gap-2.5">
          <Landmark className="w-5 h-5 text-emerald-500" />
          <CardTitle className="text-base font-bold text-slate-200">
            Portfolio Performance Chart
          </CardTitle>
        </div>
        {metrics && (
          <Badge
            className={`font-extrabold px-2.5 py-0.5 text-[10px] uppercase flex items-center gap-1 border ${
              isPnLPositive
                ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                : "bg-rose-500/10 text-rose-400 border-rose-500/20"
            }`}
          >
            {isPnLPositive ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
            {isPnLPositive ? "Profit Trend" : "Loss Trend"}
          </Badge>
        )}
      </CardHeader>
      <CardContent className="p-6 flex-grow flex items-center justify-center">
        {renderChart()}
      </CardContent>
    </Card>
  );
}
