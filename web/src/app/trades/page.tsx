"use client";

import { useEffect, useState } from "react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { History, RefreshCw, TrendingUp, TrendingDown, Landmark, Percent } from "lucide-react";

interface Trade {
  trade_id: string;
  symbol: string;
  action: string;
  entry_time: string;
  entry_price: number;
  close_time: string;
  close_price: number;
  size_usd: number;
  realized_pnl: number;
  result: string;
}

interface EquityPoint {
  timestamp: string;
  balance: number;
}

export default function Trades() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [filter, setFilter] = useState<"all" | "win" | "loss">("all");

  async function fetchTradesData() {
    setRefreshing(true);
    try {
      const [tradesRes, metricsRes] = await Promise.all([
        fetch("/api/trades"),
        fetch("/api/metrics"),
      ]);

      if (tradesRes.ok) {
        setTrades(await tradesRes.json());
      }
      if (metricsRes.ok) {
        setMetrics(await metricsRes.json());
      }
    } catch (err) {
      console.error("Error fetching trades data:", err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  useEffect(() => {
    fetchTradesData();
  }, []);

  const filteredTrades = trades.filter((t) => {
    if (filter === "all") return true;
    return t.result.toLowerCase() === filter;
  });

  // Render SVG Equity Curve Chart
  const renderEquityCurve = () => {
    if (!metrics || !metrics.equity_curve || metrics.equity_curve.length < 2) {
      return (
        <div className="h-64 flex items-center justify-center text-slate-600 text-xs font-semibold border border-dashed border-slate-900 rounded-lg">
          Not enough historical equity points to display curve.
        </div>
      );
    }

    const curve: EquityPoint[] = metrics.equity_curve;
    const balances = curve.map((p) => p.balance);
    const minVal = Math.min(...balances) * 0.99; // 1% padding
    const maxVal = Math.max(...balances) * 1.01; // 1% padding
    const range = maxVal - minVal;

    const width = 1000;
    const height = 240;
    const padding = 20;

    // Map points to SVG coordinates
    const points = curve.map((p, idx) => {
      const x = padding + (idx / (curve.length - 1)) * (width - padding * 2);
      const y = height - padding - ((p.balance - minVal) / range) * (height - padding * 2);
      return { x, y };
    });

    // Create polyline path string
    const linePath = points.map((p) => `${p.x},${p.y}`).join(" ");
    
    // Create closed path string for gradient fill
    const fillPath = `${points[0].x},${height - padding} ` + linePath + ` ${points[points.length - 1].x},${height - padding}`;

    return (
      <div className="relative w-full overflow-hidden">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-64 overflow-visible">
          <defs>
            <linearGradient id="equity-gradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#10b981" stopOpacity="0.25" />
              <stop offset="100%" stopColor="#10b981" stopOpacity="0.0" />
            </linearGradient>
          </defs>

          {/* Grid lines */}
          <line x1={padding} y1={padding} x2={width - padding} y2={padding} stroke="#1e293b" strokeDasharray="3,3" />
          <line x1={padding} y1={height / 2} x2={width - padding} y2={height / 2} stroke="#1e293b" strokeDasharray="3,3" />
          <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#334155" />

          {/* Shaded Area Under Line */}
          <polygon points={fillPath} fill="url(#equity-gradient)" />

          {/* Main Line */}
          <polyline points={linePath} fill="none" stroke="#10b981" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />

          {/* Highlight Circles for Points */}
          {points.map((p, idx) => (
            <circle
              key={idx}
              cx={p.x}
              cy={p.y}
              r="4"
              className="fill-slate-950 stroke-emerald-400 stroke-2 hover:r-6 cursor-pointer transition-all"
            />
          ))}
        </svg>
        <div className="flex justify-between text-[10px] text-slate-500 font-semibold px-4 pt-1">
          <span>{new Date(curve[0].timestamp).toLocaleDateString()}</span>
          <span>{new Date(curve[curve.length - 1].timestamp).toLocaleDateString()}</span>
        </div>
      </div>
    );
  };

  return (
    <div className="p-8 space-y-8 flex-1">
      {/* Header */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <h2 className="text-3xl font-extrabold text-slate-100 tracking-tight flex items-center gap-3">
            <History className="w-8 h-8 text-emerald-500" />
            Closed Trade Ledger
          </h2>
          <p className="text-sm font-medium text-slate-400 mt-1">
            Auditable history of all completed systematic agent trades and performance metrics.
          </p>
        </div>

        <Button
          onClick={fetchTradesData}
          disabled={refreshing}
          variant="outline"
          className="bg-slate-900/60 border-slate-900 text-slate-300 font-bold hover:bg-slate-900/80 gap-2"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? "animate-spin" : ""}`} />
          Refresh Ledger
        </Button>
      </div>

      {/* Grid: Equity Curve Chart */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <Card className="bg-slate-900/40 border-slate-900/60 backdrop-blur-sm shadow-xl xl:col-span-2">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-bold text-slate-300 flex items-center gap-2">
              <Landmark className="w-4 h-4 text-emerald-400" />
              Equity Growth Curve ($)
            </CardTitle>
          </CardHeader>
          <CardContent>{renderEquityCurve()}</CardContent>
        </Card>

        {/* Closed Stats Summary Card */}
        <Card className="bg-slate-900/40 border-slate-900/60 backdrop-blur-sm shadow-xl font-semibold text-sm">
          <CardHeader>
            <CardTitle className="text-sm font-bold text-slate-300">Ledger Metrics</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-between border-b border-slate-900 pb-2">
              <span className="text-slate-500">Total Closed Positions</span>
              <span className="text-slate-300">{trades.length}</span>
            </div>
            <div className="flex justify-between border-b border-slate-900 pb-2">
              <span className="text-slate-500">Gross Win Rate</span>
              <span className="text-emerald-400">{metrics?.win_rate || "0.0"}%</span>
            </div>
            <div className="flex justify-between border-b border-slate-900 pb-2">
              <span className="text-slate-500">Realized Return (PnL)</span>
              <span className={`font-bold ${metrics?.total_pnl >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                {metrics?.total_pnl >= 0 ? "+" : ""}${metrics?.total_pnl.toLocaleString() || "0.00"}
              </span>
            </div>
            <div className="flex justify-between border-b border-slate-900 pb-2">
              <span className="text-slate-500">Average Return / Trade</span>
              <span className={`font-bold ${metrics?.avg_return >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                {metrics?.avg_return >= 0 ? "+" : ""}${metrics?.avg_return.toLocaleString() || "0.00"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Profit Factor</span>
              <span className="text-cyan-400">{metrics?.profit_factor || "1.00"}</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Trades History Table Card */}
      <Card className="bg-slate-900/40 border-slate-900/60 backdrop-blur-sm shadow-xl">
        <CardHeader className="flex flex-row items-center justify-between pb-3 border-b border-slate-900/60">
          <CardTitle className="text-base font-bold text-slate-200">Trades History</CardTitle>
          
          <div className="flex items-center gap-1">
            {["all", "win", "loss"].map((type) => (
              <Badge
                key={type}
                onClick={() => setFilter(type as any)}
                className={`cursor-pointer px-2.5 py-0.5 font-bold uppercase transition-all text-[9px] ${
                  filter === type
                    ? "bg-slate-100 text-slate-950 hover:bg-slate-200 border-none"
                    : "bg-slate-950/40 text-slate-500 hover:text-slate-300 border border-slate-900"
                }`}
              >
                {type}
              </Badge>
            ))}
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {loading ? (
            <div className="py-8 text-center text-slate-500 text-sm font-medium">
              Loading trade logs...
            </div>
          ) : filteredTrades.length === 0 ? (
            <div className="py-12 text-center text-slate-500 text-sm font-medium">
              No completed trades match the filter.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader className="bg-slate-950/40 border-b border-slate-900/80">
                  <TableRow className="border-b-0">
                    <TableHead className="text-slate-400 font-bold h-11 text-xs">Symbol</TableHead>
                    <TableHead className="text-slate-400 font-bold h-11 text-xs">Side</TableHead>
                    <TableHead className="text-slate-400 font-bold h-11 text-xs">Entry Price</TableHead>
                    <TableHead className="text-slate-400 font-bold h-11 text-xs">Close Price</TableHead>
                    <TableHead className="text-slate-400 font-bold h-11 text-xs">Position Size</TableHead>
                    <TableHead className="text-slate-400 font-bold h-11 text-xs">PnL ($)</TableHead>
                    <TableHead className="text-slate-400 font-bold h-11 text-xs">Execution Times</TableHead>
                    <TableHead className="text-slate-400 font-bold h-11 text-xs text-right">Result</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody className="divide-y divide-slate-900/60 font-semibold text-xs text-slate-300">
                  {filteredTrades.map((trade) => {
                    const isLong = trade.action.toLowerCase() === "long";
                    const isWin = trade.realized_pnl >= 0;
                    return (
                      <TableRow key={trade.trade_id} className="border-b-0 hover:bg-slate-900/10">
                        <TableCell className="font-bold text-slate-200">{trade.symbol}</TableCell>
                        <TableCell>
                          <Badge
                            className={`font-bold px-1.5 py-0.5 text-[10px] uppercase ${
                              isLong
                                ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/10"
                                : "bg-rose-500/10 text-rose-400 border border-rose-500/10"
                            }`}
                          >
                            {trade.action}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          ${trade.entry_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                        </TableCell>
                        <TableCell>
                          ${trade.close_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                        </TableCell>
                        <TableCell>
                          ${trade.size_usd.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                        </TableCell>
                        <TableCell className={isWin ? "text-emerald-400" : "text-rose-400"}>
                          {isWin ? "+" : ""}${trade.realized_pnl.toFixed(2)}
                        </TableCell>
                        <TableCell className="text-[10px] text-slate-500">
                          <div>
                            In:{" "}
                            {new Date(trade.entry_time).toLocaleString(undefined, {
                              dateStyle: "short",
                              timeStyle: "short",
                            })}
                          </div>
                          <div className="mt-0.5">
                            Out:{" "}
                            {new Date(trade.close_time).toLocaleString(undefined, {
                              dateStyle: "short",
                              timeStyle: "short",
                            })}
                          </div>
                        </TableCell>
                        <TableCell className="text-right">
                          <Badge
                            className={`font-extrabold px-2 py-0.5 text-[9px] uppercase ${
                              isWin
                                ? "bg-emerald-500/15 text-emerald-400"
                                : "bg-rose-500/15 text-rose-400"
                            }`}
                          >
                            {trade.result}
                          </Badge>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
