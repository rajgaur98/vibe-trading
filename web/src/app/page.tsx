"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import MetricsGrid from "@/components/MetricsGrid";
import PositionsList from "@/components/PositionsList";
import EquityChart from "@/components/EquityChart";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { RefreshCw, PlayCircle, ChevronRight, BrainCircuit } from "lucide-react";

interface Decision {
  decision_id: string;
  timestamp: string;
  symbol: string;
  action: string;
  stop_loss_strategy: string;
  take_profit_strategy: string;
  risk_reward_ratio: number;
  reasoning_summary: string;
}

export default function Dashboard() {
  const [metrics, setMetrics] = useState<any>(null);
  const [positions, setPositions] = useState<any[]>([]);
  const [decisions, setDecisions] = useState<Decision[]>([]);
  
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [triggering, setTriggering] = useState(false);
  const [notification, setNotification] = useState<{ type: "success" | "error" | "info" | null; message: string }>({ type: null, message: "" });

  async function triggerOnDemandTrade() {
    setTriggering(true);
    setNotification({ type: "info", message: "Analyzing market and running dynamic trade scan. Please wait..." });
    try {
      const res = await fetch("/api/trigger-tick", { method: "POST" });
      if (res.ok) {
        await fetchDashboardData();
        setNotification({ type: "success", message: "On-demand execution tick completed! Check the decisions and positions below." });
        setTimeout(() => setNotification({ type: null, message: "" }), 6000);
      } else {
        const errData = await res.json().catch(() => ({ detail: "Unknown error" }));
        setNotification({ type: "error", message: `Scan failed: ${errData.detail || "Server error"}` });
        setTimeout(() => setNotification({ type: null, message: "" }), 8000);
      }
    } catch (err) {
      console.error("Error triggering scan:", err);
      setNotification({ type: "error", message: "Network error triggering scan." });
      setTimeout(() => setNotification({ type: null, message: "" }), 6000);
    } finally {
      setTriggering(false);
    }
  }


  async function fetchDashboardData() {
    setRefreshing(true);
    try {
      const [metricsRes, posRes, decRes] = await Promise.all([
        fetch("/api/metrics"),
        fetch("/api/positions"),
        fetch("/api/decisions?limit=5"),
      ]);

      if (metricsRes.ok) setMetrics(await metricsRes.json());
      if (posRes.ok) setPositions(await posRes.json());
      if (decRes.ok) setDecisions(await decRes.json());
    } catch (err) {
      console.error("Error fetching dashboard data:", err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  useEffect(() => {
    fetchDashboardData();
    // Poll every 30 seconds
    const interval = setInterval(fetchDashboardData, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-8 space-y-8 flex-1">
      {/* Top Banner / Headers */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <h2 className="text-3xl font-extrabold text-slate-100 tracking-tight">
            Vibe Trading Terminal
          </h2>
          <p className="text-sm font-medium text-slate-400 mt-1">
            Real-time multi-agent crypto swing-trading and risk ledger.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Button
            onClick={triggerOnDemandTrade}
            disabled={triggering || refreshing}
            className="bg-emerald-600 hover:bg-emerald-500 text-slate-100 font-bold gap-2 cursor-pointer transition-colors"
          >
            <PlayCircle className={`w-3.5 h-3.5 ${triggering ? "animate-spin" : ""}`} />
            {triggering ? "Scanning Market..." : "Scan & Trade"}
          </Button>

          <Button
            onClick={fetchDashboardData}
            disabled={refreshing || triggering}
            variant="outline"
            className="bg-slate-900/60 border-slate-900 text-slate-300 font-bold hover:bg-slate-900/80 gap-2"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </div>
      </div>

      {notification.type && (
        <div className={`p-4 rounded-xl border font-semibold text-sm transition-all duration-300 flex items-center justify-between shadow-lg ${
          notification.type === "success" 
            ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400" 
            : notification.type === "error" 
            ? "bg-rose-500/10 border-rose-500/20 text-rose-400" 
            : "bg-cyan-500/15 border-cyan-500/30 text-cyan-400 animate-pulse"
        }`}>
          <span>{notification.message}</span>
          <button 
            onClick={() => setNotification({ type: null, message: "" })} 
            className="text-slate-400 hover:text-slate-200 ml-4 font-bold text-base cursor-pointer"
          >
            ✕
          </button>
        </div>
      )}

      {/* Metrics Row */}
      <MetricsGrid metrics={metrics} />

      {/* Charts & Positions Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Portfolio Equity Chart */}
        <EquityChart metrics={metrics} />

        {/* Active Open Positions Panel */}
        <PositionsList positions={positions} loading={loading} />
      </div>

      {/* Recent Decisions Feed */}
      <Card className="bg-slate-900/40 border-slate-900/60 backdrop-blur-sm shadow-xl">
        <CardHeader className="flex flex-row items-center justify-between pb-3 border-b border-slate-900/60">
          <div className="flex items-center gap-2.5">
            <BrainCircuit className="w-5 h-5 text-emerald-500" />
            <CardTitle className="text-base font-bold text-slate-200">
              Recent Agent Decisions
            </CardTitle>
          </div>
          <Link
            href="/decisions"
            className="text-xs font-bold text-emerald-400 hover:text-emerald-300 flex items-center gap-1 transition-all"
          >
            Explore all transcripts
            <ChevronRight className="w-3.5 h-3.5" />
          </Link>
        </CardHeader>
        <CardContent className="p-0">
          {loading ? (
            <div className="py-8 text-center text-slate-500 text-sm font-medium">
              Loading decisions...
            </div>
          ) : decisions.length === 0 ? (
            <div className="py-12 text-center text-slate-500 text-sm font-medium">
              No decisions logged yet. Run a trade execution tick to generate logs.
            </div>
          ) : (
            <div className="divide-y divide-slate-900">
              {decisions.map((dec) => {
                const isFlat = dec.action.toLowerCase() === "flat";
                const isClose = dec.action.toLowerCase() === "close";
                const isLong = dec.action.toLowerCase() === "long";
                return (
                  <div key={dec.decision_id} className="p-6 hover:bg-slate-900/20 transition-all">
                    <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-3 mb-3">
                      <div className="flex items-center gap-3">
                        <span className="font-bold text-sm text-slate-300">{dec.symbol}</span>
                        <Badge
                          className={`font-extrabold px-2 py-0.5 uppercase text-[10px] ${
                            isFlat
                              ? "bg-slate-950/40 text-slate-400 border border-slate-900"
                              : isClose
                              ? "bg-amber-500/10 text-amber-400 border border-amber-500/20"
                              : isLong
                              ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                              : "bg-rose-500/10 text-rose-400 border border-rose-500/20"
                          }`}
                        >
                          {dec.action}
                        </Badge>
                      </div>

                      <span className="text-[10px] text-slate-500 font-semibold uppercase tracking-wider">
                        {dec.timestamp
                          ? new Date(dec.timestamp).toLocaleString(undefined, {
                              dateStyle: "medium",
                              timeStyle: "short",
                            })
                          : "Unknown"}
                      </span>
                    </div>

                    <p className="text-slate-400 text-xs leading-relaxed font-medium">
                      {dec.reasoning_summary}
                    </p>

                    {!isFlat && !isClose && (
                      <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-2 text-[10px] font-semibold text-slate-500">
                        <span>
                          Stop Strategy:{" "}
                          <strong className="text-slate-400 uppercase">
                            {dec.stop_loss_strategy.replace("_", " ")}
                          </strong>
                        </span>
                        <span>
                          Target Profit:{" "}
                          <strong className="text-slate-400 uppercase">
                            {dec.take_profit_strategy.replace("_", " ")}
                          </strong>
                        </span>
                        <span>
                          Risk/Reward Ratio:{" "}
                          <strong className="text-slate-400">{dec.risk_reward_ratio}x</strong>
                        </span>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
