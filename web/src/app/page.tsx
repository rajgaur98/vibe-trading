"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import MetricsGrid from "@/components/MetricsGrid";
import PositionsList from "@/components/PositionsList";
import PortfolioChart from "@/components/PortfolioChart";
import PageHeader from "@/components/PageHeader";
import ActionBadge from "@/components/ActionBadge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { PlayCircle, ChevronRight, FileText, Info, X, BarChart3 } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

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
  const [costs, setCosts] = useState<any>(null);
  const [positions, setPositions] = useState<any[]>([]);
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [livePrices, setLivePrices] = useState<Record<string, number>>({});
  const [wsConnected, setWsConnected] = useState(false);
  const [wsAttempt, setWsAttempt] = useState(0);

  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [triggering, setTriggering] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [notification, setNotification] = useState<{ type: "success" | "error" | "info" | null; message: string }>({ type: null, message: "" });

  // The /api/trigger-tick endpoint is localhost-only (see web/security.py on the API),
  // so the "Scan & Trade" button is shown ONLY when the dashboard is viewed from
  // localhost. On the public (Vercel) deployment it stays hidden — it would 403 anyway.
  // Defaults to false so SSR and first client render match (no hydration mismatch).
  const [isLocalhost, setIsLocalhost] = useState(false);
  useEffect(() => {
    const h = window.location.hostname;
    setIsLocalhost(h === "localhost" || h === "127.0.0.1" || h === "::1");
  }, []);

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
      const [metricsRes, posRes, decRes, costsRes] = await Promise.all([
        fetch("/api/metrics"),
        fetch("/api/positions"),
        fetch("/api/decisions?limit=5"),
        fetch("/api/costs"),
      ]);

      if (metricsRes.ok) setMetrics(await metricsRes.json());
      if (posRes.ok) setPositions(await posRes.json());
      if (decRes.ok) setDecisions(await decRes.json());
      if (costsRes.ok) setCosts(await costsRes.json());
      setLastUpdated(new Date());
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

  // Binance miniTicker WebSocket — streams ~1Hz close-price updates for active positions.
  // Symbols are converted to Binance combined-stream format (e.g. PENGU/USDT -> pengusdt@miniTicker).
  // Reconnects on disconnect (covers Binance's 24h server-initiated close + transient network drops).
  const intentionalClose = useRef(false);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (positions.length === 0) {
      setWsConnected(false);
      return;
    }

    // Map: Binance uppercase concatenated symbol (PENGUUSDT) -> original bot symbol (PENGU/USDT)
    const symbolMap: Record<string, string> = {};
    positions.forEach((pos) => {
      const binSymbol = pos.symbol.replace("/", "").toUpperCase();
      symbolMap[binSymbol] = pos.symbol;
    });

    const streams = Object.keys(symbolMap)
      .map((s) => s.toLowerCase() + "@miniTicker")
      .join("/");

    if (!streams) return;

    intentionalClose.current = false;
    const ws = new WebSocket(`wss://stream.binance.com:9443/stream?streams=${streams}`);

    ws.onopen = () => setWsConnected(true);

    ws.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        const data = parsed.data ?? parsed;
        if (!data || !data.s || data.c === undefined) return;
        const originalSymbol = symbolMap[data.s];
        if (!originalSymbol) return;
        const price = parseFloat(data.c);
        if (!Number.isFinite(price)) return;
        setLivePrices((prev) =>
          prev[originalSymbol] === price ? prev : { ...prev, [originalSymbol]: price }
        );
      } catch (err) {
        console.error("Binance WS parse error:", err);
      }
    };

    ws.onerror = () => setWsConnected(false);

    ws.onclose = () => {
      setWsConnected(false);
      if (!intentionalClose.current) {
        reconnectTimer.current = setTimeout(() => {
          setWsAttempt((a) => a + 1);
        }, 5000);
      }
    };

    return () => {
      intentionalClose.current = true;
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
        reconnectTimer.current = null;
      }
      ws.close();
    };
  }, [positions.map((p) => p.symbol).sort().join(","), wsAttempt]);

  // Merge polled positions with live ticker prices for rendering — live overrides stale API value.
  const positionsWithLivePrice = positions.map((pos) => ({
    ...pos,
    current_price: livePrices[pos.symbol] ?? pos.current_price,
  }));

  return (
    <div className="space-y-8 p-6 sm:p-8">
      <PageHeader
        eyebrow="Trading Terminal"
        title="Vibe Trading Terminal"
        subtitle="Real-time multi-agent crypto swing-trading and risk ledger."
        lastUpdated={lastUpdated}
        onRefresh={fetchDashboardData}
        refreshing={refreshing}
      >
        {isLocalhost && (
          <Button
            onClick={triggerOnDemandTrade}
            disabled={triggering || refreshing}
            size="sm"
          >
            <PlayCircle className={triggering ? "animate-spin" : ""} />
            {triggering ? "Scanning…" : "Scan & Trade"}
          </Button>
        )}
      </PageHeader>

      {notification.type && (
        <div
          className={cn(
            "flex items-center justify-between gap-4 rounded-xl border px-4 py-3 text-sm font-medium",
            notification.type === "success" && "border-gain/30 bg-gain/10 text-gain",
            notification.type === "error" && "border-loss/30 bg-loss/10 text-loss",
            notification.type === "info" && "border-border bg-muted text-foreground"
          )}
        >
          <span>{notification.message}</span>
          <button
            onClick={() => setNotification({ type: null, message: "" })}
            className="text-muted-foreground transition-colors hover:text-foreground"
            aria-label="Dismiss"
          >
            <X className="size-4" />
          </button>
        </div>
      )}

      <MetricsGrid metrics={metrics} costs={costs} />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-4">
        <div className="lg:col-span-3">
          <PortfolioChart
            data={metrics?.equity_curve}
            title="Portfolio Performance"
            subtitle="Total portfolio value over time"
            icon={BarChart3}
          />
        </div>
        <div className="lg:col-span-1">
          <PositionsList positions={positionsWithLivePrice} loading={loading} wsConnected={wsConnected} />
        </div>
      </div>

      {/* Recent Agent Activity */}
      <Card>
        <CardHeader className="flex flex-row items-start justify-between border-b [.border-b]:pb-4">
          <div className="flex items-start gap-2.5">
            <span className="mt-0.5 flex size-8 items-center justify-center rounded-lg border border-border text-muted-foreground">
              <FileText className="size-4" />
            </span>
            <div className="space-y-0.5">
              <CardTitle className="text-base font-semibold">Recent Agent Activity</CardTitle>
              <CardDescription>Latest agent decisions and system activity</CardDescription>
            </div>
          </div>
          <Link
            href="/decisions"
            className="flex items-center gap-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
          >
            View All
            <ChevronRight className="size-3.5" />
          </Link>
        </CardHeader>
        <CardContent className="pt-4">
          <div className="mb-4 flex items-start gap-2 rounded-lg bg-muted/60 p-3 text-xs text-muted-foreground">
            <Info className="mt-0.5 size-3.5 shrink-0" />
            <span>Agent decision logs, trade history, and detailed reasoning can be found in their respective sections.</span>
          </div>

          {loading ? (
            <div className="divide-y divide-border">
              {[...Array(3)].map((_, i) => (
                <div key={i} className="space-y-3 py-4 first:pt-0">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Skeleton className="h-4 w-16" />
                      <Skeleton className="h-4 w-12" />
                    </div>
                    <Skeleton className="h-3 w-32" />
                  </div>
                  <Skeleton className="h-3.5 w-full" />
                  <Skeleton className="h-3.5 w-5/6" />
                </div>
              ))}
            </div>
          ) : decisions.length === 0 ? (
            <div className="py-8 text-center text-sm text-muted-foreground">
              No decisions logged yet. Run a trade execution tick to generate logs.
            </div>
          ) : (
            <div className="divide-y divide-border">
              {decisions.map((dec) => {
                const isTradable = !["flat", "close"].includes(dec.action.toLowerCase());
                return (
                  <div key={dec.decision_id} className="space-y-2.5 py-4 first:pt-0">
                    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                      <div className="flex items-center gap-2.5">
                        <span className="text-sm font-semibold text-foreground">{dec.symbol}</span>
                        <ActionBadge action={dec.action} />
                      </div>
                      <span suppressHydrationWarning className="text-xs text-muted-foreground">
                        {dec.timestamp
                          ? new Date(dec.timestamp).toLocaleString(undefined, {
                              dateStyle: "medium",
                              timeStyle: "short",
                            })
                          : "Unknown"}
                      </span>
                    </div>
                    <p className="text-sm leading-relaxed text-muted-foreground">{dec.reasoning_summary}</p>
                    {isTradable && (
                      <div className="flex flex-wrap gap-x-6 gap-y-1.5 text-xs text-muted-foreground">
                        <span>
                          Stop:{" "}
                          <span className="font-medium text-foreground uppercase">
                            {dec.stop_loss_strategy.replace("_", " ")}
                          </span>
                        </span>
                        <span>
                          Target:{" "}
                          <span className="font-medium text-foreground uppercase">
                            {dec.take_profit_strategy.replace("_", " ")}
                          </span>
                        </span>
                        <span>
                          R/R:{" "}
                          <span className="font-medium text-foreground">{dec.risk_reward_ratio}x</span>
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
