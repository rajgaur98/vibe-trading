"use client";

import { useEffect, useState } from "react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import PageHeader from "@/components/PageHeader";
import PortfolioChart from "@/components/PortfolioChart";
import SegmentedControl from "@/components/SegmentedControl";
import { BarChart3, FileText } from "lucide-react";
import { cn } from "@/lib/utils";

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

type Filter = "all" | "win" | "loss";

const FILTER_OPTIONS = [
  { value: "all" as const, label: "All" },
  { value: "win" as const, label: "Win" },
  { value: "loss" as const, label: "Loss" },
];

function formatUsd(v: number) {
  const abs = Math.abs(v);
  const maxDigits = abs !== 0 && abs < 1 ? 4 : 2;
  return `$${v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: maxDigits })}`;
}

function LedgerMetric({
  label,
  value,
  tone = "neutral",
}: {
  label: string;
  value: React.ReactNode;
  tone?: "neutral" | "gain" | "loss";
}) {
  return (
    <div className="flex items-center justify-between py-3 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span
        className={cn(
          "font-semibold tabular-nums",
          tone === "neutral" && "text-foreground",
          tone === "gain" && "text-gain",
          tone === "loss" && "text-loss"
        )}
      >
        {value}
      </span>
    </div>
  );
}

export default function Trades() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [filter, setFilter] = useState<Filter>("all");

  async function fetchTradesData() {
    setRefreshing(true);
    try {
      const [tradesRes, metricsRes] = await Promise.all([
        fetch("/api/trades"),
        fetch("/api/metrics"),
      ]);
      if (tradesRes.ok) setTrades(await tradesRes.json());
      if (metricsRes.ok) setMetrics(await metricsRes.json());
      setLastUpdated(new Date());
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

  const filteredTrades = trades.filter((t) => (filter === "all" ? true : t.result.toLowerCase() === filter));

  const pnl = metrics?.total_pnl ?? 0;
  const avg = metrics?.avg_return ?? 0;

  const columns = ["Symbol", "Side", "Entry Price", "Close Price", "Position Size", "PnL ($)", "Execution Times", "Result"];

  return (
    <div className="space-y-8 p-6 sm:p-8">
      <PageHeader
        eyebrow="Trade History"
        title="Closed Trade Ledger"
        subtitle="Auditable history of all completed systematic agent trades and performance metrics."
        lastUpdated={lastUpdated}
        onRefresh={fetchTradesData}
        refreshing={refreshing}
        refreshLabel="Refresh Ledger"
      />

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        <div className="xl:col-span-2">
          <PortfolioChart
            data={metrics?.equity_curve}
            title="Equity Growth Curve ($)"
            subtitle="Cumulative portfolio value from closed trades"
            icon={BarChart3}
            height={260}
          />
        </div>

        <Card className="gap-0 py-0">
          <CardHeader className="border-b [.border-b]:pb-4 pt-5">
            <CardTitle className="flex items-center gap-2 text-base font-semibold">
              <BarChart3 className="size-4 text-muted-foreground" />
              Ledger Metrics
            </CardTitle>
          </CardHeader>
          <CardContent className="py-1">
            <LedgerMetric label="Total Closed Positions" value={trades.length} />
            <Separator />
            <LedgerMetric label="Gross Win Rate" value={`${metrics?.win_rate ?? "0.0"}%`} />
            <Separator />
            <LedgerMetric label="Realized Return (PnL)" tone={pnl >= 0 ? "gain" : "loss"} value={`${pnl >= 0 ? "+" : ""}$${pnl.toLocaleString()}`} />
            <Separator />
            <LedgerMetric label="Average Return / Trade" tone={avg >= 0 ? "gain" : "loss"} value={`${avg >= 0 ? "+" : ""}$${avg.toLocaleString()}`} />
            <Separator />
            <LedgerMetric label="Profit Factor" value={metrics?.profit_factor ?? "1.00"} />
          </CardContent>
        </Card>
      </div>

      <Card className="gap-0 py-0">
        <CardHeader className="flex flex-row items-start justify-between gap-4 border-b [.border-b]:pb-4 pt-5">
          <div className="flex items-start gap-2.5">
            <span className="mt-0.5 flex size-8 items-center justify-center rounded-lg border border-border text-muted-foreground">
              <FileText className="size-4" />
            </span>
            <div className="space-y-0.5">
              <CardTitle className="text-base font-semibold">Trades History</CardTitle>
              <CardDescription>All completed trades, sorted by most recent first</CardDescription>
            </div>
          </div>
          <SegmentedControl aria-label="Filter trades" options={FILTER_OPTIONS} value={filter} onChange={setFilter} />
        </CardHeader>
        <CardContent className="px-0 pb-0">
          {loading ? (
            <div className="space-y-3 p-5">
              {[...Array(6)].map((_, i) => (
                <Skeleton key={i} className="h-10 w-full" />
              ))}
            </div>
          ) : filteredTrades.length === 0 ? (
            <div className="py-14 text-center text-sm text-muted-foreground">No completed trades match the filter.</div>
          ) : (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow className="hover:bg-transparent">
                    {columns.map((c, i) => (
                      <TableHead
                        key={c}
                        className={cn("h-11 text-xs font-medium text-muted-foreground", i === columns.length - 1 && "text-right")}
                      >
                        {c}
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredTrades.map((trade) => {
                    const isWin = trade.realized_pnl >= 0;
                    return (
                      <TableRow key={trade.trade_id}>
                        <TableCell className="font-semibold text-foreground">{trade.symbol}</TableCell>
                        <TableCell>
                          <Badge variant="secondary" className="uppercase">{trade.action}</Badge>
                        </TableCell>
                        <TableCell className="tabular-nums">{formatUsd(trade.entry_price)}</TableCell>
                        <TableCell className="tabular-nums">{formatUsd(trade.close_price)}</TableCell>
                        <TableCell className="tabular-nums">{formatUsd(trade.size_usd)}</TableCell>
                        <TableCell className={cn("font-semibold tabular-nums", isWin ? "text-gain" : "text-loss")}>
                          {isWin ? "+" : ""}${trade.realized_pnl.toFixed(2)}
                        </TableCell>
                        <TableCell suppressHydrationWarning className="text-xs text-muted-foreground">
                          <div>In: {new Date(trade.entry_time).toLocaleString(undefined, { dateStyle: "short", timeStyle: "short" })}</div>
                          <div className="mt-0.5">Out: {new Date(trade.close_time).toLocaleString(undefined, { dateStyle: "short", timeStyle: "short" })}</div>
                        </TableCell>
                        <TableCell className="text-right">
                          <Badge
                            className={cn(
                              "border-transparent uppercase",
                              isWin ? "bg-gain/10 text-gain" : "bg-loss/10 text-loss"
                            )}
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
