import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { RefreshCw, PlayCircle } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";

interface Position {
  symbol: string;
  side: string;
  entry_time: string;
  entry_price: number;
  size_usd: number;
  stop_price: number | null;        // null when the exchange has no readable resting bracket
  take_profit_price: number | null; // (e.g. an unprotected position, or brackets not found)
  current_price?: number | null;
}

export default function PositionsList({
  positions,
  loading,
  wsConnected = false,
}: {
  positions: Position[];
  loading: boolean;
  wsConnected?: boolean;
}) {
  return (
    <Card className="bg-slate-900/40 border-slate-900/60 backdrop-blur-sm shadow-xl h-full flex flex-col justify-between">
      <div>
        <CardHeader className="flex flex-row items-center justify-between pb-3">
          <CardTitle className="text-base font-bold text-slate-200 flex items-center gap-2">
            Active Positions
            {!loading && positions.length > 0 && (
              <span
                className="flex h-2 w-2 relative"
                title={wsConnected ? "Live Binance ticker connected" : "Live ticker disconnected — showing cached price"}
              >
                {wsConnected && (
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                )}
                <span
                  className={`relative inline-flex rounded-full h-2 w-2 ${wsConnected ? "bg-emerald-500" : "bg-slate-600"}`}
                ></span>
              </span>
            )}
          </CardTitle>
          {loading && <RefreshCw className="w-4 h-4 text-slate-500 animate-spin" />}
        </CardHeader>
        <CardContent className="space-y-4 flex-1">
          {loading ? (
            <div className="space-y-3">
              {[...Array(2)].map((_, i) => (
                <div key={i} className="border border-slate-900/80 rounded-lg bg-slate-950/20 p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Skeleton className="h-4 w-16" />
                      <Skeleton className="h-4 w-12" />
                    </div>
                    <Skeleton className="h-3 w-20" />
                  </div>
                  <div className="grid grid-cols-2 gap-y-3 gap-x-2">
                    <div className="space-y-1">
                      <Skeleton className="h-3 w-12" />
                      <Skeleton className="h-4 w-16" />
                    </div>
                    <div className="space-y-1">
                      <Skeleton className="h-3 w-16" />
                      <Skeleton className="h-4 w-14" />
                    </div>
                    <div className="space-y-1">
                      <Skeleton className="h-3 w-16" />
                      <Skeleton className="h-4 w-16" />
                    </div>
                    <div className="space-y-1">
                      <Skeleton className="h-3 w-16" />
                      <Skeleton className="h-4 w-20" />
                    </div>
                  </div>
                  <div className="pt-2 border-t border-slate-900/60 grid grid-cols-2 gap-2">
                    <div className="space-y-1">
                      <Skeleton className="h-3 w-14" />
                      <Skeleton className="h-3.5 w-16" />
                    </div>
                    <div className="space-y-1">
                      <Skeleton className="h-3 w-14" />
                      <Skeleton className="h-3.5 w-16" />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : positions.length === 0 ? (
            <div className="py-8 flex flex-col items-center justify-center text-center text-slate-500 border border-dashed border-slate-900 rounded-lg p-6">
              <Badge variant="outline" className="mb-2 bg-slate-950/40 border-slate-900 text-slate-400 font-semibold px-2 py-0.5">
                FLAT
              </Badge>
              <p className="text-sm font-medium text-slate-400">No Active Positions</p>
              <p className="text-xs text-slate-600 mt-1 max-w-[200px]">
                Head Trader decided Flat or Risk Manager rejected proposals.
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {positions.map((pos, idx) => {
                const isLong = pos.side.toLowerCase() === "long";
                return (
                  <div
                    key={idx}
                    className="border border-slate-900/80 rounded-lg bg-slate-950/20 p-4 space-y-3"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-slate-200 text-sm">{pos.symbol}</span>
                        <Badge
                          className={`font-bold px-2 py-0.5 uppercase ${
                            isLong
                              ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                              : "bg-rose-500/10 text-rose-400 border border-rose-500/20"
                          }`}
                        >
                          {pos.side}
                        </Badge>
                      </div>
                      <span suppressHydrationWarning className="text-[10px] text-slate-500 font-medium">
                        {pos.entry_time
                          ? new Date(pos.entry_time).toLocaleString(undefined, {
                              month: "short",
                              day: "numeric",
                              hour: "2-digit",
                              minute: "2-digit",
                            })
                          : "Unknown"}
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-y-3 gap-x-2 text-xs font-medium">
                      <div>
                        <p className="text-slate-500">Entry Price</p>
                        <p className="text-slate-300 font-bold mt-0.5">
                          {pos.entry_price > 0 ? `$${pos.entry_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}` : "Pending"}
                        </p>
                      </div>
                      <div>
                        <p className="text-slate-500">Current Price</p>
                        <p className="text-slate-300 font-bold mt-0.5">
                          {pos.current_price ? `$${pos.current_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}` : "N/A"}
                        </p>
                      </div>
                      <div>
                        <p className="text-slate-500">Position Size</p>
                        <p className="text-slate-300 font-bold mt-0.5">
                          ${pos.size_usd.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                        </p>
                      </div>
                      <div>
                        <p className="text-slate-500">Unrealized PnL</p>
                        {pos.entry_price > 0 && pos.current_price ? (() => {
                          const returnPct = isLong
                            ? (pos.current_price - pos.entry_price) / pos.entry_price
                            : (pos.entry_price - pos.current_price) / pos.entry_price;
                          const pnlUsd = pos.size_usd * returnPct;
                          const isProfit = pnlUsd >= 0;
                          return (
                            <p className={`font-bold mt-0.5 ${isProfit ? "text-emerald-400" : "text-rose-400"}`}>
                              {isProfit ? "+" : ""}${pnlUsd.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ({isProfit ? "+" : ""}{(returnPct * 100).toFixed(2)}%)
                            </p>
                          );
                        })() : (
                          <p className="text-slate-400 font-bold mt-0.5">-</p>
                        )}
                      </div>
                    </div>

                    <div className="pt-2 border-t border-slate-900/60 grid grid-cols-2 gap-2 text-[11px] font-semibold">
                      <div>
                        <p className="text-rose-500/80 font-bold">Stop Loss</p>
                        <p className="text-slate-400 mt-0.5">
                          {pos.stop_price != null
                            ? `$${pos.stop_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}`
                            : "—"}
                        </p>
                      </div>
                      <div>
                        <p className="text-emerald-500/80 font-bold">Take Profit</p>
                        <p className="text-slate-400 mt-0.5">
                          {pos.take_profit_price != null
                            ? `$${pos.take_profit_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}`
                            : "—"}
                        </p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </CardContent>
      </div>

      <div className="p-6 pt-0">
        <div className="text-[10px] text-slate-600 bg-slate-950/40 border border-slate-900 p-2.5 rounded-md flex items-start gap-2">
          <PlayCircle className="w-3.5 h-3.5 text-slate-500 shrink-0 mt-0.5" />
          <span>
            Stop-loss and take-profit are placed as native exchange bracket orders — the exchange fills whichever triggers first, in real time, and the close is recorded via the live order stream.
          </span>
        </div>
      </div>
    </Card>
  );
}
