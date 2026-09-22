import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Target, Info } from "lucide-react";
import { cn } from "@/lib/utils";

interface Position {
  symbol: string;
  side: string;
  entry_time: string;
  entry_price: number;
  size_usd: number;
  stop_price: number | null;        // live exchange stop trigger; null when no resting stop bracket
  take_profit_price: number | null; // live exchange take-profit trigger; null when none
  current_price?: number | null;
  stop_live?: boolean;              // is there a LIVE stop resting on the exchange?
  intended_stop_price?: number | null; // stop recorded at entry, shown when the live stop is gone
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="mt-0.5 text-sm font-medium text-foreground">{children}</p>
    </div>
  );
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
    <Card className="flex h-full flex-col">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base font-semibold">
          <Target className="size-4 text-muted-foreground" />
          Active Positions
          {!loading && (
            <Badge variant="secondary" className="tabular-nums">
              {positions.length}
            </Badge>
          )}
          {!loading && positions.length > 0 && (
            <span
              className="relative flex size-2"
              title={wsConnected ? "Live Binance ticker connected" : "Live ticker disconnected — showing cached price"}
            >
              {wsConnected && (
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-gain opacity-60" />
              )}
              <span className={cn("relative inline-flex size-2 rounded-full", wsConnected ? "bg-gain" : "bg-muted-foreground")} />
            </span>
          )}
        </CardTitle>
      </CardHeader>

      <CardContent className="flex-1 space-y-3">
        {loading ? (
          [...Array(2)].map((_, i) => (
            <div key={i} className="space-y-3 rounded-lg border border-border p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Skeleton className="h-4 w-16" />
                  <Skeleton className="h-4 w-12" />
                </div>
                <Skeleton className="h-3 w-16" />
              </div>
              <div className="grid grid-cols-2 gap-3">
                {[...Array(4)].map((_, j) => (
                  <div key={j} className="space-y-1.5">
                    <Skeleton className="h-3 w-16" />
                    <Skeleton className="h-4 w-20" />
                  </div>
                ))}
              </div>
            </div>
          ))
        ) : positions.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border p-8 text-center">
            <Badge variant="secondary">FLAT</Badge>
            <p className="text-sm font-medium text-foreground">No active positions</p>
            <p className="max-w-[220px] text-xs text-muted-foreground">
              Head Trader decided Flat, or Risk Manager rejected the proposals.
            </p>
          </div>
        ) : (
          positions.map((pos, idx) => {
            const isLong = pos.side.toLowerCase() === "long";
            const hasPnl = pos.entry_price > 0 && pos.current_price;
            const returnPct = hasPnl
              ? isLong
                ? (pos.current_price! - pos.entry_price) / pos.entry_price
                : (pos.entry_price - pos.current_price!) / pos.entry_price
              : 0;
            const pnlUsd = pos.size_usd * returnPct;
            const isProfit = pnlUsd >= 0;

            return (
              <div key={idx} className="space-y-3 rounded-lg border border-border p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-foreground">{pos.symbol}</span>
                    <Badge variant="secondary" className="uppercase">
                      {pos.side}
                    </Badge>
                  </div>
                  <span suppressHydrationWarning className="text-xs text-muted-foreground">
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

                <div className="grid grid-cols-2 gap-x-4 gap-y-3">
                  <Field label="Entry Price">
                    {pos.entry_price > 0
                      ? `$${pos.entry_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}`
                      : "Pending"}
                  </Field>
                  <Field label="Current Price">
                    {pos.current_price
                      ? `$${pos.current_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}`
                      : "N/A"}
                  </Field>
                  <Field label="Position Size">
                    ${pos.size_usd.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                  </Field>
                  <div>
                    <p className="text-xs text-muted-foreground">Unrealized PnL</p>
                    {hasPnl ? (
                      <p className={cn("mt-0.5 text-sm font-semibold", isProfit ? "text-gain" : "text-loss")}>
                        {isProfit ? "+" : ""}${pnlUsd.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}{" "}
                        ({isProfit ? "+" : ""}{(returnPct * 100).toFixed(2)}%)
                      </p>
                    ) : (
                      <p className="mt-0.5 text-sm font-medium text-muted-foreground">—</p>
                    )}
                  </div>
                </div>

                <Separator />

                <div className="grid grid-cols-2 gap-x-4">
                  <div>
                    <p className="text-xs text-muted-foreground">Stop Loss</p>
                    {pos.stop_price != null ? (
                      <p className="mt-0.5 text-sm font-medium text-foreground">
                        ${pos.stop_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                      </p>
                    ) : pos.intended_stop_price != null ? (
                      <p
                        className="mt-0.5 flex items-center gap-1.5 text-sm font-medium text-loss"
                        title="Stop was placed at entry but is no longer live on the exchange — this position is unprotected."
                      >
                        ${pos.intended_stop_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                        <span className="rounded border border-loss/40 px-1 py-px text-[9px] font-semibold uppercase tracking-wide text-loss">
                          not live
                        </span>
                      </p>
                    ) : (
                      <p className="mt-0.5 text-sm font-medium text-muted-foreground">—</p>
                    )}
                  </div>
                  <Field label="Take Profit">
                    {pos.take_profit_price != null
                      ? `$${pos.take_profit_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}`
                      : "—"}
                  </Field>
                </div>
              </div>
            );
          })
        )}
      </CardContent>

      <div className="px-4 pb-4">
        <div className="flex items-start gap-2 rounded-lg bg-muted/60 p-3 text-xs text-muted-foreground">
          <Info className="mt-0.5 size-3.5 shrink-0" />
          <span>
            Stop-loss and take-profit are placed as native exchange bracket orders — the exchange fills
            whichever triggers first, and the close is recorded via the live order stream.
          </span>
        </div>
      </div>
    </Card>
  );
}
