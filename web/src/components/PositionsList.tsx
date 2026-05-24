import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { RefreshCw, PlayCircle } from "lucide-react";

interface Position {
  symbol: string;
  side: string;
  entry_time: string;
  entry_price: number;
  size_usd: number;
  stop_price: number;
  take_profit_price: number;
}

export default function PositionsList({
  positions,
  loading,
}: {
  positions: Position[];
  loading: boolean;
}) {
  return (
    <Card className="bg-slate-900/40 border-slate-900/60 backdrop-blur-sm shadow-xl h-full flex flex-col justify-between">
      <div>
        <CardHeader className="flex flex-row items-center justify-between pb-3">
          <CardTitle className="text-base font-bold text-slate-200">
            Active Positions
          </CardTitle>
          {loading && <RefreshCw className="w-4 h-4 text-slate-500 animate-spin" />}
        </CardHeader>
        <CardContent className="space-y-4 flex-1">
          {loading ? (
            <div className="py-8 text-center text-slate-500 text-sm font-medium">
              Loading active positions...
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
                      <span className="text-[10px] text-slate-500 font-medium">
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

                    <div className="grid grid-cols-2 gap-2 text-xs font-medium">
                      <div>
                        <p className="text-slate-500">Entry Price</p>
                        <p className="text-slate-300 font-bold mt-0.5">
                          ${pos.entry_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                        </p>
                      </div>
                      <div>
                        <p className="text-slate-500">Position Size</p>
                        <p className="text-slate-300 font-bold mt-0.5">
                          ${pos.size_usd.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                        </p>
                      </div>
                    </div>

                    <div className="pt-2 border-t border-slate-900/60 grid grid-cols-2 gap-2 text-[11px] font-semibold">
                      <div>
                        <p className="text-rose-500/80 font-bold">Stop Loss</p>
                        <p className="text-slate-400 mt-0.5">
                          ${pos.stop_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                        </p>
                      </div>
                      <div>
                        <p className="text-emerald-500/80 font-bold">Take Profit</p>
                        <p className="text-slate-400 mt-0.5">
                          ${pos.take_profit_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}
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
            Paper account simulates dynamic market checks against order stop loss and take profit values every 4 hours.
          </span>
        </div>
      </div>
    </Card>
  );
}
