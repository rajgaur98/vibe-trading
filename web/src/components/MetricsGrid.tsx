import { Card, CardContent } from "@/components/ui/card";
import { TrendingUp, TrendingDown, DollarSign, Percent, Award, ShieldAlert, Cpu } from "lucide-react";

import { Skeleton } from "@/components/ui/skeleton";

interface MetricsData {
  total_trades: number;
  win_rate: number;
  total_pnl: number;
  avg_return: number;
  profit_factor: number;
  balance: number;
  peak_balance: number;
  drawdown: number;
}

interface CostData {
  today_usd: number;
  calls: number;
  tokens: number;
  avg_cost_per_call: number;
  projected_monthly_usd: number;
}

export default function MetricsGrid({
  metrics,
  costs,
}: {
  metrics: MetricsData | null;
  costs?: CostData | null;
}) {
  const data = [
    {
      title: "Portfolio Balance",
      value: metrics ? `$${metrics.balance.toLocaleString()}` : <Skeleton className="h-8 w-28" />,
      icon: DollarSign,
      desc: metrics ? `Peak: $${metrics.peak_balance.toLocaleString()}` : <Skeleton className="h-3 w-20 mt-1" />,
      iconColor: "text-emerald-400",
      iconBg: "bg-emerald-500/10",
    },
    {
      title: "Total Net Profit",
      value: metrics ? `${metrics.total_pnl >= 0 ? "+" : ""}$${metrics.total_pnl.toLocaleString()}` : <Skeleton className="h-8 w-28" />,
      icon: metrics && metrics.total_pnl >= 0 ? TrendingUp : TrendingDown,
      desc: metrics ? `Total Trades: ${metrics.total_trades}` : <Skeleton className="h-3 w-20 mt-1" />,
      iconColor: metrics && metrics.total_pnl >= 0 ? "text-emerald-400" : "text-rose-400",
      iconBg: metrics && metrics.total_pnl >= 0 ? "bg-emerald-500/10" : "bg-rose-500/10",
      textColor: metrics ? (metrics.total_pnl >= 0 ? "text-emerald-400" : "text-rose-400") : "",
    },
    {
      title: "Win Rate",
      value: metrics ? `${metrics.win_rate}%` : <Skeleton className="h-8 w-16" />,
      icon: Percent,
      desc: metrics ? `Profit Factor: ${metrics.profit_factor}` : <Skeleton className="h-3 w-20 mt-1" />,
      iconColor: "text-cyan-400",
      iconBg: "bg-cyan-500/10",
    },
    {
      title: "Max Drawdown",
      value: metrics ? `${metrics.drawdown.toFixed(2)}%` : <Skeleton className="h-8 w-16" />,
      icon: ShieldAlert,
      desc: "Relative to peak equity",
      iconColor: "text-amber-400",
      iconBg: "bg-amber-500/10",
    },
    {
      title: "LLM Spend (today)",
      value: costs ? `$${(costs.today_usd ?? 0).toFixed(4)}` : <Skeleton className="h-8 w-20" />,
      icon: Cpu,
      desc: costs
        ? `~$${(costs.projected_monthly_usd ?? 0).toFixed(2)}/mo · ${costs.calls ?? 0} calls`
        : <Skeleton className="h-3 w-20 mt-1" />,
      iconColor: "text-violet-400",
      iconBg: "bg-violet-500/10",
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {data.map((item, idx) => {
        const Icon = item.icon;
        return (
          <Card key={idx} className="bg-slate-900/40 border-slate-900/60 backdrop-blur-sm shadow-xl">
            <CardContent className="p-6">
              <div className="flex items-center justify-between mb-4">
                <span className="text-sm font-semibold text-slate-400">{item.title}</span>
                <div className={`p-2.5 rounded-lg ${item.iconBg}`}>
                  <Icon className={`w-4 h-4 ${item.iconColor}`} />
                </div>
              </div>
              <div>
                {/* div (not h3/p): `value`/`desc` can be a <Skeleton> (a <div>) while loading,
                    and a <div> nested in <p>/<h3> is invalid HTML -> React hydration error. */}
                <div className={`text-2xl font-bold tracking-tight text-slate-100 ${item.textColor || ""}`}>
                  {item.value}
                </div>
                {item.desc && (
                  <div className="text-xs text-slate-500 mt-1 font-medium">{item.desc}</div>
                )}
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
