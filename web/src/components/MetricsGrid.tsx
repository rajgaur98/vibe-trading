import type { LucideIcon } from "lucide-react";
import {
  Wallet,
  DollarSign,
  TrendingUp,
  TrendingDown,
  ArrowUpRight,
  ArrowDownRight,
  Target,
  Percent,
  ShieldAlert,
  BarChart3,
  Cpu,
  FileText,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
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

interface Metric {
  title: string;
  value: React.ReactNode;
  desc: React.ReactNode;
  icon: LucideIcon;
  glyph: LucideIcon;
}

export default function MetricsGrid({
  metrics,
  costs,
}: {
  metrics: MetricsData | null;
  costs?: CostData | null;
}) {
  const pnlUp = metrics ? metrics.total_pnl >= 0 : true;

  const data: Metric[] = [
    {
      title: "Portfolio Balance",
      value: metrics ? `$${metrics.balance.toLocaleString()}` : <Skeleton className="h-8 w-28" />,
      desc: metrics ? `Peak: $${metrics.peak_balance.toLocaleString()}` : <Skeleton className="mt-1 h-3 w-20" />,
      icon: Wallet,
      glyph: DollarSign,
    },
    {
      title: "Total Net Profit",
      value: metrics
        ? `${metrics.total_pnl >= 0 ? "+" : ""}$${metrics.total_pnl.toLocaleString()}`
        : <Skeleton className="h-8 w-28" />,
      desc: metrics ? `Total Trades: ${metrics.total_trades}` : <Skeleton className="mt-1 h-3 w-20" />,
      icon: pnlUp ? TrendingUp : TrendingDown,
      glyph: pnlUp ? ArrowUpRight : ArrowDownRight,
    },
    {
      title: "Win Rate",
      value: metrics ? `${metrics.win_rate}%` : <Skeleton className="h-8 w-16" />,
      desc: metrics ? `Profit Factor: ${metrics.profit_factor}` : <Skeleton className="mt-1 h-3 w-24" />,
      icon: Target,
      glyph: Percent,
    },
    {
      title: "Max Drawdown",
      value: metrics ? `${metrics.drawdown.toFixed(2)}%` : <Skeleton className="h-8 w-16" />,
      desc: "Relative to peak equity",
      icon: ShieldAlert,
      glyph: BarChart3,
    },
    {
      title: "LLM Spend (today)",
      value: costs ? `$${(costs.today_usd ?? 0).toFixed(4)}` : <Skeleton className="h-8 w-20" />,
      desc: costs
        ? `~$${(costs.projected_monthly_usd ?? 0).toFixed(2)}/mo · ${costs.calls ?? 0} calls`
        : <Skeleton className="mt-1 h-3 w-24" />,
      icon: Cpu,
      glyph: FileText,
    },
  ];

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
      {data.map((item, idx) => {
        const Icon = item.icon;
        const Glyph = item.glyph;
        return (
          <Card key={idx} className="gap-0 py-0">
            <CardContent className="p-5">
              <div className="mb-4 flex items-start justify-between">
                <span className="flex size-9 items-center justify-center rounded-lg border border-border text-muted-foreground">
                  <Icon className="size-4" />
                </span>
                <span className="flex size-7 items-center justify-center rounded-md border border-border text-muted-foreground">
                  <Glyph className="size-3.5" />
                </span>
              </div>
              {/* div (not p/h3): value/desc can be a <Skeleton> (a div) while loading,
                  and a div nested in p/h3 is invalid HTML → hydration error. */}
              <div className="text-sm font-medium text-muted-foreground">{item.title}</div>
              <div className="mt-1 text-2xl font-semibold tracking-tight text-foreground">
                {item.value}
              </div>
              {item.desc && (
                <div className="mt-1 text-xs text-muted-foreground">{item.desc}</div>
              )}
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
