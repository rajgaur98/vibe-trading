import { Card, CardContent } from "@/components/ui/card";
import { TrendingUp, TrendingDown, DollarSign, Percent, Award, ShieldAlert } from "lucide-react";

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

export default function MetricsGrid({ metrics }: { metrics: MetricsData | null }) {
  const data = [
    {
      title: "Portfolio Balance",
      value: metrics ? `$${metrics.balance.toLocaleString()}` : "Loading...",
      icon: DollarSign,
      desc: metrics ? `Peak: $${metrics.peak_balance.toLocaleString()}` : "",
      iconColor: "text-emerald-400",
      iconBg: "bg-emerald-500/10",
    },
    {
      title: "Total Net Profit",
      value: metrics ? `${metrics.total_pnl >= 0 ? "+" : ""}$${metrics.total_pnl.toLocaleString()}` : "Loading...",
      icon: metrics && metrics.total_pnl >= 0 ? TrendingUp : TrendingDown,
      desc: metrics ? `Total Trades: ${metrics.total_trades}` : "",
      iconColor: metrics && metrics.total_pnl >= 0 ? "text-emerald-400" : "text-rose-400",
      iconBg: metrics && metrics.total_pnl >= 0 ? "bg-emerald-500/10" : "bg-rose-500/10",
      textColor: metrics ? (metrics.total_pnl >= 0 ? "text-emerald-400" : "text-rose-400") : "",
    },
    {
      title: "Win Rate",
      value: metrics ? `${metrics.win_rate}%` : "Loading...",
      icon: Percent,
      desc: metrics ? `Profit Factor: ${metrics.profit_factor}` : "",
      iconColor: "text-cyan-400",
      iconBg: "bg-cyan-500/10",
    },
    {
      title: "Max Drawdown",
      value: metrics ? `${metrics.drawdown.toFixed(2)}%` : "Loading...",
      icon: ShieldAlert,
      desc: "Relative to peak equity",
      iconColor: "text-amber-400",
      iconBg: "bg-amber-500/10",
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
                <h3 className={`text-2xl font-bold tracking-tight text-slate-100 ${item.textColor || ""}`}>
                  {item.value}
                </h3>
                {item.desc && (
                  <p className="text-xs text-slate-500 mt-1 font-medium">{item.desc}</p>
                )}
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
