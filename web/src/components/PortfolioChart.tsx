"use client";

import { useId, useMemo } from "react";
import type { LucideIcon } from "lucide-react";
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";

export interface EquityPoint {
  timestamp: string;
  balance: number;
}

const chartConfig = {
  balance: { label: "Balance", color: "var(--foreground)" },
} satisfies ChartConfig;

export default function PortfolioChart({
  data,
  title,
  subtitle,
  icon: Icon,
  height = 300,
}: {
  data: EquityPoint[] | undefined | null;
  title: string;
  subtitle?: string;
  icon?: LucideIcon;
  height?: number;
}) {
  const gradientId = useId().replace(/:/g, "");

  const sorted = useMemo(() => {
    if (!data) return [];
    return [...data].sort(
      (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );
  }, [data]);

  const hasData = sorted.length >= 2;

  return (
    <Card className="flex flex-col">
      <CardHeader className="border-b [.border-b]:pb-4">
        <div className="flex items-start gap-2.5">
          {Icon && (
            <span className="mt-0.5 flex size-8 items-center justify-center rounded-lg border border-border text-muted-foreground">
              <Icon className="size-4" />
            </span>
          )}
          <div className="space-y-0.5">
            <CardTitle className="text-base font-semibold">{title}</CardTitle>
            {subtitle && <CardDescription>{subtitle}</CardDescription>}
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex-1 pt-4">
        {hasData ? (
          <ChartContainer
            config={chartConfig}
            className="aspect-auto w-full"
            style={{ height }}
          >
            <AreaChart data={sorted} margin={{ left: 4, right: 12, top: 8, bottom: 0 }}>
              <defs>
                <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--color-balance)" stopOpacity={0.18} />
                  <stop offset="100%" stopColor="var(--color-balance)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid vertical={false} strokeDasharray="3 3" />
              <XAxis
                dataKey="timestamp"
                tickLine={false}
                axisLine={false}
                tickMargin={10}
                minTickGap={48}
                interval="preserveStartEnd"
                tickFormatter={(value) =>
                  new Date(value).toLocaleDateString(undefined, {
                    month: "short",
                    day: "numeric",
                    year: "numeric",
                  })
                }
              />
              <YAxis
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                width={64}
                tickFormatter={(value) =>
                  `$${Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 })}`
                }
              />
              <ChartTooltip
                cursor={{ strokeDasharray: "3 3" }}
                content={
                  <ChartTooltipContent
                    labelFormatter={(value) =>
                      new Date(value as string).toLocaleString(undefined, {
                        dateStyle: "medium",
                        timeStyle: "short",
                      })
                    }
                    formatter={(value) => (
                      <div className="flex w-full items-center justify-between gap-4">
                        <span className="text-muted-foreground">Balance</span>
                        <span className="font-mono font-medium text-foreground">
                          $
                          {Number(value).toLocaleString(undefined, {
                            minimumFractionDigits: 2,
                            maximumFractionDigits: 2,
                          })}
                        </span>
                      </div>
                    )}
                  />
                }
              />
              <Area
                dataKey="balance"
                type="monotone"
                stroke="var(--color-balance)"
                strokeWidth={2}
                fill={`url(#${gradientId})`}
                dot={false}
                activeDot={{ r: 4 }}
              />
            </AreaChart>
          </ChartContainer>
        ) : (
          <div
            className="flex items-center justify-center rounded-lg border border-dashed border-border text-sm text-muted-foreground"
            style={{ height }}
          >
            No equity data available yet.
          </div>
        )}
      </CardContent>
    </Card>
  );
}
