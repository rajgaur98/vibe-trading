import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function PageHeader({
  eyebrow,
  title,
  subtitle,
  lastUpdated,
  onRefresh,
  refreshing = false,
  refreshLabel = "Refresh",
  children,
}: {
  eyebrow?: string;
  title: string;
  subtitle?: string;
  lastUpdated?: Date | null;
  onRefresh?: () => void;
  refreshing?: boolean;
  refreshLabel?: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
      <div className="space-y-1">
        {eyebrow && (
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">
            {eyebrow}
          </p>
        )}
        <h1 className="text-3xl font-bold tracking-tight text-foreground">{title}</h1>
        {subtitle && <p className="text-sm text-muted-foreground">{subtitle}</p>}
      </div>

      <div className="flex items-center gap-3">
        {children}
        {onRefresh && (
          <Button variant="outline" size="sm" onClick={onRefresh} disabled={refreshing}>
            <RefreshCw className={refreshing ? "animate-spin" : ""} />
            {refreshLabel}
          </Button>
        )}
        {lastUpdated !== undefined && (
          <div className="text-right leading-tight">
            <p className="text-[11px] text-muted-foreground">Last updated</p>
            <p className="text-xs font-medium text-foreground" suppressHydrationWarning>
              {lastUpdated
                ? lastUpdated.toLocaleString(undefined, {
                    dateStyle: "medium",
                    timeStyle: "short",
                  })
                : "—"}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
