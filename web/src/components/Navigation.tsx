"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { LayoutDashboard, Brain, History } from "lucide-react";
import { cn } from "@/lib/utils";

export default function Navigation() {
  const pathname = usePathname();
  const [status, setStatus] = useState<{ status: string; mode: string } | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchStatus() {
      try {
        const res = await fetch("/api/status");
        if (res.ok) {
          const data = await res.json();
          setStatus(data);
        }
      } catch (err) {
        console.error("Failed to fetch system status:", err);
      } finally {
        setLoading(false);
      }
    }
    fetchStatus();
    // Poll every 10 seconds for online status
    const interval = setInterval(fetchStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const links = [
    { href: "/", label: "Dashboard", icon: LayoutDashboard },
    { href: "/decisions", label: "Agent Decisions", icon: Brain },
    { href: "/trades", label: "Trade History", icon: History },
  ];

  // Friendly labels for the raw TRADING_MODE enum (LIVE_TESTNET = Binance Futures demo).
  const MODE_LABELS: Record<string, string> = {
    LIVE_TESTNET: "DEMO TRADING",
    LIVE_SANDBOX: "LIVE SANDBOX",
    PAPER: "PAPER MODE",
  };
  const modeLabel = status?.mode ? MODE_LABELS[status.mode] ?? `${status.mode} MODE` : "Offline";
  const isOnline = status?.status === "online";
  const statusText = loading
    ? "Checking status…"
    : isOnline
      ? "All systems operational"
      : "Systems offline";

  return (
    <aside className="flex w-64 shrink-0 flex-col justify-between border-r border-border bg-card px-4 py-6">
      <div className="space-y-8">
        {/* Brand / Logo */}
        <div className="flex items-center gap-3 px-2">
          <div className="flex size-9 items-center justify-center rounded-xl bg-primary text-base font-bold text-primary-foreground">
            V
          </div>
          <div className="leading-tight">
            <h1 className="text-sm font-semibold tracking-tight text-foreground">
              Vibe Trading
            </h1>
            <p className="text-[10px] font-medium uppercase tracking-[0.16em] text-muted-foreground">
              Agentic Quant Bot
            </p>
          </div>
        </div>

        {/* Navigation Links */}
        <nav className="space-y-1">
          {links.map((link) => {
            const Icon = link.icon;
            const isActive = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                aria-current={isActive ? "page" : undefined}
                className={cn(
                  "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-secondary text-foreground"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground"
                )}
              >
                <Icon className="size-4 shrink-0" />
                {link.label}
              </Link>
            );
          })}
        </nav>
      </div>

      {/* System Status Footer */}
      <div
        className="flex items-center gap-2.5 rounded-xl border border-border bg-card px-3 py-2.5"
        title={loading ? undefined : `Trading mode: ${modeLabel}`}
      >
        <span className="relative flex size-2.5">
          {isOnline && (
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-gain opacity-60" />
          )}
          <span
            className={cn(
              "relative inline-flex size-2.5 rounded-full",
              isOnline ? "bg-gain" : "bg-muted-foreground"
            )}
          />
        </span>
        <div className="leading-tight">
          <p className="text-xs font-semibold text-foreground">System Status</p>
          <p className="text-[11px] font-medium text-muted-foreground">{statusText}</p>
        </div>
      </div>
    </aside>
  );
}
