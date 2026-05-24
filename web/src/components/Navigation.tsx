"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { LayoutDashboard, Brain, History, RefreshCw, Activity } from "lucide-react";

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

  return (
    <aside className="w-64 border-r border-emerald-950/20 bg-slate-950/40 backdrop-blur-md flex flex-col justify-between p-6">
      <div className="space-y-8">
        {/* Brand/Logo */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-emerald-500 to-cyan-500 flex items-center justify-center font-bold text-white shadow-[0_0_15px_rgba(16,185,129,0.3)]">
            V
          </div>
          <div>
            <h1 className="text-lg font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
              VIBE TRADING
            </h1>
            <p className="text-[10px] text-slate-500 font-medium tracking-widest uppercase">
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
                className={`flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200 ${
                  isActive
                    ? "bg-gradient-to-r from-emerald-950/40 to-slate-900 border-l-2 border-emerald-500 text-emerald-400 shadow-[0_0_15px_rgba(16,185,129,0.05)]"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/50"
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? "text-emerald-400" : "text-slate-400"}`} />
                {link.label}
              </Link>
            );
          })}
        </nav>
      </div>

      {/* System Status Footer */}
      <div className="border-t border-slate-900 pt-6">
        <div className="flex items-center justify-between rounded-lg bg-slate-900/40 border border-slate-900/60 p-3">
          <div className="flex items-center gap-2">
            <Activity className={`w-4 h-4 ${status?.status === "online" ? "text-emerald-500 animate-pulse" : "text-slate-500"}`} />
            <div>
              <p className="text-xs font-semibold text-slate-300">System Status</p>
              <p className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
                {loading ? "Checking..." : status?.mode ? `${status.mode} MODE` : "Offline"}
              </p>
            </div>
          </div>
          {!loading && status?.status === "online" && (
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_8px_#10b981]" />
          )}
        </div>
      </div>
    </aside>
  );
}
