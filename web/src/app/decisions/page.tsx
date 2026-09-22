"use client";

import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Skeleton } from "@/components/ui/skeleton";
import PageHeader from "@/components/PageHeader";
import ActionBadge from "@/components/ActionBadge";
import { Calendar, FileText, BarChart3, Activity, LineChart, Layers } from "lucide-react";

interface Decision {
  decision_id: string;
  timestamp: string;
  symbol: string;
  action: string;
  stop_loss_strategy: string;
  take_profit_strategy: string;
  risk_reward_ratio: number;
  reasoning_summary: string;
  agent_transcripts: any;
}

function SnapshotRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium text-foreground">{value}</span>
    </div>
  );
}

export default function Decisions() {
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  async function fetchDecisions() {
    setRefreshing(true);
    try {
      const res = await fetch("/api/decisions?limit=30");
      if (res.ok) {
        setDecisions(await res.json());
        setLastUpdated(new Date());
      }
    } catch (err) {
      console.error("Error fetching decisions:", err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  useEffect(() => {
    fetchDecisions();
  }, []);

  return (
    <div className="space-y-8 p-6 sm:p-8">
      <PageHeader
        eyebrow="Reasoning Trace"
        title="Agent Decision Logs"
        subtitle="Historical trace of Gemini multi-agent reasoning, indicators snapshot, and risk approvals."
        lastUpdated={lastUpdated}
        onRefresh={fetchDecisions}
        refreshing={refreshing}
        refreshLabel="Refresh"
      />

      {loading ? (
        <div className="space-y-4">
          {[...Array(4)].map((_, i) => (
            <Card key={i} className="gap-0 px-5 py-4">
              <div className="flex items-center justify-between">
                <div className="space-y-2">
                  <div className="flex items-center gap-2.5">
                    <Skeleton className="h-5 w-28" />
                    <Skeleton className="h-5 w-14" />
                  </div>
                  <Skeleton className="h-3.5 w-40" />
                </div>
                <Skeleton className="h-6 w-32 rounded-full" />
              </div>
              <Separator className="my-4" />
              <Skeleton className="h-3.5 w-full" />
              <Skeleton className="mt-2 h-3.5 w-5/6" />
            </Card>
          ))}
        </div>
      ) : decisions.length === 0 ? (
        <Card className="p-10 text-center text-sm text-muted-foreground">
          No decisions recorded. Run the bot scheduler or trigger on-demand runs to log decisions.
        </Card>
      ) : (
        <div className="space-y-4">
          {decisions.map((dec, idx) => {
            const isTradable = !["flat", "close"].includes(dec.action.toLowerCase());
            const snapshot = dec.agent_transcripts || {};
            const hasSnapshot = Object.keys(snapshot).length > 0;

            return (
              <Card key={dec.decision_id} className="gap-0 overflow-hidden py-0">
                <Accordion defaultValue={idx === 0 ? ["content"] : []}>
                  <AccordionItem value="content" className="border-b-0">
                    <AccordionTrigger className="items-center px-5 py-4 hover:no-underline">
                      <div className="flex flex-1 items-center justify-between gap-4 pr-3">
                        <div className="space-y-1.5">
                          <div className="flex items-center gap-2.5">
                            <span className="text-base font-semibold text-foreground">{dec.symbol}</span>
                            <ActionBadge action={dec.action} />
                          </div>
                          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                            <Calendar className="size-3.5" />
                            <span suppressHydrationWarning>{new Date(dec.timestamp).toLocaleString()}</span>
                          </div>
                        </div>
                        <span className="hidden items-center gap-1.5 rounded-full border border-border bg-muted/60 px-2.5 py-1 text-xs font-medium text-muted-foreground sm:flex">
                          <span className="size-1.5 rounded-full bg-gain" />
                          Decision Logged
                        </span>
                      </div>
                    </AccordionTrigger>

                    <AccordionContent className="px-5">
                      <Separator className="mb-4" />

                      <div className="space-y-2">
                        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                          <FileText className="size-3.5" />
                          Trader Decision Reasoning
                        </div>
                        <p className="text-sm leading-relaxed text-muted-foreground">{dec.reasoning_summary}</p>
                      </div>

                      {isTradable && (
                        <div className="mt-4 flex flex-wrap gap-x-8 gap-y-2 rounded-lg border border-border bg-muted/40 p-3 text-xs">
                          <div>
                            <p className="text-muted-foreground">Stop Strategy</p>
                            <p className="mt-0.5 font-semibold uppercase text-foreground">{dec.stop_loss_strategy.replace("_", " ")}</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">Target Profit</p>
                            <p className="mt-0.5 font-semibold uppercase text-foreground">{dec.take_profit_strategy.replace("_", " ")}</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">Risk / Reward</p>
                            <p className="mt-0.5 font-semibold text-foreground">{dec.risk_reward_ratio}x</p>
                          </div>
                        </div>
                      )}

                      {hasSnapshot && (
                        <div className="mt-4">
                          <Accordion>
                            <AccordionItem value="snapshot" className="border-b-0">
                              <AccordionTrigger className="py-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground hover:no-underline">
                                <span className="flex items-center gap-2">
                                  <BarChart3 className="size-4" />
                                  Market Snapshot Data
                                </span>
                              </AccordionTrigger>
                              <AccordionContent className="pt-3">
                                <div className="grid grid-cols-1 gap-3 text-xs md:grid-cols-3">
                                  <div className="space-y-2.5 rounded-lg border border-border bg-muted/40 p-4">
                                    <h5 className="flex items-center gap-1.5 border-b border-border pb-1.5 font-semibold text-foreground">
                                      <Activity className="size-3.5 text-muted-foreground" />
                                      Momentum
                                    </h5>
                                    <SnapshotRow label="RSI (14)" value={snapshot.rsi_14 ? `${snapshot.rsi_14.toFixed(2)} (${snapshot.rsi_regime})` : "N/A"} />
                                    <SnapshotRow label="ADX (14)" value={snapshot.adx_14 ? `${snapshot.adx_14.toFixed(2)} (${snapshot.adx_regime})` : "N/A"} />
                                    <SnapshotRow label="OBV Trend" value={<span className="capitalize">{snapshot.obv_trend || "N/A"}</span>} />
                                  </div>
                                  <div className="space-y-2.5 rounded-lg border border-border bg-muted/40 p-4">
                                    <h5 className="flex items-center gap-1.5 border-b border-border pb-1.5 font-semibold text-foreground">
                                      <LineChart className="size-3.5 text-muted-foreground" />
                                      Support &amp; Resistance
                                    </h5>
                                    <SnapshotRow label="Support" value={snapshot.support_price ? `$${snapshot.support_price.toLocaleString()}` : "N/A"} />
                                    <SnapshotRow label="Resistance" value={snapshot.resistance_price ? `$${snapshot.resistance_price.toLocaleString()}` : "N/A"} />
                                    <SnapshotRow label="Pattern" value={<span className="capitalize">{snapshot.candlestick_pattern ? snapshot.candlestick_pattern.replace(/_/g, " ") : "None"}</span>} />
                                  </div>
                                  <div className="space-y-2.5 rounded-lg border border-border bg-muted/40 p-4">
                                    <h5 className="flex items-center gap-1.5 border-b border-border pb-1.5 font-semibold text-foreground">
                                      <Layers className="size-3.5 text-muted-foreground" />
                                      Derivatives &amp; Macro
                                    </h5>
                                    <SnapshotRow label="Funding Rate" value={<span className="uppercase">{snapshot.funding_rate || "Neutral"}</span>} />
                                    <SnapshotRow label="Open Interest" value={<span className="capitalize">{snapshot.open_interest_trend || "Neutral"}</span>} />
                                    <SnapshotRow
                                      label="Macro Event Today"
                                      value={<span className={snapshot.is_macro_event_today ? "font-semibold text-loss" : ""}>{snapshot.is_macro_event_today ? "YES" : "NO"}</span>}
                                    />
                                  </div>
                                </div>

                                <pre className="mt-4 max-h-60 overflow-auto rounded-lg border border-border bg-muted/60 p-4 font-mono text-[11px] leading-relaxed text-foreground">
                                  {JSON.stringify(snapshot, null, 2)}
                                </pre>
                              </AccordionContent>
                            </AccordionItem>
                          </Accordion>
                        </div>
                      )}
                    </AccordionContent>
                  </AccordionItem>
                </Accordion>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
