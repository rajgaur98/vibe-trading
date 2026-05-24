"use client";

import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Brain, RefreshCw, Calendar, ArrowRight, BarChart4, Cpu } from "lucide-react";

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

export default function Decisions() {
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  async function fetchDecisions() {
    setRefreshing(true);
    try {
      const res = await fetch("/api/decisions?limit=30");
      if (res.ok) {
        setDecisions(await res.json());
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
    <div className="p-8 space-y-8 flex-1">
      {/* Header */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <h2 className="text-3xl font-extrabold text-slate-100 tracking-tight flex items-center gap-3">
            <Brain className="w-8 h-8 text-emerald-500" />
            Agent Decision Logs
          </h2>
          <p className="text-sm font-medium text-slate-400 mt-1">
            Historical trace of Gemini multi-agent reasoning, indicators snapshot, and risk approvals.
          </p>
        </div>

        <Button
          onClick={fetchDecisions}
          disabled={refreshing}
          variant="outline"
          className="bg-slate-900/60 border-slate-900 text-slate-300 font-bold hover:bg-slate-900/80 gap-2"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? "animate-spin" : ""}`} />
          Refresh Log
        </Button>
      </div>

      {/* Decisions Timeline */}
      {loading ? (
        <div className="py-12 text-center text-slate-500 text-sm font-medium">
          Loading decision log...
        </div>
      ) : decisions.length === 0 ? (
        <Card className="bg-slate-900/40 border-slate-900/60 backdrop-blur-sm shadow-xl p-8 text-center text-slate-500 text-sm">
          No decisions recorded. Run the bot scheduler or trigger on-demand `trade-once` runs to log decisions.
        </Card>
      ) : (
        <div className="space-y-6">
          {decisions.map((dec) => {
            const isFlat = dec.action.toLowerCase() === "flat";
            const isClose = dec.action.toLowerCase() === "close";
            const isLong = dec.action.toLowerCase() === "long";
            const snapshot = dec.agent_transcripts || {};

            return (
              <Card
                key={dec.decision_id}
                className="bg-slate-900/40 border-slate-900/60 backdrop-blur-sm shadow-xl overflow-hidden"
              >
                {/* Panel Header */}
                <div className="p-6 border-b border-slate-900 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                  <div className="space-y-1.5">
                    <div className="flex items-center gap-3">
                      <span className="font-extrabold text-slate-200 text-base">{dec.symbol}</span>
                      <Badge
                        className={`font-extrabold px-2.5 py-0.5 uppercase text-[10px] ${
                          isFlat
                            ? "bg-slate-950/40 text-slate-400 border border-slate-900"
                            : isClose
                            ? "bg-amber-500/10 text-amber-400 border border-amber-500/20"
                            : isLong
                            ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                            : "bg-rose-500/10 text-rose-400 border border-rose-500/20"
                        }`}
                      >
                        {dec.action}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2 text-xs font-semibold text-slate-500">
                      <Calendar className="w-3.5 h-3.5 text-slate-500" />
                      <span>{new Date(dec.timestamp).toLocaleString()}</span>
                    </div>
                  </div>

                  {!isFlat && !isClose && (
                    <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-xs font-semibold text-slate-400 bg-slate-950/20 border border-slate-900 rounded-lg p-3">
                      <div>
                        <span className="text-slate-500 font-bold block mb-0.5">Stop Strategy</span>
                        <span className="uppercase font-extrabold text-slate-300">
                          {dec.stop_loss_strategy.replace("_", " ")}
                        </span>
                      </div>
                      <div className="border-l border-slate-900/80 pl-4">
                        <span className="text-slate-500 font-bold block mb-0.5">Target Profit</span>
                        <span className="uppercase font-extrabold text-slate-300">
                          {dec.take_profit_strategy.replace("_", " ")}
                        </span>
                      </div>
                      <div className="border-l border-slate-900/80 pl-4">
                        <span className="text-slate-500 font-bold block mb-0.5">Risk/Reward</span>
                        <span className="font-extrabold text-slate-300">{dec.risk_reward_ratio}x</span>
                      </div>
                    </div>
                  )}
                </div>

                {/* Panel Body */}
                <CardContent className="p-6 space-y-6">
                  {/* Executive Summary */}
                  <div>
                    <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2 flex items-center gap-2">
                      <ArrowRight className="w-3.5 h-3.5 text-emerald-500" />
                      Trader Decision Reasoning
                    </h4>
                    <p className="text-sm text-slate-300 leading-relaxed font-medium pl-5 border-l-2 border-slate-900">
                      {dec.reasoning_summary}
                    </p>
                  </div>

                  {/* Accordion for Snapshots and Logs */}
                  {Object.keys(snapshot).length > 0 && (
                    <Accordion className="w-full border-t border-slate-900 pt-4">
                      <AccordionItem value="snapshot" className="border-b-0">
                        <AccordionTrigger className="text-xs font-bold text-slate-400 uppercase tracking-wider hover:text-slate-300 hover:no-underline py-2">
                          <span className="flex items-center gap-2">
                            <BarChart4 className="w-4 h-4 text-emerald-500" />
                            Market Snapshot Data
                          </span>
                        </AccordionTrigger>
                        <AccordionContent className="pt-4 pb-2">
                          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
                            {/* Trend Indicators */}
                            <div className="bg-slate-950/20 border border-slate-900 rounded-lg p-4 space-y-3 font-semibold">
                              <h5 className="font-bold text-slate-400 border-b border-slate-900 pb-1.5 flex items-center gap-1.5">
                                <Cpu className="w-3.5 h-3.5 text-emerald-400" />
                                Momentum Indicators
                              </h5>
                              <div className="flex justify-between">
                                <span className="text-slate-500">RSI (14)</span>
                                <span className="text-slate-300">
                                  {snapshot.rsi_14 ? `${snapshot.rsi_14.toFixed(2)} (${snapshot.rsi_regime})` : "N/A"}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-slate-500">ADX (14)</span>
                                <span className="text-slate-300">
                                  {snapshot.adx_14 ? `${snapshot.adx_14.toFixed(2)} (${snapshot.adx_regime})` : "N/A"}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-slate-500">OBV Trend</span>
                                <span className="text-slate-300 capitalize">{snapshot.obv_trend || "N/A"}</span>
                              </div>
                            </div>

                            {/* Technical Structure */}
                            <div className="bg-slate-950/20 border border-slate-900 rounded-lg p-4 space-y-3 font-semibold">
                              <h5 className="font-bold text-slate-400 border-b border-slate-900 pb-1.5 flex items-center gap-1.5">
                                <BarChart4 className="w-3.5 h-3.5 text-cyan-400" />
                                Support & Resistance
                              </h5>
                              <div className="flex justify-between">
                                <span className="text-slate-500">Support Price</span>
                                <span className="text-slate-300">
                                  {snapshot.support_price ? `$${snapshot.support_price.toLocaleString()}` : "N/A"}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-slate-500">Resistance Price</span>
                                <span className="text-slate-300">
                                  {snapshot.resistance_price ? `$${snapshot.resistance_price.toLocaleString()}` : "N/A"}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-slate-500">Pattern Detected</span>
                                <span className="text-slate-300 capitalize">
                                  {snapshot.candlestick_pattern ? snapshot.candlestick_pattern.replace(/_/g, " ") : "None"}
                                </span>
                              </div>
                            </div>

                            {/* Macro & Derivatives */}
                            <div className="bg-slate-950/20 border border-slate-900 rounded-lg p-4 space-y-3 font-semibold col-span-1 md:col-span-2 lg:col-span-1">
                              <h5 className="font-bold text-slate-400 border-b border-slate-900 pb-1.5 flex items-center gap-1.5">
                                <ArrowRight className="w-3.5 h-3.5 text-amber-400" />
                                Derivatives & Macro
                              </h5>
                              <div className="flex justify-between">
                                <span className="text-slate-500">Funding Rate</span>
                                <span className="text-slate-300 uppercase">{snapshot.funding_rate || "Neutral"}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-slate-500">Open Interest Trend</span>
                                <span className="text-slate-300 capitalize">{snapshot.open_interest_trend || "Neutral"}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-slate-500">Macro Event Today</span>
                                <span className={`font-bold ${snapshot.is_macro_event_today ? "text-amber-400 animate-pulse" : "text-slate-400"}`}>
                                  {snapshot.is_macro_event_today ? "YES" : "NO"}
                                </span>
                              </div>
                            </div>
                          </div>
                          
                          {/* Raw JSON Snapshot code block */}
                          <div className="mt-4">
                            <pre className="bg-slate-950/80 border border-slate-900 rounded-lg p-4 text-[10px] text-emerald-400 overflow-x-auto max-h-60 font-mono">
                              {JSON.stringify(snapshot, null, 2)}
                            </pre>
                          </div>
                        </AccordionContent>
                      </AccordionItem>
                    </Accordion>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
