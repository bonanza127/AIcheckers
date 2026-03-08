"use client";

import { useState, useEffect, useCallback, useMemo } from "react";

/* ---------- types ---------- */
type GenerationRecord = {
  generation: number;
  best_fitness: number;
  avg_fitness: number;
  objectives: Record<string, number>;
  created_at: string;
};

type Breakthrough = {
  generation: number;
  signals: string[];
  agent_id: string;
  fitness: number;
  created_at: string;
};

type GpuInfo = { memoryUsed: number; memoryTotal: number; utilization: number };

type EvolutionData = {
  status: "ACTIVE" | "OFFLINE" | "STALE";
  history: GenerationRecord[];
  breakthroughs: Breakthrough[];
  latestGenLine: string;
  recentLog: string[];
  gpu: GpuInfo;
  updatedAt: number;
};

/* ---------- NERV color tokens ---------- */
const C = {
  void: "#000000",
  voidWarm: "#0A0A08",
  voidPanel: "#111110",
  orange: "#FF9830",
  orangeDim: "#D08028",
  orangeHot: "#FFCC50",
  green: "#50FF50",
  greenDim: "#30BB30",
  greenFaint: "rgba(80,255,80,0.08)",
  cyan: "#20F0FF",
  cyanDim: "#10A8B8",
  cyanGlow: "rgba(32,240,255,0.15)",
  red: "#FF4840",
  redHot: "#FF6858",
  redFill: "rgba(255,72,64,0.18)",
  steel: "#E0E0D8",
  steelDim: "#9A9A90",
  steelFaint: "rgba(224,224,216,0.08)",
};

const F = {
  sys: "'IBM Plex Mono', 'Courier New', monospace",
  label: "'Bebas Neue', 'Arial Narrow', sans-serif",
  kanji: "'Noto Sans JP', sans-serif",
};

/* ---------- derived stats ---------- */
function computeStats(data: EvolutionData | null) {
  if (!data || data.history.length === 0)
    return {
      totalGens: 0, bestEver: 0, avgLatest: 0, improvement: 0,
      improvementPct: 0, genPerHour: 0, uptimeStr: "---",
      improvementStreak: 0, bestGeneration: 0,
      objectiveMaxes: {} as Record<string, { max: number; gen: number }>,
      objectiveTrends: {} as Record<string, number[]>,
      fitnessGain: 0,
    };

  const h = data.history;
  const totalGens = h.length;
  const bestEver = Math.max(...h.map((r) => r.best_fitness));
  const bestGeneration = h.reduce((best, r) => (r.best_fitness > best.best_fitness ? r : best), h[0]).generation;
  const avgLatest = h[h.length - 1].avg_fitness;
  const prev = h.length >= 2 ? h[h.length - 2].best_fitness : h[0].best_fitness;
  const curr = h[h.length - 1].best_fitness;
  const improvement = curr - prev;
  const improvementPct = prev > 0 ? (improvement / prev) * 100 : 0;
  const fitnessGain = curr - h[0].best_fitness;

  // Gen/hour from timestamps
  let genPerHour = 0;
  let uptimeStr = "---";
  if (h.length >= 2 && h[0].created_at && h[h.length - 1].created_at) {
    const t0 = new Date(h[0].created_at).getTime();
    const tN = new Date(h[h.length - 1].created_at).getTime();
    const elapsedMs = tN - t0;
    if (elapsedMs > 0) {
      genPerHour = ((h.length - 1) / elapsedMs) * 3600000;
      const mins = Math.floor(elapsedMs / 60000);
      const hrs = Math.floor(mins / 60);
      uptimeStr = hrs > 0 ? `${hrs}h ${mins % 60}m` : `${mins}m`;
    }
  }

  // Improvement streak
  let improvementStreak = 0;
  for (let i = h.length - 1; i > 0; i--) {
    if (h[i].best_fitness > h[i - 1].best_fitness + 1e-9) improvementStreak++;
    else break;
  }

  // Objective maxes and trends
  const objectiveMaxes: Record<string, { max: number; gen: number }> = {};
  const objectiveTrends: Record<string, number[]> = {};
  for (const rec of h) {
    for (const [name, score] of Object.entries(rec.objectives)) {
      if (!objectiveMaxes[name] || score > objectiveMaxes[name].max) {
        objectiveMaxes[name] = { max: score, gen: rec.generation };
      }
      if (!objectiveTrends[name]) objectiveTrends[name] = [];
      objectiveTrends[name].push(score);
    }
  }

  return {
    totalGens, bestEver, avgLatest, improvement, improvementPct,
    genPerHour, uptimeStr, improvementStreak, bestGeneration,
    objectiveMaxes, objectiveTrends, fitnessGain,
  };
}

/* ---------- Section Header ---------- */
function SectionHeader({ jp, en }: { jp: string; en: string }) {
  return (
    <div style={{
      fontSize: "0.875rem", fontWeight: 700, letterSpacing: "0.12em",
      textTransform: "uppercase" as const, color: C.orange, marginBottom: 8,
      borderBottom: `1px solid ${C.orangeDim}`, paddingBottom: 4,
      display: "flex", justifyContent: "space-between", alignItems: "baseline",
    }}>
      <span>{jp}</span>
      <span style={{ fontSize: "0.625rem", color: C.orangeDim, letterSpacing: "0.08em" }}>{en}</span>
    </div>
  );
}

/* ---------- StatBlock (compact) ---------- */
function StatBlock({
  label, value, unit, color, sub,
}: {
  label: string; value: string; unit?: string; color?: string; sub?: string;
}) {
  return (
    <div style={{ padding: "6px 8px", borderRight: `1px solid ${C.steelFaint}` }}>
      <div style={{
        fontFamily: F.sys, fontSize: "0.5625rem", fontWeight: 400,
        letterSpacing: "0.08em", textTransform: "uppercase" as const, color: C.orangeDim,
      }}>
        {label}
      </div>
      <div style={{
        fontFamily: F.sys, fontSize: "1.25rem", fontWeight: 700,
        color: color || C.green, lineHeight: 1.2, fontVariantNumeric: "tabular-nums",
      }}>
        {value}
        {unit && <span style={{ fontSize: "0.625rem", color: C.steelDim, marginLeft: 3 }}>{unit}</span>}
      </div>
      {sub && (
        <div style={{ fontFamily: F.sys, fontSize: "0.5625rem", color: C.steelDim, marginTop: 1 }}>
          {sub}
        </div>
      )}
    </div>
  );
}

/* ---------- GPU Meter ---------- */
function GpuMeter({ gpu }: { gpu: GpuInfo }) {
  const pct = gpu.memoryTotal > 0 ? (gpu.memoryUsed / gpu.memoryTotal) * 100 : 0;
  const barColor = pct > 80 ? C.red : pct > 60 ? C.orangeHot : C.cyan;
  return (
    <div style={{ border: `1px solid ${C.cyanDim}`, padding: "8px 10px", background: C.voidWarm }}>
      <div style={{
        fontFamily: F.sys, fontSize: "0.5625rem", color: C.orangeDim,
        letterSpacing: "0.08em", textTransform: "uppercase" as const, marginBottom: 6,
        display: "flex", justifyContent: "space-between",
      }}>
        <span>GPU VRAM</span>
        <span style={{ color: C.cyan }}>GTX 1660 SUPER</span>
      </div>
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <div style={{ flex: 1, height: 8, background: "rgba(32,240,255,0.08)" }}>
          <div style={{
            height: "100%", width: `${pct}%`, background: barColor,
            boxShadow: `0 0 6px ${barColor}`, transition: "width 0.5s",
          }} />
        </div>
        <span style={{ fontFamily: F.sys, fontSize: "0.75rem", color: barColor, fontWeight: 700, width: 42, textAlign: "right" }}>
          {pct.toFixed(0)}%
        </span>
      </div>
      <div style={{
        fontFamily: F.sys, fontSize: "0.625rem", color: C.steel,
        display: "flex", justifyContent: "space-between", marginTop: 4,
      }}>
        <span>{gpu.memoryUsed} / {gpu.memoryTotal} MiB</span>
        <span style={{ color: C.cyan }}>UTIL {gpu.utilization}%</span>
      </div>
    </div>
  );
}

/* ---------- Fitness Chart ---------- */
function FitnessChart({ history }: { history: GenerationRecord[] }) {
  if (history.length === 0) return null;
  const maxFit = Math.max(...history.map((h) => h.best_fitness), 0.01);
  const minFit = Math.min(...history.map((h) => h.avg_fitness));
  const width = 600;
  const height = 180;
  const pad = { top: 10, right: 10, bottom: 24, left: 44 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const range = maxFit - minFit || 0.01;

  const toY = (v: number) => pad.top + innerH - ((v - minFit) / range) * innerH;

  const points = history.map((h, i) => ({
    x: pad.left + (i / Math.max(history.length - 1, 1)) * innerW,
    yBest: toY(h.best_fitness),
    yAvg: toY(h.avg_fitness),
  }));

  const bestPath = points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.yBest}`).join(" ");
  const avgPath = points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.yAvg}`).join(" ");

  // Fill area between best and avg
  const fillPath = bestPath + " " +
    [...points].reverse().map((p, i) => `${i === 0 ? "L" : "L"}${p.x},${p.yAvg}`).join(" ") + " Z";

  return (
    <svg viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", height: "auto" }}>
      {/* Grid */}
      {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
        const val = minFit + range * frac;
        const y = toY(val);
        return (
          <g key={frac}>
            <line x1={pad.left} y1={y} x2={width - pad.right} y2={y} stroke={C.steelFaint} />
            <text x={pad.left - 4} y={y + 3} textAnchor="end" fill={C.steelDim}
              fontSize="8" fontFamily={F.sys}>{val.toFixed(3)}</text>
          </g>
        );
      })}
      {/* X-axis labels */}
      {history.length > 1 && [0, 0.5, 1].map((frac) => {
        const idx = Math.round(frac * (history.length - 1));
        const x = pad.left + (idx / Math.max(history.length - 1, 1)) * innerW;
        return (
          <text key={frac} x={x} y={height - 4} textAnchor="middle" fill={C.steelDim}
            fontSize="8" fontFamily={F.sys}>GEN {history[idx].generation}</text>
        );
      })}
      {/* Fill */}
      <path d={fillPath} fill={C.green} opacity={0.04} />
      {/* Avg line */}
      <path d={avgPath} fill="none" stroke={C.cyanDim} strokeWidth={1} opacity={0.5} />
      {/* Best line */}
      <path d={bestPath} fill="none" stroke={C.green} strokeWidth={1.5} />
      <path d={bestPath} fill="none" stroke={C.green} strokeWidth={5} opacity={0.12} />
      {/* Latest point glow */}
      {points.length > 0 && (
        <circle cx={points[points.length - 1].x} cy={points[points.length - 1].yBest}
          r={3} fill={C.green} opacity={0.9}>
          <animate attributeName="r" values="3;5;3" dur="2s" repeatCount="indefinite" />
        </circle>
      )}
    </svg>
  );
}

/* ---------- Sparkline (mini chart for objectives) ---------- */
function Sparkline({ data, color, width = 60, height = 16 }: {
  data: number[]; color: string; width?: number; height?: number;
}) {
  if (data.length < 2) return null;
  const max = Math.max(...data, 0.001);
  const min = Math.min(...data);
  const range = max - min || 0.001;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / range) * height;
    return `${x},${y}`;
  });
  return (
    <svg width={width} height={height} style={{ verticalAlign: "middle" }}>
      <polyline points={pts.join(" ")} fill="none" stroke={color} strokeWidth={1} opacity={0.7} />
    </svg>
  );
}

/* ---------- Objective Table ---------- */
function ObjectiveTable({
  current, maxes, trends,
}: {
  current: Record<string, number>;
  maxes: Record<string, { max: number; gen: number }>;
  trends: Record<string, number[]>;
}) {
  const entries = Object.entries(current).sort(([, a], [, b]) => b - a);
  if (entries.length === 0) return null;

  return (
    <table style={{
      width: "100%", borderCollapse: "collapse", fontFamily: F.sys, fontSize: "0.75rem",
    }}>
      <thead>
        <tr>
          {["OBJECTIVE", "CURRENT", "BEST", "GEN", "TREND"].map((h) => (
            <th key={h} style={{
              fontSize: "0.5625rem", letterSpacing: "0.1em", textTransform: "uppercase" as const,
              textAlign: h === "OBJECTIVE" ? "left" : "right", padding: "4px 6px",
              color: C.orange, borderBottom: `1px solid ${C.orangeDim}`, fontWeight: 400,
            }}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {entries.map(([name, score]) => {
          const m = maxes[name];
          const t = trends[name] || [];
          const isAtMax = m && Math.abs(score - m.max) < 1e-6;
          const isTF = name === "transformer_failure";
          return (
            <tr key={name} style={{ borderBottom: `1px solid ${C.greenFaint}` }}>
              <td style={{
                padding: "3px 6px", color: C.orangeDim, textTransform: "uppercase" as const,
                letterSpacing: "0.04em", fontSize: "0.625rem",
              }}>
                {isTF ? "★ " : ""}{name}
              </td>
              <td style={{
                padding: "3px 6px", textAlign: "right",
                color: isTF && score >= 0.7 ? C.orangeHot : C.green, fontWeight: 500,
                fontVariantNumeric: "tabular-nums",
              }}>
                {score.toFixed(4)}
              </td>
              <td style={{
                padding: "3px 6px", textAlign: "right",
                color: isAtMax ? C.orangeHot : C.steelDim, fontVariantNumeric: "tabular-nums",
              }}>
                {m ? m.max.toFixed(4) : "---"}
              </td>
              <td style={{
                padding: "3px 6px", textAlign: "right", color: C.steelDim,
                fontVariantNumeric: "tabular-nums",
              }}>
                {m ? m.gen : "---"}
              </td>
              <td style={{ padding: "3px 6px", textAlign: "right" }}>
                <Sparkline data={t} color={isTF ? C.orangeHot : C.greenDim} />
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/* ---------- Breakthrough Panel ---------- */
function BreakthroughPanel({ breakthroughs }: { breakthroughs: Breakthrough[] }) {
  if (breakthroughs.length === 0) {
    return (
      <div style={{
        border: `1px solid ${C.cyanDim}`, padding: 12, background: C.voidWarm, textAlign: "center",
      }}>
        <div style={{ fontFamily: F.sys, fontSize: "0.6875rem", color: C.steelDim }}>
          パラダイムシフト未検出
        </div>
        <div style={{ fontFamily: F.sys, fontSize: "0.5625rem", color: C.steelDim, marginTop: 2 }}>
          MONITORING FOR TRANSFORMER-EXCEEDING SIGNALS...
        </div>
      </div>
    );
  }
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {breakthroughs.map((bt, i) => (
        <div key={i} style={{
          border: `2px solid ${C.red}`, padding: 10,
          background: C.redFill, animation: "alertPulse 3s infinite",
        }}>
          <div style={{
            fontFamily: F.label, fontSize: "1rem", color: C.red, letterSpacing: "0.1em",
          }}>
            ★ BREAKTHROUGH — GEN {bt.generation}
          </div>
          <div style={{ fontFamily: F.sys, fontSize: "0.6875rem", color: C.steel, marginTop: 2 }}>
            FITNESS: {bt.fitness.toFixed(4)} | AGENT: {bt.agent_id}
          </div>
          <div style={{ fontFamily: F.sys, fontSize: "0.625rem", color: C.orangeHot, marginTop: 2 }}>
            {bt.signals.join(" | ")}
          </div>
        </div>
      ))}
    </div>
  );
}

/* ---------- Log Console ---------- */
function LogConsole({ lines }: { lines: string[] }) {
  return (
    <div style={{
      background: C.void, border: `1px solid ${C.cyanDim}`,
      padding: 6, maxHeight: 220, overflowY: "auto",
      fontFamily: F.sys, fontSize: "0.625rem", lineHeight: 1.5,
    }}>
      {lines.length === 0 && (
        <div style={{ color: C.steelDim, padding: 8, textAlign: "center" }}>
          ログはローカルプッシュ後に表示されます
        </div>
      )}
      {lines.map((line, i) => {
        let color = C.steelDim;
        if (line.includes("ERROR")) color = C.red;
        else if (line.includes("WARNING") || line.includes("BREAKTHROUGH")) color = C.orangeHot;
        else if (line.includes("GEN ")) color = C.green;
        else if (line.includes("INFO")) color = C.greenDim;
        return (
          <div key={i} style={{ color, whiteSpace: "pre-wrap", wordBreak: "break-all" }}>{line}</div>
        );
      })}
    </div>
  );
}

/* ---------- History Table ---------- */
function HistoryTable({ history }: { history: GenerationRecord[] }) {
  const recent = history.slice(-10).reverse();
  if (recent.length === 0) return null;
  return (
    <table style={{
      width: "100%", borderCollapse: "collapse", fontFamily: F.sys, fontSize: "0.6875rem",
    }}>
      <thead>
        <tr>
          {["GEN", "BEST", "AVG", "Δ BEST", "TOP OBJ"].map((h) => (
            <th key={h} style={{
              fontSize: "0.5625rem", letterSpacing: "0.1em", textTransform: "uppercase" as const,
              textAlign: h === "TOP OBJ" ? "left" : "right", padding: "3px 5px",
              color: C.orange, borderBottom: `1px solid ${C.orangeDim}`, fontWeight: 400,
            }}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {recent.map((r, i) => {
          const prev = i < recent.length - 1 ? recent[i + 1] : null;
          const delta = prev ? r.best_fitness - prev.best_fitness : 0;
          const topObj = Object.entries(r.objectives).sort(([, a], [, b]) => b - a)[0];
          return (
            <tr key={r.generation} style={{ borderBottom: `1px solid ${C.greenFaint}` }}>
              <td style={{ padding: "2px 5px", textAlign: "right", color: C.steelDim, fontVariantNumeric: "tabular-nums" }}>
                {String(r.generation).padStart(4, "0")}
              </td>
              <td style={{ padding: "2px 5px", textAlign: "right", color: C.green, fontWeight: 500, fontVariantNumeric: "tabular-nums" }}>
                {r.best_fitness.toFixed(4)}
              </td>
              <td style={{ padding: "2px 5px", textAlign: "right", color: C.cyanDim, fontVariantNumeric: "tabular-nums" }}>
                {r.avg_fitness.toFixed(4)}
              </td>
              <td style={{
                padding: "2px 5px", textAlign: "right", fontVariantNumeric: "tabular-nums",
                color: delta > 0 ? C.green : delta < -1e-9 ? C.red : C.steelDim,
              }}>
                {delta > 0 ? "+" : ""}{delta.toFixed(4)}
              </td>
              <td style={{ padding: "2px 5px", color: C.orangeDim, fontSize: "0.5625rem", textTransform: "uppercase" as const }}>
                {topObj ? `${topObj[0].slice(0, 12)} ${topObj[1].toFixed(2)}` : "---"}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/* ---------- Ticker ---------- */
function Ticker({ data, stats }: { data: EvolutionData; stats: ReturnType<typeof computeStats> }) {
  const text = [
    `GEN ${stats.totalGens}`,
    `BEST ${stats.bestEver.toFixed(4)}`,
    `${stats.genPerHour.toFixed(1)} GEN/HR`,
    `UPTIME ${stats.uptimeStr}`,
    `GPU ${data.gpu.utilization}%`,
    `VRAM ${data.gpu.memoryUsed}/${data.gpu.memoryTotal} MiB`,
    `STREAK +${stats.improvementStreak}`,
    `GAIN ${stats.fitnessGain >= 0 ? "+" : ""}${stats.fitnessGain.toFixed(4)}`,
    `BREAKTHROUGHS ${data.breakthroughs.length}`,
  ].join("  ///  ");
  return (
    <div style={{
      overflow: "hidden", whiteSpace: "nowrap", background: C.void,
      borderTop: `1px solid ${C.orangeDim}`, borderBottom: `1px solid ${C.orangeDim}`,
      padding: "3px 0", fontFamily: F.sys, fontSize: "0.5625rem",
      letterSpacing: "0.06em", textTransform: "uppercase" as const, color: C.orange,
    }}>
      <div style={{
        display: "inline-block",
        animation: "ticker-scroll 40s linear infinite",
        paddingLeft: "100%",
      }}>
        {text}  ///  {text}
      </div>
    </div>
  );
}

/* ========== MAIN PAGE ========== */
export default function TesterPage() {
  const [data, setData] = useState<EvolutionData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch("/api/evolution", { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
      setError(null);
    } catch (err) {
      setError(String(err));
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const stats = useMemo(() => computeStats(data), [data]);
  const latest = data?.history?.[data.history.length - 1];
  const statusColor = data?.status === "ACTIVE" ? C.green : data?.status === "STALE" ? C.orangeHot : C.red;

  return (
    <>
      <link
        href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=Bebas+Neue&family=Noto+Sans+JP:wght@400;700&display=swap"
        rel="stylesheet"
      />
      <style>{`
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        @keyframes alertPulse { 0%,100%{border-color:${C.red}} 50%{border-color:${C.redHot}} }
        @keyframes ticker-scroll { from{transform:translateX(0)} to{transform:translateX(-50%)} }
        body { margin:0; padding:0; }
        @media (prefers-reduced-motion:reduce) {
          .crt-flicker,.alert-pulse { animation:none!important; }
        }
      `}</style>

      {/* CRT scanline overlay */}
      <div style={{
        position: "fixed", inset: 0, pointerEvents: "none", zIndex: 9999,
        background: "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.05) 2px, rgba(0,0,0,0.05) 4px)",
      }} />

      {/* CRT vignette */}
      <div style={{
        position: "fixed", inset: 0, pointerEvents: "none", zIndex: 9998,
        background: "radial-gradient(ellipse at center, transparent 60%, rgba(0,0,0,0.3) 100%)",
      }} />

      <div style={{
        minHeight: "100vh", background: C.void, color: C.steel, fontFamily: F.sys,
      }}>
        {/* HEADER */}
        <div style={{
          borderBottom: `1px solid ${C.orangeDim}`, padding: "12px 20px",
          display: "flex", justifyContent: "space-between", alignItems: "center",
        }}>
          <div>
            <div style={{
              fontFamily: F.label, fontSize: "clamp(1.2rem, 3vw, 1.8rem)",
              letterSpacing: "0.15em", color: C.orange, textTransform: "uppercase" as const,
            }}>
              進化監視システム / EVOLUTION MONITOR
            </div>
            <div style={{
              fontFamily: F.sys, fontSize: "0.5625rem", color: C.steelDim,
              letterSpacing: "0.08em", textTransform: "uppercase" as const, marginTop: 2,
            }}>
              NERV PARADIGM SEARCH DIVISION — REAL-TIME TELEMETRY — {new Date().toISOString().slice(0, 10)}
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{
              width: 8, height: 8, borderRadius: "50%",
              backgroundColor: statusColor,
              boxShadow: `0 0 8px ${statusColor}`,
              animation: data?.status === "ACTIVE" ? "pulse 2s infinite" : "none",
            }} />
            <span style={{
              fontFamily: F.sys, fontSize: "0.75rem", fontWeight: 500,
              color: statusColor, letterSpacing: "0.08em",
            }}>
              {data?.status || "OFFLINE"}
            </span>
            <span style={{
              fontFamily: F.sys, fontSize: "0.5625rem", color: C.steelDim, marginLeft: 8,
            }}>
              {data?.updatedAt ? new Date(data.updatedAt).toLocaleTimeString("ja-JP") : "---"}
            </span>
          </div>
        </div>

        {/* TICKER */}
        {data && stats.totalGens > 0 && <Ticker data={data} stats={stats} />}

        {/* ERROR */}
        {error && (
          <div style={{
            margin: "8px 20px", padding: 8, border: `1px solid ${C.red}`,
            color: C.red, fontSize: "0.6875rem",
          }}>
            CONNECTION ERROR: {error}
          </div>
        )}

        {/* STAT STRIP */}
        <div style={{
          display: "grid", gridTemplateColumns: "repeat(8, 1fr)", gap: 0,
          padding: "8px 20px", borderBottom: `1px solid ${C.steelFaint}`,
        }}>
          <StatBlock label="世代 / GEN" value={stats.totalGens > 0 ? String(latest?.generation ?? 0) : "---"} />
          <StatBlock label="最高適応度 / BEST" value={stats.bestEver > 0 ? stats.bestEver.toFixed(4) : "---"} />
          <StatBlock label="平均 / AVG" value={stats.avgLatest > 0 ? stats.avgLatest.toFixed(4) : "---"} color={C.cyan} />
          <StatBlock
            label="前世代比 / Δ"
            value={stats.totalGens > 1 ? `${stats.improvement >= 0 ? "+" : ""}${stats.improvement.toFixed(4)}` : "---"}
            color={stats.improvement > 0 ? C.green : stats.improvement < 0 ? C.red : C.steelDim}
          />
          <StatBlock
            label="速度 / RATE"
            value={stats.genPerHour > 0 ? stats.genPerHour.toFixed(1) : "---"}
            unit="gen/h"
            color={C.cyan}
          />
          <StatBlock label="稼働時間 / UPTIME" value={stats.uptimeStr} color={C.steel} />
          <StatBlock
            label="連続改善 / STREAK"
            value={stats.improvementStreak > 0 ? `+${stats.improvementStreak}` : "0"}
            color={stats.improvementStreak >= 3 ? C.orangeHot : C.steelDim}
          />
          <StatBlock
            label="総ゲイン / GAIN"
            value={stats.totalGens > 0 ? `${stats.fitnessGain >= 0 ? "+" : ""}${stats.fitnessGain.toFixed(4)}` : "---"}
            color={stats.fitnessGain > 0 ? C.green : C.steelDim}
            sub={stats.improvementPct !== 0 ? `${stats.improvementPct >= 0 ? "+" : ""}${stats.improvementPct.toFixed(1)}% vs prev` : undefined}
          />
        </div>

        {/* MAIN GRID */}
        <div style={{
          display: "grid", gridTemplateColumns: "5fr 3fr", gap: 12, padding: "12px 20px",
        }}>
          {/* LEFT COLUMN */}
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {/* Fitness chart */}
            <div style={{ border: `1px solid ${C.cyanDim}`, padding: 10, background: C.voidWarm }}>
              <SectionHeader jp="適応度推移" en="FITNESS TRAJECTORY" />
              {data?.history && data.history.length > 0 ? (
                <FitnessChart history={data.history} />
              ) : (
                <div style={{ color: C.steelDim, fontSize: "0.6875rem", padding: 16, textAlign: "center" }}>
                  AWAITING DATA...
                </div>
              )}
              {data?.history && data.history.length > 0 && (
                <div style={{
                  display: "flex", justifyContent: "space-between", marginTop: 4,
                  fontSize: "0.5625rem", color: C.steelDim,
                }}>
                  <span>BEST: {C.green} / AVG: {C.cyanDim}</span>
                  <span>BEST GEN: {stats.bestGeneration}</span>
                </div>
              )}
            </div>

            {/* Objective table */}
            <div style={{ border: `1px solid ${C.cyanDim}`, padding: 10, background: C.voidWarm }}>
              <SectionHeader jp="目標別スコア" en="OBJECTIVE BREAKDOWN" />
              {latest?.objectives ? (
                <ObjectiveTable
                  current={latest.objectives}
                  maxes={stats.objectiveMaxes}
                  trends={stats.objectiveTrends}
                />
              ) : (
                <div style={{ color: C.steelDim, fontSize: "0.6875rem" }}>NO DATA</div>
              )}
            </div>

            {/* Generation history table */}
            <div style={{ border: `1px solid ${C.cyanDim}`, padding: 10, background: C.voidWarm }}>
              <SectionHeader jp="世代履歴" en="GENERATION HISTORY (RECENT 10)" />
              {data?.history ? (
                <HistoryTable history={data.history} />
              ) : (
                <div style={{ color: C.steelDim, fontSize: "0.6875rem" }}>NO DATA</div>
              )}
            </div>
          </div>

          {/* RIGHT COLUMN */}
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {/* GPU */}
            {data?.gpu && <GpuMeter gpu={data.gpu} />}

            {/* Run Config ID block */}
            <div style={{
              border: `1px solid ${C.orangeDim}`, padding: "8px 10px", background: C.void,
            }}>
              <div style={{
                fontFamily: F.label, fontSize: "1rem", letterSpacing: "0.12em",
                color: C.orange, textTransform: "uppercase" as const, marginBottom: 4,
              }}>
                実行構成 / RUN CONFIG
              </div>
              {[
                ["CONFIG", "pilot_paradigm_search_v1"],
                ["BASE MODEL", "arc_v1_public/step_518071"],
                ["POPULATION", "8"],
                ["WORKERS", "8"],
                ["MAX GENS", "300"],
                ["SELECTION", "cma_map_elites"],
                ["MUTATION", "gaussian"],
                ["CROSSOVER", "linear"],
              ].map(([k, v]) => (
                <div key={k} style={{
                  display: "flex", justifyContent: "space-between", padding: "1px 0",
                  fontSize: "0.5625rem", letterSpacing: "0.06em", textTransform: "uppercase" as const,
                }}>
                  <span style={{ color: C.steelDim }}>{k}</span>
                  <span style={{ color: C.green }}>{v}</span>
                </div>
              ))}
            </div>

            {/* Breakthrough panel */}
            <div>
              <SectionHeader jp="パラダイムシフト" en="BREAKTHROUGHS" />
              <BreakthroughPanel breakthroughs={data?.breakthroughs || []} />
            </div>

            {/* Log console */}
            <div>
              <SectionHeader jp="実行ログ" en="CONSOLE OUTPUT" />
              <LogConsole lines={data?.recentLog || []} />
            </div>

            {/* Latest gen line */}
            {data?.latestGenLine && (
              <div style={{
                border: `1px solid ${C.cyanDim}`, padding: 8, background: C.voidWarm,
              }}>
                <div style={{
                  fontSize: "0.5625rem", color: C.orangeDim, letterSpacing: "0.08em",
                  textTransform: "uppercase" as const, marginBottom: 2,
                }}>
                  LATEST READOUT
                </div>
                <div style={{
                  fontSize: "0.6875rem", color: C.green, whiteSpace: "pre-wrap", wordBreak: "break-all",
                }}>
                  {data.latestGenLine}
                </div>
              </div>
            )}

            {/* Footer info */}
            <div style={{
              fontSize: "0.5rem", color: C.steelDim, textAlign: "right",
              letterSpacing: "0.05em", textTransform: "uppercase" as const,
            }}>
              AUTO-REFRESH: 10s | SOURCE: SUPABASE REALTIME
              <br />
              LAST SYNC: {data?.updatedAt ? new Date(data.updatedAt).toLocaleTimeString("ja-JP") : "---"}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
