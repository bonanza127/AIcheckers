"use client";

import { useState, useEffect, useCallback, useMemo } from "react";

/* ================================================================
   TYPES
   ================================================================ */
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
  run_id: string;
};
type GpuInfo = { memoryUsed: number; memoryTotal: number; utilization: number };
type RunRecord = {
  run_id: string;
  started_at: string;
  ended_at: string | null;
  final_generation: number;
  best_fitness: number;
  config_name: string;
  status: string;
};
type EvolutionData = {
  status: "ACTIVE" | "OFFLINE" | "STALE";
  history: GenerationRecord[];
  breakthroughs: Breakthrough[];
  runs: RunRecord[];
  latestGenLine: string;
  recentLog: string[];
  gpu: GpuInfo;
  currentRunId: string;
  runStartedAt: string | null;
  maxGenerations: number;
  totalRuns: number;
  updatedAt: number;
};

/* ================================================================
   NERV DESIGN TOKENS
   ================================================================ */
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
  sys: "'IBM Plex Mono','Courier New',monospace",
  label: "'Bebas Neue','Arial Narrow',sans-serif",
  kanji: "'Noto Sans JP',sans-serif",
};

/* ================================================================
   DERIVED STATISTICS
   ================================================================ */
function computeStats(data: EvolutionData | null) {
  const empty = {
    totalGens: 0, bestEver: 0, avgLatest: 0, improvement: 0,
    improvementPct: 0, genPerHour: 0, uptimeStr: "---",
    improvementStreak: 0, bestGeneration: 0, fitnessGain: 0,
    objectiveMaxes: {} as Record<string, { max: number; gen: number }>,
    objectiveTrends: {} as Record<string, number[]>,
    progressPct: 0, eta: "---",
  };
  if (!data || data.history.length === 0) return empty;

  const h = data.history;
  const totalGens = h.length;
  const bestEver = Math.max(...h.map((r) => r.best_fitness));
  const bestGeneration = h.reduce((b, r) => (r.best_fitness > b.best_fitness ? r : b), h[0]).generation;
  const avgLatest = h[h.length - 1].avg_fitness;
  const prev = h.length >= 2 ? h[h.length - 2].best_fitness : h[0].best_fitness;
  const curr = h[h.length - 1].best_fitness;
  const improvement = curr - prev;
  const improvementPct = prev > 0 ? (improvement / prev) * 100 : 0;
  const fitnessGain = curr - h[0].best_fitness;

  let genPerHour = 0;
  let uptimeStr = "---";
  let eta = "---";
  if (h.length >= 2 && h[0].created_at && h[h.length - 1].created_at) {
    const t0 = new Date(h[0].created_at).getTime();
    const tN = new Date(h[h.length - 1].created_at).getTime();
    const elapsedMs = tN - t0;
    if (elapsedMs > 0) {
      genPerHour = ((h.length - 1) / elapsedMs) * 3600000;
      const mins = Math.floor(elapsedMs / 60000);
      const hrs = Math.floor(mins / 60);
      uptimeStr = hrs > 0 ? `${hrs}h ${mins % 60}m` : `${mins}m`;
      const remaining = data.maxGenerations - (latest(h)?.generation ?? 0);
      if (genPerHour > 0) {
        const etaHrs = remaining / genPerHour;
        eta = etaHrs > 1 ? `${etaHrs.toFixed(1)}h` : `${Math.round(etaHrs * 60)}m`;
      }
    }
  }

  let improvementStreak = 0;
  for (let i = h.length - 1; i > 0; i--) {
    if (h[i].best_fitness > h[i - 1].best_fitness + 1e-9) improvementStreak++;
    else break;
  }

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

  const progressPct = data.maxGenerations > 0
    ? ((latest(h)?.generation ?? 0) / data.maxGenerations) * 100 : 0;

  return {
    totalGens, bestEver, avgLatest, improvement, improvementPct,
    genPerHour, uptimeStr, improvementStreak, bestGeneration,
    objectiveMaxes, objectiveTrends, fitnessGain, progressPct, eta,
  };
}

function latest(h: GenerationRecord[]) { return h[h.length - 1]; }

/* ================================================================
   COMPONENTS
   ================================================================ */

/* --- Section Header (bilingual) --- */
function SH({ jp, en }: { jp: string; en: string }) {
  return (
    <div style={{
      fontSize: "0.75rem", fontWeight: 700, letterSpacing: "0.12em",
      textTransform: "uppercase" as const, color: C.orange,
      borderBottom: `1px solid ${C.orangeDim}`, paddingBottom: 3, marginBottom: 6,
      display: "flex", justifyContent: "space-between", alignItems: "baseline",
    }}>
      <span style={{ fontFamily: F.kanji }}>{jp}</span>
      <span style={{ fontSize: "0.5625rem", color: C.orangeDim, fontFamily: F.sys }}>{en}</span>
    </div>
  );
}

/* --- Stat Cell --- */
function Stat({ label, value, unit, color, sub }: {
  label: string; value: string; unit?: string; color?: string; sub?: string;
}) {
  return (
    <div style={{ padding: "4px 6px", borderRight: `1px solid ${C.steelFaint}`, minWidth: 0 }}>
      <div style={{
        fontFamily: F.sys, fontSize: "0.5rem", fontWeight: 400,
        letterSpacing: "0.1em", textTransform: "uppercase" as const, color: C.orangeDim,
        whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
      }}>{label}</div>
      <div style={{
        fontFamily: F.sys, fontSize: "1.1rem", fontWeight: 700,
        color: color || C.green, lineHeight: 1.1, fontVariantNumeric: "tabular-nums",
      }}>
        {value}
        {unit && <span style={{ fontSize: "0.5rem", color: C.steelDim, marginLeft: 2 }}>{unit}</span>}
      </div>
      {sub && <div style={{ fontFamily: F.sys, fontSize: "0.5rem", color: C.steelDim }}>{sub}</div>}
    </div>
  );
}

/* --- Progress Bar (mission progress to max_generations) --- */
function ProgressBar({ pct, label }: { pct: number; label: string }) {
  return (
    <div style={{ padding: "4px 6px" }}>
      <div style={{
        fontFamily: F.sys, fontSize: "0.5rem", letterSpacing: "0.1em",
        textTransform: "uppercase" as const, color: C.orangeDim, marginBottom: 2,
      }}>{label}</div>
      <div style={{ height: 6, background: C.steelFaint, position: "relative" }}>
        <div style={{
          height: "100%", width: `${Math.min(pct, 100)}%`,
          background: pct > 90 ? C.orangeHot : C.cyan,
          boxShadow: `0 0 4px ${pct > 90 ? C.orangeHot : C.cyan}`,
          transition: "width 0.5s",
        }} />
      </div>
      <div style={{
        fontFamily: F.sys, fontSize: "0.5625rem", color: C.steel, marginTop: 1, textAlign: "right",
      }}>{pct.toFixed(1)}%</div>
    </div>
  );
}

/* --- GPU Panel --- */
function GpuPanel({ gpu }: { gpu: GpuInfo }) {
  const pct = gpu.memoryTotal > 0 ? (gpu.memoryUsed / gpu.memoryTotal) * 100 : 0;
  const barColor = pct > 80 ? C.red : pct > 60 ? C.orangeHot : C.cyan;
  return (
    <div style={{ border: `1px solid ${C.cyanDim}`, padding: "6px 8px", background: C.voidWarm }}>
      <div style={{
        fontFamily: F.sys, fontSize: "0.5rem", color: C.orangeDim,
        letterSpacing: "0.1em", textTransform: "uppercase" as const,
        display: "flex", justifyContent: "space-between", marginBottom: 4,
      }}>
        <span>GPU VRAM / メモリ</span>
        <span style={{ color: C.cyan }}>GTX 1660S</span>
      </div>
      <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
        <div style={{ flex: 1, height: 8, background: "rgba(32,240,255,0.06)" }}>
          <div style={{
            height: "100%", width: `${pct}%`, background: barColor,
            boxShadow: `0 0 4px ${barColor}`, transition: "width 0.5s",
          }} />
        </div>
        <span style={{
          fontFamily: F.sys, fontSize: "0.6875rem", color: barColor, fontWeight: 700, width: 36, textAlign: "right",
        }}>{pct.toFixed(0)}%</span>
      </div>
      <div style={{
        fontFamily: F.sys, fontSize: "0.5625rem", color: C.steel,
        display: "flex", justifyContent: "space-between", marginTop: 2,
      }}>
        <span>{gpu.memoryUsed}/{gpu.memoryTotal} MiB</span>
        <span style={{ color: C.cyan }}>UTIL {gpu.utilization}%</span>
      </div>
    </div>
  );
}

/* --- Fitness Chart (SVG) --- */
function FitnessChart({ history }: { history: GenerationRecord[] }) {
  if (history.length < 2) return <div style={{ color: C.steelDim, fontSize: "0.625rem", padding: 12, textAlign: "center" }}>AWAITING MULTI-GEN DATA...</div>;
  const maxFit = Math.max(...history.map((h) => h.best_fitness), 0.01);
  const minFit = Math.min(...history.map((h) => h.avg_fitness));
  const W = 600, H = 160;
  const pad = { top: 8, right: 8, bottom: 20, left: 40 };
  const iW = W - pad.left - pad.right, iH = H - pad.top - pad.bottom;
  const range = maxFit - minFit || 0.01;
  const toY = (v: number) => pad.top + iH - ((v - minFit) / range) * iH;

  const pts = history.map((h, i) => ({
    x: pad.left + (i / Math.max(history.length - 1, 1)) * iW,
    yB: toY(h.best_fitness), yA: toY(h.avg_fitness),
  }));
  const bestP = pts.map((p, i) => `${i ? "L" : "M"}${p.x},${p.yB}`).join(" ");
  const avgP = pts.map((p, i) => `${i ? "L" : "M"}${p.x},${p.yA}`).join(" ");
  const fillP = bestP + " " + [...pts].reverse().map((p, i) => `${i ? "L" : "L"}${p.x},${p.yA}`).join(" ") + " Z";

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }}>
      {[0, 0.25, 0.5, 0.75, 1].map((f) => {
        const val = minFit + range * f, y = toY(val);
        return (
          <g key={f}>
            <line x1={pad.left} y1={y} x2={W - pad.right} y2={y} stroke={C.steelFaint} />
            <text x={pad.left - 3} y={y + 3} textAnchor="end" fill={C.steelDim} fontSize="7" fontFamily={F.sys}>{val.toFixed(3)}</text>
          </g>
        );
      })}
      {history.length > 2 && [0, 0.5, 1].map((f) => {
        const idx = Math.round(f * (history.length - 1));
        const x = pad.left + (idx / Math.max(history.length - 1, 1)) * iW;
        return <text key={f} x={x} y={H - 4} textAnchor="middle" fill={C.steelDim} fontSize="7" fontFamily={F.sys}>{history[idx].generation}</text>;
      })}
      <path d={fillP} fill={C.green} opacity={0.03} />
      <path d={avgP} fill="none" stroke={C.cyanDim} strokeWidth={1} opacity={0.4} />
      <path d={bestP} fill="none" stroke={C.green} strokeWidth={1.5} />
      <path d={bestP} fill="none" stroke={C.green} strokeWidth={4} opacity={0.1} />
      <circle cx={pts[pts.length - 1].x} cy={pts[pts.length - 1].yB} r={2.5} fill={C.green}>
        <animate attributeName="r" values="2.5;4;2.5" dur="2s" repeatCount="indefinite" />
      </circle>
    </svg>
  );
}

/* --- Sparkline --- */
function Spark({ data, color, w = 50, h = 12 }: { data: number[]; color: string; w?: number; h?: number }) {
  if (data.length < 2) return null;
  const max = Math.max(...data, 1e-4), min = Math.min(...data), r = max - min || 1e-4;
  const pts = data.map((v, i) => `${(i / (data.length - 1)) * w},${h - ((v - min) / r) * h}`).join(" ");
  return <svg width={w} height={h} style={{ verticalAlign: "middle" }}><polyline points={pts} fill="none" stroke={color} strokeWidth={1} opacity={0.7} /></svg>;
}

/* --- Objective Table --- */
function ObjTable({ current, maxes, trends }: {
  current: Record<string, number>;
  maxes: Record<string, { max: number; gen: number }>;
  trends: Record<string, number[]>;
}) {
  const rows = Object.entries(current).sort(([, a], [, b]) => b - a);
  if (!rows.length) return null;
  const th = { fontSize: "0.5rem", letterSpacing: "0.1em", textTransform: "uppercase" as const, padding: "2px 4px", color: C.orange, borderBottom: `1px solid ${C.orangeDim}`, fontWeight: 400 as const };
  return (
    <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: F.sys, fontSize: "0.6875rem" }}>
      <thead><tr>
        <th style={{ ...th, textAlign: "left" }}>OBJ</th>
        <th style={{ ...th, textAlign: "right" }}>NOW</th>
        <th style={{ ...th, textAlign: "right" }}>MAX</th>
        <th style={{ ...th, textAlign: "right" }}>@GEN</th>
        <th style={{ ...th, textAlign: "right" }}>TREND</th>
      </tr></thead>
      <tbody>
        {rows.map(([name, score]) => {
          const m = maxes[name], t = trends[name] || [];
          const atMax = m && Math.abs(score - m.max) < 1e-6;
          const tf = name === "transformer_failure";
          return (
            <tr key={name} style={{ borderBottom: `1px solid ${C.greenFaint}` }}>
              <td style={{ padding: "2px 4px", color: C.orangeDim, fontSize: "0.5625rem", textTransform: "uppercase" as const, letterSpacing: "0.04em" }}>
                {tf && <span style={{ color: C.orangeHot }}>★ </span>}{name.replace(/_/g, " ")}
              </td>
              <td style={{ padding: "2px 4px", textAlign: "right", color: tf && score >= 0.7 ? C.orangeHot : C.green, fontWeight: 500, fontVariantNumeric: "tabular-nums" }}>
                {score.toFixed(4)}
              </td>
              <td style={{ padding: "2px 4px", textAlign: "right", color: atMax ? C.orangeHot : C.steelDim, fontVariantNumeric: "tabular-nums" }}>
                {m ? m.max.toFixed(4) : "---"}
              </td>
              <td style={{ padding: "2px 4px", textAlign: "right", color: C.steelDim, fontVariantNumeric: "tabular-nums" }}>
                {m ? m.gen : "-"}
              </td>
              <td style={{ padding: "2px 4px", textAlign: "right" }}>
                <Spark data={t} color={tf ? C.orangeHot : C.greenDim} />
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/* --- History Table --- */
function HistTable({ history }: { history: GenerationRecord[] }) {
  const recent = history.slice(-8).reverse();
  if (!recent.length) return null;
  const th = { fontSize: "0.5rem", letterSpacing: "0.1em", textTransform: "uppercase" as const, padding: "2px 4px", color: C.orange, borderBottom: `1px solid ${C.orangeDim}`, fontWeight: 400 as const };
  return (
    <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: F.sys, fontSize: "0.625rem" }}>
      <thead><tr>
        <th style={{ ...th, textAlign: "right" }}>GEN</th>
        <th style={{ ...th, textAlign: "right" }}>BEST</th>
        <th style={{ ...th, textAlign: "right" }}>AVG</th>
        <th style={{ ...th, textAlign: "right" }}>Δ</th>
        <th style={{ ...th, textAlign: "left" }}>TOP OBJ</th>
      </tr></thead>
      <tbody>
        {recent.map((r, i) => {
          const prev = i < recent.length - 1 ? recent[i + 1] : null;
          const d = prev ? r.best_fitness - prev.best_fitness : 0;
          const top = Object.entries(r.objectives).sort(([, a], [, b]) => b - a)[0];
          return (
            <tr key={r.generation} style={{ borderBottom: `1px solid ${C.greenFaint}` }}>
              <td style={{ padding: "1px 4px", textAlign: "right", color: C.steelDim, fontVariantNumeric: "tabular-nums" }}>{String(r.generation).padStart(4, "0")}</td>
              <td style={{ padding: "1px 4px", textAlign: "right", color: C.green, fontWeight: 500, fontVariantNumeric: "tabular-nums" }}>{r.best_fitness.toFixed(4)}</td>
              <td style={{ padding: "1px 4px", textAlign: "right", color: C.cyanDim, fontVariantNumeric: "tabular-nums" }}>{r.avg_fitness.toFixed(4)}</td>
              <td style={{ padding: "1px 4px", textAlign: "right", fontVariantNumeric: "tabular-nums", color: d > 1e-9 ? C.green : d < -1e-9 ? C.red : C.steelDim }}>{d > 0 ? "+" : ""}{d.toFixed(4)}</td>
              <td style={{ padding: "1px 4px", color: C.orangeDim, fontSize: "0.5rem", textTransform: "uppercase" as const }}>{top ? `${top[0].slice(0, 14)} ${top[1].toFixed(2)}` : ""}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/* --- Run History (past runs) --- */
function RunHistory({ runs }: { runs: RunRecord[]; }) {
  if (!runs.length) return <div style={{ color: C.steelDim, fontSize: "0.5625rem" }}>NO RUNS RECORDED</div>;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      {runs.map((r) => {
        const isActive = r.status === "ACTIVE";
        return (
          <div key={r.run_id} style={{
            display: "flex", justifyContent: "space-between", alignItems: "center",
            padding: "2px 4px", fontSize: "0.5625rem", fontFamily: F.sys,
            borderLeft: `2px solid ${isActive ? C.green : C.steelDim}`,
            background: isActive ? C.greenFaint : "transparent",
          }}>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <span style={{ color: isActive ? C.green : C.steelDim, width: 6, height: 6, borderRadius: "50%", background: isActive ? C.green : C.steelDim, display: "inline-block", boxShadow: isActive ? `0 0 4px ${C.green}` : "none" }} />
              <span style={{ color: C.steelDim, fontVariantNumeric: "tabular-nums" }}>
                {new Date(r.started_at).toLocaleDateString("ja-JP")} {new Date(r.started_at).toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit" })}
              </span>
              <span style={{ color: C.orangeDim }}>{r.config_name || "---"}</span>
            </div>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <span style={{ color: C.green, fontVariantNumeric: "tabular-nums" }}>G{r.final_generation}</span>
              <span style={{ color: C.cyan, fontVariantNumeric: "tabular-nums" }}>{r.best_fitness.toFixed(4)}</span>
              <span style={{ color: isActive ? C.green : C.steelDim, fontSize: "0.5rem", letterSpacing: "0.06em" }}>
                {r.status}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* --- Breakthrough Panel --- */
function Breakthroughs({ items }: { items: Breakthrough[] }) {
  if (!items.length) return (
    <div style={{ border: `1px solid ${C.cyanDim}`, padding: 10, background: C.voidWarm, textAlign: "center" }}>
      <div style={{ fontFamily: F.sys, fontSize: "0.625rem", color: C.steelDim }}>パラダイムシフト未検出</div>
      <div style={{ fontFamily: F.sys, fontSize: "0.5rem", color: C.steelDim, marginTop: 2 }}>MONITORING FOR TRANSFORMER-EXCEEDING SIGNALS...</div>
    </div>
  );
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {items.map((bt, i) => (
        <div key={i} style={{ border: `2px solid ${C.red}`, padding: 8, background: C.redFill, animation: "alertPulse 3s infinite" }}>
          <div style={{ fontFamily: F.label, fontSize: "0.875rem", color: C.red, letterSpacing: "0.1em" }}>★ BREAKTHROUGH — GEN {bt.generation}</div>
          <div style={{ fontFamily: F.sys, fontSize: "0.625rem", color: C.steel, marginTop: 2 }}>FITNESS: {bt.fitness.toFixed(4)} | AGENT: {bt.agent_id}</div>
          <div style={{ fontFamily: F.sys, fontSize: "0.5625rem", color: C.orangeHot, marginTop: 1 }}>{bt.signals.join(" | ")}</div>
        </div>
      ))}
    </div>
  );
}

/* --- Log Console --- */
function LogConsole({ lines }: { lines: string[] }) {
  return (
    <div style={{
      background: C.void, border: `1px solid ${C.cyanDim}`,
      padding: 4, maxHeight: 180, overflowY: "auto",
      fontFamily: F.sys, fontSize: "0.5625rem", lineHeight: 1.4,
    }}>
      {!lines.length && <div style={{ color: C.steelDim, padding: 6, textAlign: "center", fontSize: "0.5rem" }}>ログ待機中...</div>}
      {lines.map((line, i) => {
        let c = C.steelDim;
        if (line.includes("ERROR")) c = C.red;
        else if (line.includes("WARNING") || line.includes("BREAKTHROUGH")) c = C.orangeHot;
        else if (line.includes("GEN ")) c = C.green;
        else if (line.includes("INFO")) c = C.greenDim;
        return <div key={i} style={{ color: c, whiteSpace: "pre-wrap", wordBreak: "break-all" }}>{line}</div>;
      })}
    </div>
  );
}

/* --- Ticker Tape --- */
function Ticker({ data, stats }: { data: EvolutionData; stats: ReturnType<typeof computeStats> }) {
  const segs = [
    `GEN ${data.history[data.history.length - 1]?.generation ?? "?"}/${data.maxGenerations}`,
    `BEST ${stats.bestEver.toFixed(4)}`,
    `${stats.genPerHour.toFixed(1)} GEN/HR`,
    `UPTIME ${stats.uptimeStr}`,
    `ETA ${stats.eta}`,
    `GPU ${data.gpu.utilization}%`,
    `VRAM ${data.gpu.memoryUsed}/${data.gpu.memoryTotal}`,
    `STREAK +${stats.improvementStreak}`,
    `GAIN ${stats.fitnessGain >= 0 ? "+" : ""}${stats.fitnessGain.toFixed(4)}`,
    `RUNS ${data.totalRuns}`,
    `BT ${data.breakthroughs.length}`,
  ];
  const text = segs.join("  ///  ");
  return (
    <div style={{
      overflow: "hidden", whiteSpace: "nowrap", background: C.void,
      borderTop: `1px solid ${C.orangeDim}`, borderBottom: `1px solid ${C.orangeDim}`,
      padding: "2px 0", fontFamily: F.sys, fontSize: "0.5rem",
      letterSpacing: "0.06em", textTransform: "uppercase" as const, color: C.orange,
    }}>
      <div style={{ display: "inline-block", animation: "ticker-scroll 45s linear infinite", paddingLeft: "100%" }}>
        {text}  ///  {text}
      </div>
    </div>
  );
}

/* ================================================================
   MAIN PAGE
   ================================================================ */
export default function TesterPage() {
  const [data, setData] = useState<EvolutionData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch("/api/evolution", { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
      setError(null);
    } catch (err) { setError(String(err)); }
  }, []);

  useEffect(() => {
    fetchData();
    const iv = setInterval(fetchData, 10000);
    return () => clearInterval(iv);
  }, [fetchData]);

  const stats = useMemo(() => computeStats(data), [data]);
  const lat = data?.history?.[data.history.length - 1];
  const sc = data?.status === "ACTIVE" ? C.green : data?.status === "STALE" ? C.orangeHot : C.red;

  return (
    <>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=Bebas+Neue&family=Noto+Sans+JP:wght@400;700&display=swap" rel="stylesheet" />
      <style>{`
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
        @keyframes alertPulse{0%,100%{border-color:${C.red}}50%{border-color:${C.redHot}}}
        @keyframes ticker-scroll{from{transform:translateX(0)}to{transform:translateX(-50%)}}
        @keyframes phosphor{0%,100%{opacity:1}50%{opacity:0.97}}
        body{margin:0;padding:0}
        @media(prefers-reduced-motion:reduce){*{animation:none!important}}
      `}</style>

      {/* CRT scanlines */}
      <div style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 9999, background: "repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.05) 2px,rgba(0,0,0,0.05) 4px)" }} />
      {/* CRT vignette */}
      <div style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 9998, background: "radial-gradient(ellipse at center,transparent 65%,rgba(0,0,0,0.25) 100%)" }} />

      <div style={{ minHeight: "100vh", background: C.void, color: C.steel, fontFamily: F.sys, animation: "phosphor 0.08s infinite" }}>

        {/* === HEADER === */}
        <div style={{
          borderBottom: `2px solid ${C.orange}`, padding: "10px 16px",
          display: "flex", justifyContent: "space-between", alignItems: "center",
        }}>
          <div>
            <div style={{ fontFamily: F.label, fontSize: "clamp(1rem, 2.5vw, 1.6rem)", letterSpacing: "0.15em", color: C.orange, textTransform: "uppercase" as const }}>
              進化監視システム / EVOLUTION MONITOR
            </div>
            <div style={{ fontFamily: F.sys, fontSize: "0.5rem", color: C.steelDim, letterSpacing: "0.08em", textTransform: "uppercase" as const, marginTop: 1 }}>
              NERV PARADIGM SEARCH DIVISION — REAL-TIME TELEMETRY
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", backgroundColor: sc, boxShadow: `0 0 6px ${sc}`, animation: data?.status === "ACTIVE" ? "pulse 2s infinite" : "none" }} />
            <span style={{ fontFamily: F.sys, fontSize: "0.6875rem", fontWeight: 700, color: sc, letterSpacing: "0.08em" }}>{data?.status || "OFFLINE"}</span>
            <span style={{ fontFamily: F.sys, fontSize: "0.5rem", color: C.steelDim, marginLeft: 6 }}>
              {data?.updatedAt ? new Date(data.updatedAt).toLocaleTimeString("ja-JP") : "---"}
            </span>
          </div>
        </div>

        {/* === TICKER === */}
        {data && stats.totalGens > 0 && <Ticker data={data} stats={stats} />}

        {/* === ERROR === */}
        {error && <div style={{ margin: "6px 16px", padding: 6, border: `1px solid ${C.red}`, color: C.red, fontSize: "0.625rem" }}>CONNECTION ERROR: {error}</div>}

        {/* === STAT STRIP === */}
        <div style={{
          display: "grid", gridTemplateColumns: "repeat(10, 1fr)", gap: 0,
          padding: "4px 16px", borderBottom: `1px solid ${C.steelFaint}`,
        }}>
          <Stat label="世代 / GEN" value={lat ? `${lat.generation}` : "---"} sub={`/ ${data?.maxGenerations ?? 300}`} />
          <Stat label="最高 / BEST" value={stats.bestEver > 0 ? stats.bestEver.toFixed(4) : "---"} sub={`@gen ${stats.bestGeneration}`} />
          <Stat label="平均 / AVG" value={stats.avgLatest > 0 ? stats.avgLatest.toFixed(4) : "---"} color={C.cyan} />
          <Stat label="前世代比 / Δ" value={stats.totalGens > 1 ? `${stats.improvement >= 0 ? "+" : ""}${stats.improvement.toFixed(4)}` : "---"} color={stats.improvement > 0 ? C.green : stats.improvement < -1e-9 ? C.red : C.steelDim} />
          <Stat label="速度 / RATE" value={stats.genPerHour > 0 ? stats.genPerHour.toFixed(1) : "---"} unit="g/h" color={C.cyan} />
          <Stat label="稼働 / UP" value={stats.uptimeStr} color={C.steel} />
          <Stat label="ETA" value={stats.eta} color={C.steelDim} />
          <Stat label="連続↑ / STREAK" value={`+${stats.improvementStreak}`} color={stats.improvementStreak >= 3 ? C.orangeHot : C.steelDim} />
          <Stat label="総ゲイン / GAIN" value={stats.fitnessGain !== 0 ? `${stats.fitnessGain >= 0 ? "+" : ""}${stats.fitnessGain.toFixed(4)}` : "---"} color={stats.fitnessGain > 0 ? C.green : C.steelDim} />
          <ProgressBar pct={stats.progressPct} label="進捗 / PROGRESS" />
        </div>

        {/* === MAIN 3-COLUMN GRID === */}
        <div style={{ display: "grid", gridTemplateColumns: "5fr 3fr 2fr", gap: 10, padding: "10px 16px" }}>

          {/* --- COL 1: Charts & Tables --- */}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ border: `1px solid ${C.cyanDim}`, padding: 8, background: C.voidWarm }}>
              <SH jp="適応度推移" en="FITNESS TRAJECTORY" />
              <FitnessChart history={data?.history || []} />
              {data?.history && data.history.length > 1 && (
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.5rem", color: C.steelDim, marginTop: 2 }}>
                  <span><span style={{ color: C.green }}>━</span> BEST &nbsp; <span style={{ color: C.cyanDim }}>━</span> AVG</span>
                  <span>BEST @ GEN {stats.bestGeneration}</span>
                </div>
              )}
            </div>

            <div style={{ border: `1px solid ${C.cyanDim}`, padding: 8, background: C.voidWarm }}>
              <SH jp="目標別スコア" en="OBJECTIVE BREAKDOWN" />
              {lat?.objectives ? <ObjTable current={lat.objectives} maxes={stats.objectiveMaxes} trends={stats.objectiveTrends} /> : <div style={{ color: C.steelDim, fontSize: "0.625rem" }}>NO DATA</div>}
            </div>

            <div style={{ border: `1px solid ${C.cyanDim}`, padding: 8, background: C.voidWarm }}>
              <SH jp="世代履歴" en="GENERATION HISTORY" />
              <HistTable history={data?.history || []} />
            </div>
          </div>

          {/* --- COL 2: GPU, Breakthroughs, Log --- */}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {data?.gpu && <GpuPanel gpu={data.gpu} />}

            <div>
              <SH jp="パラダイムシフト" en="BREAKTHROUGHS" />
              <Breakthroughs items={data?.breakthroughs || []} />
            </div>

            <div>
              <SH jp="実行ログ" en="CONSOLE" />
              <LogConsole lines={data?.recentLog || []} />
            </div>

            {data?.latestGenLine && (
              <div style={{ border: `1px solid ${C.cyanDim}`, padding: 6, background: C.voidWarm }}>
                <div style={{ fontSize: "0.5rem", color: C.orangeDim, letterSpacing: "0.08em", textTransform: "uppercase" as const, marginBottom: 1 }}>LATEST READOUT</div>
                <div style={{ fontSize: "0.625rem", color: C.green, whiteSpace: "pre-wrap", wordBreak: "break-all" }}>{data.latestGenLine}</div>
              </div>
            )}
          </div>

          {/* --- COL 3: Run Info --- */}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {/* Run Config */}
            <div style={{ border: `1px solid ${C.orangeDim}`, padding: "6px 8px", background: C.void }}>
              <div style={{ fontFamily: F.label, fontSize: "0.875rem", letterSpacing: "0.12em", color: C.orange, textTransform: "uppercase" as const, marginBottom: 3 }}>
                実行構成 / RUN CONFIG
              </div>
              {[
                ["RUN ID", data?.currentRunId?.split("_").slice(-1)[0] || "---"],
                ["CONFIG", "paradigm_v1"],
                ["MODEL", "arc_v1/518071"],
                ["POP", "8"],
                ["WORKERS", "8"],
                ["MAX GEN", String(data?.maxGenerations ?? 300)],
                ["SELECT", "cma_map_elites"],
                ["MUTATE", "gaussian"],
                ["CROSS", "linear+ties+dare"],
              ].map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "0.5px 0", fontSize: "0.5rem", letterSpacing: "0.06em", textTransform: "uppercase" as const }}>
                  <span style={{ color: C.steelDim }}>{k}</span>
                  <span style={{ color: C.green, fontVariantNumeric: "tabular-nums" }}>{v}</span>
                </div>
              ))}
            </div>

            {/* Run History */}
            <div style={{ border: `1px solid ${C.cyanDim}`, padding: "6px 8px", background: C.voidWarm }}>
              <SH jp="実行履歴" en="RUN HISTORY" />
              <RunHistory runs={data?.runs || []} />
            </div>

            {/* System Info */}
            <div style={{ border: `1px solid ${C.steelFaint}`, padding: "6px 8px", background: C.void }}>
              <div style={{ fontSize: "0.5rem", color: C.orangeDim, letterSpacing: "0.1em", textTransform: "uppercase" as const, marginBottom: 3 }}>
                SYSTEM / システム情報
              </div>
              {[
                ["REFRESH", "10s AUTO"],
                ["SOURCE", "SUPABASE"],
                ["GPU", "GTX 1660 SUPER 6GB"],
                ["RUNS", String(data?.totalRuns ?? 0)],
                ["LAST SYNC", data?.updatedAt ? new Date(data.updatedAt).toLocaleTimeString("ja-JP") : "---"],
              ].map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "0.5px 0", fontSize: "0.5rem", textTransform: "uppercase" as const }}>
                  <span style={{ color: C.steelDim }}>{k}</span>
                  <span style={{ color: C.steel }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
