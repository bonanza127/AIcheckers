"use client";

import { useState, useEffect, useCallback, useMemo } from "react";

/* ================================================================
   TYPES
   ================================================================ */
type GenRecord = {
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
  history: GenRecord[];
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
  orange: "#FF9830",
  orangeDim: "#D08028",
  orangeHot: "#FFCC50",
  green: "#50FF50",
  greenDim: "#30BB30",
  greenFaint: "rgba(80,255,80,0.08)",
  cyan: "#20F0FF",
  cyanDim: "#10A8B8",
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
   DERIVED STATS
   ================================================================ */
function deriveStats(data: EvolutionData | null) {
  if (!data?.history?.length) return null;

  const h = data.history;
  const cur = h[h.length - 1];
  const bestEver = Math.max(...h.map((r) => r.best_fitness));
  const bestGen = h.reduce((b, r) => (r.best_fitness > b.best_fitness ? r : b), h[0]).generation;
  const tfScore = cur.objectives?.transformer_failure ?? 0;

  // Uptime from first history entry
  let uptimeStr = "---";
  let eta = "---";
  let genPerHour = 0;
  if (h.length >= 2 && h[0].created_at && cur.created_at) {
    const elapsed = new Date(cur.created_at).getTime() - new Date(h[0].created_at).getTime();
    if (elapsed > 0) {
      genPerHour = ((h.length - 1) / elapsed) * 3_600_000;
      const m = Math.floor(elapsed / 60_000);
      uptimeStr = m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : `${m}m`;
      const rem = data.maxGenerations - cur.generation;
      if (genPerHour > 0) {
        const etaH = rem / genPerHour;
        eta = etaH > 1 ? `${etaH.toFixed(1)}h` : `${Math.round(etaH * 60)}m`;
      }
    }
  }
  // Also count total uptime across all runs
  let totalUptimeMs = 0;
  for (const run of data.runs) {
    const start = new Date(run.started_at).getTime();
    const end = run.ended_at ? new Date(run.ended_at).getTime() : Date.now();
    totalUptimeMs += end - start;
  }
  const totalMins = Math.floor(totalUptimeMs / 60_000);
  const totalUptimeStr = totalMins >= 60
    ? `${Math.floor(totalMins / 60)}h ${totalMins % 60}m`
    : `${totalMins}m`;

  const progressPct = data.maxGenerations > 0 ? (cur.generation / data.maxGenerations) * 100 : 0;

  // Objective trends
  const objTrends: Record<string, number[]> = {};
  const objMaxes: Record<string, { max: number; gen: number }> = {};
  for (const rec of h) {
    for (const [name, score] of Object.entries(rec.objectives)) {
      if (!objTrends[name]) objTrends[name] = [];
      objTrends[name].push(score);
      if (!objMaxes[name] || score > objMaxes[name].max) objMaxes[name] = { max: score, gen: rec.generation };
    }
  }

  return {
    cur, bestEver, bestGen, tfScore, uptimeStr, totalUptimeStr, eta,
    progressPct, objTrends, objMaxes,
  };
}

/* ================================================================
   COMPONENTS
   ================================================================ */

function SH({ jp, en }: { jp: string; en: string }) {
  return (
    <div style={{
      fontSize: "0.6875rem", fontWeight: 700, letterSpacing: "0.12em",
      textTransform: "uppercase" as const, color: C.orange,
      borderBottom: `1px solid ${C.orangeDim}`, paddingBottom: 2, marginBottom: 6,
      display: "flex", justifyContent: "space-between", alignItems: "baseline",
    }}>
      <span style={{ fontFamily: F.kanji }}>{jp}</span>
      <span style={{ fontSize: "0.5rem", color: C.orangeDim, fontFamily: F.sys }}>{en}</span>
    </div>
  );
}

/* --- Big KPI: transformer_failure score --- */
function TFGauge({ score, threshold }: { score: number; threshold: number }) {
  const pct = Math.min(score / threshold, 1) * 100;
  const isBreakthrough = score >= threshold;
  const color = isBreakthrough ? C.orangeHot : score > threshold * 0.5 ? C.orange : C.steelDim;
  return (
    <div style={{
      border: `2px solid ${isBreakthrough ? C.orangeHot : C.cyanDim}`,
      padding: "10px 12px", background: isBreakthrough ? "rgba(255,204,80,0.06)" : C.voidWarm,
      animation: isBreakthrough ? "alertPulse 2s infinite" : "none",
    }}>
      <div style={{
        fontFamily: F.sys, fontSize: "0.5rem", letterSpacing: "0.1em",
        textTransform: "uppercase" as const, color: C.orangeDim, marginBottom: 4,
      }}>
        ★ TRANSFORMER FAILURE SCORE / 変革指標
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
        <span style={{
          fontFamily: F.sys, fontSize: "2.2rem", fontWeight: 700,
          color, lineHeight: 1, fontVariantNumeric: "tabular-nums",
        }}>
          {score.toFixed(4)}
        </span>
        <span style={{ fontFamily: F.sys, fontSize: "0.625rem", color: C.steelDim }}>
          / {threshold.toFixed(1)} THRESHOLD
        </span>
      </div>
      <div style={{ height: 6, background: C.steelFaint, marginTop: 6 }}>
        <div style={{
          height: "100%", width: `${pct}%`,
          background: color, boxShadow: `0 0 4px ${color}`, transition: "width 0.5s",
        }} />
      </div>
      {isBreakthrough && (
        <div style={{
          fontFamily: F.label, fontSize: "1.2rem", color: C.orangeHot,
          letterSpacing: "0.15em", marginTop: 6, textAlign: "center",
        }}>
          ◀ ◀ ◀ PARADIGM SHIFT DETECTED ▶ ▶ ▶
        </div>
      )}
    </div>
  );
}

/* --- Progress / Mission Status --- */
function MissionStatus({ gen, max, pct, eta, uptime, totalUptime, totalRuns }: {
  gen: number; max: number; pct: number; eta: string; uptime: string; totalUptime: string; totalRuns: number;
}) {
  return (
    <div style={{ border: `1px solid ${C.cyanDim}`, padding: "8px 10px", background: C.voidWarm }}>
      <div style={{
        display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 6,
      }}>
        <span style={{ fontFamily: F.sys, fontSize: "0.5rem", color: C.orangeDim, letterSpacing: "0.1em", textTransform: "uppercase" as const }}>
          探索進捗 / SEARCH PROGRESS
        </span>
        <span style={{ fontFamily: F.sys, fontSize: "0.5rem", color: C.steelDim }}>
          RUN {totalRuns}
        </span>
      </div>
      {/* Big gen counter */}
      <div style={{ display: "flex", alignItems: "baseline", gap: 4, marginBottom: 4 }}>
        <span style={{ fontFamily: F.sys, fontSize: "1.8rem", fontWeight: 700, color: C.green, lineHeight: 1, fontVariantNumeric: "tabular-nums" }}>
          {gen}
        </span>
        <span style={{ fontFamily: F.sys, fontSize: "0.75rem", color: C.steelDim }}>/ {max}</span>
      </div>
      {/* Progress bar */}
      <div style={{ height: 8, background: C.steelFaint, marginBottom: 6 }}>
        <div style={{
          height: "100%", width: `${Math.min(pct, 100)}%`,
          background: pct > 90 ? C.orangeHot : C.cyan,
          boxShadow: `0 0 4px ${pct > 90 ? C.orangeHot : C.cyan}`,
          transition: "width 0.5s",
        }} />
      </div>
      {/* Stats row */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 0, fontFamily: F.sys, fontSize: "0.5625rem" }}>
        {[
          ["稼働時間", uptime, C.steel],
          ["総稼働", totalUptime, C.steelDim],
          ["残り", eta, C.cyan],
        ].map(([label, val, c]) => (
          <div key={label as string} style={{ borderRight: `1px solid ${C.steelFaint}`, padding: "2px 4px" }}>
            <div style={{ fontSize: "0.4375rem", color: C.orangeDim, letterSpacing: "0.08em", textTransform: "uppercase" as const }}>{label}</div>
            <div style={{ color: c as string, fontWeight: 500 }}>{val}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* --- GPU Mini --- */
function GpuMini({ gpu }: { gpu: GpuInfo }) {
  const pct = gpu.memoryTotal > 0 ? (gpu.memoryUsed / gpu.memoryTotal) * 100 : 0;
  const c = pct > 80 ? C.red : pct > 60 ? C.orangeHot : C.cyan;
  return (
    <div style={{ border: `1px solid ${C.cyanDim}`, padding: "6px 8px", background: C.voidWarm }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontFamily: F.sys, fontSize: "0.4375rem", color: C.orangeDim, letterSpacing: "0.1em", textTransform: "uppercase" as const, marginBottom: 3 }}>
        <span>GPU / GTX 1660S</span>
        <span style={{ color: C.cyan }}>UTIL {gpu.utilization}%</span>
      </div>
      <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
        <div style={{ flex: 1, height: 5, background: C.steelFaint }}>
          <div style={{ height: "100%", width: `${pct}%`, background: c, transition: "width 0.5s" }} />
        </div>
        <span style={{ fontFamily: F.sys, fontSize: "0.5625rem", color: c, fontWeight: 700 }}>{gpu.memoryUsed}/{gpu.memoryTotal}</span>
      </div>
    </div>
  );
}

/* --- Fitness Chart --- */
function FitnessChart({ history }: { history: GenRecord[] }) {
  if (history.length < 2) return <div style={{ color: C.steelDim, fontSize: "0.5625rem", padding: 10, textAlign: "center" }}>COLLECTING DATA...</div>;
  const vals = history.map((h) => h.best_fitness);
  const mx = Math.max(...vals, 0.01), mn = Math.min(...vals);
  const W = 580, H = 140, p = { t: 6, r: 6, b: 18, l: 38 };
  const iW = W - p.l - p.r, iH = H - p.t - p.b, rng = mx - mn || 0.01;
  const toY = (v: number) => p.t + iH - ((v - mn) / rng) * iH;
  const pts = history.map((h, i) => ({ x: p.l + (i / Math.max(history.length - 1, 1)) * iW, y: toY(h.best_fitness) }));
  const path = pts.map((pt, i) => `${i ? "L" : "M"}${pt.x},${pt.y}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }}>
      {[0, 0.25, 0.5, 0.75, 1].map((f) => {
        const v = mn + rng * f, y = toY(v);
        return <g key={f}><line x1={p.l} y1={y} x2={W - p.r} y2={y} stroke={C.steelFaint} /><text x={p.l - 3} y={y + 3} textAnchor="end" fill={C.steelDim} fontSize="7" fontFamily={F.sys}>{v.toFixed(3)}</text></g>;
      })}
      {history.length > 2 && [0, 0.5, 1].map((f) => {
        const idx = Math.round(f * (history.length - 1)), x = p.l + (idx / Math.max(history.length - 1, 1)) * iW;
        return <text key={f} x={x} y={H - 3} textAnchor="middle" fill={C.steelDim} fontSize="7" fontFamily={F.sys}>{history[idx].generation}</text>;
      })}
      <path d={path} fill="none" stroke={C.green} strokeWidth={1.5} />
      <path d={path} fill="none" stroke={C.green} strokeWidth={4} opacity={0.1} />
      <circle cx={pts[pts.length - 1].x} cy={pts[pts.length - 1].y} r={2.5} fill={C.green}>
        <animate attributeName="r" values="2.5;4;2.5" dur="2s" repeatCount="indefinite" />
      </circle>
    </svg>
  );
}

/* --- Sparkline --- */
function Spark({ data, color, w = 48, h = 11 }: { data: number[]; color: string; w?: number; h?: number }) {
  if (data.length < 2) return null;
  const mx = Math.max(...data, 1e-4), mn = Math.min(...data), r = mx - mn || 1e-4;
  const pts = data.map((v, i) => `${(i / (data.length - 1)) * w},${h - ((v - mn) / r) * h}`).join(" ");
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
  const thS: React.CSSProperties = { fontSize: "0.4375rem", letterSpacing: "0.1em", textTransform: "uppercase", padding: "2px 3px", color: C.orange, borderBottom: `1px solid ${C.orangeDim}`, fontWeight: 400 };
  return (
    <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: F.sys, fontSize: "0.625rem" }}>
      <thead><tr>
        <th style={{ ...thS, textAlign: "left" }}>OBJECTIVE</th>
        <th style={{ ...thS, textAlign: "right" }}>NOW</th>
        <th style={{ ...thS, textAlign: "right" }}>BEST</th>
        <th style={{ ...thS, textAlign: "right" }}>TREND</th>
      </tr></thead>
      <tbody>
        {rows.map(([name, score]) => {
          const m = maxes[name], t = trends[name] || [];
          const atMax = m && Math.abs(score - m.max) < 1e-6;
          const tf = name === "transformer_failure";
          return (
            <tr key={name} style={{ borderBottom: `1px solid ${C.greenFaint}` }}>
              <td style={{ padding: "2px 3px", color: tf ? C.orangeHot : C.orangeDim, fontSize: "0.5rem", textTransform: "uppercase" as const }}>
                {tf && "★ "}{name.replace(/_/g, " ")}
              </td>
              <td style={{ padding: "2px 3px", textAlign: "right", color: tf && score >= 0.7 ? C.orangeHot : C.green, fontWeight: 500, fontVariantNumeric: "tabular-nums" }}>
                {score.toFixed(4)}
              </td>
              <td style={{ padding: "2px 3px", textAlign: "right", color: atMax ? C.orangeHot : C.steelDim, fontVariantNumeric: "tabular-nums" }}>
                {m ? m.max.toFixed(4) : "---"}
              </td>
              <td style={{ padding: "2px 3px", textAlign: "right" }}>
                <Spark data={t} color={tf ? C.orangeHot : C.greenDim} />
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/* --- Breakthrough Panel (prominent) --- */
function BreakthroughPanel({ items }: { items: Breakthrough[] }) {
  if (!items.length) return (
    <div style={{
      border: `1px dashed ${C.cyanDim}`, padding: "14px 12px", background: C.voidWarm, textAlign: "center",
    }}>
      <div style={{ fontFamily: F.kanji, fontSize: "0.75rem", color: C.steelDim }}>パラダイムシフト未検出</div>
      <div style={{ fontFamily: F.sys, fontSize: "0.5rem", color: C.steelDim, marginTop: 3, letterSpacing: "0.06em" }}>
        SCANNING FOR SIGNALS EXCEEDING TRANSFORMER BASELINE...
      </div>
    </div>
  );
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {items.map((bt, i) => (
        <div key={i} style={{
          border: `2px solid ${C.red}`, padding: 10, background: C.redFill,
          animation: "alertPulse 2s infinite",
        }}>
          <div style={{
            fontFamily: F.label, fontSize: "1.1rem", color: C.red, letterSpacing: "0.15em",
          }}>
            ◀ ◀ BREAKTHROUGH — GEN {bt.generation} ▶ ▶
          </div>
          <div style={{ fontFamily: F.sys, fontSize: "0.625rem", color: C.steel, marginTop: 3 }}>
            FITNESS: {bt.fitness.toFixed(4)} | AGENT: {bt.agent_id}
          </div>
          <div style={{ fontFamily: F.sys, fontSize: "0.5625rem", color: C.orangeHot, marginTop: 2 }}>
            {bt.signals.join(" | ")}
          </div>
        </div>
      ))}
    </div>
  );
}

/* --- Run History --- */
function RunHistory({ runs }: { runs: RunRecord[] }) {
  if (!runs.length) return null;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
      {runs.slice(0, 10).map((r) => {
        const active = r.status === "ACTIVE";
        const dur = (() => {
          const s = new Date(r.started_at).getTime();
          const e = r.ended_at ? new Date(r.ended_at).getTime() : Date.now();
          const m = Math.floor((e - s) / 60_000);
          return m >= 60 ? `${Math.floor(m / 60)}h${m % 60}m` : `${m}m`;
        })();
        return (
          <div key={r.run_id} style={{
            display: "flex", justifyContent: "space-between", alignItems: "center",
            padding: "2px 3px", fontSize: "0.5rem", fontFamily: F.sys,
            borderLeft: `2px solid ${active ? C.green : C.steelDim}`,
            background: active ? C.greenFaint : "transparent",
          }}>
            <span style={{ color: C.steelDim }}>
              {new Date(r.started_at).toLocaleDateString("ja-JP", { month: "numeric", day: "numeric" })}
              {" "}
              {new Date(r.started_at).toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit" })}
            </span>
            <span style={{ color: C.steelDim }}>{dur}</span>
            <span style={{ color: C.green, fontVariantNumeric: "tabular-nums" }}>G{r.final_generation}</span>
            <span style={{ color: C.cyan, fontVariantNumeric: "tabular-nums" }}>{r.best_fitness.toFixed(4)}</span>
            <span style={{ color: active ? C.green : C.steelDim, fontSize: "0.4375rem" }}>{active ? "●" : "○"}</span>
          </div>
        );
      })}
    </div>
  );
}

/* --- Log Console --- */
function LogConsole({ lines }: { lines: string[] }) {
  return (
    <div style={{
      background: C.void, border: `1px solid ${C.cyanDim}`,
      padding: 4, maxHeight: 160, overflowY: "auto",
      fontFamily: F.sys, fontSize: "0.5rem", lineHeight: 1.4,
    }}>
      {!lines.length && <div style={{ color: C.steelDim, padding: 6, textAlign: "center" }}>ログ待機中...</div>}
      {lines.map((l, i) => {
        let c = C.steelDim;
        if (l.includes("ERROR")) c = C.red;
        else if (l.includes("BREAKTHROUGH")) c = C.orangeHot;
        else if (l.includes("GEN ")) c = C.green;
        else if (l.includes("INFO")) c = C.greenDim;
        return <div key={i} style={{ color: c, whiteSpace: "pre-wrap", wordBreak: "break-all" }}>{l}</div>;
      })}
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
    const iv = setInterval(fetchData, 10_000);
    return () => clearInterval(iv);
  }, [fetchData]);

  const stats = useMemo(() => deriveStats(data), [data]);
  const sc = data?.status === "ACTIVE" ? C.green : data?.status === "STALE" ? C.orangeHot : C.red;

  return (
    <>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=Bebas+Neue&family=Noto+Sans+JP:wght@400;700&display=swap" rel="stylesheet" />
      <style>{`
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
        @keyframes alertPulse{0%,100%{border-color:${C.red}}50%{border-color:${C.redHot}}}
        body{margin:0;padding:0}
        @media(prefers-reduced-motion:reduce){*{animation:none!important}}
      `}</style>

      {/* CRT scanlines */}
      <div style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 9999, background: "repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.04) 2px,rgba(0,0,0,0.04) 4px)" }} />
      {/* Vignette */}
      <div style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 9998, background: "radial-gradient(ellipse at center,transparent 65%,rgba(0,0,0,0.2) 100%)" }} />

      <div style={{ minHeight: "100vh", background: C.void, color: C.steel, fontFamily: F.sys }}>

        {/* === HEADER === */}
        <div style={{
          borderBottom: `2px solid ${C.orange}`, padding: "10px 16px",
          display: "flex", justifyContent: "space-between", alignItems: "center",
        }}>
          <div>
            <div style={{ fontFamily: F.label, fontSize: "clamp(1rem, 2.5vw, 1.5rem)", letterSpacing: "0.15em", color: C.orange, textTransform: "uppercase" as const }}>
              パラダイム探索監視 / PARADIGM SEARCH MONITOR
            </div>
            <div style={{ fontFamily: F.sys, fontSize: "0.4375rem", color: C.steelDim, letterSpacing: "0.08em", textTransform: "uppercase" as const, marginTop: 1 }}>
              NERV MAGI DIVISION — SEARCHING FOR TRANSFORMER-EXCEEDING ARCHITECTURES
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", backgroundColor: sc, boxShadow: `0 0 6px ${sc}`, animation: data?.status === "ACTIVE" ? "pulse 2s infinite" : "none" }} />
            <span style={{ fontFamily: F.sys, fontSize: "0.625rem", fontWeight: 700, color: sc }}>{data?.status || "OFFLINE"}</span>
          </div>
        </div>

        {/* === ERROR === */}
        {error && <div style={{ margin: "6px 16px", padding: 6, border: `1px solid ${C.red}`, color: C.red, fontSize: "0.5625rem" }}>LINK ERROR: {error}</div>}

        {/* === MAIN 2-COLUMN === */}
        <div style={{ display: "grid", gridTemplateColumns: "3fr 2fr", gap: 10, padding: "10px 16px" }}>

          {/* --- LEFT: Core Monitoring --- */}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>

            {/* TF Gauge — the most important thing */}
            <TFGauge score={stats?.tfScore ?? 0} threshold={0.7} />

            {/* Best fitness */}
            <div style={{ border: `1px solid ${C.cyanDim}`, padding: "8px 10px", background: C.voidWarm }}>
              <div style={{ fontFamily: F.sys, fontSize: "0.4375rem", color: C.orangeDim, letterSpacing: "0.1em", textTransform: "uppercase" as const, marginBottom: 2 }}>
                最高適応度 / BEST FITNESS
              </div>
              <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
                <span style={{ fontFamily: F.sys, fontSize: "1.5rem", fontWeight: 700, color: C.green, fontVariantNumeric: "tabular-nums" }}>
                  {stats?.bestEver?.toFixed(4) ?? "---"}
                </span>
                <span style={{ fontFamily: F.sys, fontSize: "0.5rem", color: C.steelDim }}>
                  @ GEN {stats?.bestGen ?? "---"}
                </span>
              </div>
            </div>

            {/* Fitness chart */}
            <div style={{ border: `1px solid ${C.cyanDim}`, padding: 8, background: C.voidWarm }}>
              <SH jp="適応度推移" en="FITNESS TRAJECTORY" />
              <FitnessChart history={data?.history || []} />
            </div>

            {/* Objective table */}
            <div style={{ border: `1px solid ${C.cyanDim}`, padding: 8, background: C.voidWarm }}>
              <SH jp="目標別スコア" en="OBJECTIVES" />
              {stats?.cur?.objectives
                ? <ObjTable current={stats.cur.objectives} maxes={stats.objMaxes} trends={stats.objTrends} />
                : <div style={{ color: C.steelDim, fontSize: "0.5625rem" }}>NO DATA</div>
              }
            </div>
          </div>

          {/* --- RIGHT: Status & History --- */}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>

            {/* Mission Status */}
            <MissionStatus
              gen={stats?.cur?.generation ?? 0}
              max={data?.maxGenerations ?? 300}
              pct={stats?.progressPct ?? 0}
              eta={stats?.eta ?? "---"}
              uptime={stats?.uptimeStr ?? "---"}
              totalUptime={stats?.totalUptimeStr ?? "---"}
              totalRuns={data?.totalRuns ?? 0}
            />

            {/* GPU */}
            {data?.gpu && <GpuMini gpu={data.gpu} />}

            {/* Breakthroughs — prominent */}
            <div>
              <SH jp="パラダイムシフト検知" en="BREAKTHROUGH DETECTION" />
              <BreakthroughPanel items={data?.breakthroughs || []} />
            </div>

            {/* Run History */}
            {(data?.runs?.length ?? 0) > 0 && (
              <div style={{ border: `1px solid ${C.cyanDim}`, padding: "6px 8px", background: C.voidWarm }}>
                <SH jp="実行履歴" en="RUN HISTORY" />
                <RunHistory runs={data?.runs || []} />
              </div>
            )}

            {/* Log */}
            <div>
              <SH jp="ログ" en="CONSOLE" />
              <LogConsole lines={data?.recentLog || []} />
            </div>

            {/* Footer */}
            <div style={{
              fontSize: "0.4375rem", color: C.steelDim, textAlign: "right",
              letterSpacing: "0.05em", textTransform: "uppercase" as const,
            }}>
              AUTO-REFRESH 10s | SUPABASE REALTIME | {data?.updatedAt ? new Date(data.updatedAt).toLocaleTimeString("ja-JP") : "---"}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
