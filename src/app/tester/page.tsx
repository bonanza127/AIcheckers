"use client";

import { useState, useEffect, useCallback } from "react";

type GenerationRecord = {
  generation: number;
  best_fitness: number;
  avg_fitness: number;
  objectives: Record<string, number>;
};

type Breakthrough = {
  generation: number;
  signals: string[];
  agent_id: string;
  fitness: number;
  created_at: string;
};

type GpuInfo = {
  memoryUsed: number;
  memoryTotal: number;
  utilization: number;
};

type EvolutionData = {
  status: "ACTIVE" | "OFFLINE" | "STALE";
  history: GenerationRecord[];
  breakthroughs: Breakthrough[];
  latestGenLine: string;
  recentLog: string[];
  gpu: GpuInfo;
  updatedAt: number;
};

// NERV color tokens
const C = {
  void: "#000000",
  voidWarm: "#0A0A08",
  voidPanel: "#111110",
  orange: "#FF9830",
  orangeDim: "#D08028",
  orangeHot: "#FFCC50",
  green: "#50FF50",
  greenDim: "#30BB30",
  cyan: "#20F0FF",
  cyanDim: "#10A8B8",
  red: "#FF4840",
  redHot: "#FF6858",
  steel: "#E0E0D8",
  steelDim: "#9A9A90",
};

function NervHeader({ status }: { status: string }) {
  return (
    <div
      style={{
        borderBottom: `1px solid ${C.orangeDim}`,
        padding: "16px 24px",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}
    >
      <div>
        <div
          style={{
            fontFamily: "'Bebas Neue', 'Arial Narrow', sans-serif",
            fontSize: "clamp(1.2rem, 3vw, 2rem)",
            letterSpacing: "0.15em",
            color: C.orange,
            textTransform: "uppercase" as const,
          }}
        >
          進化監視システム / EVOLUTION MONITOR
        </div>
        <div
          style={{
            fontFamily: "'IBM Plex Mono', monospace",
            fontSize: "0.6875rem",
            color: C.steelDim,
            letterSpacing: "0.08em",
            textTransform: "uppercase" as const,
            marginTop: 4,
          }}
        >
          NERV PARADIGM SEARCH DIVISION — REAL-TIME TELEMETRY
        </div>
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        <div
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            backgroundColor: status === "ACTIVE" ? C.green : C.red,
            boxShadow: `0 0 8px ${status === "ACTIVE" ? C.green : C.red}`,
            animation: status === "ACTIVE" ? "pulse 2s infinite" : "none",
          }}
        />
        <span
          style={{
            fontFamily: "'IBM Plex Mono', monospace",
            fontSize: "0.8125rem",
            fontWeight: 500,
            color: status === "ACTIVE" ? C.green : C.red,
            letterSpacing: "0.08em",
          }}
        >
          {status}
        </span>
      </div>
    </div>
  );
}

function StatBlock({ label, value, unit, color }: { label: string; value: string; unit?: string; color?: string }) {
  return (
    <div style={{ padding: "8px 0" }}>
      <div
        style={{
          fontFamily: "'IBM Plex Mono', monospace",
          fontSize: "0.6875rem",
          fontWeight: 400,
          letterSpacing: "0.08em",
          textTransform: "uppercase" as const,
          color: C.orangeDim,
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontFamily: "'IBM Plex Mono', monospace",
          fontSize: "1.5rem",
          fontWeight: 700,
          color: color || C.green,
          lineHeight: 1.2,
        }}
      >
        {value}
        {unit && (
          <span style={{ fontSize: "0.75rem", color: C.steelDim, marginLeft: 4 }}>{unit}</span>
        )}
      </div>
    </div>
  );
}

function GpuMeter({ gpu }: { gpu: GpuInfo }) {
  const pct = gpu.memoryTotal > 0 ? (gpu.memoryUsed / gpu.memoryTotal) * 100 : 0;
  const barColor = pct > 80 ? C.red : pct > 60 ? C.orangeHot : C.cyan;
  return (
    <div
      style={{
        border: `1px solid ${C.cyanDim}`,
        padding: 12,
        background: C.voidWarm,
      }}
    >
      <div
        style={{
          fontFamily: "'IBM Plex Mono', monospace",
          fontSize: "0.6875rem",
          color: C.orangeDim,
          letterSpacing: "0.08em",
          textTransform: "uppercase" as const,
          marginBottom: 8,
        }}
      >
        GPU VRAM
      </div>
      <div
        style={{
          height: 6,
          background: "rgba(32, 240, 255, 0.1)",
          marginBottom: 4,
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${pct}%`,
            background: barColor,
            boxShadow: `0 0 6px ${barColor}`,
            transition: "width 0.5s",
          }}
        />
      </div>
      <div
        style={{
          fontFamily: "'IBM Plex Mono', monospace",
          fontSize: "0.75rem",
          color: C.steel,
          display: "flex",
          justifyContent: "space-between",
        }}
      >
        <span>
          {gpu.memoryUsed} / {gpu.memoryTotal} MiB
        </span>
        <span style={{ color: C.cyan }}>UTIL {gpu.utilization}%</span>
      </div>
    </div>
  );
}

function FitnessChart({ history }: { history: GenerationRecord[] }) {
  if (history.length === 0) return null;
  const maxFit = Math.max(...history.map((h) => h.best_fitness), 0.01);
  const width = 600;
  const height = 200;
  const pad = { top: 10, right: 10, bottom: 20, left: 40 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;

  const points = history.map((h, i) => ({
    x: pad.left + (i / Math.max(history.length - 1, 1)) * innerW,
    yBest: pad.top + innerH - (h.best_fitness / maxFit) * innerH,
    yAvg: pad.top + innerH - (h.avg_fitness / maxFit) * innerH,
  }));

  const bestPath = points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.yBest}`).join(" ");
  const avgPath = points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.yAvg}`).join(" ");

  return (
    <svg viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", height: "auto" }}>
      {/* Grid */}
      {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
        const y = pad.top + innerH * (1 - frac);
        return (
          <g key={frac}>
            <line x1={pad.left} y1={y} x2={width - pad.right} y2={y} stroke="rgba(224,224,216,0.08)" />
            <text
              x={pad.left - 4}
              y={y + 3}
              textAnchor="end"
              fill={C.steelDim}
              fontSize="9"
              fontFamily="'IBM Plex Mono', monospace"
            >
              {(maxFit * frac).toFixed(2)}
            </text>
          </g>
        );
      })}
      {/* Avg line */}
      <path d={avgPath} fill="none" stroke={C.cyanDim} strokeWidth={1} opacity={0.6} />
      {/* Best line */}
      <path d={bestPath} fill="none" stroke={C.green} strokeWidth={1.5} />
      {/* Glow */}
      <path d={bestPath} fill="none" stroke={C.green} strokeWidth={4} opacity={0.15} />
    </svg>
  );
}

function ObjectiveBreakdown({ breakdown }: { breakdown: Record<string, number> }) {
  const entries = Object.entries(breakdown).sort(([, a], [, b]) => b - a);
  if (entries.length === 0) return null;
  const maxVal = Math.max(...entries.map(([, v]) => v), 0.01);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {entries.map(([name, score]) => (
        <div key={name} style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div
            style={{
              fontFamily: "'IBM Plex Mono', monospace",
              fontSize: "0.6875rem",
              color: C.orangeDim,
              width: 160,
              textAlign: "right",
              letterSpacing: "0.05em",
              textTransform: "uppercase" as const,
              flexShrink: 0,
            }}
          >
            {name}
          </div>
          <div style={{ flex: 1, height: 6, background: "rgba(80,255,80,0.08)" }}>
            <div
              style={{
                height: "100%",
                width: `${(score / maxVal) * 100}%`,
                background: name === "transformer_failure" && score >= 0.7 ? C.orangeHot : C.green,
                boxShadow: name === "transformer_failure" && score >= 0.7 ? `0 0 8px ${C.orangeHot}` : "none",
              }}
            />
          </div>
          <div
            style={{
              fontFamily: "'IBM Plex Mono', monospace",
              fontSize: "0.8125rem",
              fontWeight: 500,
              color: C.green,
              width: 48,
              textAlign: "right",
            }}
          >
            {score.toFixed(3)}
          </div>
        </div>
      ))}
    </div>
  );
}

function BreakthroughPanel({ breakthroughs }: { breakthroughs: Breakthrough[] }) {
  if (breakthroughs.length === 0) {
    return (
      <div
        style={{
          border: `1px solid ${C.cyanDim}`,
          padding: 16,
          background: C.voidWarm,
          textAlign: "center",
        }}
      >
        <div
          style={{
            fontFamily: "'IBM Plex Mono', monospace",
            fontSize: "0.8125rem",
            color: C.steelDim,
          }}
        >
          NO PARADIGM SHIFTS DETECTED
        </div>
        <div
          style={{
            fontFamily: "'IBM Plex Mono', monospace",
            fontSize: "0.625rem",
            color: C.steelDim,
            marginTop: 4,
          }}
        >
          MONITORING...
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {breakthroughs.map((bt, i) => (
        <div
          key={i}
          style={{
            border: `2px solid ${C.red}`,
            padding: 12,
            background: "rgba(255, 72, 64, 0.08)",
            animation: "alertPulse 3s infinite",
          }}
        >
          <div
            style={{
              fontFamily: "'Bebas Neue', sans-serif",
              fontSize: "1.2rem",
              color: C.red,
              letterSpacing: "0.1em",
            }}
          >
            ★ BREAKTHROUGH — GEN {bt.generation}
          </div>
          <div
            style={{
              fontFamily: "'IBM Plex Mono', monospace",
              fontSize: "0.75rem",
              color: C.steel,
              marginTop: 4,
            }}
          >
            FITNESS: {bt.fitness.toFixed(4)} | AGENT: {bt.agent_id}
          </div>
          <div
            style={{
              fontFamily: "'IBM Plex Mono', monospace",
              fontSize: "0.6875rem",
              color: C.orangeHot,
              marginTop: 4,
            }}
          >
            {bt.signals.join(" | ")}
          </div>
        </div>
      ))}
    </div>
  );
}

function LogConsole({ lines }: { lines: string[] }) {
  return (
    <div
      style={{
        background: C.void,
        border: `1px solid ${C.cyanDim}`,
        padding: 8,
        maxHeight: 300,
        overflowY: "auto",
        fontFamily: "'IBM Plex Mono', monospace",
        fontSize: "0.6875rem",
        lineHeight: 1.6,
      }}
    >
      {lines.map((line, i) => {
        let color = C.steelDim;
        if (line.includes("ERROR")) color = C.red;
        else if (line.includes("WARNING") || line.includes("BREAKTHROUGH")) color = C.orangeHot;
        else if (line.includes("GEN ")) color = C.green;
        else if (line.includes("INFO")) color = C.greenDim;
        return (
          <div key={i} style={{ color, whiteSpace: "pre-wrap", wordBreak: "break-all" }}>
            {line}
          </div>
        );
      })}
    </div>
  );
}

export default function TesterPage() {
  const [data, setData] = useState<EvolutionData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch("/api/evolution", { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json);
      setError(null);
    } catch (err) {
      setError(String(err));
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000); // poll every 10s
    return () => clearInterval(interval);
  }, [fetchData]);

  const latest = data?.history?.[data.history.length - 1];

  return (
    <>
      <link
        href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=Bebas+Neue&family=Noto+Sans+JP:wght@400;700&display=swap"
        rel="stylesheet"
      />
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
        @keyframes alertPulse {
          0%, 100% { border-color: ${C.red}; }
          50% { border-color: ${C.redHot}; }
        }
        @keyframes scanline {
          0% { transform: translateY(-100%); }
          100% { transform: translateY(100vh); }
        }
        body { margin: 0; padding: 0; }
      `}</style>

      {/* CRT scanline overlay */}
      <div
        style={{
          position: "fixed",
          inset: 0,
          background: `repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.06) 2px, rgba(0,0,0,0.06) 4px)`,
          pointerEvents: "none",
          zIndex: 9999,
        }}
      />

      <div
        style={{
          minHeight: "100vh",
          background: C.void,
          color: C.steel,
          fontFamily: "'IBM Plex Mono', monospace",
        }}
      >
        <NervHeader status={data?.status || "OFFLINE"} />

        {error && (
          <div
            style={{
              margin: "16px 24px",
              padding: 12,
              border: `1px solid ${C.red}`,
              color: C.red,
              fontSize: "0.75rem",
            }}
          >
            CONNECTION ERROR: {error}
          </div>
        )}

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr 1fr 1fr",
            gap: 1,
            padding: "16px 24px",
            borderBottom: `1px solid rgba(255,152,48,0.2)`,
          }}
        >
          <StatBlock
            label="世代 / GENERATION"
            value={latest ? String(latest.generation) : "---"}
          />
          <StatBlock
            label="最高適応度 / BEST FITNESS"
            value={latest ? latest.best_fitness.toFixed(4) : "---"}
          />
          <StatBlock
            label="平均適応度 / AVG FITNESS"
            value={latest ? latest.avg_fitness.toFixed(4) : "---"}
            color={C.cyan}
          />
          <StatBlock
            label="目標数 / OBJECTIVES"
            value={latest ? String(Object.keys(latest.objectives).length) : "---"}
            color={C.steel}
          />
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "2fr 1fr",
            gap: 16,
            padding: "16px 24px",
          }}
        >
          {/* Left column */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {/* Fitness chart */}
            <div
              style={{
                border: `1px solid ${C.cyanDim}`,
                padding: 12,
                background: C.voidWarm,
              }}
            >
              <div
                style={{
                  fontSize: "0.875rem",
                  fontWeight: 700,
                  letterSpacing: "0.12em",
                  textTransform: "uppercase" as const,
                  color: C.orange,
                  marginBottom: 8,
                }}
              >
                適応度推移 / FITNESS TRAJECTORY
              </div>
              {data?.history && data.history.length > 0 ? (
                <FitnessChart history={data.history} />
              ) : (
                <div style={{ color: C.steelDim, fontSize: "0.75rem", padding: 20, textAlign: "center" }}>
                  AWAITING DATA...
                </div>
              )}
            </div>

            {/* Objective breakdown */}
            <div
              style={{
                border: `1px solid ${C.cyanDim}`,
                padding: 12,
                background: C.voidWarm,
              }}
            >
              <div
                style={{
                  fontSize: "0.875rem",
                  fontWeight: 700,
                  letterSpacing: "0.12em",
                  textTransform: "uppercase" as const,
                  color: C.orange,
                  marginBottom: 8,
                }}
              >
                目標別スコア / OBJECTIVE BREAKDOWN
              </div>
              {latest?.objectives ? (
                <ObjectiveBreakdown breakdown={latest.objectives} />
              ) : (
                <div style={{ color: C.steelDim, fontSize: "0.75rem" }}>NO DATA</div>
              )}
            </div>

            {/* Log console */}
            <div>
              <div
                style={{
                  fontSize: "0.875rem",
                  fontWeight: 700,
                  letterSpacing: "0.12em",
                  textTransform: "uppercase" as const,
                  color: C.orange,
                  marginBottom: 8,
                }}
              >
                実行ログ / CONSOLE OUTPUT
              </div>
              <LogConsole lines={data?.recentLog || []} />
            </div>
          </div>

          {/* Right column */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {/* GPU */}
            {data?.gpu && <GpuMeter gpu={data.gpu} />}

            {/* Breakthrough panel */}
            <div>
              <div
                style={{
                  fontSize: "0.875rem",
                  fontWeight: 700,
                  letterSpacing: "0.12em",
                  textTransform: "uppercase" as const,
                  color: C.orange,
                  marginBottom: 8,
                }}
              >
                パラダイムシフト / BREAKTHROUGHS
              </div>
              <BreakthroughPanel breakthroughs={data?.breakthroughs || []} />
            </div>

            {/* Latest gen info */}
            {data?.latestGenLine && (
              <div
                style={{
                  border: `1px solid ${C.cyanDim}`,
                  padding: 12,
                  background: C.voidWarm,
                }}
              >
                <div
                  style={{
                    fontSize: "0.6875rem",
                    color: C.orangeDim,
                    letterSpacing: "0.08em",
                    textTransform: "uppercase" as const,
                    marginBottom: 4,
                  }}
                >
                  LATEST GENERATION
                </div>
                <div
                  style={{
                    fontSize: "0.75rem",
                    color: C.green,
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-all",
                  }}
                >
                  {data.latestGenLine}
                </div>
              </div>
            )}

            {/* Update timestamp */}
            <div
              style={{
                fontSize: "0.625rem",
                color: C.steelDim,
                textAlign: "right",
                letterSpacing: "0.05em",
              }}
            >
              LAST UPDATE: {data?.updatedAt ? new Date(data.updatedAt).toLocaleTimeString("ja-JP") : "---"}
              <br />
              AUTO-REFRESH: 10s
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
