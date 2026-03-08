import { NextResponse } from "next/server";

const SUPABASE_URL =
  process.env.NEXT_PUBLIC_SUPABASE_URL ||
  "https://dmlupxahissxtlvpbwwv.supabase.co";
const SUPABASE_ANON_KEY =
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ||
  "sb_publishable_U_3vJBebGH6N-0FmMacxFg_RIEqjZM6";

async function sbFetch<T>(table: string, query: string = ""): Promise<T | null> {
  try {
    const res = await fetch(`${SUPABASE_URL}/rest/v1/${table}?${query}`, {
      headers: {
        apikey: SUPABASE_ANON_KEY,
        Authorization: `Bearer ${SUPABASE_ANON_KEY}`,
      },
      next: { revalidate: 0 },
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

interface EvolutionState {
  generation: number;
  best_fitness: number;
  avg_fitness: number;
  objectives: Record<string, number>;
  population_size: number;
  gpu_memory_used: number;
  gpu_memory_total: number;
  gpu_utilization: number;
  status: string;
  recent_log: string[];
  updated_at: string;
  run_id: string;
  run_started_at: string;
  max_generations: number;
  total_runs: number;
}

interface HistoryRow {
  run_id: string;
  generation: number;
  best_fitness: number;
  avg_fitness: number;
  objectives: Record<string, number>;
  created_at: string;
}

interface BreakthroughRow {
  run_id: string;
  generation: number;
  signals: string[];
  fitness: number;
  agent_id: string;
  created_at: string;
}

interface RunRow {
  run_id: string;
  started_at: string;
  ended_at: string | null;
  final_generation: number;
  best_fitness: number;
  config_name: string;
  status: string;
}

export async function GET() {
  const [stateRows, runs] = await Promise.all([
    sbFetch<EvolutionState[]>("evolution_state", "select=*&id=eq.current"),
    sbFetch<RunRow[]>("evolution_runs", "select=*&order=started_at.desc&limit=20"),
  ]);

  const state = stateRows?.[0] ?? null;
  const currentRunId = state?.run_id || "";

  // Fetch history + breakthroughs for current run
  const [history, breakthroughs] = await Promise.all([
    currentRunId
      ? sbFetch<HistoryRow[]>(
          "evolution_history",
          `select=*&run_id=eq.${currentRunId}&order=generation.asc`
        )
      : sbFetch<HistoryRow[]>("evolution_history", "select=*&order=generation.asc"),
    sbFetch<BreakthroughRow[]>(
      "evolution_breakthroughs",
      "select=*&order=created_at.desc&limit=20"
    ),
  ]);

  // Determine status from updated_at freshness
  let status = "OFFLINE";
  if (state?.updated_at) {
    const age = Date.now() - new Date(state.updated_at).getTime();
    status = age < 5 * 60 * 1000 ? "ACTIVE" : "STALE";
  }
  if (state?.status === "OFFLINE") status = "OFFLINE";

  // Build latestGenLine
  let latestGenLine = "";
  if (state) {
    const parts = Object.entries(state.objectives ?? {})
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([k, v]) => `${k.slice(0, 3)}=${(v as number).toFixed(2)}`)
      .join(" ");
    latestGenLine = `GEN ${String(state.generation).padStart(4, "0")} | best=${state.best_fitness.toFixed(4)} | ${parts}`;
  }

  return NextResponse.json({
    status,
    history: history ?? [],
    breakthroughs: breakthroughs ?? [],
    runs: runs ?? [],
    latestGenLine,
    recentLog: state?.recent_log ?? [],
    gpu: {
      memoryUsed: state?.gpu_memory_used ?? 0,
      memoryTotal: state?.gpu_memory_total ?? 0,
      utilization: state?.gpu_utilization ?? 0,
    },
    currentRunId,
    runStartedAt: state?.run_started_at ?? null,
    maxGenerations: state?.max_generations ?? 300,
    totalRuns: (runs ?? []).length,
    updatedAt: state?.updated_at ? new Date(state.updated_at).getTime() : Date.now(),
  });
}
