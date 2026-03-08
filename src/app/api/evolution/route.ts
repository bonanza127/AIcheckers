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
}

interface HistoryRow {
  generation: number;
  best_fitness: number;
  avg_fitness: number;
  objectives: Record<string, number>;
}

interface BreakthroughRow {
  generation: number;
  signals: string[];
  fitness: number;
  agent_id: string;
  created_at: string;
}

export async function GET() {
  const [stateRows, history, breakthroughs] = await Promise.all([
    sbFetch<EvolutionState[]>("evolution_state", "select=*&id=eq.current"),
    sbFetch<HistoryRow[]>(
      "evolution_history",
      "select=*&order=generation.asc"
    ),
    sbFetch<BreakthroughRow[]>(
      "evolution_breakthroughs",
      "select=*&order=created_at.desc&limit=20"
    ),
  ]);

  const state = stateRows?.[0] ?? null;

  // Determine if active: check updated_at within last 5 minutes
  let status = "OFFLINE";
  if (state?.updated_at) {
    const age = Date.now() - new Date(state.updated_at).getTime();
    status = age < 5 * 60 * 1000 ? "ACTIVE" : "STALE";
  }
  if (state?.status === "ACTIVE" && status !== "OFFLINE") {
    status = "ACTIVE";
  }

  // Build latestGenLine from state
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
    latestGenLine,
    recentLog: state?.recent_log ?? [],
    gpu: {
      memoryUsed: state?.gpu_memory_used ?? 0,
      memoryTotal: state?.gpu_memory_total ?? 0,
      utilization: state?.gpu_utilization ?? 0,
    },
    updatedAt: state?.updated_at ? new Date(state.updated_at).getTime() : Date.now(),
  });
}
