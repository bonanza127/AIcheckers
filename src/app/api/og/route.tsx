import { ImageResponse } from "next/og";
import { NextRequest } from "next/server";

export const runtime = "edge";

export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;

  const verdict = searchParams.get("verdict") || "AI DETECTED";
  const score = searchParams.get("score") || "98";

  // 3状態判定: AI / UNKNOWN / HUMAN
  const verdictType = verdict.includes("AI")
    ? "ai"
    : verdict.includes("UNKNOWN")
      ? "unknown"
      : "human";

  // カラー定義
  const colors = {
    ai: { primary: "#EF4444", bg: "rgba(239, 68, 68, 0.2)", bgLight: "rgba(239, 68, 68, 0.1)" },
    unknown: { primary: "#F59E0B", bg: "rgba(245, 158, 11, 0.2)", bgLight: "rgba(245, 158, 11, 0.1)" },
    human: { primary: "#10B981", bg: "rgba(16, 185, 129, 0.2)", bgLight: "rgba(16, 185, 129, 0.1)" },
  };
  const color = colors[verdictType];

  const confidenceLabel = verdictType === "ai" ? "AI Generated" : verdictType === "unknown" ? "Uncertain" : "Human Made";

  return new ImageResponse(
    (
      <div
        style={{
          height: "100%",
          width: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          backgroundColor: "#0C1117",
          backgroundImage:
            "radial-gradient(circle at 25% 25%, #1a1f2e 0%, transparent 50%), radial-gradient(circle at 75% 75%, #1a1f2e 0%, transparent 50%)",
        }}
      >
        {/* Logo */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            marginBottom: 30,
          }}
        >
          <div
            style={{
              width: 60,
              height: 60,
              borderRadius: 12,
              background: "linear-gradient(135deg, #A78BFA, #8B5CF6)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              marginRight: 16,
              boxShadow: "0 0 40px rgba(139, 92, 246, 0.4)",
              fontSize: 28,
              color: "white",
            }}
          >
            AI
          </div>
          <div
            style={{
              fontSize: 36,
              fontWeight: 800,
              color: "#E6E9EE",
              display: "flex",
            }}
          >
            AI Checkers
          </div>
        </div>

        {/* Result Banner */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            padding: "40px 80px",
            borderRadius: 24,
            background: `linear-gradient(135deg, ${color.bg}, ${color.bgLight})`,
            border: `3px solid ${color.primary}`,
            boxShadow: `0 0 60px ${color.bg}`,
          }}
        >
          <div
            style={{
              fontSize: 64,
              fontWeight: 900,
              color: color.primary,
              display: "flex",
              textAlign: "center",
            }}
          >
            {verdict}
          </div>
          <div
            style={{
              fontSize: 80,
              fontWeight: 900,
              color: "#E6E9EE",
              marginTop: 10,
              display: "flex",
            }}
          >
            {score}%
          </div>
          <div
            style={{
              fontSize: 24,
              color: "#8B949E",
              marginTop: 10,
              display: "flex",
            }}
          >
            {confidenceLabel} Confidence
          </div>
        </div>

        {/* Footer */}
        <div
          style={{
            display: "flex",
            marginTop: 40,
            fontSize: 20,
            color: "#8B949E",
          }}
        >
          aicheckers.net - Anime AI Image Detector
        </div>
      </div>
    ),
    {
      width: 1200,
      height: 630,
    }
  );
}
