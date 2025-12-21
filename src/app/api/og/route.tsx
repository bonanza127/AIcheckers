import { ImageResponse } from "next/og";
import { NextRequest } from "next/server";

export const runtime = "edge";

export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;

  const verdict = searchParams.get("verdict") || "AI DETECTED";
  const score = searchParams.get("score") || "98";

  // 3段階判定: AI DETECTED / UNKNOWN / HUMAN CONFIRMED
  const isAI = verdict.includes("AI");
  const isUnknown = verdict.includes("UNKNOWN");

  // 色の設定
  const accentColor = isAI ? "#EF4444" : isUnknown ? "#6B7280" : "#10B981";
  const glowRgba = isAI
    ? "rgba(239, 68, 68, 0.3)"
    : isUnknown
      ? "rgba(107, 114, 128, 0.3)"
      : "rgba(16, 185, 129, 0.3)";
  const bgGradient = isAI
    ? "linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.1))"
    : isUnknown
      ? "linear-gradient(135deg, rgba(107, 114, 128, 0.2), rgba(107, 114, 128, 0.1))"
      : "linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(16, 185, 129, 0.1))";

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
        {/* Logo/Icon area */}
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
            }}
          >
            <svg
              width="36"
              height="36"
              viewBox="0 0 24 24"
              fill="none"
              stroke="white"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="11" cy="11" r="8" />
              <path d="m21 21-4.3-4.3" />
            </svg>
          </div>
          <div
            style={{
              fontSize: 36,
              fontWeight: 800,
              color: "#E6E9EE",
              letterSpacing: -1,
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
            background: bgGradient,
            border: `3px solid ${accentColor}`,
            boxShadow: `0 0 60px ${glowRgba}`,
          }}
        >
          <div
            style={{
              fontSize: 56,
              fontWeight: 900,
              color: accentColor,
              textAlign: "center",
              letterSpacing: 2,
              textShadow: `0 0 30px ${glowRgba}`,
            }}
          >
            {verdict}
          </div>
          <div
            style={{
              fontSize: 96,
              fontWeight: 900,
              color: "#E6E9EE",
              marginTop: 10,
            }}
          >
            {score}%
          </div>
          <div
            style={{
              fontSize: 24,
              color: "#8B949E",
              marginTop: 10,
            }}
          >
            AI Possibility
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
          aicheckers.net - 二次元特化AIイラスト判定ツール
        </div>
      </div>
    ),
    {
      width: 1200,
      height: 630,
    }
  );
}
