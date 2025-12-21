import { ImageResponse } from "next/og";
import { NextRequest } from "next/server";

export const runtime = "edge";

export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;

  const verdict = searchParams.get("verdict") || "AI DETECTED";
  const score = searchParams.get("score") || "98";
  const trace = searchParams.get("trace") || "";

  // 3状態判定: AI / UNKNOWN / HUMAN
  const verdictType = verdict.includes("AI")
    ? "ai"
    : verdict.includes("UNKNOWN")
      ? "unknown"
      : "human";

  // カラー定義
  const colors = {
    ai: { primary: "#EF4444", bg: "rgba(239, 68, 68, 0.15)", glow: "rgba(239, 68, 68, 0.3)" },
    unknown: { primary: "#F59E0B", bg: "rgba(245, 158, 11, 0.15)", glow: "rgba(245, 158, 11, 0.3)" },
    human: { primary: "#10B981", bg: "rgba(16, 185, 129, 0.15)", glow: "rgba(16, 185, 129, 0.3)" },
  };
  const color = colors[verdictType];

  const possibilityLabel = verdictType === "ai" ? "AI POSSIBILITY" : verdictType === "unknown" ? "UNCERTAINTY" : "HUMAN POSSIBILITY";

  return new ImageResponse(
    (
      <div
        style={{
          height: "100%",
          width: "100%",
          display: "flex",
          flexDirection: "column",
          backgroundColor: "#0C1117",
          padding: "40px 60px",
          fontFamily: "sans-serif",
        }}
      >
        {/* Header */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginBottom: 30,
          }}
        >
          {/* Logo */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
            }}
          >
            <div
              style={{
                width: 48,
                height: 48,
                borderRadius: 10,
                background: "linear-gradient(135deg, #A78BFA, #8B5CF6)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                marginRight: 12,
                fontSize: 22,
                fontWeight: 700,
                color: "white",
              }}
            >
              AI
            </div>
            <div
              style={{
                fontSize: 28,
                fontWeight: 700,
                color: "#E6E9EE",
                display: "flex",
              }}
            >
              AI Checkers
            </div>
          </div>

          {/* Model Badge */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              fontSize: 18,
              color: "#8B949E",
            }}
          >
            <span style={{ display: "flex" }}>使用モデル: </span>
            <span style={{ display: "flex", color: "#A78BFA", fontWeight: 600, marginLeft: 6 }}>Moonlight V1.3</span>
          </div>
        </div>

        {/* Main Panel */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            flex: 1,
            backgroundColor: "#161B22",
            borderRadius: 16,
            border: `2px solid ${color.primary}`,
            boxShadow: `0 0 40px ${color.glow}`,
            padding: "30px 40px",
          }}
        >
          {/* Trace Info (if provided) */}
          {trace && (
            <div
              style={{
                display: "flex",
                fontSize: 18,
                color: "#8B949E",
                marginBottom: 20,
              }}
            >
              検出された痕跡: {trace}
            </div>
          )}

          {/* Progress Section */}
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              marginBottom: 30,
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: 12,
              }}
            >
              <span style={{ display: "flex", fontSize: 22, color: color.primary, fontWeight: 600 }}>
                {possibilityLabel}
              </span>
              <span style={{ display: "flex", fontSize: 28, color: "#E6E9EE", fontWeight: 700 }}>
                {score}%
              </span>
            </div>
            {/* Progress Bar */}
            <div
              style={{
                display: "flex",
                width: "100%",
                height: 16,
                backgroundColor: "#374151",
                borderRadius: 8,
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  display: "flex",
                  width: `${score}%`,
                  height: "100%",
                  backgroundColor: color.primary,
                  borderRadius: 8,
                  boxShadow: `0 0 12px ${color.primary}`,
                }}
              />
            </div>
          </div>

          {/* Classification */}
          <div
            style={{
              display: "flex",
              alignItems: "baseline",
              marginTop: "auto",
            }}
          >
            <span style={{ display: "flex", fontSize: 28, color: "#8B949E", marginRight: 20 }}>
              CLASSIFICATION:
            </span>
            <span
              style={{
                display: "flex",
                fontSize: 72,
                fontWeight: 900,
                color: color.primary,
                textShadow: `0 0 30px ${color.glow}`,
              }}
            >
              {verdict}
            </span>
          </div>
        </div>

        {/* Footer */}
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            marginTop: 20,
            fontSize: 18,
            color: "#6E7681",
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
