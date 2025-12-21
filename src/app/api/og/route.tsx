import { ImageResponse } from "next/og";
import { NextRequest } from "next/server";

export const runtime = "edge";

export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;

  const verdict = searchParams.get("verdict") || "AI DETECTED";
  const score = searchParams.get("score") || "98";
  const trace = searchParams.get("trace") || "中程度のAttention集中";
  const time = searchParams.get("time") || "0.00";

  // 3状態判定: AI / UNKNOWN / HUMAN
  const verdictType = verdict.includes("AI")
    ? "ai"
    : verdict.includes("UNKNOWN")
      ? "unknown"
      : "human";

  // カラー定義
  const colors = {
    ai: { primary: "#EF4444", glow: "rgba(239, 68, 68, 0.4)" },
    unknown: { primary: "#F59E0B", glow: "rgba(245, 158, 11, 0.4)" },
    human: { primary: "#10B981", glow: "rgba(16, 185, 129, 0.4)" },
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
          backgroundColor: "#161B22",
          padding: "40px 50px",
          fontFamily: "sans-serif",
        }}
      >
        {/* Header Row - 最終判定 by AIチェッカー */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            borderBottom: "2px solid rgba(139, 92, 246, 0.3)",
            paddingBottom: 24,
            marginBottom: 24,
          }}
        >
          <div style={{ display: "flex", fontSize: 52, fontWeight: 800, color: "#E6E9EE", letterSpacing: "-0.02em" }}>
            最終判定
          </div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              fontSize: 28,
              marginLeft: 20,
            }}
          >
            <span style={{ display: "flex", color: "#6E7681" }}>by</span>
            <span
              style={{
                display: "flex",
                alignItems: "center",
                marginLeft: 12,
                color: "#E6E9EE",
                background: "linear-gradient(135deg, rgba(139, 92, 246, 0.3), rgba(167, 139, 250, 0.2))",
                padding: "8px 20px",
                borderRadius: 8,
                border: "1px solid rgba(139, 92, 246, 0.5)",
              }}
            >
              <span style={{ display: "flex", color: "#A78BFA", fontWeight: 700 }}>AI</span>
              <span style={{ display: "flex", marginLeft: 6 }}>チェッカー</span>
            </span>
          </div>
        </div>

        {/* Info Row 1 - BATCH STATUS (左) + 使用モデル (右) */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            fontSize: 24,
            color: "#8B949E",
            marginBottom: 16,
          }}
        >
          <span style={{ display: "flex" }}>
            BATCH STATUS: <span style={{ color: "#E6E9EE", marginLeft: 8, display: "flex" }}>1 / 1</span>
          </span>
          <span style={{ display: "flex" }}>
            使用モデル: <span style={{ color: "#A78BFA", fontWeight: 600, marginLeft: 8, display: "flex" }}>Moonlight V1.3</span>
          </span>
        </div>

        {/* Info Row 2 - PROCESSING TIME (左) + ロジック (右) */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            fontSize: 24,
            color: "#8B949E",
            marginBottom: 20,
          }}
        >
          <span style={{ display: "flex" }}>
            PROCESSING TIME: <span style={{ color: "#E6E9EE", marginLeft: 8, display: "flex" }}>{time}s</span>
          </span>
          <span style={{ display: "flex" }}>
            ロジック: <span style={{ color: "#E6E9EE", marginLeft: 8, display: "flex" }}>カスケード方式</span>
          </span>
        </div>

        {/* Trace */}
        <div
          style={{
            display: "flex",
            fontSize: 26,
            color: "#8B949E",
            marginBottom: 20,
          }}
        >
          検出された痕跡: <span style={{ color: "#E6E9EE", marginLeft: 12, display: "flex" }}>{trace}</span>
        </div>

        {/* Progress Section */}
        <div style={{ display: "flex", flexDirection: "column", marginBottom: 36 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 12,
            }}
          >
            <span style={{ display: "flex", fontSize: 20, color: color.primary, fontWeight: 600 }}>
              {possibilityLabel}
            </span>
            <span style={{ display: "flex", fontSize: 26, color: "#E6E9EE", fontWeight: 700 }}>
              {score}%
            </span>
          </div>
          {/* Progress Bar */}
          <div
            style={{
              display: "flex",
              width: "100%",
              height: 14,
              backgroundColor: "#374151",
              borderRadius: 7,
            }}
          >
            <div
              style={{
                display: "flex",
                width: `${score}%`,
                height: "100%",
                backgroundColor: color.primary,
                borderRadius: 7,
                boxShadow: `0 0 10px ${color.primary}`,
              }}
            />
          </div>
        </div>

        {/* Classification - Main Focus */}
        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            flex: 1,
          }}
        >
          <span style={{ display: "flex", fontSize: 28, color: "#8B949E", marginRight: 24 }}>
            CLASSIFICATION:
          </span>
          <span
            style={{
              display: "flex",
              fontSize: 96,
              fontWeight: 900,
              color: color.primary,
              textShadow: `0 0 40px ${color.glow}`,
              lineHeight: 1,
            }}
          >
            {verdict}
          </span>
        </div>

        {/* Footer */}
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            fontSize: 16,
            color: "#6E7681",
            borderTop: "1px solid rgba(255,255,255,0.1)",
            paddingTop: 16,
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
