import { ImageResponse } from "next/og";

export const runtime = "edge";

export const alt = "AI Checkers - AIイラスト判定ツール";
export const size = {
  width: 1200,
  height: 630,
};
export const contentType = "image/png";

export default async function Image() {
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
            marginBottom: 40,
          }}
        >
          <div
            style={{
              width: 80,
              height: 80,
              borderRadius: 16,
              background: "linear-gradient(135deg, #A78BFA, #8B5CF6)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              marginRight: 20,
              boxShadow: "0 0 40px rgba(139, 92, 246, 0.4)",
            }}
          >
            <svg
              width="48"
              height="48"
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
              fontSize: 48,
              fontWeight: 800,
              color: "#E6E9EE",
              letterSpacing: -1,
            }}
          >
            AI Checkers
          </div>
        </div>

        {/* Main title */}
        <div
          style={{
            fontSize: 56,
            fontWeight: 800,
            color: "#E6E9EE",
            textAlign: "center",
            marginBottom: 20,
            lineHeight: 1.2,
          }}
        >
          二次元に特化した
          <br />
          AIイラストチェッカー
        </div>

        {/* Subtitle */}
        <div
          style={{
            fontSize: 24,
            color: "#8B949E",
            textAlign: "center",
            maxWidth: 800,
          }}
        >
          画像をアップロードするだけで、AI生成画像を高精度判定
        </div>

        {/* Badge */}
        <div
          style={{
            display: "flex",
            marginTop: 40,
            gap: 16,
          }}
        >
          <div
            style={{
              padding: "12px 24px",
              backgroundColor: "rgba(139, 92, 246, 0.2)",
              border: "2px solid #8B5CF6",
              borderRadius: 8,
              color: "#A78BFA",
              fontSize: 18,
              fontWeight: 600,
            }}
          >
            🎨 アニメ特化
          </div>
          <div
            style={{
              padding: "12px 24px",
              backgroundColor: "rgba(16, 185, 129, 0.2)",
              border: "2px solid #10B981",
              borderRadius: 8,
              color: "#10B981",
              fontSize: 18,
              fontWeight: 600,
            }}
          >
            ✨ 無料で使える
          </div>
          <div
            style={{
              padding: "12px 24px",
              backgroundColor: "rgba(239, 68, 68, 0.2)",
              border: "2px solid #EF4444",
              borderRadius: 8,
              color: "#EF4444",
              fontSize: 18,
              fontWeight: 600,
            }}
          >
            🔍 高精度判定
          </div>
        </div>
      </div>
    ),
    {
      ...size,
    }
  );
}
