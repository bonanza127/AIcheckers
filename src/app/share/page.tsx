import { Metadata } from "next";
import Link from "next/link";

type Props = {
  searchParams: Promise<{
    verdict?: string; score?: string; time?: string; trace?: string;
    v?: string; s?: string; t?: string;  // 短縮版
  }>;
};

// 判定タイプヘルパー（5段階対応）
function getVerdictType(verdict: string): "ai" | "high" | "middle" | "low" | "human" {
  if (verdict.includes("AI DETECTED")) return "ai";
  if (verdict.includes("HIGH ALERT")) return "high";
  if (verdict.includes("MIDDLE CAUTION")) return "middle";
  if (verdict.includes("MINOR CAUTION") || verdict.includes("LOW")) return "low";
  return "human";
}

// 短縮コードからverdictに変換
function expandVerdict(v: string): string {
  if (v === "ai") return "AI DETECTED";
  if (v === "ha") return "HIGH ALERT";
  if (v === "mc") return "MIDDLE CAUTION";
  if (v === "ls") return "LOW SIMILARITY";
  if (v === "h") return "HUMAN CONFIRMED";
  return v;
}

export async function generateMetadata({ searchParams }: Props): Promise<Metadata> {
  const params = await searchParams;
  // 短縮パラメータ優先、なければ通常パラメータ
  const verdict = params.v ? expandVerdict(params.v) : (params.verdict || "AI DETECTED");
  const score = params.s || params.score || "98";
  const time = params.t || params.time || "0.00";
  const trace = params.trace || "";
  const verdictType = getVerdictType(verdict);

  const title = `${verdict} (${score}%) - AI Checkers`;
  const descriptions: Record<typeof verdictType, string> = {
    ai: `この画像はAI生成の可能性が${score}%です。AI Checkersで判定しました。`,
    high: `この画像はAI生成の疑いが高いです（${score}%）。AI Checkersで判定しました。`,
    middle: `この画像は判定が微妙です（${score}%）。AI Checkersで判定しました。`,
    low: `この画像は人間作の可能性が高いです（${100 - parseInt(score)}%）。AI Checkersで判定しました。`,
    human: `この画像は人間作の可能性が${100 - parseInt(score)}%です。AI Checkersで判定しました。`,
  };
  const description = descriptions[verdictType];

  const ogParams = new URLSearchParams({
    verdict,
    score,
    time,
    ...(trace && { trace }),
  });
  const ogImageUrl = `https://www.aicheckers.net/api/og?${ogParams.toString()}`;

  return {
    title,
    description,
    openGraph: {
      title,
      description,
      images: [
        {
          url: ogImageUrl,
          width: 1200,
          height: 630,
          alt: title,
        },
      ],
      type: "website",
      siteName: "AI Checkers",
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
      images: [ogImageUrl],
    },
  };
}

export default async function SharePage({ searchParams }: Props) {
  const params = await searchParams;
  const verdict = params.v ? expandVerdict(params.v) : (params.verdict || "AI DETECTED");
  const score = params.s || params.score || "98";
  const verdictType = getVerdictType(verdict);

  // カラー定義（5段階対応）
  const colorClasses = {
    ai: { bg: "bg-red-500/10", border: "border-red-500", text: "text-red-500" },
    high: { bg: "bg-orange-600/10", border: "border-orange-600", text: "text-orange-500" },
    middle: { bg: "bg-yellow-500/10", border: "border-yellow-500", text: "text-yellow-400" },
    low: { bg: "bg-green-500/10", border: "border-green-500", text: "text-green-500" },
    human: { bg: "bg-blue-500/10", border: "border-blue-500", text: "text-blue-500" },
  };
  const colors = colorClasses[verdictType];

  const confidenceLabels: Record<typeof verdictType, string> = {
    ai: "AI生成",
    high: "疑わしい",
    middle: "判定困難",
    low: "可能性低",
    human: "人間作",
  };
  const confidenceLabel = confidenceLabels[verdictType];

  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center p-4">
      <div className="max-w-lg w-full text-center">
        {/* Logo */}
        <div className="flex items-center justify-center gap-3 mb-8">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-400 to-purple-600 flex items-center justify-center shadow-lg shadow-purple-500/30">
            <svg
              className="w-7 h-7 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <circle cx="11" cy="11" r="8" strokeWidth="2" />
              <path d="m21 21-4.3-4.3" strokeWidth="2" />
            </svg>
          </div>
          <span className="text-2xl font-bold text-text-primary">AI Checkers</span>
        </div>

        {/* Result Card */}
        <div
          className={`p-8 rounded-2xl border-2 ${colors.bg} ${colors.border}`}
        >
          <div
            className={`text-4xl font-black mb-2 ${colors.text}`}
          >
            {verdict}
          </div>
          <div className="text-6xl font-black text-text-primary mb-2">
            {score}%
          </div>
          <div className="text-muted">
            {confidenceLabel}確信度
          </div>
        </div>

        {/* CTA */}
        <div className="mt-8 space-y-4">
          <p className="text-muted">
            あなたも画像をチェックしてみませんか？
          </p>
          <Link
            href="/"
            className="inline-block px-8 py-3 rounded-xl bg-accent text-white font-semibold hover:bg-accent/80 transition-colors"
          >
            AI Checkersを使う
          </Link>
        </div>
      </div>
    </div>
  );
}
