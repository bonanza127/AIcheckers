import { Metadata } from "next";
import { redirect } from "next/navigation";

type Props = {
  searchParams: Promise<{
    verdict?: string; score?: string; time?: string; trace?: string;
    v?: string; s?: string; t?: string; tr?: string;  // 短縮版
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

// スコアからverdictを計算（本番と同じ閾値）
function getVerdictFromScore(score: number): string {
  if (score >= 80) return "AI DETECTED";
  if (score >= 60) return "HIGH ALERT";
  if (score >= 40) return "MIDDLE CAUTION";
  if (score >= 20) return "LOW SIMILARITY";
  return "HUMAN CONFIRMED";
}

// 短縮traceコードを展開
function expandTrace(tr: string): string {
  if (tr === "ai") return "均一テクスチャ、不自然なエッジ処理";
  if (tr === "mx") return "特徴混在 - 追加検証を推奨";
  if (tr === "hu") return "有機的筆致、自然なテクスチャ";
  return tr;
}

export async function generateMetadata({ searchParams }: Props): Promise<Metadata> {
  const params = await searchParams;
  // スコアからverdictを計算（本番と同じロジック）
  const score = params.s || params.score || "98";
  const scoreNum = parseInt(score, 10) || 98;
  const verdict = getVerdictFromScore(scoreNum);
  const time = params.t || params.time || "0.00";
  const trace = params.tr ? expandTrace(params.tr) : (params.trace || "");
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
  const ogImageUrl = `https://aicheckers.net/api/og?${ogParams.toString()}`;

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

// OGPクローラー以外（人間）は即座にトップページにリダイレクト
export default async function SharePage({ searchParams }: Props) {
  // OGPメタデータはgenerateMetadataで生成済み
  // 人間がアクセスした場合は即座にトップページへ
  redirect("/");
}
