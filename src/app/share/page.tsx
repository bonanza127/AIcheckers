import { Metadata } from "next";
import Link from "next/link";

type Props = {
  searchParams: Promise<{ verdict?: string; score?: string }>;
};

export async function generateMetadata({ searchParams }: Props): Promise<Metadata> {
  const params = await searchParams;
  const verdict = params.verdict || "AI DETECTED";
  const score = params.score || "98";
  const isAI = verdict.includes("AI");
  const isUnknown = verdict.includes("UNKNOWN");

  const title = `${verdict} (${score}%) - AI Checkers`;
  const description = isAI
    ? `この画像はAI生成の可能性が${score}%です。AI Checkersで判定しました。`
    : isUnknown
      ? `この画像の判定は困難です（${score}%）。AI Checkersで判定しました。`
      : `この画像は人間作の可能性が高いです（AI: ${score}%）。AI Checkersで判定しました。`;

  const ogImageUrl = `https://www.aicheckers.net/api/og?verdict=${encodeURIComponent(verdict)}&score=${score}`;

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
  const verdict = params.verdict || "AI DETECTED";
  const score = params.score || "98";
  const isAI = verdict.includes("AI");
  const isUnknown = verdict.includes("UNKNOWN");

  // 色の設定
  const cardClass = isAI
    ? "bg-red-500/10 border-red-500"
    : isUnknown
      ? "bg-gray-500/10 border-gray-500"
      : "bg-green-500/10 border-green-500";
  const textClass = isAI
    ? "text-red-500"
    : isUnknown
      ? "text-gray-400"
      : "text-green-500";

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
        <div className={`p-8 rounded-2xl border-2 ${cardClass}`}>
          <div className={`text-4xl font-black mb-2 ${textClass}`}>
            {verdict}
          </div>
          <div className="text-6xl font-black text-text-primary mb-2">
            {score}%
          </div>
          <div className="text-muted">
            AI Possibility
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
