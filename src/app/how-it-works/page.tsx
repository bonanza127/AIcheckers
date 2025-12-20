import Link from "next/link";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "仕組み | AIチェッカー",
  description: "AIチェッカーがどのようにAI生成画像を検出するかを解説します。",
};

export default function HowItWorks() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="site-header border-b border-gray-700">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/" className="text-xl font-bold text-accent hover:opacity-80">
            AIチェッカー
          </Link>
          <nav className="text-sm text-muted">
            <Link href="/" className="hover:text-accent">← トップへ戻る</Link>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12 max-w-3xl">
        <h1 className="text-3xl font-bold mb-8 text-center">AIチェッカーの仕組み</h1>

        {/* Overview */}
        <section className="mb-10">
          <h2 className="text-xl font-bold mb-4 text-accent border-b border-gray-700 pb-2">概要</h2>
          <p className="text-muted leading-relaxed">
            AIチェッカーは、二次元イラストに特化したAI生成画像検出ツールです。
            独自にファインチューニングした<span className="text-accent font-bold">Moonlight V1.3</span>モデルを使用し、
            <span className="text-accent font-bold">98.35%</span>の精度でAI生成画像を判別します。
          </p>
        </section>

        {/* Detection Flow */}
        <section className="mb-10">
          <h2 className="text-xl font-bold mb-4 text-accent border-b border-gray-700 pb-2">検出の流れ</h2>
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <span className="w-8 h-8 rounded-full bg-accent text-background flex items-center justify-center font-bold text-sm">1</span>
              <div>
                <p className="font-bold">画像アップロード</p>
                <p className="text-muted text-sm">PNG, JPG, WebP形式に対応</p>
              </div>
            </div>
            <div className="flex justify-center text-muted">↓</div>
            <div className="flex items-center gap-4">
              <span className="w-8 h-8 rounded-full bg-accent text-background flex items-center justify-center font-bold text-sm">2</span>
              <div>
                <p className="font-bold">特徴抽出（Vision Transformer）</p>
                <p className="text-muted text-sm">画像から768次元の特徴ベクトルを抽出</p>
              </div>
            </div>
            <div className="flex justify-center text-muted">↓</div>
            <div className="flex items-center gap-4">
              <span className="w-8 h-8 rounded-full bg-accent text-background flex items-center justify-center font-bold text-sm">3</span>
              <div>
                <p className="font-bold">AI判定（Linear Probe分類器）</p>
                <p className="text-muted text-sm">学習済み分類器がAI/手描きを判定</p>
              </div>
            </div>
            <div className="flex justify-center text-muted">↓</div>
            <div className="flex items-center gap-4">
              <span className="w-8 h-8 rounded-full bg-accent text-background flex items-center justify-center font-bold text-sm">4</span>
              <div>
                <p className="font-bold">結果表示</p>
                <p className="text-muted text-sm">AI確率と最終判定を表示</p>
              </div>
            </div>
          </div>
        </section>

        {/* Supported Models */}
        <section className="mb-10">
          <h2 className="text-xl font-bold mb-4 text-accent border-b border-gray-700 pb-2">対応AIモデル</h2>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="bg-gray-800/50 p-3 rounded">
              <p className="font-bold">Illustrious</p>
              <p className="text-muted">高精度で検出</p>
            </div>
            <div className="bg-gray-800/50 p-3 rounded">
              <p className="font-bold">Pony Diffusion</p>
              <p className="text-muted">高精度で検出</p>
            </div>
            <div className="bg-gray-800/50 p-3 rounded">
              <p className="font-bold">SDXL 1.0</p>
              <p className="text-muted">高精度で検出</p>
            </div>
            <div className="bg-gray-800/50 p-3 rounded">
              <p className="font-bold">NovelAI</p>
              <p className="text-muted">対応強化中</p>
            </div>
          </div>
        </section>

        {/* Accuracy Note */}
        <section className="mb-10">
          <h2 className="text-xl font-bold mb-4 text-accent border-b border-gray-700 pb-2">精度について</h2>
          <p className="text-muted leading-relaxed">
            98.35%の精度は、10,000枚以上の画像による検証結果です。
            ただし、以下のケースでは判定が難しい場合があります：
          </p>
          <ul className="list-disc list-inside text-muted mt-3 space-y-1">
            <li>AI生成後に大幅な加筆修正が行われた画像</li>
            <li>極端に低解像度の画像</li>
            <li>学習データに含まれていない新しいAIモデルの画像</li>
          </ul>
        </section>

        {/* Back Link */}
        <div className="text-center pt-6 border-t border-gray-700">
          <Link
            href="/"
            className="inline-block px-6 py-3 bg-accent text-background font-bold rounded hover:opacity-90 transition"
          >
            画像をチェックする →
          </Link>
        </div>
      </main>

      {/* Footer */}
      <footer className="site-footer p-4 mt-12">
        <div className="container mx-auto text-center text-xs text-muted">
          <p><a href="/disclaimer" className="hover:underline">免責事項</a> | &copy; 2025 AIチェッカー All rights reserved. | <a href="mailto:contact@aicheckers.net" className="text-accent hover:underline">お問い合わせ</a></p>
        </div>
      </footer>
    </div>
  );
}
