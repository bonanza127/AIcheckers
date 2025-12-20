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
        <h1 className="text-3xl font-bold mb-8 text-center">How it works?</h1>

        {/* Section 1: Problem Statement */}
        <section className="mb-12">
          <h2 className="text-xl font-bold mb-4 text-accent border-b border-gray-700 pb-2">現状の問題</h2>
          <p className="text-muted leading-relaxed">
            生成AIの急速な発展により、AIの判別が非常に難しくなっています。
          </p>
          <p className="text-muted leading-relaxed mt-3">
            AIそのものの賛否はともかくとして、AI生成画像を自作と偽ってAI禁止のプラットフォームへ投稿したり、不正にマネタイズする行為は明確に非難されるべきです。
          </p>
          <p className="text-muted leading-relaxed mt-3">
            海外にもAIチェッカーの類いはありますが、その多くはDeepfakeを念頭にした実写画像・動画が対象のものがほとんどで、アニメイラストをきちんと判別できるようなものは見当たりませんでした。
          </p>
          <p className="text-muted leading-relaxed mt-3">
            そこでこうした状況に対策を講じるべく、<span className="text-accent font-bold">二次元イラストに特化した日本向けのAIチェッカー</span>を開発しました。
          </p>
        </section>

        {/* Section 2: Vision Transformer */}
        <section className="mb-12">
          <h2 className="text-xl font-bold mb-4 text-accent border-b border-gray-700 pb-2">使用技術：Vision Transformer</h2>
          <p className="text-muted leading-relaxed">
            Vision Transformer（ViT）は、画像を小さなパッチに分割し、それぞれの関係性を解析する革新的なアーキテクチャです。
            従来の畳み込みニューラルネットワーク（CNN）と異なり、画像全体のコンテキストを同時に理解できるため、
            AI生成画像特有の微細なパターンを高精度で検出できます。
          </p>
          <p className="text-muted leading-relaxed mt-3">
            本サービスでは、このViTを<span className="text-accent font-bold">10万枚以上のアニメ画像でファインチューニング</span>し、
            二次元イラストの特徴を専門的に学習させた独自モデル「<span className="text-foreground font-bold">Moonlight</span>」を使用しています。
          </p>
        </section>

        {/* Section 3: Detection Flow - ViT Internal */}
        <section className="mb-12">
          <h2 className="text-xl font-bold mb-4 text-accent border-b border-gray-700 pb-2">検出の仕組み</h2>

          {/* ViT Processing Diagram */}
          <div className="bg-gray-800/30 rounded-lg p-6 mb-6">
            <div className="space-y-6">
              {/* Step 1: Patch Embedding */}
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-lg bg-accent/20 border border-accent flex items-center justify-center font-bold text-accent shrink-0">1</div>
                <div className="flex-1">
                  <p className="font-bold text-foreground">パッチ分割</p>
                  <p className="text-muted text-sm mt-1">画像を196個の小さなパッチ（14×14グリッド）に分割し、それぞれを768次元の埋め込みベクトルに変換します。</p>
                  <div className="mt-2 flex items-center gap-2 text-xs text-muted">
                    <div className="grid grid-cols-4 gap-0.5">
                      {[...Array(16)].map((_, i) => (
                        <div key={i} className="w-3 h-3 bg-accent/30 rounded-sm"></div>
                      ))}
                    </div>
                    <span>→</span>
                    <div className="flex gap-0.5">
                      {[...Array(4)].map((_, i) => (
                        <div key={i} className="w-1 h-6 bg-accent/50 rounded-sm"></div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Step 2: Self-Attention */}
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-lg bg-accent/20 border border-accent flex items-center justify-center font-bold text-accent shrink-0">2</div>
                <div className="flex-1">
                  <p className="font-bold text-foreground">Self-Attention（自己注意機構）</p>
                  <p className="text-muted text-sm mt-1">12層のTransformerブロックを通過。各パッチが他の全パッチとの関係性を計算し、画像全体の文脈を理解します。AI生成画像特有の「均一すぎるテクスチャ」や「不自然なエッジ」がここで検出されます。</p>
                  <div className="mt-2 text-xs font-mono text-accent/70">
                    Layer 1 → Layer 2 → ... → Layer 12
                  </div>
                </div>
              </div>

              {/* Step 3: Attention Map */}
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-lg bg-accent/20 border border-accent flex items-center justify-center font-bold text-accent shrink-0">3</div>
                <div className="flex-1">
                  <p className="font-bold text-foreground">Attention Map生成</p>
                  <p className="text-muted text-sm mt-1">モデルが「どこに注目したか」を可視化。AI画像では特定領域への注目が収束しやすく、人間の作品では注目が分散する傾向があります。</p>
                </div>
              </div>

              {/* Step 4: Classification */}
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-lg bg-accent/20 border border-accent flex items-center justify-center font-bold text-accent shrink-0">4</div>
                <div className="flex-1">
                  <p className="font-bold text-foreground">分類判定</p>
                  <p className="text-muted text-sm mt-1">最終層から出力された768次元の特徴ベクトルを、学習済み分類器に入力。AI/人間の確率スコアを算出します。</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 4: Supported Models */}
        <section className="mb-12">
          <h2 className="text-xl font-bold mb-4 text-accent border-b border-gray-700 pb-2">対応AIモデル</h2>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="bg-gray-800/50 p-4 rounded-lg border border-gray-700">
              <p className="font-bold text-foreground">SDXL</p>
              <p className="text-success text-sm mt-1">攻略完了</p>
            </div>
            <div className="bg-gray-800/50 p-4 rounded-lg border border-gray-700">
              <p className="font-bold text-foreground">Illustrious</p>
              <p className="text-success text-sm mt-1">派生版含め高頻度で検出</p>
            </div>
            <div className="bg-gray-800/50 p-4 rounded-lg border border-gray-700">
              <p className="font-bold text-foreground">Pony Diffusion</p>
              <p className="text-success text-sm mt-1">現在メインのv6まで高頻度で検出</p>
            </div>
            <div className="bg-gray-800/50 p-4 rounded-lg border border-gray-700">
              <p className="font-bold text-foreground">NovelAI</p>
              <p className="text-yellow-400 text-sm mt-1">やや弱い？（7.5割程度）対応強化中</p>
            </div>
          </div>
        </section>

        {/* Section 5: Accuracy */}
        <section className="mb-12">
          <h2 className="text-xl font-bold mb-4 text-accent border-b border-gray-700 pb-2">検出精度について</h2>
          <p className="text-muted leading-relaxed">
            現在主流のモデルやLoRAに対しては<span className="text-accent font-bold">高精度で検出</span>します。
            ただし、以下のケースでは判定が難しい場合があります：
          </p>
          <ul className="list-none mt-4 space-y-2 text-muted">
            <li className="flex items-start gap-2">
              <span className="text-gray-500 mt-1">•</span>
              <span>AI生成後に加筆修正が行われた画像</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-gray-500 mt-1">•</span>
              <span>極端に解像度が低い画像</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-gray-500 mt-1">•</span>
              <span>大幅に改変された独自モデルを使用した画像</span>
            </li>
          </ul>

          <div className="mt-6 p-4 bg-gray-800/30 rounded-lg border-l-4 border-accent">
            <p className="text-foreground font-medium">推定無罪の原則</p>
            <p className="text-muted text-sm mt-2">
              人間の作品がAIと誤判定されることを防ぐため、本サービスでは<span className="text-foreground">確信度が高い場合のみAI判定</span>を下すよう設計しています。
              しかし、それでも100%の精度ではありません。あくまでも判断材料の一つとしてご活用いただければ幸いです。
            </p>
          </div>
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
          <p><a href="/disclaimer" className="hover:underline">免責事項</a> | &copy; 2025 AIチェッカー All rights reserved. | <a href="mailto:contact@aicheckers.net" className="hover:underline">お問い合わせ</a></p>
        </div>
      </footer>
    </div>
  );
}
