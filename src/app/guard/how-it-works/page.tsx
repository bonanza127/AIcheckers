import Link from "next/link";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "仕組み | AIイラストガード",
  description: "AIイラストガードがどのようにAI学習から作品を保護するかを解説します。",
};

export default function GuardHowItWorks() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="site-header border-b border-gray-700">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/guard" className="flex items-center gap-1.5 hover:opacity-80">
            <img src="/logo-transparent.png" alt="AI Checkers" className="w-10 h-10" />
            <span className="text-2xl font-bold text-foreground">AIイラストガード</span>
          </Link>
          <nav className="text-sm text-muted">
            <Link href="/guard" className="hover:text-accent">← ガードへ戻る</Link>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12 max-w-3xl">
        <h1 className="text-4xl font-extrabold mb-10 text-center">
          <span className="bg-gradient-to-r from-accent via-purple-400 to-accent bg-clip-text text-transparent">How it works?</span>
        </h1>

        {/* Section 1: Problem Statement */}
        <section className="mb-12">
          <h2 className="text-xl font-bold mb-4 text-foreground border-b border-gray-700 pb-2">無断学習の問題</h2>
          <p className="text-muted leading-relaxed">
            生成AIの急速な発展により、インターネット上に公開されたイラストがAIの学習データとして無断で使用されるケースが増加しています。
          </p>
          <p className="text-muted leading-relaxed mt-3">
            一度学習されたデータは取り消すことが難しく、自分の作風や技術が無断で模倣される可能性があります。
          </p>
          <p className="text-muted leading-relaxed mt-3">
            そこで、このような状況に対策を講じるべく、<span className="text-accent font-bold">AI学習を妨害する防壁技術「MoonKnight」</span>を開発しました。
          </p>
        </section>

        {/* Section 2: Technology Overview */}
        <section className="mb-12">
          <h2 className="text-xl font-bold mb-4 text-foreground border-b border-gray-700 pb-2">MoonKnight V3 について</h2>
          <p className="text-muted leading-relaxed">
            MoonKnightは、画像に人間の目には見えない微細なノイズパターンを埋め込むことで、AI画像生成器の学習効果を大幅に低減する技術です。
          </p>
          <p className="text-muted leading-relaxed mt-3">
            この技術は、<span className="text-accent font-bold">離散ウェーブレット変換（DWT）</span>と<span className="text-accent font-bold">知覚マスキング</span>を組み合わせることで、
            画質を損なうことなく効果的な防壁を構築します。
          </p>
        </section>

        {/* Section 3: Protection Flow */}
        <section className="mb-12">
          <h2 className="text-xl font-bold mb-4 text-foreground border-b border-gray-700 pb-2">防壁構築の仕組み</h2>

          {/* Processing Diagram */}
          <div className="bg-gray-800/30 rounded-lg p-6 mb-6">
            <div className="space-y-6">
              {/* Step 1: DWT Transform */}
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-lg bg-accent/20 border border-accent flex items-center justify-center font-bold text-accent shrink-0">1</div>
                <div className="flex-1">
                  <p className="font-bold text-foreground">DWT変換（離散ウェーブレット変換）</p>
                  <p className="text-muted text-sm mt-1">画像を周波数成分に分解し、LL（低周波）・LH・HL・HH（高周波）のサブバンドに分離します。</p>
                  <div className="mt-2 text-xs font-mono text-accent/70">
                    Image → LL / LH / HL / HH
                  </div>
                </div>
              </div>

              {/* Step 2: Perceptual Masking */}
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-lg bg-accent/20 border border-accent flex items-center justify-center font-bold text-accent shrink-0">2</div>
                <div className="flex-1">
                  <p className="font-bold text-foreground">知覚マスキング（JND計算）</p>
                  <p className="text-muted text-sm mt-1">人間の視覚特性に基づいて「Just Noticeable Difference（JND）」閾値を計算。人間が知覚できないレベルの変化量を特定します。</p>
                </div>
              </div>

              {/* Step 3: Noise Injection */}
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-lg bg-accent/20 border border-accent flex items-center justify-center font-bold text-accent shrink-0">3</div>
                <div className="flex-1">
                  <p className="font-bold text-foreground">学習妨害パターン生成</p>
                  <p className="text-muted text-sm mt-1">高周波帯域（HH/LH/HL）に、AI学習を妨害する特殊なノイズパターンを生成。このパターンは人間には見えませんが、AIの学習プロセスを大きく乱します。</p>
                </div>
              </div>

              {/* Step 4: Signature Embedding */}
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-lg bg-accent/20 border border-accent flex items-center justify-center font-bold text-accent shrink-0">4</div>
                <div className="flex-1">
                  <p className="font-bold text-foreground">署名埋込</p>
                  <p className="text-muted text-sm mt-1">MoonKnight署名を画像に埋め込み、保護済みであることを証明可能にします。</p>
                </div>
              </div>

              {/* Step 5: Inverse DWT */}
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-lg bg-accent/20 border border-accent flex items-center justify-center font-bold text-accent shrink-0">5</div>
                <div className="flex-1">
                  <p className="font-bold text-foreground">逆DWT変換 & 品質検証</p>
                  <p className="text-muted text-sm mt-1">変換を逆適用して画像を再構成。SSIM（構造類似性指標）が0.99以上を維持していることを確認し、画質劣化がないことを保証します。</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 4: Effectiveness */}
        <section className="mb-12">
          <h2 className="text-xl font-bold mb-4 text-foreground border-b border-gray-700 pb-2">防壁の効果</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div className="bg-gray-800/50 p-4 rounded-lg border border-gray-700">
              <p className="font-bold text-foreground">学習効果の低減</p>
              <p className="text-muted text-sm mt-1">保護された画像をAIが学習しても、元の作風を正確に再現することが困難になります。</p>
            </div>
            <div className="bg-gray-800/50 p-4 rounded-lg border border-gray-700">
              <p className="font-bold text-foreground">画質維持</p>
              <p className="text-muted text-sm mt-1">SSIM 0.99以上を維持し、人間の目には変化が分かりません。</p>
            </div>
            <div className="bg-gray-800/50 p-4 rounded-lg border border-gray-700">
              <p className="font-bold text-foreground">保護証明</p>
              <p className="text-muted text-sm mt-1">MoonKnight署名により、画像が保護済みであることを確認できます。</p>
            </div>
            <div className="bg-gray-800/50 p-4 rounded-lg border border-gray-700">
              <p className="font-bold text-foreground">SNS対応</p>
              <p className="text-muted text-sm mt-1">JPEG再圧縮やリサイズに対しても一定の耐性があります。</p>
            </div>
          </div>
        </section>

        {/* Section 5: Limitations */}
        <section className="mb-12">
          <h2 className="text-xl font-bold mb-4 text-foreground border-b border-gray-700 pb-2">制限事項</h2>
          <p className="text-muted leading-relaxed">
            MoonKnightは強力な防壁を提供しますが、以下の点にご注意ください：
          </p>
          <ul className="list-none mt-4 space-y-2 text-muted">
            <li className="flex items-start gap-2">
              <span className="text-gray-500 mt-1">•</span>
              <span>100%の保護を保証するものではありません</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-gray-500 mt-1">•</span>
              <span>既に学習されたデータを取り消すことはできません</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-gray-500 mt-1">•</span>
              <span>極端な画像処理（大幅なリサイズ等）で効果が低下する場合があります</span>
            </li>
          </ul>

          <div className="mt-6 p-4 bg-gray-800/30 rounded-lg border-l-4 border-accent">
            <p className="text-foreground font-medium">予防的な対策として</p>
            <p className="text-muted text-sm mt-2">
              本サービスは、AI学習に対する<span className="text-foreground">予防的な対策</span>を提供します。
              作品を公開する前に保護を適用することで、<span className="text-accent font-bold">将来的なリスクを軽減</span>することができます。
            </p>
          </div>
        </section>

        {/* Back Link */}
        <div className="text-center pt-8 border-t border-gray-700">
          <Link
            href="/guard"
            className="inline-block px-8 py-3 bg-gradient-to-r from-accent to-purple-500 text-white font-bold rounded-lg hover:from-accent/90 hover:to-purple-500/90 transition-all shadow-lg shadow-accent/20"
          >
            作品を保護する →
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
